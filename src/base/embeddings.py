from typing import List
import os
import asyncio

import torch
from openai import AsyncOpenAI

from src.init.logger import logger


class EmbeddingManager:
    def __init__(self, config=None):
        if config:
            self.model_name = getattr(config, "model", "baai/bge-m3")
            self.dimensions = getattr(config, "dimensions", 1024)
            self.max_token_size = getattr(config, "max_token_size", 8192)
            self.embed_batch_size = getattr(config, "embed_batch_size", 100)
            self.embedding_func_max_async = getattr(config, "embedding_func_max_async", 10)
            self.api_key = getattr(config, "api_key", None) or os.getenv("OPENROUTER_API_KEY")
        else:
            self.model_name = "baai/bge-m3"
            self.dimensions = 1024
            self.max_token_size = 8192
            self.embed_batch_size = 100
            self.embedding_func_max_async = 10
            self.api_key = os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError("Missing OPENROUTER_API_KEY")

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Using device for similarity calculation: {self.device}")
        logger.info(f"Using OpenRouter embedding model: {self.model_name}")

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=texts,
            )

            data = sorted(response.data, key=lambda x: x.index)
            return [item.embedding for item in data]

        except Exception as e:
            logger.error(f"Failed to call OpenRouter embedding API: {e}")
            raise

    async def get_embeddings(
        self,
        texts: List[str],
        need_tensor: bool = False,
    ):
        if not texts:
            empty_tensor = torch.empty(
                (0, self.dimensions),
                dtype=torch.float32,
                device=self.device,
            )
            return empty_tensor if need_tensor else []

        batches = [
            texts[i : i + self.embed_batch_size]
            for i in range(0, len(texts), self.embed_batch_size)
        ]

        semaphore = asyncio.Semaphore(self.embedding_func_max_async)

        async def run_batch(batch: List[str]) -> List[List[float]]:
            async with semaphore:
                return await self._embed_batch(batch)

        batch_results = await asyncio.gather(
            *(run_batch(batch) for batch in batches)
        )

        embeddings = [
            embedding
            for batch in batch_results
            for embedding in batch
        ]

        embeddings_tensor = torch.tensor(
            embeddings,
            dtype=torch.float32,
            device=self.device,
        )

        logger.info(f"Embedding tensor shape: {embeddings_tensor.shape}")
        logger.info(f"Embedding tensor device: {embeddings_tensor.device}")

        if need_tensor:
            return embeddings_tensor

        return embeddings_tensor.cpu().tolist()

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        try:
            vec1_tensor = torch.tensor(vec1, dtype=torch.float32, device=self.device)
            vec2_tensor = torch.tensor(vec2, dtype=torch.float32, device=self.device)

            eps = 1e-8

            dot_product = torch.dot(vec1_tensor, vec2_tensor)
            norm_vec1 = torch.linalg.norm(vec1_tensor).clamp_min(eps)
            norm_vec2 = torch.linalg.norm(vec2_tensor).clamp_min(eps)

            similarity = dot_product / (norm_vec1 * norm_vec2)

            return similarity.item()

        except Exception as e:
            logger.error(f"Failed to calculate cosine similarity: {e}")
            return 0.0

    def cosine_similarity_tensor(self, vec1, vec2):
        try:
            eps = 1e-8

            if not isinstance(vec1, torch.Tensor):
                vec1 = torch.tensor(vec1, dtype=torch.float32, device=self.device)
            else:
                vec1 = vec1.to(self.device)

            if not isinstance(vec2, torch.Tensor):
                vec2 = torch.tensor(vec2, dtype=torch.float32, device=self.device)
            else:
                vec2 = vec2.to(self.device)

            if vec1.dim() == 1:
                vec1 = vec1.unsqueeze(0)

            if vec2.dim() == 1:
                vec2 = vec2.unsqueeze(0)

            norm1 = torch.linalg.norm(vec1, dim=1, keepdim=True).clamp_min(eps)
            norm2 = torch.linalg.norm(vec2, dim=1, keepdim=True).clamp_min(eps)

            vec1_normed = vec1 / norm1
            vec2_normed = vec2 / norm2

            similarity = torch.matmul(vec1_normed, vec2_normed.T)

            logger.info(f"similarity shape: {similarity.shape}")

            return similarity

        except Exception as e:
            logger.error(f"Failed to calculate similarity with torch: {e}")
            return torch.empty((0, 0), dtype=torch.float32, device=self.device)

    def transfer_to_tensor(self, embeddings: List[List[float]]) -> torch.Tensor:
        return torch.tensor(
            embeddings,
            dtype=torch.float32,
            device=self.device,
        )

    def batch_cosine_similarity(
        self,
        query_vec: List[float],
        candidate_vecs: List[List[float]],
    ) -> List[float]:
        try:
            if not candidate_vecs:
                return []

            eps = 1e-8

            query = torch.tensor(
                query_vec,
                dtype=torch.float32,
                device=self.device,
            )

            candidates = self.transfer_to_tensor(candidate_vecs)

            query_norm = query / torch.linalg.norm(query).clamp_min(eps)

            candidates_norm = candidates / torch.linalg.norm(
                candidates,
                dim=1,
                keepdim=True,
            ).clamp_min(eps)

            similarities = torch.matmul(candidates_norm, query_norm)

            return similarities.cpu().tolist()

        except Exception as e:
            logger.error(f"Failed to calculate batch similarity with torch: {e}")
            return [self.cosine_similarity(query_vec, vec) for vec in candidate_vecs]