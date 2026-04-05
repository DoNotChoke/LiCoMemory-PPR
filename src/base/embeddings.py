from typing import List, Dict, Any
import torch

from src.init.logger import logger


class EmbeddingManager:
    def __init__(self, config=None):
        if config:
            self.model_name = config.model
            self.dimensions = config.dimensions
            self.max_token_size = config.max_token_size
            self.embed_batch_size = config.embed_batch_size
            self.embedding_func_max_async = config.embedding_func_max_async

        else:
            self.model_name = "BAAI/bge-m3"
            self.max_token_size = 1024
            self.embed_batch_size = 100
            self.embedding_func_max_async = 10

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device for embedding calculation: {self.device}")

        try:
            from sentence_transformers import SentenceTransformer

            self.client = SentenceTransformer(self.model_name, device=str(self.device))
            logger.info(f"HuggingFace embedding model loaded on device: {self.device}")
        except Exception as e:
            logger.warning(f"Failed to load HuggingFace embedding model: {e}")

    async def get_embeddings(self, texts: List[str], need_tensor=False) -> List[List[float]]:
        embeddings = self.client.encode(texts, convert_to_tensor=True, device=str(self.device))
        logger.info(f"Self device: {self.device}")
        logger.info(f"Embedding: {embeddings.shape}")
        logger.info(f"Embedding device: {embeddings.device}")
        if need_tensor:
            return embeddings
        return embeddings.cpu().tolist() if isinstance(embeddings, torch.Tensor) else embeddings.tolist()

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        try:
            vec1_tensor = torch.tensor(vec1, dtype=torch.float32, device=self.device)
            vec2_tensor = torch.tensor(vec2, dtype=torch.float32, device=self.device)

            dot_product = torch.dot(vec1_tensor, vec2_tensor)
            norm_vec1 = torch.linalg.norm(vec1_tensor)
            norm_vec2 = torch.linalg.norm(vec2_tensor)

            similarity = dot_product / (norm_vec1 * norm_vec2)

            return similarity.item()
        except Exception as e:
            logger.error(f"Failed to calculate cosine similarity: {e}")
            return 0,0

    def cosine_similarity_tensor(self, vec1, vec2):
        """Calculate cosine similarity between two tensors with shape (n, d) and (m, d) using torch on GPU."""
        try:
            eps = 1e-8
            norm1 = torch.linalg.norm(vec1, dim=1, keepdim=True).clamp_min(eps)
            norm2 = torch.linalg.norm(vec2, dim=1, keepdim=True).clamp_min(eps)
            vec1_normed = vec1 / norm1
            vec2_normed = vec2 / norm2
            similarity = torch.matmul(vec1_normed, vec2_normed.T)
            logger.info(f"similarity shape: {similarity.shape}")
            return similarity
        except Exception as e:
            logger.error(f"Failed to calculate similarity with torch: {e}")
            return 0.0

    def transfer_to_tensor(self, embeddings: List[List[float]]) -> torch.Tensor:
        return torch.tensor(embeddings, dtype=torch.float32, device=self.device)

    def batch_cosine_similarity(self, query_vec: List[float], candidate_vecs: List[List[float]]) -> List[float]:
        try:
            query = torch.tensor(query_vec, dtype=torch.float32, device=self.device)
            candidates = self.transfer_to_tensor(candidate_vecs)

            query_norm = query / torch.linalg.norm(query)
            candidates_norm = candidates / torch.linalg.norm(candidates, dim=1, keepdim=True)

            similarities = torch.matmul(candidates_norm, query_norm)

            return similarities.cpu().tolist()
        except Exception as e:
            logger.error(f"Failed to calculate batch similarity with torch: {e}")
            # Fallback to individual calculations
            return [self.cosine_similarity(query_vec, vec) for vec in candidate_vecs]