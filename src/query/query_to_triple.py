from src.base.embeddings import EmbeddingManager
from src.coregraph.dynamic_memory import DynamicMemory
from src.init.config import Config
from src.init.logger import logger

from typing import List, Dict, Any, Sequence, Tuple


class QueryToTriple:
    def __init__(self, config: Config, dynamic_memory: DynamicMemory, embedding_manager: EmbeddingManager):
        self.config = config
        self.dynamic_memory = dynamic_memory
        self.graph = dynamic_memory.graph_builder.graph
        self.embedding = embedding_manager

    def retrieve_global_triples(self) -> List[Dict[str, Any]]:
        if self.graph is None:
            return []
        triples: List[Dict[str, Any]] = []

        for src, tgt, data in self.graph.edges(data=True):
            relation = data.get("relation_name", "relates_to")
            triple_text = f"{src} {relation} {tgt}"

            timestamp = self._extract_and_format_timestamp(data)

            triple = {
                "src": src,
                "tgt": tgt,
                "relation": relation,
                "triple_text": triple_text,
                "chunk_id": data.get("chunk_id", ""),
                "chunk_ids": data.get("chunk_ids", []),
                "session_id": data.get("session_id", ""),
                "session_ids": data.get("session_ids", []),
                "timestamp": timestamp,
                "src_in_entities": False,
                "tgt_in_entities": False,
            }

            if "session_time" in data:
                triple["session_time"] = data.get("session_time", "")
            if "session_times" in data:
                triple["session_times"] = data.get("session_times", [])
            if "description" in data:
                triple["description"] = data.get("description")

            triples.append(triple)

        logger.info("Found %s triples in global graph", len(triples))
        return triples

    @staticmethod
    def _extract_and_format_timestamp(edge_data: Dict[str, Any]) -> str:
        """
        Same logic style as QueryProcessor._extract_and_format_timestamp:
        - prefer session_time / timestamp
        - if session_times exists, take the latest valid timestamp
        - normalize to YYYY/MM/DD when possible
        """
        timestamp = edge_data.get("session_time", "") or edge_data.get("timestamp", "")

        if not timestamp and "session_times" in edge_data:
            session_times = edge_data.get("session_times", [])
            valid_times = [t for t in session_times if t]
            if valid_times:
                timestamp = sorted(valid_times)[-1]

        if timestamp:
            try:
                date_part = timestamp.split()[0] if " " in timestamp else timestamp
                formatted_timestamp = date_part.replace("-", "/")
                return formatted_timestamp
            except Exception as e:
                logger.warning("Failed to format timestamp '%s': %s", timestamp, e)
                return ""
        return ""

    @staticmethod
    def _extract_session_ids(triple_or_edge: Dict[str, Any]) -> List[str]:
        session_ids = triple_or_edge.get("session_ids", []) or []
        if not isinstance(session_ids, list):
            session_ids = [session_ids]

        session_id = triple_or_edge.get("session_id", "")
        if session_id and session_id not in session_ids:
            session_ids.append(session_id)

        return [str(session) for session in session_ids if session]

    async def query_to_triples(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        triples = self.retrieve_global_triples()
        if not triples:
            logger.error("No triples found in graph for global retrieval")
            return []

        if not self.embedding:
            logger.error("Embedding manager not initialized")
            return []

        triple_embeddings = []
        texts_to_embed = []
        indices_to_compute = []

        for i, triple in enumerate(triples):
            src = triple["src"]
            tgt = triple["tgt"]

            if self.graph.has_edge(src, tgt):
                edge_data = self.graph.edges[src, tgt]
                if "embedding" in edge_data:
                    triple_embeddings.append(edge_data["embedding"])
                else:
                    triple_embeddings.append(None)
                    texts_to_embed.append(triple["triple_text"])
                    indices_to_compute.append(i)
            else:
                triple_embeddings.append(None)
                texts_to_embed.append(triple["triple_text"])
                indices_to_compute.append(i)

        if texts_to_embed:
            computed_embeddings = await self.embedding.get_embeddings(texts_to_embed)
            if computed_embeddings and len(computed_embeddings) == len(texts_to_embed):
                for idx, computed_emb in zip(indices_to_compute, computed_embeddings):
                    triple_embeddings[idx] = computed_emb
            else:
                logger.error("Failed to compute embeddings")
                return []
        question_embedding = await self.embedding.get_embeddings([question], need_tensor=True)
        triple_embeddings = self.embedding.transfer_to_tensor(triple_embeddings)
        similarities = self.embedding.cosine_similarity_tensor(
            question_embedding, triple_embeddings
        )

        for i, triple in enumerate(triples):
            similarity_score = float(similarities[0][i])
            triple["similarity_score"] = similarity_score
            triple["final_score"] = similarity_score

        triples.sort(key=lambda x: x["final_score"], reverse=True)
        selected = triples[:top_k]
        logger.info("Selected %s global triples", len(selected))
        return selected

    def score_sessions_from_selected_triples(
        self,
        triples: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        session_scores: Dict[str, float] = {}
        for triple in triples:
            score = float(triple.get("similarity_score", triple.get("final_score", 0.0)))
            for session_id in self._extract_session_ids(triple):
                if not session_id:
                    continue
                if session_id not in session_scores:
                    session_scores[session_id] = score

        ranked_scores = dict(sorted(session_scores.items(), key=lambda x: x[1], reverse=True))
        logger.info("Ranked %s sessions from selected triples", len(ranked_scores))
        return ranked_scores

    def merge_session_candidates(
        self,
        summary_sessions: Dict[str, float],
        query_to_triple_sessions: Dict[str, float],
        top_summary_k: int = 10,
        top_query_k: int = 8,
        final_top_k: int = 15,
    ) -> Tuple[List[str], Dict[str, float]]:
        ranked_summary_sessions = sorted(summary_sessions.items(), key=lambda x: x[1], reverse=True)
        top_summary_sessions = ranked_summary_sessions[:top_summary_k]
        top_query_to_triple_sessions = list(query_to_triple_sessions.items())[:top_query_k]

        merged_scores: Dict[str, float] = {}
        for session_id, summary_score in top_summary_sessions:
            merged_scores[session_id] = float(summary_score)

        for session_id, query_score in top_query_to_triple_sessions:
            if session_id not in merged_scores:
                merged_scores[session_id] = 0.0
            merged_scores[session_id] = 0.7 * merged_scores[session_id] + 0.3 * float(query_score)

        ranked_merged_scores = dict(sorted(merged_scores.items(), key=lambda x: x[1], reverse=True))
        final_sessions = list(ranked_merged_scores.keys())[:final_top_k]
        final_scores = {session_id: ranked_merged_scores[session_id] for session_id in final_sessions}
        logger.info(f"Retrieved {len(final_sessions)} final sessions")
        return final_sessions, final_scores

    def merger_session_candidates(
        self,
        summary_sessions: Dict[str, float],
        query_to_triple_sessions: Dict[str, float],
        top_summary_k: int = 10,
        top_query_k: int = 8,
        final_top_k: int = 15,
    ) -> Tuple[List[str], Dict[str, float]]:
        return self.merge_session_candidates(
            summary_sessions=summary_sessions,
            query_to_triple_sessions=query_to_triple_sessions,
            top_summary_k=top_summary_k,
            top_query_k=top_query_k,
            final_top_k=final_top_k,
        )

    async def retrieve_top_sessions(
        self,
        question: str,
        summary_sessions: Dict[str, float],
        top_k_triples: int = 15,
        top_summary_k: int = 10,
        top_query_k: int = 8,
        final_top_k: int = 15,
    ) -> Tuple[List[str], Dict[str, float]]:
        triples = self.retrieve_global_triples()
        if not triples:
            logger.warning("No global triples available for session retrieval")
            return [], {}

        selected_triples = await self.query_to_triples(
            question=question,
            top_k=top_k_triples,
        )
        if not selected_triples:
            logger.warning("No triples selected from query_to_triples")
            return [], {}

        query_to_triple_sessions = self.score_sessions_from_selected_triples(selected_triples)
        return self.merger_session_candidates(
            summary_sessions=summary_sessions,
            query_to_triple_sessions=query_to_triple_sessions,
            top_summary_k=top_summary_k,
            top_query_k=top_query_k,
            final_top_k=final_top_k,
        )

    def get_top_k_sessions(self, session_scores: Sequence[Tuple[str, float]], top_k: int) -> List[Dict[str, Any]]:
        return [
            {"session_id": session_id, "score": score}
            for session_id, score in list(session_scores)[:top_k]
        ]

    def merge_session_candidates_rrf(
            self,
            summary_sessions: Dict[str, float],
            query_to_triple_sessions: Dict[str, float],
            top_summary_k: int = 15,
            top_query_k: int = 10,
            final_top_k: int = 15,
            rrf_k: int = 60,
            summary_weight: float = 0.6,
            query_to_triple_weight: float = 0.4,
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Merge session candidates from summary retrieval and query-to-triple retrieval
        using weighted Reciprocal Rank Fusion (RRF).

        RRF score(session) =
            summary_weight * 1 / (rrf_k + rank_summary)
            + query_to_triple_weight * 1 / (rrf_k + rank_query)

        Notes:
        - rank starts from 1
        - only top_summary_k and top_query_k candidates are considered
        - higher RRF score is better
        """

        ranked_summary_sessions = sorted(
            summary_sessions.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_summary_k]

        ranked_query_sessions = sorted(
            query_to_triple_sessions.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_query_k]

        rrf_scores: Dict[str, float] = {}

        for rank, (session_id, _) in enumerate(ranked_summary_sessions, start=1):
            rrf_scores[session_id] = rrf_scores.get(session_id, 0.0) + (
                    summary_weight * (1.0 / (rrf_k + rank))
            )

        for rank, (session_id, _) in enumerate(ranked_query_sessions, start=1):
            rrf_scores[session_id] = rrf_scores.get(session_id, 0.0) + (
                    query_to_triple_weight * (1.0 / (rrf_k + rank))
            )

        ranked_rrf_scores = dict(
            sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        )

        final_sessions = list(ranked_rrf_scores.keys())[:final_top_k]
        final_scores = {
            session_id: ranked_rrf_scores[session_id]
            for session_id in final_sessions
        }

        logger.info("Retrieved %s final sessions with RRF merging", len(final_sessions))
        return final_sessions, final_scores

