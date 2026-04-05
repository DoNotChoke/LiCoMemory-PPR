from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime

from src.base.embeddings import EmbeddingManager
from src.init.config import Config
from src.init.logger import logger


class TripleReranker:
    def __init__(self, config: Config) -> None:
        self.config = config

        self.use_cognirank = getattr(self.config.retriever, "enable_CogniRank", False)
        self.rerank_k = getattr(self.config.retriever, "rerank_k", 0.5)

        # Weights used in reranking
        self.weights = {
            "similarity": 1.0,   # triple semantic similarity
            "summary": 0.0,      # session/summary relevance
            "ppr": 0.0           # graph propagation relevance
        }

        self.embedding_manager = EmbeddingManager(config.embedding)
        self._configure_weights()

        strategy_name = (
            "CogniRank (hierarchical temporal-semantic + PPR)"
            if self.use_cognirank
            else "SimpleRank (weighted + PPR)"
        )
        logger.info(f"Triple Reranker initialized with strategy {strategy_name}")
        if self.use_cognirank:
            logger.info(f"  CogniRank parameters: k={self.rerank_k}")
        logger.info(f"  Active weights: {self.weights}")

    def _configure_weights(self):
        """
        Configure weights for similarity / summary / ppr.

        Recommended config:
            retriever:
              enable_summary: True
              summary_weight: 0.25
              ppr_weight: 0.25

        Then similarity weight = 1 - summary_weight - ppr_weight.
        """
        enable_summary = getattr(self.config.retriever, "enable_summary", False)
        summary_weight = getattr(self.config.retriever, "summary_weight", 0.0) if enable_summary else 0.0
        ppr_weight = getattr(self.config.retriever, "ppr_weight", 0.0)

        # clamp
        summary_weight = max(0.0, float(summary_weight))
        ppr_weight = max(0.0, float(ppr_weight))

        remaining = 1.0 - summary_weight - ppr_weight
        if remaining < 0.0:
            logger.warning(
                f"summary_weight ({summary_weight}) + ppr_weight ({ppr_weight}) > 1.0. "
                f"Will normalize weights."
            )
            self.weights["summary"] = summary_weight
            self.weights["ppr"] = ppr_weight
            self.weights["similarity"] = 0.0
            self._normalize_weights(target_total=1.0)
        else:
            self.weights["summary"] = summary_weight
            self.weights["ppr"] = ppr_weight
            self.weights["similarity"] = remaining

        logger.info(
            "Reranking weights configured: "
            f"similarity={self.weights['similarity']:.2f}, "
            f"summary={self.weights['summary']:.2f}, "
            f"ppr={self.weights['ppr']:.2f}"
        )

    def rerank_triples(
        self,
        triples: List[Dict[str, Any]],
        summaries: Optional[List[Dict[str, Any]]] = None,
        summary_rankings: Optional[Dict[str, float]] = None,
        question_time: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if not triples:
            return triples

        strategy = "CogniRank" if self.use_cognirank else "SimpleRank"
        logger.info(f"Reranking {len(triples)} triples using strategy {strategy}")

        if summary_rankings:
            logger.info(f"📊 Available summary rankings for {len(summary_rankings)} sessions:")
            sorted_rankings = sorted(summary_rankings.items(), key=lambda x: x[1], reverse=True)
            for session_id, score in sorted_rankings[:5]:
                logger.info(f"   Session {session_id}: {score:.3f}")
        else:
            logger.info("📊 No summary rankings available")

        triple_session_ids = set()
        for triple in triples[:5]:
            session_id = triple.get("session_id", "NO_SESSION_ID")
            triple_session_ids.add(session_id)
            logger.debug(f"Triple '{triple.get('triple_text', '')[:30]}' has session_id: {session_id}")

        logger.info(f"🔍 Found {len(triple_session_ids)} unique session_ids in triples: {list(triple_session_ids)}")

        if self.use_cognirank:
            self._apply_cognirank(triples, summary_rankings, question_time)
        else:
            # FIX: code cũ truyền nhầm question_time vào vị trí summary_rankings
            self._apply_simplerank(triples, summaries, summary_rankings)

        triples.sort(key=lambda x: x["reranked_score"], reverse=True)

        logger.info(f"Top reranked triples ({strategy}):")
        for i, triple in enumerate(triples[:5]):
            if self.use_cognirank:
                Ss = triple.get("session_similarity", 0.0)
                St = triple.get("triple_similarity", 0.0)
                Sp = triple.get("ppr_score_used", 0.0)
                S_sem = triple.get("semantic_score", 0.0)
                w_t = triple.get("temporal_weight", 1.0)
                logger.info(
                    f"  {i + 1}. {triple.get('triple_text', '')} "
                    f"(Ss={Ss:.3f}, St={St:.3f}, Sp={Sp:.3f}, "
                    f"S_sem={S_sem:.3f}, w(Δτ)={w_t:.3f}, R={triple['reranked_score']:.3f})"
                )
            else:
                summary_score = triple.get("summary_bonus", 0.0)
                similarity_score = triple.get("similarity_score", triple.get("original_score", 0.0))
                ppr_score = triple.get("ppr_score", 0.0)
                logger.info(
                    f"  {i + 1}. {triple.get('triple_text', '')} "
                    f"(similarity={similarity_score:.3f}, "
                    f"summary={summary_score:.3f}, "
                    f"ppr={ppr_score:.3f}, "
                    f"reranked={triple['reranked_score']:.3f})"
                )

        return triples

    def _apply_simplerank(
        self,
        triples: List[Dict[str, Any]],
        summaries: Optional[List[Dict[str, Any]]] = None,
        summary_rankings: Optional[Dict[str, float]] = None
    ):
        """
        SimpleRank:
            R = w_sim * St + w_sum * Ss + w_ppr * Sp
        """
        has_summary = self.weights["summary"] > 0 and summary_rankings is not None and summaries is not None
        has_ppr = self.weights["ppr"] > 0

        for triple in triples:
            St = triple.get("similarity_score", triple.get("final_score", 0.0))
            Sp = triple.get("ppr_score", 0.0) if has_ppr else 0.0

            weighted_score = self.weights["similarity"] * St
            summary_bonus = 0.0

            if has_summary:
                summary_bonus = self._calculate_summary_bonus(triple, summary_rankings)
                weighted_score += self.weights["summary"] * summary_bonus
                triple["summary_bonus"] = summary_bonus
            else:
                triple["summary_bonus"] = 0.0

            if has_ppr:
                weighted_score += self.weights["ppr"] * Sp

            triple["triple_similarity"] = St
            triple["ppr_score_used"] = Sp
            triple["reranked_score"] = weighted_score
            triple["original_score"] = triple.get("final_score", 0.0)
            triple["final_score"] = weighted_score

            logger.debug(
                f"SimpleRank for {triple.get('triple_text', '')[:40]}: "
                f"St={St:.3f}*{self.weights['similarity']:.2f} + "
                f"Ss={summary_bonus:.3f}*{self.weights['summary']:.2f} + "
                f"Sp={Sp:.3f}*{self.weights['ppr']:.2f} = "
                f"{weighted_score:.3f}"
            )

    def _apply_cognirank(
        self,
        triples: List[Dict[str, Any]],
        summary_rankings: Optional[Dict[str, float]] = None,
        question_time: Optional[str] = None
    ):
        """
        CogniRank:
            S_sem = w_sum * Ss + w_sim * St + w_ppr * Sp
            R_t   = S_sem * exp(- (Δτ / median_gap)^k )
        """
        time_gaps = []
        missing_timestamps = 0

        if question_time:
            question_time = question_time.split()[0] if " " in question_time else question_time
            question_time = question_time[:10] if len(question_time) >= 10 else question_time

        logger.info(f"⏰ Query time (normalized): {question_time if question_time else 'NOT PROVIDED'}")

        for triple in triples:
            triple_time = triple.get("timestamp", "")
            if question_time and triple_time:
                gap = self._calculate_time_gap_days(question_time, triple_time)
                time_gaps.append(gap)
            else:
                if not triple_time:
                    missing_timestamps += 1
                time_gaps.append(0.0)

        if missing_timestamps > 0:
            logger.warning(f"⚠️  {missing_timestamps}/{len(triples)} triples are missing timestamps!")

        median_gap = np.median(time_gaps) if time_gaps else 1.0
        median_gap = max(float(median_gap), 1.0)

        logger.info(
            f"📊 CogniRank temporal analysis: median gap τ̂={median_gap:.1f} days, "
            f"time gaps range: [{min(time_gaps):.1f}, {max(time_gaps):.1f}] days"
        )
        logger.info(
            f"📊 CogniRank semantic weights: "
            f"summary={self.weights['summary']:.2f}, "
            f"triple={self.weights['similarity']:.2f}, "
            f"ppr={self.weights['ppr']:.2f}"
        )

        for i, triple in enumerate(triples):
            St = triple.get("similarity_score", triple.get("final_score", 0.0))
            Sp = triple.get("ppr_score", 0.0)

            session_id = triple.get("session_id", "")
            Ss = summary_rankings.get(session_id, 0.0) if summary_rankings else 0.0
            if Ss == 0.0:
                Ss = 0.01

            S_sem = (
                self.weights["summary"] * Ss +
                self.weights["similarity"] * St +
                self.weights["ppr"] * Sp
            )

            delta_tau = time_gaps[i]
            if delta_tau >= 0:
                w_t = np.exp(-((delta_tau / median_gap) ** self.rerank_k))
            else:
                w_t = 1.0

            R_t = S_sem * w_t

            triple["session_similarity"] = Ss
            triple["triple_similarity"] = St
            triple["ppr_score_used"] = Sp
            triple["semantic_score"] = S_sem
            triple["temporal_weight"] = w_t
            triple["time_gap_days"] = delta_tau
            triple["reranked_score"] = R_t
            triple["original_score"] = triple.get("final_score", 0.0)
            triple["final_score"] = R_t

            logger.debug(
                f"CogniRank for {triple.get('triple_text', '')[:40]}: "
                f"Ss={Ss:.3f}, St={St:.3f}, Sp={Sp:.3f}, S_sem={S_sem:.3f}, "
                f"Δτ={delta_tau:.1f}days, w(Δτ)={w_t:.3f}, R={R_t:.3f}"
            )

    def _calculate_time_gap_days(self, time1: str, time2: str) -> float:
        try:
            date1 = datetime.strptime(time1, "%Y/%m/%d")
            date2 = datetime.strptime(time2, "%Y/%m/%d")
            delta = abs((date1 - date2).days)
            return float(delta)
        except Exception as e:
            logger.warning(f"Failed to parse dates '{time1}' and '{time2}': {e}")
            return 0.0

    def _calculate_summary_bonus(self, triple: Dict[str, Any], summary_rankings: Dict[str, float]) -> float:
        session_id = triple.get("session_id", "")
        if not session_id:
            logger.debug(f"Triple has no session_id: {triple.get('triple_text', '')[:50]}")
            return 0.0

        ranking_score = summary_rankings.get(session_id, 0.0)
        logger.debug(f"Summary bonus for session {session_id}: {ranking_score:.4f}")
        return ranking_score

    def get_top_k_triples(
        self,
        reranked_triples: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = getattr(self.config.retriever, "top_k_triples", 10)
        return reranked_triples[:top_k]

    def create_summary_rankings(
        self,
        summaries: List[Dict[str, Any]],
        question_embedding=None,
        summary_embeddings=None
    ) -> Dict[str, float]:
        if not summaries:
            return {}

        rankings = {}

        if question_embedding is not None and summary_embeddings is not None and hasattr(self, "embedding_manager"):
            for i, summary in enumerate(summaries):
                session_id = summary.get("session_id", "")
                if session_id and i < len(summary_embeddings):
                    similarity = self.embedding_manager.cosine_similarity(question_embedding, summary_embeddings[i])
                    rankings[session_id] = float(similarity)

            logger.info(f"Created similarity-based summary rankings for {len(rankings)} sessions")
        else:
            for i, summary in enumerate(summaries):
                session_id = summary.get("session_id", "")
                if session_id:
                    score = 0.9 * (0.9 ** i)
                    rankings[session_id] = score

            logger.info(f"Created rank-based summary rankings for {len(rankings)} sessions")

        return rankings

    def add_reranking_factor(self, factor_name: str, weight: float):
        self.weights[factor_name] = weight
        self._normalize_weights(target_total=1.0)
        logger.info(f"Added reranking factor '{factor_name}' with weight {weight}")

    def _normalize_weights(self, target_total: float = 1.0):
        total_weight = sum(self.weights.values())
        if total_weight <= 0:
            self.weights = {"similarity": 1.0, "summary": 0.0, "ppr": 0.0}
            return

        factor = target_total / total_weight
        for key in self.weights:
            self.weights[key] *= factor

        logger.info(f"Normalized reranking weights: {self.weights}")
