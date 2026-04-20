from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from src.base.embeddings import EmbeddingManager
from src.coregraph.dynamic_memory import DynamicMemory
from src.init.config import Config
from src.init.logger import logger


class PPRRetriever:
    def __init__(
        self,
        config: Config,
        dynamic_memory: Optional[DynamicMemory] = None,
        global_graph=None,
        embedding_manager: Optional[EmbeddingManager] = None,
    ):
        self.config = config
        self.dynamic_memory = dynamic_memory
        self.global_graph = (
            global_graph
            if global_graph is not None
            else getattr(getattr(dynamic_memory, "graph_builder", None), "graph", None)
        )
        self.embedding_manager = (
            embedding_manager
            or getattr(dynamic_memory, "embedding_manager", None)
            or EmbeddingManager(config.embedding)
        )

    def retrieve_all_triples(self) -> List[Dict[str, Any]]:
        if self.global_graph is None:
            return []

        triples: List[Dict[str, Any]] = []
        for src, tgt, data in self.global_graph.edges(data=True):
            triples.append(self._edge_to_triple(src, tgt, data))
        return triples

    def sessions_to_triples(self, sessions: List[str]) -> List[Dict[str, Any]]:
        if self.global_graph is None or not sessions:
            return []

        session_set = {str(session_id) for session_id in sessions if str(session_id).strip()}
        triples: List[Dict[str, Any]] = []

        for src, tgt, data in self.global_graph.edges(data=True):
            edge_session_ids = self._extract_session_ids(data)
            matched_sessions = [session_id for session_id in edge_session_ids if session_id in session_set]
            if not matched_sessions:
                continue

            triple = self._edge_to_triple(src, tgt, data)
            triple["matched_session_ids"] = matched_sessions
            triple["occurrence_count"] = max(len(matched_sessions), 1)
            triples.append(triple)

        logger.info("Collected %s triples from %s sessions", len(triples), len(session_set))
        return triples

    async def query_to_triples(
        self,
        question: str,
        triples: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        if not question or not triples:
            return []

        triple_embeddings: List[Optional[List[float]]] = []
        texts_to_embed: List[str] = []
        indices_to_compute: List[int] = []

        for index, triple in enumerate(triples):
            cached_embedding = triple.get("embedding")
            if cached_embedding is not None:
                triple_embeddings.append(cached_embedding)
                continue

            src = triple.get("src", "")
            tgt = triple.get("tgt", "")
            if self.global_graph is not None and self.global_graph.has_edge(src, tgt):
                edge_data = self.global_graph.edges[src, tgt]
                if "embedding" in edge_data:
                    triple_embeddings.append(edge_data["embedding"])
                    continue

            triple_embeddings.append(None)
            texts_to_embed.append(triple.get("triple_text", ""))
            indices_to_compute.append(index)

        if texts_to_embed:
            computed_embeddings = await self.embedding_manager.get_embeddings(texts_to_embed)
            if not computed_embeddings or len(computed_embeddings) != len(texts_to_embed):
                logger.error("Failed to compute triple embeddings for query scoring")
                return []
            for index, embedding in zip(indices_to_compute, computed_embeddings):
                triple_embeddings[index] = embedding
                triples[index]["embedding"] = embedding

        if any(embedding is None for embedding in triple_embeddings):
            logger.error("Missing triple embeddings after scoring preparation")
            return []

        question_embedding = await self.embedding_manager.get_embeddings([question], need_tensor=True)
        triple_embedding_tensor = self.embedding_manager.transfer_to_tensor(triple_embeddings)  # type: ignore[arg-type]
        similarities = self.embedding_manager.cosine_similarity_tensor(
            question_embedding,
            triple_embedding_tensor,
        )

        scored_triples: List[Dict[str, Any]] = []
        for index, triple in enumerate(triples):
            scored_triple = triple.copy()
            scored_triple["similarity_score"] = float(similarities[0][index])
            scored_triple["final_score"] = scored_triple["similarity_score"]
            scored_triples.append(scored_triple)

        scored_triples.sort(key=self._get_similarity_score, reverse=True)
        return scored_triples[:top_k]

    def build_local_graph(
        self,
        triples: List[Dict[str, Any]],
        similarity_threshold: Optional[float] = None,
    ) -> nx.MultiDiGraph:
        threshold = similarity_threshold
        if threshold is None:
            threshold = getattr(self.config.retriever, "synonym_threshold", None)
        if threshold is None:
            threshold = getattr(self.config.graph, "entity_merge_threshold", 0.85)

        local_graph = nx.MultiDiGraph()

        for triple in triples:
            src = triple.get("src", "")
            tgt = triple.get("tgt", "")
            if not src or not tgt:
                continue

            self._ensure_node_with_synonym_edges(local_graph, src, float(threshold))
            self._ensure_node_with_synonym_edges(local_graph, tgt, float(threshold))
            self._add_or_increment_relation_edge(local_graph, triple)

        logger.info(
            "Built local graph with %s nodes and %s edges",
            local_graph.number_of_nodes(),
            local_graph.number_of_edges(),
        )
        return local_graph

    def assign_reset_probabilities(
        self,
        local_graph: nx.MultiDiGraph,
        seed_triples: List[Dict[str, Any]],
        symbolic_anchor_bonus: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        node_to_scores: Dict[str, List[float]] = defaultdict(list)
        symbolic_anchor_bonus = symbolic_anchor_bonus or {}

        for triple in seed_triples:
            similarity_score = self._get_similarity_score(triple)
            src = triple.get("src", "")
            tgt = triple.get("tgt", "")
            if src in local_graph:
                node_to_scores[src].append(similarity_score)
            if tgt in local_graph:
                node_to_scores[tgt].append(similarity_score)

        personalization: Dict[str, float] = {}
        for node in local_graph.nodes():
            scores = node_to_scores.get(node, [])
            average_score = sum(scores) / len(scores) if scores else 0.0
            anchor_bonus = float(symbolic_anchor_bonus.get(node, 0.0))
            local_graph.nodes[node]["seed_score"] = average_score
            local_graph.nodes[node]["symbolic_anchor_bonus"] = anchor_bonus
            personalization[node] = average_score + anchor_bonus

        total = sum(personalization.values())
        if total <= 0:
            uniform = 1.0 / max(local_graph.number_of_nodes(), 1)
            for node in local_graph.nodes():
                local_graph.nodes[node]["reset_probability"] = uniform
                personalization[node] = uniform
            return personalization

        for node, score in personalization.items():
            normalized_score = score / total
            local_graph.nodes[node]["reset_probability"] = normalized_score
            personalization[node] = normalized_score

        return personalization

    async def symbolic_anchoring(
        self,
        local_graph: nx.MultiDiGraph,
        entities: List[str],
    ) -> Dict[str, float]:
        if local_graph.number_of_nodes() == 0 or not entities:
            return {}

        normalized_entities = [entity.strip() for entity in entities if isinstance(entity, str) and entity.strip()]
        if not normalized_entities:
            return {}

        node_names: List[str] = []
        node_embeddings: List[List[float]] = []
        for node_name, node_data in local_graph.nodes(data=True):
            embedding = node_data.get("embedding")
            if embedding is None:
                continue
            node_names.append(node_name)
            node_embeddings.append(embedding)

        if not node_embeddings:
            logger.warning("Skipping symbolic anchoring because local graph nodes have no embeddings")
            return {}

        entity_embeddings = await self.embedding_manager.get_embeddings(normalized_entities)
        if not entity_embeddings or len(entity_embeddings) != len(normalized_entities):
            logger.warning("Skipping symbolic anchoring because entity embeddings could not be computed")
            return {}

        anchor_bonus: Dict[str, float] = defaultdict(float)
        anchor_entities: Dict[str, List[str]] = defaultdict(list)

        for entity, entity_embedding in zip(normalized_entities, entity_embeddings):
            similarities = self.embedding_manager.batch_cosine_similarity(entity_embedding, node_embeddings)
            if not similarities:
                continue

            best_index = max(range(len(similarities)), key=lambda index: similarities[index])
            best_node = node_names[best_index]

            anchor_bonus[best_node] += 0.2
            anchor_entities[best_node].append(entity)

        for node_name in local_graph.nodes():
            local_graph.nodes[node_name]["symbolic_anchor_bonus"] = float(anchor_bonus.get(node_name, 0.0))
            if node_name in anchor_entities:
                local_graph.nodes[node_name]["anchored_entities"] = anchor_entities[node_name]

        logger.info(
            "Applied symbolic anchoring for %s entities across %s nodes",
            len(normalized_entities),
            len(anchor_bonus),
        )
        return dict(anchor_bonus)

    def run_ppr(
        self,
        local_graph: nx.MultiDiGraph,
        personalization: Dict[str, float],
    ) -> Dict[str, float]:
        if local_graph.number_of_nodes() == 0:
            return {}

        alpha = getattr(self.config.retriever, "ppr_alpha", None)
        if alpha is None:
            walk_prob = getattr(self.config.retriever, "walk_prob", 0.2)
            alpha = 1.0 - float(walk_prob)

        node_scores = nx.pagerank(
            local_graph,
            alpha=float(alpha),
            personalization=personalization,
            weight="weight",
        )

        for node, score in node_scores.items():
            local_graph.nodes[node]["ppr_score"] = float(score)

        return node_scores

    def rank_triples_from_ppr(
        self,
        local_graph: nx.MultiDiGraph,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        ranked_triples: List[Dict[str, Any]] = []

        for src, tgt, _, data in local_graph.edges(keys=True, data=True):
            if data.get("edge_type") == "synonymy":
                continue

            src_score = float(local_graph.nodes[src].get("ppr_score", 0.0))
            tgt_score = float(local_graph.nodes[tgt].get("ppr_score", 0.0))
            if src_score + tgt_score > 0:
                triple_score = 2 * src_score * tgt_score / (src_score + tgt_score)
            else:
                triple_score = 0.0

            ranked_triple = data.get("triple", {}).copy()
            ranked_triple["ppr_score"] = triple_score
            ranked_triple["final_score"] = triple_score
            ranked_triples.append(ranked_triple)

        ranked_triples.sort(key=lambda triple: float(triple.get("final_score", 0.0)), reverse=True)
        return ranked_triples[:top_k]

    async def retrieve_global_triples(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        global_triples = self.retrieve_all_triples()
        return await self.query_to_triples(question=question, triples=global_triples, top_k=top_k)

    async def retrieve(
        self,
        question: str,
        candidate_sessions: List[str],
        entities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:

        local_triples = self.sessions_to_triples(candidate_sessions)
        if not local_triples:
            logger.warning("No local triples found from summary sessions, falling back to global triples")
            local_triples = self.retrieve_all_triples()

        seed_top_k = getattr(self.config.retriever, "ppr_seed_top_k", 5)
        seed_triples = await self.query_to_triples(question=question, triples=local_triples, top_k=seed_top_k)
        if not seed_triples:
            return {
                "seed_triples": [],
                "local_graph": nx.MultiDiGraph(),
                "node_scores": {},
                "reranked_triples": [],
            }

        local_graph = self.build_local_graph(local_triples)
        symbolic_anchor_bonus = await self.symbolic_anchoring(local_graph, entities or [])
        personalization = self.assign_reset_probabilities(local_graph, seed_triples, symbolic_anchor_bonus)
        node_scores = self.run_ppr(local_graph, personalization)

        top_k = 15
        reranked_triples = self.rank_triples_from_ppr(local_graph, top_k=top_k)

        return {
            "seed_triples": seed_triples,
            "local_graph": local_graph,
            "symbolic_anchor_bonus": symbolic_anchor_bonus,
            "node_scores": node_scores,
            "reranked_triples": reranked_triples,
        }

    def _ensure_node_with_synonym_edges(
        self,
        local_graph: nx.MultiDiGraph,
        node_name: str,
        similarity_threshold: float,
    ) -> None:
        if node_name in local_graph:
            return

        node_attributes = {}
        if self.global_graph is not None and node_name in self.global_graph.nodes:
            node_attributes = dict(self.global_graph.nodes[node_name])
        node_attributes.setdefault("entity_name", node_name)

        existing_nodes = list(local_graph.nodes())
        local_graph.add_node(node_name, **node_attributes)

        new_embedding = local_graph.nodes[node_name].get("embedding")
        if new_embedding is None:
            return

        for existing_node in existing_nodes:
            existing_embedding = local_graph.nodes[existing_node].get("embedding")
            if existing_embedding is None:
                continue

            similarity = self.embedding_manager.cosine_similarity(new_embedding, existing_embedding)
            if similarity <= similarity_threshold:
                continue

            synonym_data = {
                "edge_type": "synonymy",
                "relation_name": "synonymy",
                "weight": float(similarity),
                "similarity_score": float(similarity),
            }
            local_graph.add_edge(node_name, existing_node, **synonym_data)
            local_graph.add_edge(existing_node, node_name, **synonym_data)

    def _add_or_increment_relation_edge(
        self,
        local_graph: nx.MultiDiGraph,
        triple: Dict[str, Any],
    ) -> None:
        src = triple.get("src", "")
        tgt = triple.get("tgt", "")
        relation = triple.get("relation", "relates_to")
        increment = max(int(triple.get("occurrence_count", 1)), 1)

        existing_key = None
        existing_data = None
        edge_bundle = local_graph.get_edge_data(src, tgt, default={})
        for key, data in edge_bundle.items():
            if data.get("edge_type") != "relation":
                continue
            if data.get("relation_name") != relation:
                continue
            existing_key = key
            existing_data = data
            break

        if existing_key is not None and existing_data is not None:
            existing_data["weight"] = float(existing_data.get("weight", 0.0)) + increment
            existing_data["occurrence_count"] = int(existing_data.get("occurrence_count", 1)) + increment
            existing_data["triple"] = self._merge_triple_metadata(existing_data.get("triple", {}), triple)
            return

        local_graph.add_edge(
            src,
            tgt,
            edge_type="relation",
            relation_name=relation,
            weight=float(increment),
            occurrence_count=increment,
            triple=triple.copy(),
        )

    def _merge_triple_metadata(
        self,
        existing: Dict[str, Any],
        new: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged = existing.copy() if existing else new.copy()

        chunk_ids = set(self._ensure_list(merged.get("chunk_ids", [])))
        chunk_ids.update(self._ensure_list(new.get("chunk_ids", [])))
        chunk_id = new.get("chunk_id", "") or merged.get("chunk_id", "")
        if chunk_id:
            chunk_ids.add(chunk_id)
        merged["chunk_ids"] = list(chunk_ids)
        merged["chunk_id"] = chunk_id

        session_ids = set(self._extract_session_ids(merged))
        session_ids.update(self._extract_session_ids(new))
        merged["session_ids"] = list(session_ids)
        merged["session_id"] = new.get("session_id", "") or merged.get("session_id", "")

        matched_session_ids = set(self._ensure_list(merged.get("matched_session_ids", [])))
        matched_session_ids.update(self._ensure_list(new.get("matched_session_ids", [])))
        if matched_session_ids:
            merged["matched_session_ids"] = list(matched_session_ids)

        merged["occurrence_count"] = int(merged.get("occurrence_count", 1)) + int(new.get("occurrence_count", 1))
        return merged

    def _edge_to_triple(self, src: str, tgt: str, data: Dict[str, Any]) -> Dict[str, Any]:
        relation = data.get("relation_name", "relates_to")
        triple_text = f"{src} {relation} {tgt}"
        session_ids = self._extract_session_ids(data)
        return {
            "src": src,
            "tgt": tgt,
            "relation": relation,
            "triple_text": triple_text,
            "chunk_id": data.get("chunk_id", ""),
            "chunk_ids": self._ensure_list(data.get("chunk_ids", [])),
            "session_id": data.get("session_id", ""),
            "session_ids": session_ids,
            "timestamp": self._extract_and_format_timestamp(data),
            "description": data.get("description", ""),
            "embedding": data.get("embedding"),
            "occurrence_count": max(len(session_ids), 1),
        }

    @staticmethod
    def _extract_and_format_timestamp(edge_data: Dict[str, Any]) -> str:
        timestamp = edge_data.get("session_time", "") or edge_data.get("timestamp", "")
        if not timestamp and "session_times" in edge_data:
            session_times = [time for time in edge_data.get("session_times", []) if time]
            if session_times:
                timestamp = sorted(session_times)[-1]

        if not timestamp:
            return ""

        try:
            date_part = timestamp.split()[0] if " " in timestamp else timestamp
            return date_part.replace("-", "/")
        except Exception as exc:
            logger.warning("Failed to format timestamp '%s': %s", timestamp, exc)
            return ""

    @staticmethod
    def _ensure_list(value: Any) -> List[Any]:
        if not value:
            return []
        if isinstance(value, list):
            return value
        return [value]

    @staticmethod
    def _extract_session_ids(triple_or_edge: Dict[str, Any]) -> List[str]:
        session_ids = triple_or_edge.get("session_ids", []) or []
        if not isinstance(session_ids, list):
            session_ids = [session_ids]

        session_id = triple_or_edge.get("session_id", "")
        if session_id and session_id not in session_ids:
            session_ids.append(session_id)

        return [str(session) for session in session_ids if session]

    @staticmethod
    def _get_similarity_score(triple: Dict[str, Any]) -> float:
        return float(triple.get("similarity_score", triple.get("final_score", 0.0)))
