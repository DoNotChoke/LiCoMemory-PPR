import networkx as nx
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
import time
import re

import numpy as np
from datetime import datetime

from src.init.logger import logger
from src.init.config import Config
from src.base.embeddings import EmbeddingManager
from src.base.llm import LLMManager
from src.prompt.recognition_memory_prompt import RECOGNITION_MEMORY_PROMPT


class PPREngine:
    """Orchestrates PPR-enhanced retrieval over a CogniGraph.

    Design decisions in this version:
    - PPR runs on a TIME-AGNOSTIC graph: relation/context/synonym edges all
      use static weights, and session-seed personalization is also time-
      agnostic (just fusion_score).
    - Temporal decay is applied AT THE END as a multiplicative rerank factor
      on top-K sessions and top-K triples. This matches LiCoMemory paper §3.2
      where R(t) = S_sem × w(Δτ).
    """

    def __init__(
        self,
        config: Config,
        embedding_manager: EmbeddingManager,
        llm_manager: Optional[LLMManager] = None,
    ):
        self.config = config
        self.embedding_manager = embedding_manager
        self.llm = llm_manager

        # --- Channel A (summary) ---
        self.top_k_summary = 15

        # --- Channel B (query-to-triple) ---
        self.top_k_triples_b = 20

        # --- Fusion ---
        self.alpha = 0.7
        self.K_final = 15

        self.synonym_threshold = 0.8
        self.w_rel = 1.0
        self.w_syn_factor = 2.0
        self.w_ctx = 0.5

        self.ppr_damping = 0.8
        self.ppr_epsilon = 0.2
        self.ppr_session_weight = 0.5
        self.ppr_max_phrase_seeds = 5

        # --- Post-PPR fusion (structural + semantic) ---
        self.beta_ppr = 1.0
        self.beta_sem = 0.0

        # --- Hub-degree penalty for triple scoring ---
        # Penalizes triples where one or both endpoints are high-degree hub
        # nodes (e.g. "User"). Without this, hub nodes accumulate PPR mass
        # simply from topology, causing generic triples like
        # (User, interested in, X) to dominate over specific triples.
        #
        # Formula: hub_penalty = 1.0 / log2(2 + max(degree_src, degree_tgt))
        #   → degree=1:  penalty = 1/log2(3)  ≈ 0.63
        #   → degree=5:  penalty = 1/log2(7)  ≈ 0.36
        #   → degree=15: penalty = 1/log2(17) ≈ 0.24
        #   → degree=30: penalty = 1/log2(32) = 0.20
        #
        # The log dampens the penalty so that moderate-degree nodes aren't
        # overly penalised — only true hubs get significant reduction.
        self.enable_hub_penalty = True

        # --- Recognition Memory (HippoRAG 2 §3.4) ---
        # DISABLED by default: on conversational memory graphs dominated by
        # behavioral/interest triples, the LLM filter lacks direct evidence
        # and misdirects phrase seeding. Enable only when graph contains
        # sufficient factual triples (e.g. financial, temporal facts).
        self.enable_recognition = False
        self.recognition_input_k = 8
        self.recognition_max_keep = 5
        self.recognition_fallback_k = 5

        # --- Key-Fact Passage Weight Enhancement (CatRAG §3.4) ---
        # Uses top-K local triples by similarity score (NOT recognition
        # filter output) to boost context edges. This avoids the dependency
        # on Recognition Memory while still concentrating PPR mass toward
        # sessions containing the most query-relevant triples.
        self.keyfact_boost_beta = 2.0
        self.keyfact_top_k = 5  # how many top-sim local triples drive the boost

        # --- Final Temporal Rerank (LiCoMemory §3.2) ---
        # Applied AFTER PPR on top-K sessions and top-K triples.
        # This is the ONLY place temporal information enters the pipeline.
        # Set enable_temporal_rerank=False to disable entirely (e.g. for
        # temporally-insensitive query types like preference recall).
        self.enable_temporal_rerank = True
        # Weibull decay coefficient. Smaller tk = gentler decay.
        self.temporal_rerank_tk = 0.2
        # How many top sessions/triples enter the rerank stage.
        self.rerank_top_k_sessions = 15
        self.rerank_top_k_triples = 20

        logger.info(
            f"PPREngine initialised  |  K_summary={self.top_k_summary}  "
            f"K_triples_b={self.top_k_triples_b}  "
            f"α={self.alpha}  K_final={self.K_final}  "
            f"damping={self.ppr_damping}  ε={self.ppr_epsilon}  "
            f"w_rel={self.w_rel}  w_syn×={self.w_syn_factor}  "
            f"w_ctx={self.w_ctx}  "
            f"recognition={'ON' if self.enable_recognition else 'OFF'}"
            f" (in={self.recognition_input_k}/keep={self.recognition_max_keep})  "
            f"keyfact_β={self.keyfact_boost_beta}  "
            f"temporal_rerank={'ON' if self.enable_temporal_rerank else 'OFF'}"
            f" (tk={self.temporal_rerank_tk})"
        )

    # ------------------------------------------------------------------
    # Channel B: Query-to-Triple retrieval
    # ------------------------------------------------------------------

    async def query_to_triple_retrieval(
        self,
        question: str,
        graph,
    ) -> Dict[str, Any]:
        """Match *question* directly against every triple in the graph.

        Returns top-K_t triples and maps them to source sessions via
        max-pooling (each session's score = its best triple's score).

        Returns
        -------
        dict with keys:
            ``triple_scores``   – list[dict] top triples with scores.
            ``session_scores``  – dict[session_id → float] max triple
                                  score per session (raw, not normalised).
            ``all_session_triple_count`` – dict[session_id → int].
        """

        if graph is None or graph.number_of_edges() == 0:
            logger.warning("Channel B: graph is empty")
            return self._empty_channel_b_result()

        # ---- 1. Collect triples + embeddings ----
        triple_records: List[Dict[str, Any]] = []
        triple_embeddings: list = []
        texts_to_embed: List[str] = []
        indices_needing_embed: List[int] = []

        for src, tgt, data in graph.edges(data=True):
            triple_text = (
                f"{src} {data.get('relation_name', 'relates_to')} {tgt}"
            )
            record = {
                'src': src,
                'tgt': tgt,
                'relation': data.get('relation_name', 'relates_to'),
                'triple_text': triple_text,
                'session_id': data.get('session_id', ''),
                'chunk_id': data.get('chunk_id', ''),
                'chunk_ids': data.get('chunk_ids', []),
                'timestamp': self._format_timestamp(data),
            }
            triple_records.append(record)

            emb = data.get('embedding', None)
            if emb is not None:
                triple_embeddings.append(emb)
            else:
                triple_embeddings.append(None)
                texts_to_embed.append(triple_text)
                indices_needing_embed.append(len(triple_records) - 1)

        if not triple_records:
            logger.warning("Channel B: no edges in graph")
            return self._empty_channel_b_result()

        logger.info(
            f"Channel B: {len(triple_records)} triples, "
            f"{len(texts_to_embed)} need embedding"
        )

        # ---- 2. Batch-embed missing triples ----
        if texts_to_embed:
            computed = await self.embedding_manager.get_embeddings(texts_to_embed)
            if computed and len(computed) == len(texts_to_embed):
                for idx, emb in zip(indices_needing_embed, computed):
                    triple_embeddings[idx] = emb
            else:
                logger.error("Channel B: batch embedding failed")

        # Drop triples still missing embeddings
        valid = [i for i, e in enumerate(triple_embeddings) if e is not None]
        if len(valid) < len(triple_records):
            logger.warning(
                f"Channel B: dropped {len(triple_records) - len(valid)} "
                f"triples (no embedding)"
            )
        triple_records = [triple_records[i] for i in valid]
        triple_embeddings = [triple_embeddings[i] for i in valid]

        if not triple_records:
            return self._empty_channel_b_result()

        # ---- 3. Cosine similarity: query vs all triples ----
        t0 = time.time()
        question_emb = await self.embedding_manager.get_embeddings([question])
        if not question_emb:
            logger.error("Channel B: failed to embed question")
            return self._empty_channel_b_result()

        similarities = self.embedding_manager.batch_cosine_similarity(
            question_emb[0], triple_embeddings
        )
        logger.info(
            f"Channel B: similarity done in {time.time() - t0:.3f}s "
            f"({len(triple_records)} triples)"
        )

        for rec, sim in zip(triple_records, similarities):
            rec['similarity_score'] = float(sim)

        # ---- 4. Top-K triples ----
        triple_records.sort(key=lambda r: r['similarity_score'], reverse=True)
        top_triples = triple_records[: self.top_k_triples_b]

        if top_triples:
            logger.info(
                "Channel B top-5:  "
                + "  |  ".join(
                    f"{t['triple_text'][:40]}… ({t['similarity_score']:.3f})"
                    for t in top_triples[:5]
                )
            )

        # ---- 5. Map triples → sessions (max-pooling) ----
        session_scores: Dict[str, float] = {}
        session_triple_count: Dict[str, int] = defaultdict(int)

        for rec in top_triples:
            sid = rec.get('session_id', '')
            if not sid:
                continue
            score = rec['similarity_score']
            if score > session_scores.get(sid, 0.0):
                session_scores[sid] = score
            session_triple_count[sid] += 1

        logger.info(
            f"Channel B → {len(session_scores)} sessions  "
            f"(top: {self._top_n_dict(session_scores, 3)})"
        )

        return {
            'triple_scores': top_triples,
            'session_scores': dict(session_scores),
            'all_session_triple_count': dict(session_triple_count),
        }

    # ==================================================================
    # Recognition Memory (HippoRAG 2 §3.4)
    # ==================================================================
    # Filter triples through an LLM "recognition" step. Called by
    # run_context_aware_ppr() with triples extracted from the LOCAL graph
    # so every verified triple is guaranteed to exist in the subgraph
    # PPR will walk on.
    # ==================================================================

    async def recognition_memory_filter(
        self,
        question: str,
        triples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """LLM-based triple filter using RECOGNITION_MEMORY_PROMPT.

        Input format to LLM (one triple per line):
            ("src", "relation", "tgt")

        Output format from LLM (same tuple format, only relevant ones):
            ("src", "relation", "tgt")
            ("src2", "relation2", "tgt2")

        The parser extracts (src, relation, tgt) from each output line and
        matches back to the original triple dicts by (src, relation, tgt)
        normalised to lowercase. Falls back to top-`recognition_fallback_k`
        of input on any failure.
        """
        if not self.enable_recognition or not self.llm:
            return triples[: self.recognition_max_keep]

        if not triples:
            return []

        candidates = triples[: self.recognition_input_k]

        # ---- 1. Format facts as ("src", "relation", "tgt") lines ----
        fact_lines = []
        for t in candidates:
            src = t.get("src", "").strip()
            rel = t.get("relation", "").strip()
            tgt = t.get("tgt", "").strip()
            fact_lines.append(f'("{src}", "{rel}", "{tgt}")')

        facts_str = "\n".join(fact_lines)

        prompt = RECOGNITION_MEMORY_PROMPT.format(
            question=question,
            facts=facts_str,
        )

        # ---- 2. Call LLM ----
        t0 = time.time()
        try:
            response = await self.llm.generate(
                prompt, task=None, max_tokens=256, temperature=0.0,
            )
        except Exception as e:
            logger.warning(f"Recognition Memory: LLM call failed ({e}), fallback")
            return candidates[: self.recognition_fallback_k]

        elapsed = time.time() - t0
        logger.info(f"Recognition Memory LLM call: {elapsed:.3f}s")

        if not response or not response.strip():
            logger.warning("Recognition Memory: empty response, fallback")
            return candidates[: self.recognition_fallback_k]

        # ---- 3. Parse output: extract (src, relation, tgt) tuples ----
        # Matches both:
        #   ("src", "relation", "tgt")
        #   ('src', 'relation', 'tgt')
        # Strips surrounding whitespace and quotes from each field.
        tuple_pattern = re.compile(
            r'\(\s*["\']([^"\']+?)["\']\s*,\s*["\']([^"\']+?)["\']\s*,\s*["\']([^"\']+?)["\']\s*\)'
        )
        parsed_tuples: List[Tuple[str, str, str]] = []
        for match in tuple_pattern.finditer(response):
            src_out = match.group(1).strip().lower()
            rel_out = match.group(2).strip().lower()
            tgt_out = match.group(3).strip().lower()
            parsed_tuples.append((src_out, rel_out, tgt_out))

        if not parsed_tuples:
            logger.warning(
                f"Recognition Memory: no tuples parsed from response "
                f"({response[:80]!r}), fallback"
            )
            return candidates[: self.recognition_fallback_k]

        # ---- 4. Match parsed tuples back to candidate dicts ----
        # Build a lookup: (src_lower, rel_lower, tgt_lower) → original dict
        candidate_lookup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for t in candidates:
            key = (
                t.get("src", "").strip().lower(),
                t.get("relation", "").strip().lower(),
                t.get("tgt", "").strip().lower(),
            )
            candidate_lookup[key] = t

        filtered: List[Dict[str, Any]] = []
        seen_keys: set = set()
        for parsed_key in parsed_tuples:
            if parsed_key in seen_keys:
                continue
            seen_keys.add(parsed_key)

            if parsed_key in candidate_lookup:
                # Exact match
                filtered.append(candidate_lookup[parsed_key])
            else:
                # Fuzzy fallback: LLM may have slightly reworded the triple.
                # Try substring matching on (src, tgt) ignoring relation.
                src_p, _, tgt_p = parsed_key
                for ckey, cval in candidate_lookup.items():
                    src_c, _, tgt_c = ckey
                    if (src_p in src_c or src_c in src_p) and \
                       (tgt_p in tgt_c or tgt_c in tgt_p):
                        if ckey not in seen_keys:
                            seen_keys.add(ckey)
                            filtered.append(cval)
                            break

        # Cap to recognition_max_keep
        filtered = filtered[: self.recognition_max_keep]

        if filtered:
            logger.info(
                f"Recognition Memory: kept {len(filtered)}/{len(candidates)} triples"
            )
            for f in filtered:
                logger.info(
                    f"  ✓ ({f.get('src','')}, {f.get('relation','')}, {f.get('tgt','')})"
                )
        else:
            # All parsed tuples were hallucinated (not in candidates) → fallback
            logger.warning(
                "Recognition Memory: no matches found in candidates, fallback"
            )
            filtered = candidates[: self.recognition_fallback_k]

        return filtered

    # ------------------------------------------------------------------
    # Dual-channel fusion
    # ------------------------------------------------------------------

    def dual_channel_session_selection(
        self,
        summary_rankings: Dict[str, float],
        channel_b_result: Dict[str, Any],
        alpha: Optional[float] = None,
        K_final: Optional[int] = None,
    ) -> Dict[str, Any]:
        alpha = alpha if alpha is not None else self.alpha
        K_final = K_final if K_final is not None else self.K_final

        triple_session_scores: Dict[str, float] = channel_b_result.get(
            'session_scores', {}
        )

        # ---- 1. Channel A pool: top-K_a by summary score ----
        sorted_a = sorted(
            summary_rankings.items(), key=lambda x: x[1], reverse=True
        )
        channel_a_pool = [sid for sid, _ in sorted_a[: self.top_k_summary]]
        channel_a_set = set(channel_a_pool)

        # ---- 2. Channel B pool: all sessions from triple mapping ----
        channel_b_pool = list(triple_session_scores.keys())
        channel_b_set = set(channel_b_pool)

        # ---- 3. Union ----
        union_ids = list(channel_a_set | channel_b_set)

        logger.info(
            f"Fusion pools: Ch-A={len(channel_a_pool)}, "
            f"Ch-B={len(channel_b_pool)}, "
            f"union={len(union_ids)}"
        )

        # ---- 4. Normalise Channel A within the union ----
        #
        # Every session in the system has a summary score (even Channel-B-
        # only sessions).  We collect those raw scores for the union,
        # then min-max normalise so that the narrow 0.4–0.5 range stretches
        # to [0, 1].
        #
        # Channel B scores are NOT normalised — their raw values (typically
        # 0.6–0.9) already sit in a natural [0, 1] range.

        union_a_raw = {
            sid: summary_rankings.get(sid, 0.0) for sid in union_ids
        }
        norm_a = self._min_max_normalize(union_a_raw)

        # ---- 5. Compute fusion scores ----
        results: List[Dict[str, Any]] = []

        for sid in union_ids:
            a_norm = norm_a.get(sid, 0.0)
            b_raw = triple_session_scores.get(sid, 0.0)

            fusion = alpha * a_norm + (1.0 - alpha) * b_raw

            # Provenance: which pool(s) contributed this session
            in_a = sid in channel_a_set
            in_b = sid in channel_b_set
            if in_a and in_b:
                source = 'AB'
            elif in_a:
                source = 'A'
            else:
                source = 'B'

            results.append({
                'session_id': sid,
                'fusion_score': fusion,
                'channel_a_raw': union_a_raw[sid],
                'channel_b_raw': b_raw,
                'channel_a_norm': a_norm,
                'source': source,
            })

        # ---- 6. Sort and cut ----
        results.sort(key=lambda r: r['fusion_score'], reverse=True)
        results = results[:K_final]

        # ---- 7. Log ----
        logger.info(
            f"Fusion → top {len(results)} from {len(union_ids)} candidates"
        )
        for r in results[:8]:
            logger.info(
                f"  {r['session_id']}  "
                f"fusion={r['fusion_score']:.4f}  "
                f"A_norm={r['channel_a_norm']:.4f}  "
                f"B_raw={r['channel_b_raw']:.4f}  "
                f"[{r['source']}]"
            )

        return {
            'selected_sessions': results,
            'channel_a_pool': channel_a_pool,
            'channel_b_pool': channel_b_pool,
        }

    async def build_local_graph(
        self,
        selected_session_ids: List[str],
        global_graph: nx.DiGraph,
    ) -> nx.DiGraph:
        t0 = time.time()
        local = nx.DiGraph()
        session_set = set(selected_session_ids)

        session_entities: Dict[str, set] = defaultdict(set)
        triple_count = 0
        for src, tgt, data in global_graph.edges(data=True):
            sid = data.get("session_id", "")

            edge_sids = set()
            if sid and sid in session_set:
                edge_sids.add(sid)
            for multi_sid in data.get("session_ids", []):
                if multi_sid in session_set:
                    edge_sids.add(multi_sid)

            if not edge_sids:
                continue

            if not local.has_node(src):
                src_data = (global_graph.nodes[src] if src in global_graph.nodes else {})
                local.add_node(
                    src,
                    node_type="entity",
                    embedding=src_data.get("embedding"),
                    entity_type=src_data.get("entity_type", ""),
                )

            if not local.has_node(tgt):
                tgt_data = (global_graph.nodes[tgt] if tgt in global_graph.nodes else {})
                local.add_node(
                    tgt,
                    node_type="entity",
                    embedding=tgt_data.get("embedding"),
                    entity_type=tgt_data.get("entity_type", ""),
                )

            if not local.has_edge(src, tgt):
                local.add_edge(
                    src, tgt,
                    weight=self.w_rel,
                    edge_type="relation",
                    relation_name=data.get("relation_name", ""),
                    session_id=sid,
                    chunk_id=data.get("chunk_id", ""),
                    chunk_ids=data.get("chunk_ids", []),
                    timestamp=self._format_timestamp(data),
                    embedding=data.get("embedding")
                )
                triple_count += 1

            for s in edge_sids:
                session_entities[s].add(src)
                session_entities[s].add(tgt)

        # ---- 2. Add session nodes + context edges ----
        for sid in selected_session_ids:
            sess_node = f"sess:{sid}"
            local.add_node(sess_node, node_type="session", session_id=sid)

            for entity_name in session_entities.get(sid, []):
                if not local.has_node(entity_name):
                    continue

                local.add_edge(
                    sess_node, entity_name,
                    weight=self.w_ctx,
                    edge_type="context"
                )
                local.add_edge(
                    entity_name, sess_node,
                    weight=self.w_ctx,
                    edge_type="context"
                )

        # ---- 3. Add synonym edges between entity nodes ----
        entity_nodes = [
            n for n, d in local.nodes(data=True)
            if d.get("node_type") == "entity"
        ]

        entity_embs: Dict[str, Any] = {}
        need_embed_names: List[str] = []

        for name in entity_nodes:
            emb = local.nodes[name].get("embedding")
            if emb is not None:
                entity_embs[name] = emb
            else:
                need_embed_names.append(name)

        if need_embed_names:
            computed = await self.embedding_manager.get_embeddings(need_embed_names)
            if computed and len(computed) == len(need_embed_names):
                for name, emb in zip(need_embed_names, computed):
                    entity_embs[name] = emb
                    local.nodes[name]["embedding"] = emb

        embeddable = [n for n in entity_nodes if n in entity_embs]
        syn_added = 0

        if len(embeddable) > 1:
            emb_list = [entity_embs[n] for n in embeddable]
            # Single tensor operation for all pairwise similarities
            emb_tensor = self.embedding_manager.transfer_to_tensor(emb_list)
            sim_matrix = self.embedding_manager.cosine_similarity_tensor(
                emb_tensor, emb_tensor
            )
            # sim_matrix is a (N, N) tensor on GPU
            for i in range(len(embeddable)):
                for j in range(i + 1, len(embeddable)):
                    sim = float(sim_matrix[i][j])
                    if sim >= self.synonym_threshold:
                        a, b = embeddable[i], embeddable[j]
                        w = self.w_syn_factor * sim
                        if not local.has_edge(a, b):
                            local.add_edge(a, b, weight=w, edge_type="synonym", sim=sim)
                        if not local.has_edge(b, a):
                            local.add_edge(b, a, weight=w, edge_type="synonym", sim=sim)
                        syn_added += 1

        elapsed = time.time() - t0
        logger.info(
            f"Local graph built in {elapsed:.3f}s  |  "
            f"nodes={local.number_of_nodes()} "
            f"(entity={len(entity_nodes)}, "
            f"session={len(selected_session_ids)})  |  "
            f"edges={local.number_of_edges()} "
            f"(rel={triple_count}, syn={syn_added}, "
            f"ctx={sum(len(v) for v in session_entities.values()) * 2})"
        )

        return local

    async def run_context_aware_ppr(
        self,
        local_graph: nx.DiGraph,
        question: str,
        query_entities: List[str],
        question_time: str,
        fusion_result: Dict[str, Any],
        channel_b_triples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run Personalized PageRank on the time-agnostic local graph.

        Pipeline:
        A. Collect relation triples from local graph (guaranteed present)
        B. Select key triples (top-K by similarity, or Recognition Memory if enabled)
        C. Key-Fact Passage Weight Enhancement (boost context edges in-place)
        D. Build personalization vector (TIME-AGNOSTIC):
           D1. Phrase seeds from key_triples
           D2. Symbolic anchors from query_entities
           D3. Session seeds = ppr_session_weight × fusion_score (no Weibull)
        E. Run PPR on the graph
        F. Score sessions and triples
        G. Final temporal rerank on top-K sessions and top-K triples
        """
        if local_graph.number_of_nodes() == 0:
            logger.warning("PPR: local graph is empty")
            return self._empty_ppr_result()

        # ==============================================================
        # Step A: Collect relation triples from LOCAL graph
        # ==============================================================
        local_triples: List[Dict[str, Any]] = []
        chanb_sim: Dict[Tuple[str, str], float] = {}
        for t in channel_b_triples:
            key = (t.get("src", ""), t.get("tgt", ""))
            chanb_sim[key] = t.get("similarity_score", 0.0)

        for src, tgt, data in local_graph.edges(data=True):
            if data.get("edge_type") != "relation":
                continue
            local_triples.append({
                "src": src,
                "tgt": tgt,
                "relation": data.get("relation_name", ""),
                "triple_text": f"{src} {data.get('relation_name', '')} {tgt}",
                "session_id": data.get("session_id", ""),
                "chunk_id": data.get("chunk_id", ""),
                "chunk_ids": data.get("chunk_ids", []),
                "timestamp": data.get("timestamp", ""),
                "embedding": data.get("embedding"),
                "similarity_score": chanb_sim.get((src, tgt), 0.0),
            })

        local_triples.sort(key=lambda t: t["similarity_score"], reverse=True)
        logger.info(
            f"Local graph has {len(local_triples)} relation triples "
            f"for Recognition Memory"
        )

        # ==============================================================
        # Step B: Select key triples for seeding and Key-Fact boost
        # ==============================================================
        # Two paths depending on whether Recognition Memory is enabled:
        #  - OFF (default): use top-K local triples by similarity score
        #  - ON: use LLM-filtered triples (may hurt on behavioral graphs)
        # The selected triples drive BOTH phrase seeding (Step D1) and
        # Key-Fact context-edge boosting (Step C).
        # ==============================================================
        if self.enable_recognition:
            key_triples = await self.recognition_memory_filter(
                question=question,
                triples=local_triples,
            )
            logger.info(
                f"Key triples: {len(key_triples)} from Recognition Memory"
            )
        else:
            key_triples = local_triples[: self.keyfact_top_k]
            logger.info(
                f"Key triples: top-{self.keyfact_top_k} by similarity "
                f"(Recognition Memory OFF)"
            )
            if key_triples:
                for kt in key_triples[:3]:
                    logger.info(
                        f"  → ({kt.get('src','')}, {kt.get('relation','')}, "
                        f"{kt.get('tgt','')})  sim={kt.get('similarity_score', 0):.3f}"
                    )

        # ==============================================================
        # Step C: Key-Fact Passage Weight Enhancement (in-place on graph)
        # Boost context edges for (entity, session) pairs that appear in
        # key_triples. This concentrates PPR mass toward sessions with
        # the most query-relevant triples.
        # ==============================================================
        keyfact_pairs: Set[Tuple[str, str]] = set()
        if key_triples and self.keyfact_boost_beta > 0:
            for vt in key_triples:
                v_sid = vt.get("session_id", "")
                if not v_sid:
                    continue
                sess_node = f"sess:{v_sid}"
                if not local_graph.has_node(sess_node):
                    continue
                for entity in (vt.get("src", ""), vt.get("tgt", "")):
                    if not entity or not local_graph.has_node(entity):
                        continue
                    keyfact_pairs.add((entity, v_sid))

            boost_count = 0
            for entity_name, sid in keyfact_pairs:
                sess_node = f"sess:{sid}"
                for u, v in [(sess_node, entity_name), (entity_name, sess_node)]:
                    if local_graph.has_edge(u, v):
                        edata = local_graph.edges[u, v]
                        if edata.get("edge_type") == "context":
                            old_w = edata["weight"]
                            edata["weight"] = old_w * (1.0 + self.keyfact_boost_beta)
                            boost_count += 1

            logger.info(
                f"Key-Fact: boosted {boost_count} context edges "
                f"from {len(keyfact_pairs)} (entity, session) pairs  "
                f"(β={self.keyfact_boost_beta})"
            )

        # ==============================================================
        # Step D: Build personalization vector (TIME-AGNOSTIC)
        # ==============================================================
        personalization: Dict[str, float] = {}

        # ---- D1. Phrase seeds from key_triples ----
        phrase_scores: Dict[str, float] = defaultdict(float)
        for triple in key_triples:
            for key in ("src", "tgt"):
                name = triple.get(key, "")
                if name and local_graph.has_node(name):
                    score = triple.get("similarity_score", 0.0)
                    if score > phrase_scores[name]:
                        phrase_scores[name] = score

        sorted_phrases = sorted(
            phrase_scores.items(), key=lambda x: x[1], reverse=True
        )[: self.ppr_max_phrase_seeds]

        for name, score in sorted_phrases:
            personalization[name] = score

        logger.info(
            f"PPR seeds — phrases ({len(sorted_phrases)} from "
            f"{len(key_triples)} key triples):  "
            + ', '.join(f'{n}={s:.3f}' for n, s in sorted_phrases[:3])
        )

        # ---- D2. Symbolic anchors (unchanged — safety net) ----
        anchors_added = 0
        for entity in query_entities:
            entity_clean = entity.strip()
            if entity_clean in local_graph.nodes:
                node_data = local_graph.nodes[entity_clean]
                if node_data.get("node_type") == "entity":
                    degree = max(local_graph.degree(entity_clean), 1)
                    anchor_weight = self.ppr_epsilon / degree

                    if entity_clean not in personalization or personalization[entity_clean] < anchor_weight:
                        personalization[entity_clean] = max(
                            personalization.get(entity_clean, 0.0), anchor_weight
                        )
                    anchors_added += 1
                    continue

            for node in local_graph.nodes:
                nd = local_graph.nodes[node]
                if nd.get("node_type") != "entity":
                    continue
                if (entity_clean.lower() in node.lower() or node.lower() in entity_clean.lower()):
                    degree = max(local_graph.degree(node), 1)
                    anchor_weight = self.ppr_epsilon / degree
                    personalization[node] = max(
                        personalization.get(node, 0.0),
                        anchor_weight
                    )
                    anchors_added += 1
                    break

        logger.info(f"PPR seeds - symbolic anchors: {anchors_added}")

        # ---- D3. Session seeds (TIME-AGNOSTIC: no Weibull here) ----
        # Each session node gets a reset probability proportional to its
        # fusion score from the dual-channel selection. We deliberately
        # do NOT modulate by time gap — that's done at the rerank stage.
        fusion_lookup: Dict[str, float] = {}
        for sess in fusion_result.get("selected_sessions", []):
            fusion_lookup[sess["session_id"]] = sess.get("fusion_score", 0.0)

        sessions_added = 0
        for node, data in local_graph.nodes(data=True):
            if data.get("node_type") != "session":
                continue
            sid = data.get("session_id", "")
            fusion_score = fusion_lookup.get(sid, 0.0)
            session_reset = self.ppr_session_weight * fusion_score
            if session_reset > 0:
                personalization[node] = session_reset
                sessions_added += 1

        logger.info(f"PPR seeds — sessions: {sessions_added} (time-agnostic)")

        if not personalization:
            logger.warning(
                "PPR: no seeds found, falling back to uniform over sessions"
            )
            for node, data in local_graph.nodes(data=True):
                if data.get("node_type") == "session":
                    personalization[node] = 1.0

        if not personalization:
            return self._empty_ppr_result()

        # ---- Normalize personalization to sum=1 ----
        total_p = sum(personalization.values())
        if total_p > 0:
            personalization = {k: v / total_p for k, v in personalization.items()}

        logger.info(
            f"PPR personalization: {len(personalization)} seeds, "
            f"top 5: {self._top_n_dict(personalization, 5)}"
        )

        # ==============================================================
        # Step E: Run PPR
        # ==============================================================
        t0 = time.time()
        try:
            ppr_scores = nx.pagerank(
                local_graph,
                alpha=self.ppr_damping,
                personalization=personalization,
                weight="weight",
                max_iter=100,
                tol=1e-6
            )
        except nx.PowerIterationFailedConvergence:
            logger.warning("PPR did not converge in 100 iterations")
            ppr_scores = nx.pagerank(
                local_graph,
                alpha=self.ppr_damping,
                personalization=personalization,
                weight='weight',
                max_iter=300,
                tol=1e-4,
            )

        elapsed = time.time() - t0
        logger.info(f"PPR converged in {elapsed:.4f}s")

        # ==============================================================
        # Step F: Extract session and triple scores
        # ==============================================================

        # ---- F1. Session scores ----
        session_results: List[Dict[str, Any]] = []
        for node, data in local_graph.nodes(data=True):
            if data.get("node_type") != "session":
                continue
            session_results.append({
                'session_id': data.get('session_id', ''),
                'ppr_score': ppr_scores.get(node, 0.0),
                'node_name': node,
            })
        session_results.sort(key=lambda x: x["ppr_score"], reverse=True)

        logger.info(
            f"PPR session ranking (top 5): "
            + '  '.join(
                f"{s['session_id']}={s['ppr_score']:.5f}"
                for s in session_results[:5]
            )
        )

        # ---- F2. Triple scores (harmonic + semantic + hub penalty) ----
        triple_results: List[Dict[str, Any]] = []
        seen_triples = set()
        triple_embeddings_for_sim: list = []
        triple_texts_for_sim: List[str] = []
        indices_needing_embed: List[int] = []

        for src, tgt, data in local_graph.edges(data=True):
            if data.get("edge_type") != "relation":
                continue
            triple_key = (src, tgt)
            if triple_key in seen_triples:
                continue
            seen_triples.add(triple_key)

            s_src = ppr_scores.get(src, 0.0)
            s_tgt = ppr_scores.get(tgt, 0.0)
            if s_src + s_tgt > 0:
                harmonic = 2.0 * s_src * s_tgt / (s_src + s_tgt)
            else:
                harmonic = 0.0

            # Hub-degree penalty: demote triples where an endpoint is a
            # high-degree hub (e.g. "User" connected to 30+ nodes).
            # This redistributes rank from generic triples toward specific
            # ones that carry more informational value.
            hub_penalty = 1.0
            if self.enable_hub_penalty and harmonic > 0:
                degree_src = local_graph.degree(src)
                degree_tgt = local_graph.degree(tgt)
                max_degree = max(degree_src, degree_tgt)
                hub_penalty = 1.0 / np.log2(2 + max_degree)
                harmonic = harmonic * hub_penalty

            triple_text = f"{src} {data.get('relation_name', '')} {tgt}"
            idx = len(triple_results)

            triple_results.append({
                'src': src,
                'tgt': tgt,
                'relation': data.get('relation_name', ''),
                'triple_text': triple_text,
                'ppr_score': harmonic,
                'ppr_score_raw': (2.0 * s_src * s_tgt / (s_src + s_tgt)
                                  if s_src + s_tgt > 0 else 0.0),
                'hub_penalty': hub_penalty,
                'src_ppr': s_src,
                'tgt_ppr': s_tgt,
                'src_degree': local_graph.degree(src),
                'tgt_degree': local_graph.degree(tgt),
                'session_id': data.get('session_id', ''),
                'chunk_id': data.get('chunk_id', ''),
                'chunk_ids': data.get('chunk_ids', []),
                'timestamp': data.get('timestamp', ''),
                'similarity_score': 0.0,
                'final_score': harmonic,
            })

            emb = data.get('embedding')
            if emb is not None:
                triple_embeddings_for_sim.append(emb)
            else:
                triple_embeddings_for_sim.append(None)
                triple_texts_for_sim.append(triple_text)
                indices_needing_embed.append(idx)

        if triple_results:
            if indices_needing_embed:
                computed = await self.embedding_manager.get_embeddings(
                    triple_texts_for_sim
                )
                if computed and len(computed) == len(triple_texts_for_sim):
                    for list_pos, result_idx in enumerate(indices_needing_embed):
                        triple_embeddings_for_sim[result_idx] = computed[list_pos]

            valid_emb_indices = [
                i for i, e in enumerate(triple_embeddings_for_sim) if e is not None
            ]
            if valid_emb_indices:
                question_emb = await self.embedding_manager.get_embeddings(
                    [question]
                )
                if question_emb:
                    valid_embs = [triple_embeddings_for_sim[i] for i in valid_emb_indices]
                    sims = self.embedding_manager.batch_cosine_similarity(
                        question_emb[0], valid_embs
                    )
                    for list_pos, result_idx in enumerate(valid_emb_indices):
                        triple_results[result_idx]['similarity_score'] = float(
                            sims[list_pos]
                        )

            ppr_harmonic_scores = {
                i: t["ppr_score"] for i, t in enumerate(triple_results)
            }
            norm_ppr = self._min_max_normalize(ppr_harmonic_scores)

            for i, triple in enumerate(triple_results):
                ppr_norm = norm_ppr.get(i, 0.0)
                sem = triple['similarity_score']
                triple['ppr_score_norm'] = ppr_norm
                triple['final_score'] = (
                        self.beta_ppr * ppr_norm + self.beta_sem * sem
                )

        triple_results.sort(key=lambda x: x['final_score'], reverse=True)

        logger.info(
            f"Pre-rerank triple ranking (top 5)"
            + (f" [hub_penalty=ON]" if self.enable_hub_penalty else "") + ":  "
            + '  '.join(
                f"{t['triple_text'][:30]} "
                f"(ppr={t.get('ppr_score_norm', 0):.3f} "
                f"hub={t.get('hub_penalty', 1):.2f} "
                f"sem={t['similarity_score']:.3f} "
                f"final={t['final_score']:.3f})"
                for t in triple_results[:5]
            )
        )

        # ==============================================================
        # Step G: Final Temporal Rerank (LiCoMemory §3.2)
        # Apply Weibull decay only on top-K sessions and top-K triples.
        # This is the ONLY place temporal information enters the pipeline.
        # ==============================================================
        ranked_sessions, ranked_triples = self._temporal_rerank(
            session_results=session_results,
            triple_results=triple_results,
            local_graph=local_graph,
            question_time=question_time,
        )

        return {
            'ppr_scores': ppr_scores,
            'ranked_sessions': ranked_sessions,
            'ranked_triples': ranked_triples,
        }

    # ==================================================================
    # Final Temporal Rerank (LiCoMemory §3.2)
    # ==================================================================
    # Applies Weibull temporal decay to top-K sessions and top-K triples
    # AFTER PPR. Multiplicative penalty on the structural/semantic score
    # — does not change graph structure or PPR walk behavior.
    #
    # Formula (paper Eq. 3.2):
    #     R(t) = S × w(Δτ)
    #     w(Δτ) = exp(-(Δτ / τ̂)^tk),  0 < tk < 1
    # where τ̂ is the median of Δτ across the rerank candidates.
    # ==================================================================

    def _temporal_rerank(
        self,
        session_results: List[Dict[str, Any]],
        triple_results: List[Dict[str, Any]],
        local_graph: nx.DiGraph,
        question_time: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Apply Weibull temporal decay on top-K sessions and top-K triples.

        Returns
        -------
        (ranked_sessions, ranked_triples) — both reordered after rerank.
        Sessions/triples beyond top-K are appended unchanged at the end.
        """
        if not self.enable_temporal_rerank or not question_time:
            logger.info("Temporal rerank: disabled or no question_time")
            return session_results, triple_results

        # ---- 1. Rerank sessions ----
        top_k_sess = self.rerank_top_k_sessions
        sess_top = session_results[:top_k_sess]
        sess_rest = session_results[top_k_sess:]

        # Compute time gaps for top-K sessions
        sess_gaps: List[float] = []
        for sess in sess_top:
            node_name = sess["node_name"]
            latest_ts = self._get_session_latest_timestamp(local_graph, node_name)
            gap = (
                self._time_gap_days(question_time, latest_ts)
                if latest_ts else 0.0
            )
            sess["time_gap_days"] = gap
            sess_gaps.append(gap)

        # Median over top-K (not full list — locality of Weibull)
        median_sess = float(np.median(sess_gaps)) if sess_gaps else 1.0
        median_sess = max(median_sess, 1.0)

        for sess in sess_top:
            gap = sess["time_gap_days"]
            if gap >= 0 and self.temporal_rerank_tk > 0:
                w_t = float(np.exp(
                    -((gap / median_sess) ** self.temporal_rerank_tk)
                ))
            else:
                w_t = 1.0
            sess["temporal_weight"] = w_t
            sess["pre_rerank_score"] = sess["ppr_score"]
            sess["rerank_score"] = sess["ppr_score"] * w_t

        sess_top.sort(key=lambda s: s["rerank_score"], reverse=True)
        ranked_sessions = sess_top + sess_rest

        logger.info(
            f"Temporal rerank — sessions (median_gap={median_sess:.1f}d, "
            f"tk={self.temporal_rerank_tk}):  top 5: "
            + '  '.join(
                f"{s['session_id']} "
                f"(ppr={s['pre_rerank_score']:.4f} "
                f"w_t={s['temporal_weight']:.3f} "
                f"→ {s['rerank_score']:.4f})"
                for s in ranked_sessions[:5]
            )
        )

        # ---- 2. Rerank triples ----
        top_k_tri = self.rerank_top_k_triples
        tri_top = triple_results[:top_k_tri]
        tri_rest = triple_results[top_k_tri:]

        tri_gaps: List[float] = []
        for triple in tri_top:
            ts = triple.get("timestamp", "")
            gap = self._time_gap_days(question_time, ts) if ts else 0.0
            triple["time_gap_days"] = gap
            tri_gaps.append(gap)

        median_tri = float(np.median(tri_gaps)) if tri_gaps else 1.0
        median_tri = max(median_tri, 1.0)

        for triple in tri_top:
            gap = triple["time_gap_days"]
            if gap >= 0 and self.temporal_rerank_tk > 0:
                w_t = float(np.exp(
                    -((gap / median_tri) ** self.temporal_rerank_tk)
                ))
            else:
                w_t = 1.0
            triple["temporal_weight"] = w_t
            triple["pre_rerank_score"] = triple["final_score"]
            triple["final_score"] = triple["pre_rerank_score"] * w_t

        tri_top.sort(key=lambda t: t["final_score"], reverse=True)
        ranked_triples = tri_top + tri_rest

        logger.info(
            f"Temporal rerank — triples (median_gap={median_tri:.1f}d):  "
            f"top 5: "
            + '  '.join(
                f"{t['triple_text'][:25]} "
                f"(pre={t['pre_rerank_score']:.3f} "
                f"w_t={t['temporal_weight']:.3f} "
                f"→ {t['final_score']:.3f})"
                for t in ranked_triples[:5]
            )
        )

        return ranked_sessions, ranked_triples

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _min_max_normalize(scores: Dict[str, float]) -> Dict[str, float]:
        """Min-max normalise to [0, 1].  All-equal → all 1.0."""
        if not scores:
            return {}
        vals = list(scores.values())
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            return {k: 1.0 for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

    @staticmethod
    def _format_timestamp(edge_data: Dict[str, Any]) -> str:
        """Best-effort timestamp extraction from an edge."""
        timestamp = (
            edge_data.get('session_time', '')
            or edge_data.get('timestamp', '')
        )
        if not timestamp and 'session_times' in edge_data:
            valid = [t for t in edge_data.get('session_times', []) if t]
            if valid:
                timestamp = sorted(valid)[-1]
        if timestamp:
            try:
                date_part = (
                    timestamp.split()[0] if ' ' in timestamp else timestamp
                )
                return date_part.replace('-', '/')
            except Exception:
                return ''
        return ''

    @staticmethod
    def _top_n_dict(d: dict, n: int = 3) -> str:
        top = sorted(d.items(), key=lambda x: x[1], reverse=True)[:n]
        return ', '.join(f'{k}: {v:.3f}' for k, v in top)

    @staticmethod
    def _empty_channel_b_result() -> Dict[str, Any]:
        return {
            'triple_scores': [],
            'session_scores': {},
            'all_session_triple_count': {},
        }

    @staticmethod
    def _empty_ppr_result() -> Dict[str, Any]:
        return {
            'ppr_scores': {},
            'ranked_sessions': [],
            'ranked_triples': [],
        }

    @staticmethod
    def _get_session_latest_timestamp(
        local_graph: nx.DiGraph, sess_node: str
    ) -> str:
        latest = ""
        for _, neighbor, data in local_graph.edges(sess_node, data=True):
            if data.get("edge_type") != "context":
                continue
            for _, _, rel_data in local_graph.edges(neighbor, data=True):
                if rel_data.get('edge_type') != 'relation':
                    continue
                ts = rel_data.get('timestamp', '')
                if ts and ts > latest:
                    latest = ts

        return latest

    @staticmethod
    def _time_gap_days(time1: str, time2: str) -> float:
        """Absolute day gap between two YYYY/MM/DD date strings."""
        for fmt in ('%Y/%m/%d', '%Y-%m-%d'):
            try:
                d1 = datetime.strptime(time1[:10], fmt)
                d2 = datetime.strptime(time2[:10], fmt)
                return float(abs((d1 - d2).days))
            except (ValueError, TypeError):
                continue
        return 0.0