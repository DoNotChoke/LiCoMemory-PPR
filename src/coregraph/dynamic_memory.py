import os
from typing import List, Dict, Any

import networkx as nx
import pickle

from src.base.embeddings import EmbeddingManager
from src.chunking.chunk_processor import ChunkProcessor
from src.base.llm import LLMManager
from src.init.logger import logger
from src.init.config import Config
from .graph_builder import GraphBuilder
from .entity_extractor import EntityExtractor
from .dialogue_extractor import DialogueExtractor
from .session_summarizer import SessionSummarizer
from src.utils.time_statistic import GraphBuildingTimeStatistic
from src.utils.cost_manager import GraphBuildingCostManager

class DynamicMemory:
    def __init__(self, config: Config, llm_manager: LLMManager, base_dir: str = "./results"):
        self.config = config
        self.base_dir = base_dir
        self.llm_manager = llm_manager
        self.graph_builder = GraphBuilder(config)
        self.entity_extractor = EntityExtractor(llm_manager)
        data_type = getattr(config, "data_type", "LongmemEval")
        self.dialogue_extractor = DialogueExtractor(llm_manager, data_type)
        self.session_summarizer = SessionSummarizer(llm_manager)
        self.embedding_manager = EmbeddingManager(config.embedding)
        self.chunk_processor = ChunkProcessor(config.chunk, data_type)
        self.time_manager = GraphBuildingTimeStatistic()
        self.cost_manager = GraphBuildingCostManager(max_budget=llm_manager.cost_manager.max_budget)
        self.chunk_storage = {}
        self.entity_name_to_index = {}

        logger.info("Dynamic Memory initialized")

    async def generate_session_summaries(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info("Starting session summarization")

        self.time_manager.start_summary_generation()
        cost_before_summary = self.cost_manager.get_costs()

        sessions_map = {}
        for doc in documents:
            session_id = doc.get("session_id", "unknown")
            if session_id not in sessions_map:
                sessions_map[session_id] = {
                    "session_id": session_id,
                    "session_time": str(doc.get("session_time", "unknown")),
                    "context": doc.get("context", "")
                }
            else:
                existing_context = sessions_map[session_id]["context"]
                new_context = doc.get("context", "")
                if new_context and new_context not in existing_context:
                    sessions_map[session_id]["context"] += "\n\n" + new_context

        sessions = list(sessions_map.values())
        logger.info(f"Found {len(sessions)} unique sessions to summarize")

        from tqdm import tqdm
        summary_progress_bar = tqdm(total=len(sessions), desc="Stage 1: Session Summary Generation", unit="calls")

        logger.info(f"Stage 1: Processing {len(sessions)} session summaries")

        summaries = await self.session_summarizer.summarize_sessions(sessions, progress_bar=summary_progress_bar)
        summary_progress_bar.close()
        cost_after_summary = self.llm_manager.cost_manager.get_costs()
        summary_prompt_tokens = cost_after_summary.total_prompt_tokens - cost_before_summary.total_prompt_tokens
        summary_completion_tokens = cost_after_summary.total_completion_tokens - cost_before_summary.total_completion_tokens
        self.cost_manager.update_summary_generation_cost(summary_prompt_tokens, summary_completion_tokens, self.llm_manager.model)

        self.time_manager.end_summary_generation()

        summary_path = os.path.join(self.base_dir, "session_summaries.json")
        self.session_summarizer.save_summaries(summaries, summary_path)

        logger.info(f"Generated and saved {len(summaries)} session summaries")
        return summaries

    async def build_graph(self, chunks: List[Dict[str, Any]], force: bool = None, add: bool = None) -> None:
        if force is None:
            force = getattr(self.config.graph, "force", False)
        if add is None:
            add = getattr(self.config.graph, "add", False)

        logger.info(f"Graph build mode: force={force}, add={add}")

        graph_exists = hasattr(self, 'graph_builder') and self.graph_builder.graph is not None

        if add:
            logger.info("🔄 Dynamic addition mode: loading existing graph and adding new chunks")
            try:
                await self._load_existing_graph()
                logger.info("✅ Successfully loaded existing graph")
            except FileNotFoundError:
                logger.info("📝 No existing graph found, building from scratch")
                await self._build_graph(chunks)
                return
            await self._dynamic_add_chunks(chunks)

        elif force or not graph_exists:
            mode_reason = "force rebuild" if force else "graph doesn't exist"
            logger.info(f"🏗️ Build from scratch mode: {mode_reason}")
            await self._build_graph(chunks)

        else:
            # Load existing mode - load from pkl file
            logger.info("📂 Load existing mode: loading graph from pkl file")
            await self._load_existing_graph()

    async def _manage_session_summaries(self, chunks: List[Dict[str, Any]], force: bool) -> None:
        if getattr(self.config.retriever, "enable_summary", False):
            summaries_path = os.path.join(self.base_dir, "session_summaries.json")
            if force and os.path.exists(summaries_path):
                logger.info(f"Removing existing session summaries file due to forced rebuild with summary enabled: {summaries_path}")
                os.remove(summaries_path)
            if force or not os.path.exists(summaries_path):
                logger.info("Generating session summaries at the start of the process")
                await self.generate_session_summaries(chunks)
            else:
                logger.info("Session summaries file already exists, skipping")
        else:
            logger.info("Summary generation is disabled, skipping")

    async def _load_existing_graph(self) -> None:
        try:
            pkl_filename = f"{self.config.index_name}.pkl"
            pkl_path = os.path.join(self.base_dir, pkl_filename)

            if not os.path.exists(pkl_path):
                logger.error(f"PKL file not found: {pkl_path}")
                raise FileNotFoundError(f"Graph file not found: {pkl_path}")

            logger.info(f"Loading graph from: {pkl_path}")
            self.load_graph(pkl_path)

            try:
                stats = self.get_graph_stats()
                logger.info(f"Loaded graph with {stats['num_nodes']} nodes and {stats['num_edges']} edges")
            except Exception as stats_error:
                logger.warning(f"Could not retrieve graph statistics after loading: {stats_error}")
                logger.info("Graph loaded successfully, but statistics unavailable")

        except Exception as e:
            logger.error(f"Failed to load existing graph: {e}")
            raise

    async def _build_graph(self, chunks: List[Dict[str, Any]]):
        logger.info(f"Starting to build graph from {len(chunks)} chunks")

        self.time_manager.start_total_graph_building()
        self.cost_manager = GraphBuildingCostManager() if self.cost_manager is None else self.cost_manager

        pkl_filename = f"{self.config.index_name}.pkl"
        pkl_path = os.path.join(self.base_dir, pkl_filename)
        if os.path.exists(pkl_path):
            logger.info(f"Removing existing graph file: {pkl_path}")
            os.remove(pkl_path)

        backup_path = pkl_path + ".backup"
        if os.path.exists(backup_path):
            logger.info(f"Removing existing backup file: {backup_path}")
            os.remove(backup_path)

        logger.info("Clearing existing graph and chunk storage...")
        self.graph_builder.graph = nx.DiGraph()
        self.chunk_storage.clear()

        logger.info("✅ Cleared existing graph and chunk storage")

        logger.info("Storing original chunk data...")
        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id", 0))
            if chunk.get("text", ""):
                self.chunk_storage[chunk_id] = chunk
                logger.debug(f"Stored chunk {chunk_id}: {len(chunk.get('text', ''))} characters")

        logger.info(f"Stored {len(self.chunk_storage)} chunks in chunk_storage")

        dialogue_mode = getattr(self.config.chunk, "dialogue_input", False)
        cost_before_add = self.llm_manager.cost_manager.get_costs()

        from tqdm import tqdm

        if dialogue_mode:
            logger.info("Extracting entities and relationships using dialogue mode...")
            cost_before_dialogue = self.llm_manager.cost_manager.get_costs()

            dialogue_progress_bar = tqdm(total=len(chunks), desc="Stage 2: Dialogue Entity & Relationship Extraction", unit="calls")
            logger.info(f"Stage 2: Processing {len(chunks)} chunks for entity and relationship extraction")

            self.time_manager.start_entity_extraction()
            entities, relationships = await self.dialogue_extractor.extract_from_chunks(chunks, progress_bar=dialogue_progress_bar)
            dialogue_progress_bar.close()
            self.time_manager.end_entity_extraction()
            self.time_manager.relationship_extraction_time = 0

            cost_after_dialogue = self.llm_manager.cost_manager.get_costs()
            dialogue_prompt_tokens = cost_after_dialogue.total_prompt_tokens - cost_before_dialogue.total_prompt_tokens
            dialogue_completion_tokens = cost_after_dialogue.total_completion_tokens - cost_before_dialogue.total_completion_tokens

            entity_tokens = dialogue_prompt_tokens // 2 + dialogue_completion_tokens // 2
            relation_tokens = dialogue_prompt_tokens - (dialogue_prompt_tokens // 2) + dialogue_completion_tokens - (
                        dialogue_completion_tokens // 2)

            self.cost_manager.update_entity_extraction_cost(dialogue_prompt_tokens // 2,
                                                            dialogue_completion_tokens // 2, self.llm_manager.model)
            self.cost_manager.update_relationship_extraction_cost(
                dialogue_prompt_tokens - (dialogue_prompt_tokens // 2),
                dialogue_completion_tokens - (dialogue_completion_tokens // 2),
                self.llm_manager.model
            )

            entities, entity_mapping = self.dialogue_extractor.deduplicate_entities(
                entities, self.config.graph.entity_merge_threshold
            )
            relationships = self.dialogue_extractor.deduplicate_relationships(
                relationships, entity_mapping, self.config.graph.relationship_merge_threshold
            )

            logger.info(f"Dialogue mode - After deduplication: {len(entities)} entities, {len(relationships)} relationships")
        else:
            logger.info("Extracting entities...")

            cost_before_entity = self.llm_manager.cost_manager.get_costs()

            progress_bar = tqdm(total=len(chunks), desc="Stage 2: Entity Extraction", unit="calls")
            logger.info(f"Stage 2: Processing {len(chunks)} chunks for entity extraction")

            self.time_manager.start_entity_extraction()
            entities = await self.entity_extractor.extract_from_chunks(chunks, progress_bar=progress_bar)

            progress_bar.close()
            self.time_manager.end_entity_extraction()

            cost_after_entity = self.llm_manager.cost_manager.get_costs()
            entity_prompt_tokens = cost_after_entity.total_prompt_tokens - cost_before_entity.total_prompt_tokens
            entity_completion_tokens = cost_after_entity.total_completion_tokens - cost_before_entity.total_completion_tokens
            self.cost_manager.update_entity_extraction_cost(entity_prompt_tokens, entity_completion_tokens,
                                                            self.llm_manager.model)

            entities = self.entity_extractor.deduplicate_entities(
                entities, self.config.graph.entity_merge_threshold
            )

            relationships = []
            logger.info("Skipping relationship extraction (unused in dialogue mode)")

        triples = self._create_triples(entities, relationships)

        self.time_manager.start_graph_construction()
        self.graph_builder.build_from_entities_and_relationships(entities, relationships)
        self.time_manager.end_graph_construction()

        self.entity_name_to_index = {entity.get("entity", ""): idx for idx, entity in enumerate(entities)}

        await self._precompute_embeddings(entities, relationships)

        self.time_manager.end_total_graph_building()
        self._log_graph_building_summary()
        logger.info("Graph built successfully")

    def _log_graph_building_summary(self):
        logger.info("=" * 80)
        logger.info("📊 GRAPH BUILDING SUMMARY")
        logger.info("=" * 80)
        time_summary = self.time_manager.get_graph_building_summary()
        logger.info(f"⏱️  TIME STATISTICS:")
        logger.info(f"   📄 Chunking Time: {time_summary['chunking_time']}s")
        logger.info(f"   🏷️  Entity and Relation Extraction Time: {time_summary['entity_extraction_time']}s")
        logger.info(f"   🕸️  Graph Construction Time: {time_summary['graph_construction_time']}s")
        logger.info(f"   📝 Summary Generation Time: {time_summary['summary_generation_time']}s")
        logger.info(f"   📊 Total Graph Building Time: {time_summary['total_graph_building_time']}s")

        if hasattr(self, 'cost_manager') and self.cost_manager:
            cost_summary = self.cost_manager.get_graph_building_summary()
            logger.info(f"💰 COST STATISTICS:")
            total_extraction_tokens = cost_summary['entity_extraction_tokens'] + cost_summary[
                'relationship_extraction_tokens']
            logger.info(f"   🏷️  Entity and Relation Extraction Tokens: {total_extraction_tokens}")
            logger.info(f"   📝 Summary Generation Tokens: {cost_summary['summary_generation_tokens']}")
            logger.info(f"   📊 Total Graph Building Tokens: {cost_summary['total_graph_building_tokens']}")
            logger.info(f"   💵 Total Cost: ${cost_summary['total_cost_usd']}")

        logger.info("=" * 80)

    def _log_dynamic_add_summary(self):
        logger.info("=" * 80)
        logger.info("📊 DYNAMIC ADDITION SUMMARY")
        logger.info("=" * 80)

        if hasattr(self, 'cost_manager') and self.cost_manager:
            cost_summary = self.cost_manager.get_graph_building_summary()
            total_extraction_tokens = cost_summary['entity_extraction_tokens'] + cost_summary[
                'relationship_extraction_tokens']
            logger.info(f"💰 COST STATISTICS:")
            logger.info(f"   🏷️  Entity and Relation Extraction Tokens: {total_extraction_tokens}")
            logger.info(f"   📝 Summary Update Tokens: {cost_summary['summary_generation_tokens']}")
            logger.info(f"   📊 Total Graph Building Tokens: {cost_summary['total_graph_building_tokens']}")
            logger.info(f"   💵 Total Cost: ${cost_summary['total_cost_usd']}")

        logger.info("=" * 80)

    async def _update_summary_for_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        session_id = session_data.get("session_id", "unknown")

        summary_path = os.path.join(self.base_dir, "session_summaries.json")
        if os.path.exists(summary_path):
            existing_summaries = self.session_summarizer.load_summaries(summary_path)
        else:
            existing_summaries = []

        existing_summary = None
        for summary in existing_summaries:
            if summary.get("session_id") == session_id:
                existing_summary = summary
                break

        if existing_summary:
            logger.info(f"📝 Updating existing summary for session: {session_id} using ADDITION_PROMPT")
            # Create a pseudo-chunk for update
            chunk_data = {
                "session_id": session_id,
                "session_time": session_data.get("session_time", "unknown"),
                "text": session_data.get("context", "")
            }
            updated_summary = self.session_summarizer.update_summary_with_chunk(chunk_data, existing_summary)
        else:
            logger.info(f"📝 Creating new summary for session: {session_id} using SUMMARY_PROMPT")
            updated_summary = await self.session_summarizer.summarize_session(session_data)

        all_summaries_dict = {s.get("session_id", ""): s for s in existing_summaries}
        all_summaries_dict[session_id] = updated_summary
        all_summaries = list(all_summaries_dict.values())
        self.session_summarizer.save_summaries(all_summaries, summary_path)

        return updated_summary

    async def add_single_session(self, session_corpus: List[Dict[str, Any]]):
        if not session_corpus:
            logger.warning("Empty session corpus, skipping")
            return

        session_id = session_corpus[0].get("session_id", "unknown")
        session_time = session_corpus[0].get("session_time", "unknown")

        logger.info("=" * 80)
        logger.info(f"🔄 Processing session: {session_id} (time: {session_time})")
        logger.info("=" * 80)

        if getattr(self.config.retriever, "enable_summary", False):
            cost_before_summary = self.llm_manager.get_costs()

            session_data = {
                "session_id": session_id,
                "session_time": str(session_time),
                "context": '\n\n'.join([doc.get("context", "") for doc in session_corpus])
            }

            await self._update_summary_for_session(session_data)

            cost_after_summary = self.llm_manager.get_costs()
            summary_prompt_tokens = cost_after_summary.total_prompt_tokens - cost_before_summary.total_prompt_tokens
            summary_completion_tokens = cost_after_summary.total_completion_tokens - cost_before_summary.total_completion_tokens
            self.cost_manager.update_summary_generation_cost(
                summary_prompt_tokens,
                summary_completion_tokens,
                self.llm_manager.model
            )

        chunks = self.chunk_processor.process_corpus(session_corpus)
        logger.info(f"Processed {len(chunks)} chunks from session {session_id}")

        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id", 0))
            if chunk.get("text", ""):
                self.chunk_storage[chunk_id] = chunk

        cost_before_extract = self.llm_manager.get_costs()

        dialogue_mode = getattr(self.config.chunk, "dialogue_input", False)

        if dialogue_mode:
            logger.info("Extracting entities and relationships using dialogue mode...")
            entities, relationships = await self.dialogue_extractor.extract_from_chunks(chunks)

            entities, entity_mapping = self.dialogue_extractor.deduplicate_entities(entities, self.config.graph.entity_merge_threshold)
            relationships = self.dialogue_extractor.deduplicate_relationships(relationships, entity_mapping, self.config.graph.relationship_merge_threshold)

            logger.info(f"Extracted and deduplicated: {len(entities)} entities, {len(relationships)} relationships")
        else:
            logger.info("Extracting entities...")
            entities = await self.entity_extractor.extract_from_chunks(chunks)
            entities = self.entity_extractor.deduplicate_entities(entities, self.config.graph.entity_merge_threshold)

            relationships = []

            logger.info(f"Extracted and deduplicated: {len(entities)} entities")

        cost_after_extract = self.llm_manager.cost_manager.get_costs()
        extract_prompt_tokens = cost_after_extract.total_prompt_tokens - cost_before_extract.total_prompt_tokens
        extract_completion_tokens = cost_after_extract.total_completion_tokens - cost_before_extract.total_completion_tokens

        self.cost_manager.update_entity_extraction_cost(
            extract_prompt_tokens // 2,
            extract_completion_tokens // 2,
            self.llm_manager.model
        )
        self.cost_manager.update_relationship_extraction_cost(
            extract_prompt_tokens - (extract_prompt_tokens // 2),
            extract_completion_tokens - (extract_completion_tokens // 2),
            self.llm_manager.model
        )

        logger.info("Adding entities and relationships to graph...")
        self.graph_builder.add_entities_and_relationships_incrementally(entities, relationships)

        for entity in entities:
            entity_name = entity.get("entity", "")
            if entity_name and entity_name not in self.entity_name_to_index:
                self.entity_name_to_index[entity_name] = len(self.entity_name_to_index)

        logger.info(f"✅ Session {session_id} processed successfully")
        logger.info("=" * 80)

    async def _dynamic_add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        logger.info(f"Dynamically adding {len(chunks)} new chunks to existing graph")

        logger.info("Storing new chunk data to chunk_storage...")
        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id", 0))
            if chunk.get("text", ""):
                self.chunk_storage[chunk_id] = chunk
                logger.debug(f"Stored new chunk {chunk_id}: {len(chunk.get('text', ''))} characters")

        logger.info(f"Stored {len(chunks)} new chunks in chunk_storage")

        dialogue_mode = getattr(self.config.chunk, "dialogue_input", False)
        cost_before_add = self.llm_manager.get_costs()

        if dialogue_mode:
            logger.info("Extracting entities and relationships from new chunks using dialogue mode...")

            entities, relationships = await self.dialogue_extractor.extract_from_chunks(chunks)
            entities, entity_mapping = self.dialogue_extractor.deduplicate_entities(entities, self.config.graph.entity_merge_threshold)
            relationships = self.dialogue_extractor.deduplicate_relationships(relationships, entity_mapping, self.config.graph.relationship_merge_threshold)

            logger.info(f"Dialogue mode - Extracted {len(entities)} entities and {len(relationships)} relationships from new chunks")

        else:
            logger.info("Extracting entities from new chunks...")

            entities = await self.dialogue_extractor.extract_from_chunks(chunks)
            entities = self.dialogue_extractor.deduplicate_entities(
                entities, self.config.graph.entity_merge_threshold
            )
            logger.info(f"Extracted {len(entities)} entities from new chunks")

            relationships = []
            logger.info("Skipping relationship extraction from new chunks (unused in dialogue mode)")

        triples = self._create_triples(entities, relationships)
        logger.info(f"Created {len(triples)} triples from new chunks")

        logger.info("Adding new entities and relationships to existing graph incrementally...")
        self.graph_builder.add_entities_and_relationships_incrementally(entities, relationships)

        start_idx = len(self.entity_name_to_index)
        for idx, entity in enumerate(entities):
            entity_name = entity.get("entity", "")
            if entity_name and entity_name not in self.entity_name_to_index:
                self.entity_name_to_index[entity_name] = start_idx + idx
        logger.info(f"Updated entity index, now has {len(self.entity_name_to_index)} entities")

        cost_after_add = self.llm_manager.cost_manager.get_costs()
        add_prompt_tokens = cost_after_add.total_prompt_tokens - cost_before_add.total_prompt_tokens
        add_completion_tokens = cost_after_add.total_completion_tokens - cost_before_add.total_completion_tokens

        self.cost_manager.update_entity_extraction_cost(add_prompt_tokens // 2, add_completion_tokens // 2,
                                                        self.llm_manager.model)
        self.cost_manager.update_relationship_extraction_cost(
            add_prompt_tokens - (add_prompt_tokens // 2),
            add_completion_tokens - (add_completion_tokens // 2),
            self.llm_manager.model
        )
        self._log_dynamic_add_summary()
        logger.info("Dynamic addition completed")

    def _create_triples(self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        triples = []

        entity_chunks = {}
        for entity in entities:
            entity_name = entity.get("entity", "")
            chunk_id = entity.get("chunk_id", 0)
            if entity_name not in entity_chunks:
                entity_chunks[entity_name] = []
            entity_chunks[entity_name].append(chunk_id)

        for relationship in relationships:
            src = relationship.get("src", "")
            tgt = relationship.get("tgt", "")
            relation = relationship.get("relation", "")
            chunk_id = relationship.get("chunk_id", 0)

            src_type = self._get_entity_type(entities, src)
            tgt_type = self._get_entity_type(entities, tgt)

            triple = {
                "src": src,
                "tgt": tgt,
                "src_type": src_type,
                "tgt_type": tgt_type,
                "relation": relation,
                "chunk_id": chunk_id,
            }
            triples.append(triple)

        return triples

    def _get_entity_type(self, entities: List[Dict[str, Any]], entity_name: str) -> str:
        for entity in entities:
            if entity.get("entity", "") == entity_name:
                return entity.get("type", "unknown")
        return "unknown"

    async def _precompute_embeddings(self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> None:
        import time
        start_time = time.time()

        try:
            await self._precompute_entity_embeddings(entities)
            await self._precompute_relationship_embeddings(relationships)
            await self._precompute_summary_embeddings()
            elapsed = time.time() - start_time
            logger.info(f"Embedding precomputation completed in {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"❌ Error during embedding precomputation: {e}")
            import traceback
            traceback.print_exc()

    async def _precompute_entity_embeddings(self, entities: List[Dict[str, Any]]) -> None:
        if not entities:
            return

        entity_texts = []
        entity_names = []
        for entity in entities:
            entity_name = entity.get("entity", "")
            entity_type = entity.get("type", "unknown")
            description = entity.get("description", "")

            if description:
                entity_text = f"{entity_name} ({entity_type}): {description}"
            else:
                entity_text = f"{entity_name} ({entity_type})"

            entity_texts.append(entity_text)
            entity_names.append(entity_name)

        embeddings = await self.embedding_manager.get_embeddings(entity_texts)

        if not embeddings or len(embeddings) != len(entity_texts):
            logger.error(f"Failed to get embeddings for entities (got {len(embeddings) if embeddings else 0}, expected {len(entity_texts)})")
            return

        for entity_name, embedding in zip(entity_names, embeddings):
            if entity_name in self.graph_builder.graph.nodes:
                self.graph_builder.graph.nodes[entity_name]["embedding"] = embedding

    async def _precompute_relationship_embeddings(self, relationships: List[Dict[str, Any]]) -> None:
        if not relationships:
            return

        triple_texts = []
        triple_keys = []

        for rel in relationships:
            src = rel.get("src", "")
            tgt = rel.get("tgt", "")
            relation = rel.get("relation", "")
            description = rel.get("description", "")

            if description:
                triple_text = f"{src} - {relation} - {tgt}: {description}"
            else:
                triple_text = f"{src} - {relation} - {tgt}"

            triple_texts.append(triple_text)
            triple_keys.append((src, tgt))

        embeddings = await self.embedding_manager.get_embeddings(triple_texts)

        if not embeddings or len(embeddings) != len(triple_texts):
            logger.error(
                f"Failed to get embeddings for relationships (got {len(embeddings) if embeddings else 0}, expected {len(triple_texts)})")
            return

        for (src, tgt), embedding in zip(triple_keys, embeddings):
            if self.graph_builder.graph.has_edge(src, tgt):
                self.graph_builder.graph.edges[src, tgt]["embedding"] = embedding

    async def _precompute_summary_embeddings(self) -> None:
        summary_path = os.path.join(self.base_dir, "session_summaries.json")

        if not os.path.exists(summary_path):
            return

        import json
        with open(summary_path, "r", encoding="utf-8") as f:
            summaries = json.load(f)

        if not summaries:
            return

        summary_texts = []
        for summary in summaries:
            text_parts = []
            session_id = summary.get("session_id", "unknown")
            session_time = summary.get("session_time", "unknown")
            text_parts.append(f"Session {session_id} ({session_time})")

            if 'keys' in summary and isinstance(summary['keys'], str):
                text_parts.append(f"Keys: {summary['keys']}")

            if 'context' in summary and isinstance(summary['context'], dict):
                context = summary['context']
                theme_keys = [k for k in context.keys() if k.startswith('theme_')]
                theme_keys.sort(key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 0)

                for theme_key in theme_keys:
                    theme_num = theme_key.split('_')[1]
                    summary_key = f'summary_{theme_num}'

                    theme_title = context.get(theme_key, '')
                    theme_summary = context.get(summary_key, '')

                    if theme_title:
                        text_parts.append(f"{theme_key}: {theme_title}")
                    if theme_summary:
                        text_parts.append(f"Summary: {theme_summary}")

            summary_texts.append(" ".join(text_parts))

        embeddings = await self.embedding_manager.get_embeddings(summary_texts)

        if not embeddings or len(embeddings) != len(summary_texts):
            logger.error(f"Failed to get embeddings for summaries (got {len(embeddings) if embeddings else 0}, expected {len(summary_texts)})")
            return

        for summary, embedding in zip(summaries, embeddings):
            summary["embedding"] = embedding

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        stats = self.graph_builder.get_graph_stats()
        return stats

    def save_graph(self, path: str) -> None:
        """Save graph and chunk storage to file."""
        graph_data = {
            'graph': self.graph_builder.graph,
            'chunk_storage': self.chunk_storage,
            'entity_name_to_index': self.entity_name_to_index,
        }
        with open(path, 'wb') as f:
            pickle.dump(graph_data, f)
        logger.info(f"Graph and chunk storage saved to {path}")

    def load_graph(self, path: str) -> None:
        """Load graph and chunk storage from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

            if isinstance(data, dict):
                self.graph_builder.graph = data.get('graph')
                self.chunk_storage = data.get('chunk_storage', {})
                self.entity_name_to_index = data.get('entity_name_to_index', {})
                logger.info(
                    f"Graph loaded from {path} with {len(self.chunk_storage)} chunks and {len(self.entity_name_to_index)} entities")
            else:
                self.graph_builder.graph = data
                self.chunk_storage = {}
                self.entity_name_to_index = {}
                logger.info(f"Graph loaded from {path} (old format, no chunk storage or entity index)")
        logger.info(f"Graph loaded from {path}")