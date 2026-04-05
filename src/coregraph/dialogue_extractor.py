from typing import List, Dict, Any, Tuple
import asyncio
import json

from src.init.logger import logger
from src.base.llm import LLMManager
from src.prompt.entity_prompt import DIALOGUE_EXTRACTION_PROMPT, LOCOMO_EXTRACTION_PROMPT


class DialogueExtractor:
    def __init__(self, llm_manager: LLMManager, data_type: str = "LongmemEval"):
        self.llm = llm_manager
        self.data_type = data_type
        if data_type == "LOCOMO":
            self.extraction_prompt = LOCOMO_EXTRACTION_PROMPT
            logger.info("Dialogue Extractor initialized with LOCOMO prompt")
        else:
            self.extraction_prompt = DIALOGUE_EXTRACTION_PROMPT
            logger.info("Dialogue Extractor initialized with LongmemEval prompt")

    async def extract_entities_and_relationships(self, text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not text:
            return [],[]

        try:
            prompt = self.extraction_prompt.format(text=text)
            entities, relationships = await self.llm.generate(prompt, task="entities_relationships_extraction")
            entities_parse = []
            relationships_parse = []
            for entity in entities:
                entity_parse = self._parse_entity(entity)
                if entity_parse:
                    entities_parse.append(entity_parse)

            for relationship in relationships:
                relationship_parse = self._parse_relationship(relationship)
                if relationship_parse:
                    relationships_parse.append(relationship_parse)

            logger.debug(f"Extracted {len(entities_parse)} entities and {len(relationships_parse)} relationships from dialogue")
            return entities_parse, relationships_parse

        except Exception as e:
            logger.error(f"Failed to extract entities and relationships from dialogue: {e}")
            logger.error(f"Input text that caused error: {text[:200]}")
            return [],[]


    def _parse_entity(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not isinstance(entity, dict):
                return None

            entity_name = entity.get("entity_name", "")
            entity_type = entity.get("entity_type", "")
            description = entity.get("description", "")

            if entity_name and entity_type:
                return {
                    "entity": entity_name,
                    "type": entity_type,
                    "description": description
                }
        except Exception as e:
            logger.warning(f"Failed to parse entity item: {entity}, error: {e}")

        return None

    def _parse_relationship(self, relationship: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not isinstance(relationship, dict):
                return None

            create_time = relationship.get("create_time", "")
            session_id = relationship.get("session_id", "")
            source_entity = relationship.get("source_entity", "")
            target_entity = relationship.get("target_entity", "")
            relationship_name = relationship.get("relationship_name", "")
            relationship_strength = relationship.get("relationship_strength", 1)

            try:
                strength = int(relationship_strength)
            except (TypeError, ValueError):
                strength = 1

            return {
                "create_time": create_time,
                "session_id": session_id,
                "src": source_entity,
                "tgt": target_entity,
                "relation": relationship_name,
                "strength": strength,
                "weight": strength / 10.0,
            }
        except Exception as e:
            logger.warning(f"Failed to parse relationship item: {relationship}, error: {e}")

        return None

    async def extract_from_chunks(self, chunks: List[Dict[str, Any]], progress_bar=None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not chunks:
            return [],[]

        async def extract_one(chunk: Dict[str, Any], index: int):
            try:
                text = chunk.get("text", "")
                session_time = str(chunk.get("session_time", "unknown"))
                session_id = str(chunk.get("session_id", "unknown"))

                enhanced_json = {
                    "text": text,
                    "session_time": session_time,
                    "session_id": session_id,
                }

                formatted_text = json.dumps(enhanced_json, ensure_ascii=False)

                entities, relationships = await self.extract_entities_and_relationships(
                    formatted_text
                )

                # gắn thêm metadata từ chunk nếu cần
                chunk_id = chunk.get("chunk_id")
                doc_id = chunk.get("doc_id")

                for entity in entities:
                    if chunk_id is not None:
                        entity["chunk_id"] = chunk_id
                    if doc_id is not None:
                        entity["doc_id"] = doc_id
                    entity["session_id"] = session_id
                    entity["session_time"] = session_time

                for rel in relationships:
                    if chunk_id is not None:
                        rel["chunk_id"] = chunk_id
                    if doc_id is not None:
                        rel["doc_id"] = doc_id
                    # ưu tiên metadata từ chunk nếu output LLM thiếu
                    if not rel.get("session_id"):
                        rel["session_id"] = session_id
                    if not rel.get("create_time"):
                        rel["create_time"] = session_time

                return entities, relationships

            except Exception as e:
                logger.error(f"Failed to extract from chunk {index}: {e}")
                return [], []
            finally:
                if progress_bar:
                    progress_bar.update(1)

        if hasattr(self.llm, "enable_concurrent") and self.llm.enable_concurrent:
            results = await asyncio.gather(
                *(extract_one(chunk, i) for i, chunk in enumerate(chunks)),
                return_exceptions=False
            )
        else:
            results = []
            for i, chunk in enumerate(chunks):
                results.append(await extract_one(chunk, i))

        all_entities = []
        all_relationships = []

        for entities, relationships in results:
            all_entities.extend(entities)
            all_relationships.extend(relationships)

        logger.info(
            f"Extracted total {len(all_entities)} entities and "
            f"{len(all_relationships)} relationships from {len(chunks)} chunks"
        )

        return all_entities, all_relationships

    def deduplicate_entities(self, entities: List[Dict[str, Any]], similarity_threshold: float = 0.85) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
        if not entities:
            return [], []

        unique_entities = []
        entity_mapping = {}

        for entity in entities:
            entity_name = entity.get("entity", "")
            entity_name_lower = entity_name.lower()
            entity_type = entity.get("type", "").lower()
            is_duplicate = False
            canonical_entity = None

            for unique_entity in unique_entities:
                unique_name = unique_entity.get("entity", "")
                unique_name_lower = unique_name.lower()
                unique_type = unique_entity.get("type", "").lower()
                if entity_name_lower == unique_name_lower and entity_type == unique_type:
                    is_duplicate = True
                    canonical_entity = unique_entity
                    self._merge_entity_sessions(unique_entity, entity)
                    break

                similarity = self._calculate_similarity(entity_name_lower, unique_name_lower)
                if similarity >= similarity_threshold and self._are_types_compatible(entity_type, unique_type):
                    is_duplicate = True
                    canonical_entity = unique_entity
                    self._merge_entity_sessions(unique_entity, entity)
                    break

            if is_duplicate:
                canonical_name = canonical_entity.get("entity", "")
                entity_mapping[entity_name] = canonical_name
                logger.debug(f"Entity mapping: '{entity_name}' -> '{canonical_name}'")
            else:
                unique_entities.append(entity)
                entity_mapping[entity_name] = entity_name

        return unique_entities, entity_mapping

    def deduplicate_relationships(self, relationships: List[Dict[str, Any]], entity_mapping: Dict[str, str] = None, similarity_threshold: float = 0.9) -> List[Dict[str, Any]]:
        if not relationships:
            return []
        if entity_mapping:
            relationships = self._update_relationship_entity_references(relationships, entity_mapping)

        unique_relationships = []

        for relationship in relationships:
            src = relationship.get("src", "").lower()
            tgt = relationship.get("tgt", "").lower()
            relation = relationship.get("relation", "").lower()
            is_duplicate = False

            for unique_relationship in unique_relationships:
                unique_src = unique_relationship.get("src", "").lower()
                unique_tgt = unique_relationship.get("tgt", "").lower()
                unique_relation = unique_relationship.get("relation", "").lower()

                if src == unique_src and tgt == unique_tgt and relation == unique_relation:
                    is_duplicate = True
                    self._merge_relationship_sessions(unique_relationship, entity_mapping)
                    break

            if not is_duplicate:
                unique_relationships.append(relationship)

        return unique_relationships

    def _update_relationship_entity_references(self, relationships: List[Dict[str, Any]],
                                               entity_mapping: Dict[str, str]) -> List[Dict[str, Any]]:
        updated_relationships = []
        updates_count = 0

        for relationship in relationships:
            updated_relationship = relationship.copy()

            # Update src entity reference
            src = relationship.get('src', '')
            if src in entity_mapping and entity_mapping[src] != src:
                updated_relationship['src'] = entity_mapping[src]
                updates_count += 1
                logger.debug(f"Updated src: '{src}' -> '{entity_mapping[src]}'")

            # Update tgt entity reference
            tgt = relationship.get('tgt', '')
            if tgt in entity_mapping and entity_mapping[tgt] != tgt:
                updated_relationship['tgt'] = entity_mapping[tgt]
                updates_count += 1
                logger.debug(f"Updated tgt: '{tgt}' -> '{entity_mapping[tgt]}'")

            updated_relationships.append(updated_relationship)

        return updated_relationships

    def _merge_entity_sessions(self, existing_entity: Dict[str, Any], new_entity: Dict[str, Any]):
        if 'session_time' in new_entity:
            existing_sessions = existing_entity.get('session_times', [])
            if 'session_time' in existing_entity:
                existing_sessions.append(existing_entity['session_time'])
                existing_entity['session_times'] = list(set(existing_sessions + [new_entity['session_time']]))
            else:
                existing_entity['session_times'] = [new_entity['session_time']]
            existing_entity['session_time'] = new_entity['session_time']  # Keep latest

        if 'session_id' in new_entity:
            existing_sessions = existing_entity.get('session_ids', [])
            if 'session_id' in existing_entity:
                existing_sessions.append(existing_entity['session_id'])
                existing_entity['session_ids'] = list(set(existing_sessions + [new_entity['session_id']]))
            else:
                existing_entity['session_ids'] = [new_entity['session_id']]
            existing_entity['session_id'] = new_entity['session_id']  # Keep latest

        if 'chunk_id' in new_entity:
            existing_chunks = existing_entity.get('chunk_ids', [])
            if 'chunk_id' in existing_entity:
                existing_chunks.append(existing_entity['chunk_id'])
                existing_entity['chunk_ids'] = list(set(existing_chunks + [new_entity['chunk_id']]))
            else:
                existing_entity['chunk_ids'] = [new_entity['chunk_id']]
            existing_entity['chunk_id'] = new_entity['chunk_id']

    def _merge_relationship_sessions(self, existing_relationship: Dict[str, Any], new_relationship: Dict[str, Any]):
        """Merge session information for duplicate relationships and update strength."""
        # Update strength (take the higher value)
        existing_strength = existing_relationship.get('strength', 1)
        new_strength = new_relationship.get('strength', 1)
        existing_relationship['strength'] = max(existing_strength, new_strength)
        existing_relationship['weight'] = existing_relationship['strength'] / 10.0

        if 'session_time' in new_relationship and new_relationship['session_time']:
            existing_sessions = existing_relationship.get('session_times', [])
            if 'session_time' in existing_relationship and existing_relationship['session_time']:
                existing_sessions.append(existing_relationship['session_time'])
                existing_relationship['session_times'] = list(
                    set(existing_sessions + [new_relationship['session_time']]))
            else:
                existing_relationship['session_times'] = [new_relationship['session_time']]
            existing_relationship['session_time'] = new_relationship['session_time']

        if 'session_id' in new_relationship:
            existing_sessions = existing_relationship.get('session_ids', [])
            if 'session_id' in existing_relationship:
                existing_sessions.append(existing_relationship['session_id'])
                existing_relationship['session_ids'] = list(set(existing_sessions + [new_relationship['session_id']]))
            else:
                existing_relationship['session_ids'] = [new_relationship['session_id']]
            existing_relationship['session_id'] = new_relationship['session_id']

        # Merge chunk IDs
        if 'chunk_id' in new_relationship:
            existing_chunks = existing_relationship.get('chunk_ids', [])
            if 'chunk_id' in existing_relationship:
                existing_chunks.append(existing_relationship['chunk_id'])
                existing_relationship['chunk_ids'] = list(set(existing_chunks + [new_relationship['chunk_id']]))
            else:
                existing_relationship['chunk_ids'] = [new_relationship['chunk_id']]
            existing_relationship['chunk_id'] = new_relationship['chunk_id']

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple similarity between two texts."""
        if not text1 or not text2:
            return 0.0
        set1 = set(text1.split())
        set2 = set(text2.split())
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _are_types_compatible(self, type1: str, type2: str) -> bool:
        """Check if two entity types are compatible for merging."""
        return type1 == type2

