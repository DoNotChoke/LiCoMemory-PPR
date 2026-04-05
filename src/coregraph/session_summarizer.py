from typing import Dict, List, Any
import json
import asyncio

from src.base.llm import LLMManager
from src.init.logger import logger
from src.prompt.summary_prompt import SUMMARY_PROMPT, ADDITION_PROMPT


class SessionSummarizer:
    def __init__(self, llm_manager: LLMManager):
        self.llm = llm_manager
        self.existing_summaries = {}
        logger.info("Session summarizer initialized")

    async def summarize_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            session_id = session_data.get("session_id", "unknown")
            session_time = str(session_data.get("session_time", "unknown"))
            context = session_data.get("context", "")

            formatted_text = (
                f"Session ID: {session_id}\n"
                f"Session Time: {session_time}\n"
                f"Context: {context}"
            )
            prompt = SUMMARY_PROMPT.replace("{text}", formatted_text)
            response = await self.llm.generate(prompt, task="summarize_session")

            summary = self._convert_structured_summary_to_legacy(
                response=response,
                session_id=session_id,
                session_time=session_time,
            )

            logger.debug(f"Generated summary for session {session_id}")
            return summary

        except Exception as e:
            logger.error(f"Failed to generate summary for session: {e}")
            return {
                "session_id": session_data.get("session_id", "unknown"),
                "session_time": str(session_data.get("session_time", "unknown")),
                "summary_status": "failed",
                "error": str(e),
            }

    async def summarize_sessions(
            self,
            sessions: List[Dict[str, Any]],
            progress_bar=None
    ) -> List[Dict[str, Any]]:
        """Generate summaries for multiple sessions using summarize_session()."""

        if not sessions:
            return []

        summaries: List[Dict[str, Any]] = []

        async def summarize_one(session: Dict[str, Any]) -> Dict[str, Any]:
            try:
                return await self.summarize_session(session)
            except Exception as e:
                logger.error(
                    f"Failed to summarize session {session.get('session_id', 'unknown')}: {e}"
                )
                return {
                    "session_id": session.get("session_id", "unknown"),
                    "session_time": str(session.get("session_time", "unknown")),
                    "summary_status": "failed",
                    "error": str(e),
                }
            finally:
                if progress_bar:
                    progress_bar.update(1)

        if hasattr(self.llm, "enable_concurrent") and self.llm.enable_concurrent:
            tasks = [summarize_one(session) for session in sessions]
            summaries = await asyncio.gather(*tasks, return_exceptions=False)
        else:
            for i, session in enumerate(sessions):
                result = await summarize_one(session)
                summaries.append(result)

        logger.info(f"Generated summaries for {len(summaries)} sessions")
        return summaries

    def save_summaries(self, summaries: List[Dict[str, Any]], output_path: str) -> None:
        """Save summaries to a JSON file."""
        try:
            def convert_timestamps(obj):
                if isinstance(obj, dict):
                    return {k: convert_timestamps(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_timestamps(item) for item in obj]
                elif hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                return obj

            summaries_serializable = convert_timestamps(summaries)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summaries_serializable, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(summaries)} summaries to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save summaries to {output_path}: {e}")

    def load_summaries(self, input_path: str) -> List[Dict[str, Any]]:
        """Load summaries from a JSON file."""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                summaries = json.load(f)
            logger.info(f"Loaded {len(summaries)} summaries from {input_path}")
            self.existing_summaries = {s.get('session_id', ''): s for s in summaries if s.get('session_id')}
            return summaries
        except Exception as e:
            logger.error(f"Failed to load summaries from {input_path}: {e}")
            return []

    async def update_summary_with_chunk(
            self,
            chunk: Dict[str, Any],
            existing_summary: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        try:
            session_id = chunk.get("session_id", "unknown")
            session_time = str(chunk.get("session_time", "unknown"))
            chunk_text = chunk.get("text", "")

            if existing_summary is None:
                existing_summary = self.existing_summaries.get(session_id)

            if existing_summary:
                logger.info(f"Updating existing summary for session {session_id}")
                summary = await self._update_existing_summary(existing_summary, chunk_text)
            else:
                logger.info(f"Creating new summary for session {session_id}")
                summary = await self.summarize_session({
                    "session_id": session_id,
                    "session_time": session_time,
                    "context": chunk_text,
                })

            self.existing_summaries[session_id] = summary
            logger.debug(f"Summary updated for session {session_id}")
            return summary

        except Exception as e:
            logger.error(f"Failed to update summary for chunk: {e}")
            return {
                "session_id": chunk.get("session_id", "unknown"),
                "session_time": str(chunk.get("session_time", "unknown")),
                "summary_status": "failed",
                "error": str(e),
            }

    async def _update_existing_summary(self, existing_summary: Dict[str, Any], new_chunk_text: str) -> Dict[str, Any]:
        try:
            existing_summary_str = json.dumps(existing_summary, ensure_ascii=False, indent=2)
            prompt = ADDITION_PROMPT.format(
                summary=existing_summary_str,
                text=new_chunk_text,
            )

            session_id = existing_summary.get("session_id", "unknown")
            session_time = str(existing_summary.get("session_time", "unknown"))

            response = await self.llm.generate(prompt, task="summarize_session")
            updated_summary = self._convert_structured_summary_to_legacy(
                response=response,
                session_id=session_id,
                session_time=session_time,
            )

            logger.debug(f"Successfully updated summary for session {session_id}")
            return updated_summary

        except Exception as e:
            logger.error(f"Failed to update existing summary: {e}")
            return existing_summary

    def _convert_structured_summary_to_legacy(
            self,
            response: Dict[str, Any],
            session_id: str,
            session_time: str,
    ) -> Dict[str, Any]:
        """
        Convert structured summary output:
        {
            "session_id": ...,
            "session_time": ...,
            "keys": [...],
            "themes": [{"title": ..., "summary": ...}, ...]
        }

        into legacy format:
        {
            "session_id": ...,
            "session_time": ...,
            "keys": "a, b, c",
            "context": {
                "theme_1": "...",
                "summary_1": "...",
                ...
            }
        }
        """
        if not response or not isinstance(response, dict):
            return {
                "session_id": session_id,
                "session_time": session_time.split(" ")[0] if " " in session_time
                else session_time.split("T")[0] if "T" in session_time
                else session_time,
                "keys": "session, conversation",
                "context": {
                    "theme_1": "General conversation",
                    "summary_1": "No structured summary was generated.",
                },
            }

        keys = response.get("keys", [])
        themes = response.get("themes", [])

        if isinstance(keys, list):
            keys_value = ", ".join(str(k) for k in keys if k)
        else:
            keys_value = str(keys) if keys else "session, conversation"

        context_dict = {}
        if isinstance(themes, list) and themes:
            for idx, theme in enumerate(themes, start=1):
                if isinstance(theme, dict):
                    context_dict[f"theme_{idx}"] = theme.get("title", f"Theme {idx}")
                    context_dict[f"summary_{idx}"] = theme.get("summary", "")
        else:
            context_dict = {
                "theme_1": "General conversation",
                "summary_1": "No theme summary available.",
            }

        return {
            "session_id": response.get("session_id", session_id),
            "session_time": response.get("session_time", session_time),
            "keys": keys_value,
            "context": context_dict,
        }