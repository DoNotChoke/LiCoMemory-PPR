from typing import Dict, Any, List
import asyncio

from .structured_output.entity import EntityExtractionResult
from src.prompt.entity_prompt import QUERY_ENTITY_EXTRACTION_PROMPT
from src.init.logger import logger
from src.utils.cost_manager import CostManager
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from .structured_output.entity_relationship import EntityRelationshipExtractionResult
from .structured_output.summary import SessionSummaryResult


class LLMManager:
    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = 32768,
        base_url: str = None,
        enable_concurrent: bool = True,
        max_concurrent: int = 8,
        timeout: int = 600,
        max_budget: float = 100.0,
    ):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.enable_concurrent = enable_concurrent
        self.timeout = timeout

        self.client = ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            timeout=self.timeout,
        )
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent) if self.enable_concurrent else None
        self.cost_manager = CostManager(max_budget=max_budget)

        logger.info(
            f"LLM Manager initialized with model: {model}, "
            f"concurrent: {enable_concurrent}, max_concurrent: {max_concurrent}"
        )

    async def generate(self, prompt: str, task: str = None, **kwargs):
        try:
            if self.semaphore:
                async with self.semaphore:
                    return await self._generate_internal(prompt, task, **kwargs)
            else:
                return await self._generate_internal(prompt, task, **kwargs)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "" if task != "entities_extraction" else []

    async def _generate_internal(self, prompt: str, task: str = None, **kwargs):
        messages = [HumanMessage(content=prompt)]

        max_tokens = kwargs.pop("max_tokens", 1000)
        temperature = kwargs.pop("temperature", 0.1)

        llm = self.client.bind(
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

        try:
            if task:
                output_schema = None
                if task == "entities_extraction":
                    output_schema = EntityExtractionResult
                elif task == "summarize_session":
                    output_schema = SessionSummaryResult
                elif task == "entities_relationships_extraction":
                    output_schema = EntityRelationshipExtractionResult

                llm = llm.with_structured_output(
                    output_schema,
                    include_raw=True,
                )
                result = await llm.ainvoke(messages)

                raw_msg = result["raw"]
                parsed = result["parsed"]
                parsing_error = result["parsing_error"]

                if parsing_error:
                    raise parsing_error

                usage = getattr(raw_msg, "usage_metadata", None)
                if usage:
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)
                else:
                    input_tokens = 0
                    output_tokens = 0

                self.cost_manager.update_cost(input_tokens, output_tokens, self.model)
                if task == "entities_extraction":
                    return [entity.model_dump() for entity in parsed.entities]
                if task == "summarize_session":
                    return parsed.model_dump()
                if task == "entities_relationships_extraction":
                    entities = [entity.model_dump() for entity in parsed.entities]
                    relationships = [entity.model_dump() for entity in parsed.relationships]
                    return entities, relationships

            response = await llm.ainvoke(messages)
            response_content = response.content.strip() if response.content else ""

            usage = getattr(response, "usage_metadata", None)
            if usage:
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
            else:
                input_tokens = 0
                output_tokens = 0

            self.cost_manager.update_cost(input_tokens, output_tokens, self.model)
            return response_content

        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise

    async def batch_generate(self, prompts: List[str], progress_bar=None, task="", **kwargs) -> List[str]:
        if not self.enable_concurrent:
            logger.info("Concurrent not enabled, using sequential processing")
            results = []
            for prompt in prompts:
                result = await self.generate(prompt, **kwargs)
                results.append(result)
                if progress_bar:
                    progress_bar.update(1)
            return results
        logger.info(f"Batch concurrent processing {len(prompts)} requests, max concurrent: {self.max_concurrent}")
        results_list = [None] * len(prompts)

        async def generate_with_progress(prompt, index):
            """Generate with progress tracking."""
            try:
                result = await self.generate(prompt, task, **kwargs)
                results_list[index] = result
            except Exception as e:
                logger.error(f"Batch processing request {index} failed: {e}")
                results_list[index] = ""
            finally:
                # Update progress bar when each request completes
                if progress_bar:
                    progress_bar.update(1)

        tasks = [generate_with_progress(prompt, i) for i, prompt in enumerate(prompts)]
        await asyncio.gather(*tasks, return_exceptions=True)

        return results_list


    async def extract_entities(self, text: str, session_time: str = "") -> List[Dict[str, Any]]:
        prompt = QUERY_ENTITY_EXTRACTION_PROMPT.format(text=text, session_time=session_time)
        logger.info(f"Sending professional entity extraction prompt to LLM (model: {self.model}, session_time: {session_time})")

        entities = await self.generate(prompt, task="entities_extraction")
        logger.info(f"Successfully extracted {len(entities)} entities")
        return entities

    async def batch_extract_entities(self, texts: List[str], progress_bar=None) -> List[List[Dict[str, Any]]]:
        if not self.enable_concurrent:
            logger.info("Concurrent not enabled, using sequential entity extraction")
            results = []
            for text in texts:
                result = await self.extract_entities(text)
                results.append(result)
                if progress_bar:
                    progress_bar.update(1)

            return results

        logger.info(f"Batch concurrent entity extraction {len(texts)} texts, max concurrent: {self.max_concurrent}")

        results_list = [None] * len(texts)

        async def extract_with_progress(text, index):
            try:
                result = await self.extract_entities(text)
                results_list[index] = result
            except Exception as e:
                logger.error(f"Batch entity extraction resquest {index} failed: {e}")
                results_list[index] = []
            finally:
                if progress_bar:
                    progress_bar.update(1)

        tasks = [extract_with_progress(text, i) for i, text in enumerate(texts)]
        await asyncio.gather(*tasks, return_exceptions=True)

        return results_list

    def get_costs(self):
        """Get current cost statistics."""
        return self.cost_manager.get_costs()

    def get_last_stage_cost(self):
        """Get last stage cost statistics."""
        return self.cost_manager.get_last_stage_cost()

    def get_cost_summary(self):
        """Get cost summary."""
        return self.cost_manager.get_cost_summary()

    def check_budget(self):
        """Check budget."""
        return self.cost_manager.check_budget()

    def set_max_budget(self, budget: float):
        """Set maximum budget."""
        self.cost_manager.max_budget = budget
        logger.info(f"💰 Max budget set to: ${budget:.2f}")