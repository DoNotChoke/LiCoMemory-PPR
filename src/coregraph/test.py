import os
import json
import asyncio
from typing import List, Dict, Any

from src.init.logger import logger
from src.base.llm import LLMManager
from src.coregraph.session_summarizer import SessionSummarizer
from tqdm import tqdm


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    sessions = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sessions.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skip invalid JSON line {line_no}: {e}")
    return sessions


async def main():
    # ==== Config ====
    corpus_path = r"src/dataset/locomo/group_1/Corpus.json"   # đổi lại nếu chạy local Windows

    # ==== Init LLMManager ====
    llm_manager = LLMManager(
        api_key="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",
        model="meta-llama/llama-3.1-8b-instruct",
        base_url="https://openrouter.ai/api/v1",
        enable_concurrent=True,
        max_concurrent=4,
        timeout=600,
        max_budget=10.0,
    )

    # ==== Init SessionSummarizer ====
    session_summarizer = SessionSummarizer(llm_manager=llm_manager)

    # ==== Load sessions ====
    sessions = load_jsonl(corpus_path)
    logger.info(f"Loaded {len(sessions)} sessions from {corpus_path}")

    if not sessions:
        logger.warning("No sessions found.")
        return

    # Test ít trước cho an toàn
    test_sessions = sessions[:]

    summary_progress_bar = tqdm(total=len(test_sessions), unit="calls")
    # ==== Summarize ====
    summaries = await session_summarizer.summarize_sessions(test_sessions, progress_bar=summary_progress_bar)

    # ==== Print results ====
    print("\n===== SUMMARY RESULTS =====\n")
    for i, summary in enumerate(summaries, start=1):
        print(f"--- Summary {i} ---")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print()

    # ==== Optional: save output ====
    out_path = "summaries_output.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved summaries to {out_path}")

    # ==== Cost summary ====
    try:
        print("===== COST SUMMARY =====")
        print(llm_manager.get_cost_summary())
    except Exception as e:
        logger.warning(f"Could not get cost summary: {e}")


if __name__ == "__main__":
    asyncio.run(main())