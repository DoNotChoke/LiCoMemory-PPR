import argparse
import asyncio
import json
import pandas as pd
from pathlib import Path

import os
from src.init.config import Config
from src.init.graph_rag import GraphRAG
from src.init.logger import update_logger_path
from src.dataset.query_dataset import RAGQueryDataset
from src.utils.final_report import FinalReportGenerator

from src.evaluation.evaluator import Evaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Dynamic Memory Graph RAG System")
    parser.add_argument("-opt", type=str, help="Path to option YAML file.")
    parser.add_argument("-dataset_name", type=str, help="Name of the dataset.")
    parser.add_argument("-external_graph", type=str, help="Path to external tree file to load from.")
    parser.add_argument("-root", type=str, default="", help="Root directory to prefix result/config/metric paths.")
    parser.add_argument("-query", type=str, default=None, help="Whether to run query and evaluation (1 to enable, 0 to disable). If not specified, will prompt user.")
    return parser.parse_args()


def check_dirs(root: str):
    if root:
        base_dir = os.path.join("./results", root)
    else:
        base_dir = "./results"

    result_dir = os.path.join(base_dir, "results")
    os.makedirs(result_dir, exist_ok=True)

    os.makedirs(base_dir, exist_ok=True)

    return base_dir, result_dir


async def process_queries_async(query_dataset, graph_rag, dataset_len, config):
    from src.evaluation.llm_evaluator import LLMEvaluator
    from src.evaluation.evaluator import Evaluator
    from src.init.logger import logger

    all_res = []

    enable_llm_eval = config.evaluation.enable_llm_eval if config else False
    if enable_llm_eval:
        llm_evaluator = LLMEvaluator(config, "", "")
        print(f"🔍 Using LLM evaluation with model: {config.evaluation.eval_model}")
        logger.info(f"🔍 Using LLM evaluation with model: {config.evaluation.eval_model}")
    else:
        evaluator_obj = Evaluator("", "", None)
        print("🔍 Using exact match evaluation")
        logger.info("🔍 Using exact match evaluation")

    total_correct_llm = 0
    total_correct_exact = 0
    total_evaluated = 0

    for i in range(dataset_len):
        query = query_dataset[i]
        try:
            question_time = query.get("question_time", "")
            res = await graph_rag.query(query["question"], question_time=question_time)

            # Extract answer and top_session_ids from result dictionary
            if isinstance(res, dict):
                query["output"] = res.get('answer', '')
                query["top_session_ids"] = res.get('top_session_ids', [])
            else:
                query["output"] = res
                query["top_session_ids"] = []

            all_res.append(query)

            # Real-time evaluation
            expected_answer = str(query.get('answer', '')).strip()
            model_output = str(query.get('output', '')).strip()

            if expected_answer and model_output:
                total_evaluated += 1

                # LLM evaluation
                if enable_llm_eval:
                    question_type = query.get('question_type', 'default')
                    is_correct_llm = await llm_evaluator.evaluate_with_llm(
                        query.get('question', ''),
                        expected_answer,
                        model_output,
                        question_type
                    )
                    if is_correct_llm:
                        total_correct_llm += 1

                # Get query statistics
                query_stats = res.get('query_summary', {}) if isinstance(res, dict) else {}
                time_stats = query_stats.get('detailed_retrieval_breakdown', {})
                cost_stats = res.get('cost_summary', {}) if isinstance(res, dict) else {}

                # Print and log results
                separator = "=" * 80
                print(f"\n{separator}")
                logger.info(separator)

                question_info = f"Question { i +1}/{dataset_len}:"
                print(question_info)
                logger.info(question_info)

                # Print time statistics
                if time_stats:
                    print("Time Statistics:")
                    logger.info("Time Statistics:")
                    for stage, time_val in time_stats.items():
                        if stage != 'detailed_total':
                            stage_name = stage.replace('_', ' ').title()
                            print(f"  {stage_name}: {time_val:.2f}s")
                            logger.info(f"  {stage_name}: {time_val:.2f}s")

                # Print token statistics
                if cost_stats:
                    print("Token Statistics:")
                    logger.info("Token Statistics:")
                    retrieval_tokens = cost_stats.get('retrieval_tokens', 0)
                    answer_tokens = cost_stats.get('answer_generation_tokens', 0)
                    total_tokens = cost_stats.get('total_query_tokens', 0)
                    print(f"  Retrieval Tokens: {retrieval_tokens}")
                    logger.info(f"  Retrieval Tokens: {retrieval_tokens}")
                    print(f"  Answer Generation Tokens: {answer_tokens}")
                    logger.info(f"  Answer Generation Tokens: {answer_tokens}")
                    print(f"  Total Tokens: {total_tokens}")
                    logger.info(f"  Total Tokens: {total_tokens}")

                # Print answers
                print(f"Expected: {expected_answer}")
                logger.info(f"Expected: {expected_answer}")
                print(f"Got: {model_output}")
                logger.info(f"Got: {model_output}")

                # Print accuracy
                if total_evaluated > 0:
                    accuracy = total_correct_llm / total_evaluated * 100 if enable_llm_eval else 0.0
                    accuracy_text = f"Current Accuracy: {total_correct_llm}/{total_evaluated} ({accuracy:.1f}%)"
                    print(accuracy_text)
                    logger.info(accuracy_text)
            else:
                separator = "=" * 80
                skip_msg = f"Question { i +1}/{dataset_len}: ⚠️  Skipped (missing answer or output)"
                print(f"\n{separator}")
                print(skip_msg)
                logger.warning(skip_msg)

        except Exception as e:
            separator = "=" * 80
            error_msg = f"❌ Error processing query { i +1}: {e}"
            print(f"\n{separator}")
            print(error_msg)
            logger.error(error_msg)
            import traceback
            traceback.print_exc()
            query["output"] = "Error processing query"
            query["top_session_ids"] = []
            all_res.append(query)

    # Final summary
    separator = "=" * 80
    print(f"\n{separator}")
    logger.info(separator)

    summary_title = "📊 FINAL EVALUATION SUMMARY"
    print(summary_title)
    logger.info(summary_title)

    print(separator)
    logger.info(separator)

    total_q = f"Total Questions: {dataset_len}"
    print(total_q)
    logger.info(total_q)

    evaluated = f"Evaluated: {total_evaluated}"
    print(evaluated)
    logger.info(evaluated)

    if enable_llm_eval:
        if total_evaluated > 0:
            final_accuracy = f"Final Accuracy: {total_correct_llm}/{total_evaluated} ({total_correct_llm /total_evaluated *100:.2f}%)"
        else:
            final_accuracy = "Final Accuracy: N/A"
        print(final_accuracy)
        logger.info(final_accuracy)

    print(f"{separator}\n")
    logger.info(separator)

    return all_res

def wrapper_query(query_dataset, graph_rag, result_dir, config=None):
    """Process queries and save results."""
    dataset_len = len(query_dataset)
    print(f"Processing {dataset_len} queries...")

    # Use single event loop for all queries
    all_res = asyncio.run(process_queries_async(query_dataset, graph_rag, dataset_len, config))

    # Save results
    all_res_df = pd.DataFrame(all_res)
    save_path = os.path.join(result_dir, "results.json")
    all_res_df.to_json(save_path, orient="records", indent=2)
    print(f"Results saved to {save_path}")

    return save_path

async def wrapper_evaluation(path, dataset_name, result_dir, config=None):
    """Run evaluation on results."""
    eval = Evaluator(path, dataset_name, config)
    res_dict = await eval.evaluate()

    save_path = os.path.join(result_dir, "metrics.json")
    with open(save_path, "w") as f:
        json.dump(res_dict, f, indent=2, ensure_ascii=False)

    print(f"Metrics saved to {save_path}")
    return res_dict

if __name__ == '__main__':
    args = parse_args()
    config = Config.parse(Path(args.opt), dataset_name=args.dataset_name)

    base_dir, result_dir = check_dirs(args.root)

    update_logger_path(base_dir)

    graph_rag = GraphRAG(config, base_dir)

    try:
        query_dataset = RAGQueryDataset(
            data_dir=os.path.join(config.data_root, config.dataset_name)
        )
        corpus = query_dataset.get_corpus()
        print(f"Loaded dataset with {len(corpus)} documents")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        corpus = []

    force = getattr(config.graph, 'force', False)
    add = getattr(config.graph, 'add', False)

    if not force and not add:
        print("Load existing mode: skipping document insertion, will load from pkl file")
        asyncio.run(graph_rag.insert([]))
    elif corpus:
        print("Inserting documents into graph...")
        asyncio.run(graph_rag.insert(corpus))
        print("Document insertion completed")
    else:
        print("No corpus provided and not in load existing mode")

    final_report = FinalReportGenerator()

    if hasattr(graph_rag.core, "graph") and graph_rag.core.graph:
        dm = graph_rag.core.graph
        if hasattr(dm, 'time_manager') and hasattr(dm, 'cost_manager'):
            final_report.set_graph_building_stats(
                dm.time_manager.get_graph_building_summary(),
                dm.cost_manager.get_graph_building_summary()
            )

    if args.query:
        print("Running query and evaluation...")
        save_path = wrapper_query(query_dataset, graph_rag, result_dir, config)
        evaluation_results = asyncio.run(wrapper_evaluation(save_path, config.dataset_name, result_dir, config))
        final_report.set_evaluation_results(evaluation_results)

        # Collect query statistics from query processor
        if hasattr(graph_rag.core, 'query_processor') and graph_rag.core.query_processor:
            qp = graph_rag.core.query_processor
            if hasattr(qp, 'time_manager') and hasattr(qp, 'cost_manager'):
                final_report.add_query_stats(
                    qp.time_manager.get_query_summary(),
                    qp.cost_manager.get_query_summary()
                )
    else:
        print("Skipping query and evaluation.")

    print("Dynamic Memory Graph RAG System execution completed!")