#!/usr/bin/env python3
"""
Simple Math Evaluator - Async evaluation with real-time verification and JSONL logging
"""

import importlib
from wrapt_timeout_decorator import timeout


def patch_target_module(
    to_patch: str,
    replace_with,
):
    to_patch = to_patch.split(".")
    assert len(to_patch) > 1, "must have an object to patch"

    to_patch, obj_name_to_patch = to_patch[:-1], to_patch[-1]
    to_patch = ".".join(to_patch)
    source = importlib.import_module(to_patch)
    setattr(source, obj_name_to_patch, replace_with)


def timeout_adapter(func=None, **kwargs):
    timeout_val = kwargs.pop("timeout_seconds", None)
    return timeout(dec_timeout=timeout_val, use_signals=False, **kwargs)


patch_target_module("math_verify.utils.timeout", timeout_adapter)
patch_target_module("math_verify.parser.timeout", timeout_adapter)


import json
import os
import asyncio
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from math_verification import (
    extract_boxed,
    format_for_math_verify,
    string_compare_answers,
    verify_answer,
)
from api_client import (
    make_api_call,
    calculate_max_tokens,
    create_openai_client,
    MODEL_CONFIG,
)
from tenacity import RetryError
import re
import typer

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Model configuration
MODEL_CONFIG = {
    "provider": "openai",
    "model_id": "gpt-3.5-turbo",
    "tokenizer_id": "gpt-3.5-turbo",
    "max_context": 128000,
    "max_output": 40960,
}

# Prompts
PROMPT_CN = "请逐步推理，并在 \\boxed{} 内给出您的最终答案。\n\n"
PROMPT_EN = (
    "Put your final answer within \\boxed{}.\n\n"
)

app = typer.Typer()


class AsyncJSONLWriter:
    """Simple async JSONL writer"""

    def __init__(self, filename: str):
        self.filename = filename
        self.queue = asyncio.Queue()
        self.file_handle = None
        self.is_running = False
        self._writer_task = None

    async def start(self):
        """Start the writer"""
        if not self.is_running:
            self.file_handle = open(self.filename, "a", encoding="utf-8")
            self.is_running = True
            self._writer_task = asyncio.create_task(self._writer_loop())

    async def _writer_loop(self):
        """Process write requests"""
        while self.is_running or not self.queue.empty():
            try:
                data = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                self.file_handle.write(json.dumps(data, ensure_ascii=False) + "\n")
                self.file_handle.flush()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error writing to JSONL: {e}")

        if self.file_handle:
            self.file_handle.close()

    async def write(self, data: Dict):
        """Queue data for writing"""
        await self.queue.put(data)

    async def stop(self):
        """Stop the writer"""
        self.is_running = False
        if self._writer_task:
            await self._writer_task


def load_completed_attempts(jsonl_file: str) -> Set[Tuple[str, int]]:
    """
    Load completed (unique_id, sample_idx) pairs from existing JSONL file.

    Returns:
        Set of tuples (unique_id, sample_idx) that have been completed
    """
    completed = set()

    if not os.path.exists(jsonl_file):
        return completed

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                unique_id = data.get("unique_id")
                sample_idx = data.get("sample_idx")
                if unique_id is not None and sample_idx is not None:
                    completed.add((unique_id, sample_idx))
            except json.JSONDecodeError:
                continue

    print(f"📖 Loaded {len(completed)} completed attempts from {jsonl_file}")
    return completed


async def process_sample(
    client,
    problem_data: Dict,
    sample_idx: int,
    is_chinese: bool,
    writer: AsyncJSONLWriter,
    semaphore: asyncio.Semaphore,
    temperature: float,
) -> None:
    """Process a single sample"""
    async with semaphore:
        # Create prompt
        problem_text = problem_data["problem"]
        prompt = PROMPT_CN if is_chinese else PROMPT_EN
        messages = [{"role": "user", "content": prompt + problem_text}]

        try:
            # Make API call
            response = await make_api_call(client, messages, temperature)

            # Extract response content and tokens
            response_content = response.choices[0].message.content
            output_tokens = response.usage.completion_tokens
            input_tokens = response.usage.prompt_tokens
            total_tokens = response.usage.total_tokens

            assert response.choices[0].finish_reason in ["stop", "length"]

            # Extract answer
            boxed_answers = extract_boxed(response_content)
            extracted = boxed_answers[0] if boxed_answers else ""

            # Verify answer
            is_correct = (
                False
                if response.choices[0].finish_reason != "stop"
                else verify_answer(extracted, problem_data["answer"])
            )

            # Write result immediately
            await writer.write(
                {
                    "unique_id": problem_data["unique_id"],
                    "sample_idx": sample_idx,
                    "problem": problem_text,
                    "gold_answer": problem_data["answer"],
                    "subject": problem_data.get("subject", "unknown"),
                    "response": response_content,
                    "raw_response": response.model_dump(),
                    "extracted_answer": extracted,
                    "is_correct": is_correct,
                    "output_tokens": output_tokens,
                    "response_length": len(response_content),
                    "timestamp": datetime.now().isoformat(),
                }
            )

            print(
                f"✓ {problem_data['unique_id']}-{sample_idx}: {'CORRECT' if is_correct else 'INCORRECT'}"
            )
            return response

        except RetryError as e:
            print(f"✗ {problem_data['unique_id']}-{sample_idx}: RETRY ERROR - {str(e)}")

            # Write error result with traceback
            await writer.write(
                {
                    "unique_id": problem_data["unique_id"],
                    "sample_idx": sample_idx,
                    "error": f"RetryError: {str(e)}",
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.now().isoformat(),
                }
            )
            return None

        except Exception as e:
            print(f"✗ {problem_data['unique_id']}-{sample_idx}: ERROR - {str(e)}")

            # Write error result with traceback
            await writer.write(
                {
                    "unique_id": problem_data["unique_id"],
                    "sample_idx": sample_idx,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.now().isoformat(),
                }
            )
            return None


async def run_evaluation(
    min_idx: int,
    max_idx: int,
    samples: int,
    temperature: float,
    concurrency: int,
    dataset: str,
    output: Optional[str],
):
    """Main evaluation function"""
    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Load dataset
    dataset_file = f"data/OlymMATH-{dataset}.jsonl"
    problems = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            problems.append(json.loads(line))

    # Select problems
    selected = problems[min_idx:max_idx]
    print(f"📊 Loaded {len(selected)} problems from index {min_idx} to {max_idx}")

    # Determine output file
    if output:
        output_file = output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"simple_eval_{timestamp}_{dataset}_{min_idx}-{max_idx}.jsonl"

    # Load completed attempts
    completed_attempts = load_completed_attempts(output_file)

    # Initialize
    is_chinese = dataset.startswith("ZH")
    client = create_openai_client()

    # Create writer
    writer = AsyncJSONLWriter(output_file)
    await writer.start()

    # Create semaphore
    semaphore = asyncio.Semaphore(concurrency)

    # Create all tasks, skipping completed ones
    tasks = []
    skipped = 0
    for problem in selected:
        for sample_idx in range(samples):
            # Skip if already completed
            if (problem["unique_id"], sample_idx) in completed_attempts:
                print(
                    f"⏭️  Skipping {problem['unique_id']}-{sample_idx} (already completed)"
                )
                skipped += 1
                continue

            task = process_sample(
                client, problem, sample_idx, is_chinese, writer, semaphore, temperature
            )
            tasks.append(task)

    if skipped > 0:
        print(f"📊 Skipped {skipped} already completed attempts")

    results = []
    if tasks:
        print(f"🏃 Running {len(tasks)} evaluations...")

        # Run all tasks
        results = await asyncio.gather(*tasks)
        print(f"🏃 Completed {len(results)} evaluations")
    else:
        print("⚠️  No new tasks to run (all already completed)")

    # Stop writer
    await writer.stop()

    print(f"\n📄 Results saved to: {output_file}")
    return results


@app.command()
def evaluate(
    min_idx: int = typer.Option(
        ..., "--min", help="Start index for problems (inclusive)"
    ),
    max_idx: int = typer.Option(
        ..., "--max", help="End index for problems (exclusive)"
    ),
    samples: int = typer.Option(10, "--samples", help="Number of samples per question"),
    temperature: float = typer.Option(
        0.6, "--temperature", help="Sampling temperature"
    ),
    concurrency: int = typer.Option(
        10, "--concurrency", help="Maximum concurrent API requests"
    ),
    dataset: str = typer.Option(
        "EN-EASY",
        "--dataset",
        help="Dataset to use (EN-EASY, EN-HARD, ZH-EASY, ZH-HARD)",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", help="Output JSONL file (default: auto-generated)"
    ),
):
    """Run simple math evaluation with async processing and JSONL output"""
    results = asyncio.run(
        run_evaluation(
            min_idx, max_idx, samples, temperature, concurrency, dataset, output
        )
    )
    from IPython import embed

    embed()


def evaluate_jsonl_results(results: List[Dict], dataset: str) -> Dict:
    """
    Evaluate JSONL results and compute metrics like local_tester.py
    """
    from collections import defaultdict
    import numpy as np

    # Filter valid results and group by unique_id
    valid_results = [r for r in results if "is_correct" in r]
    if not valid_results:
        return {"error": "No valid results found"}

    # Group results by unique_id
    problems = defaultdict(list)
    for result in valid_results:
        problems[result["unique_id"]].append(result)

    # Determine language
    is_chinese = dataset.startswith("ZH")
    language = "Chinese" if is_chinese else "English"

    # Get sample size
    sample_size = len(list(problems.values())[0]) if problems else 0

    # Initialize metrics
    metrics = {
        "total": {
            "correct": 0,
            "total": 0,
            "output_tokens": [],
            "response_lengths": [],
            "correct_output_tokens": [],
            "correct_response_lengths": [],
            "incorrect_output_tokens": [],
            "incorrect_response_lengths": [],
        },
        "by_subject": defaultdict(
            lambda: {
                "correct": 0,
                "total": 0,
                "output_tokens": [],
                "response_lengths": [],
                "correct_output_tokens": [],
                "correct_response_lengths": [],
                "incorrect_output_tokens": [],
                "incorrect_response_lengths": [],
                "consistency_correct": 0,
                "consistency_total": 0,
            }
        ),
        "consistency_correct": 0,
        "consistency_total": 0,
    }

    # Process each problem
    for unique_id, samples in problems.items():
        # Get subject from first sample
        subject = samples[0].get("subject", "unknown")

        # Process pass@1 metrics
        correct_count = 0
        for sample in samples:
            is_correct = sample["is_correct"]
            output_tokens = sample["output_tokens"]
            response_length = sample.get(
                "response_length", len(sample.get("response", ""))
            )

            metrics["total"]["output_tokens"].append(output_tokens)
            metrics["total"]["response_lengths"].append(response_length)
            metrics["by_subject"][subject]["output_tokens"].append(output_tokens)
            metrics["by_subject"][subject]["response_lengths"].append(response_length)

            if is_correct:
                correct_count += 1
                metrics["total"]["correct"] += 1
                metrics["by_subject"][subject]["correct"] += 1
                metrics["total"]["correct_output_tokens"].append(output_tokens)
                metrics["total"]["correct_response_lengths"].append(response_length)
                metrics["by_subject"][subject]["correct_output_tokens"].append(
                    output_tokens
                )
                metrics["by_subject"][subject]["correct_response_lengths"].append(
                    response_length
                )
            else:
                metrics["total"]["incorrect_output_tokens"].append(output_tokens)
                metrics["total"]["incorrect_response_lengths"].append(response_length)
                metrics["by_subject"][subject]["incorrect_output_tokens"].append(
                    output_tokens
                )
                metrics["by_subject"][subject]["incorrect_response_lengths"].append(
                    response_length
                )

            metrics["total"]["total"] += 1
            metrics["by_subject"][subject]["total"] += 1

        # Process consistency (cons@SAMPLE)
        # Group answers that are equivalent to each other
        answer_clusters = []
        formatted_answers = []

        # Format all extracted answers for comparison
        for sample in samples:
            extracted = sample["extracted_answer"]
            try:
                formatted = format_for_math_verify(extracted)
                parsed = parse(formatted)
                formatted_answers.append((extracted, parsed))
            except Exception:
                formatted_answers.append((extracted, None))

        # Form clusters of equivalent answers
        for i, (raw_answer_i, parsed_i) in enumerate(formatted_answers):
            if any(i in cluster for cluster in answer_clusters):
                continue

            new_cluster = [i]

            for j, (raw_answer_j, parsed_j) in enumerate(formatted_answers):
                if i != j and not any(j in cluster for cluster in answer_clusters):
                    try:
                        if parsed_i is not None and parsed_j is not None:
                            if verify(parsed_i, parsed_j):
                                new_cluster.append(j)
                        elif raw_answer_i == raw_answer_j:
                            new_cluster.append(j)
                    except Exception:
                        if raw_answer_i == raw_answer_j:
                            new_cluster.append(j)

            answer_clusters.append(new_cluster)

        # Find the largest cluster
        largest_cluster = max(answer_clusters, key=len) if answer_clusters else []

        # Use the first answer from the largest cluster as the consistent answer
        if largest_cluster:
            consistent_answer_idx = largest_cluster[0]
            consistent_sample = samples[consistent_answer_idx]
            consistent_answer = consistent_sample["extracted_answer"]
            gold_answer = consistent_sample["gold_answer"]

            # Check if the consistent answer is correct
            is_consistent_correct = verify_answer(consistent_answer, gold_answer)

            # Update consistency metrics
            metrics["consistency_total"] += 1
            metrics["by_subject"][subject]["consistency_total"] += 1
            if is_consistent_correct:
                metrics["consistency_correct"] += 1
                metrics["by_subject"][subject]["consistency_correct"] += 1

    # Calculate final metrics
    pass_at_1 = (
        metrics["total"]["correct"] / metrics["total"]["total"]
        if metrics["total"]["total"] > 0
        else 0
    )
    cons_at_sample = (
        metrics["consistency_correct"] / metrics["consistency_total"]
        if metrics["consistency_total"] > 0
        else 0
    )

    # Subject metrics
    subject_metrics = {}
    for subject, data in metrics["by_subject"].items():
        subject_pass_at_1 = data["correct"] / data["total"] if data["total"] > 0 else 0
        subject_cons_at_sample = (
            data["consistency_correct"] / data["consistency_total"]
            if data["consistency_total"] > 0
            else 0
        )
        avg_output_tokens = (
            np.mean(data["output_tokens"]) if data["output_tokens"] else 0
        )
        avg_response_length = (
            np.mean(data["response_lengths"]) if data["response_lengths"] else 0
        )
        avg_correct_tokens = (
            np.mean(data["correct_output_tokens"])
            if data["correct_output_tokens"]
            else 0
        )
        avg_correct_length = (
            np.mean(data["correct_response_lengths"])
            if data["correct_response_lengths"]
            else 0
        )
        avg_incorrect_tokens = (
            np.mean(data["incorrect_output_tokens"])
            if data["incorrect_output_tokens"]
            else 0
        )
        avg_incorrect_length = (
            np.mean(data["incorrect_response_lengths"])
            if data["incorrect_response_lengths"]
            else 0
        )

        subject_metrics[subject] = {
            "pass@1": subject_pass_at_1,
            "cons@SAMPLE": subject_cons_at_sample,
            "total_samples": data["total"],
            "correct_samples": data["correct"],
            "avg_output_tokens": avg_output_tokens,
            "avg_response_length": avg_response_length,
            "avg_correct_tokens": avg_correct_tokens,
            "avg_correct_length": avg_correct_length,
            "avg_incorrect_tokens": avg_incorrect_tokens,
            "avg_incorrect_length": avg_incorrect_length,
            "consistency_correct": data["consistency_correct"],
            "consistency_total": data["consistency_total"],
        }

    # Overall metrics
    total_avg_output_tokens = (
        np.mean(metrics["total"]["output_tokens"])
        if metrics["total"]["output_tokens"]
        else 0
    )
    total_avg_response_length = (
        np.mean(metrics["total"]["response_lengths"])
        if metrics["total"]["response_lengths"]
        else 0
    )
    correct_avg_tokens = (
        np.mean(metrics["total"]["correct_output_tokens"])
        if metrics["total"]["correct_output_tokens"]
        else 0
    )
    correct_avg_length = (
        np.mean(metrics["total"]["correct_response_lengths"])
        if metrics["total"]["correct_response_lengths"]
        else 0
    )
    incorrect_avg_tokens = (
        np.mean(metrics["total"]["incorrect_output_tokens"])
        if metrics["total"]["incorrect_output_tokens"]
        else 0
    )
    incorrect_avg_length = (
        np.mean(metrics["total"]["incorrect_response_lengths"])
        if metrics["total"]["incorrect_response_lengths"]
        else 0
    )

    return {
        "language": language,
        "sample_size": sample_size,
        "pass@1": pass_at_1,
        "cons@SAMPLE": cons_at_sample,
        "total_samples": metrics["total"]["total"],
        "correct_samples": metrics["total"]["correct"],
        "consistency_correct": metrics["consistency_correct"],
        "consistency_total": metrics["consistency_total"],
        "avg_output_tokens": total_avg_output_tokens,
        "avg_response_length": total_avg_response_length,
        "avg_correct_tokens": correct_avg_tokens,
        "avg_correct_length": correct_avg_length,
        "avg_incorrect_tokens": incorrect_avg_tokens,
        "avg_incorrect_length": incorrect_avg_length,
        "by_subject": subject_metrics,
    }


@app.command()
def analyze(
    jsonl_file: str = typer.Argument(..., help="Path to JSONL results file"),
    dataset: str = typer.Option(
        "EN-EASY", "--dataset", help="Dataset type for language detection"
    ),
):
    """Analyze JSONL results and print evaluation metrics like local_tester.py"""

    # Load JSONL results
    results = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"📊 Loaded {len(results)} records from {jsonl_file}")

    # Filter valid results
    valid_results = [r for r in results if "is_correct" in r]
    error_results = [r for r in results if "error" in r]

    print(f"📊 Valid results: {len(valid_results)}")
    print(f"📊 Error results: {len(error_results)}")

    if not valid_results:
        print("❌ No valid results to analyze")
        return

    # Evaluate results
    final_metrics = evaluate_jsonl_results(valid_results, dataset)

    if "error" in final_metrics:
        print(f"❌ {final_metrics['error']}")
        return

    # Print metrics in the same format as local_tester.py
    language = final_metrics["language"]
    sample_size = final_metrics["sample_size"]

    print(f"\n===== {language} EVALUATION RESULTS =====")
    print(f"pass@1: {final_metrics['pass@1']:.4f}")
    print(f"cons@{sample_size}: {final_metrics['cons@SAMPLE']:.4f}")
    print(f"Avg output tokens: {final_metrics['avg_output_tokens']:.2f}")
    print(f"Avg response length: {final_metrics['avg_response_length']:.2f}")
    print(f"Avg correct output tokens: {final_metrics['avg_correct_tokens']:.2f}")
    print(f"Avg correct response length: {final_metrics['avg_correct_length']:.2f}")
    print(f"Avg incorrect output tokens: {final_metrics['avg_incorrect_tokens']:.2f}")
    print(f"Avg incorrect response length: {final_metrics['avg_incorrect_length']:.2f}")

    print("\nResults by subject:")
    for subject, metrics_data in final_metrics["by_subject"].items():
        print(f"  {subject}:")
        print(f"    pass@1: {metrics_data['pass@1']:.4f}")
        print(f"    cons@{sample_size}: {metrics_data['cons@SAMPLE']:.4f}")
        print(f"    Avg output tokens: {metrics_data['avg_output_tokens']:.2f}")
        print(f"    Avg response length: {metrics_data['avg_response_length']:.2f}")
        print(
            f"    Avg correct output tokens: {metrics_data['avg_correct_tokens']:.2f}"
        )
        print(
            f"    Avg correct response length: {metrics_data['avg_correct_length']:.2f}"
        )
        print(
            f"    Avg incorrect output tokens: {metrics_data['avg_incorrect_tokens']:.2f}"
        )
        print(
            f"    Avg incorrect response length: {metrics_data['avg_incorrect_length']:.2f}"
        )
        print(f"    Total samples: {metrics_data['total_samples']}")

    if error_results:
        print(f"\n❌ Found {len(error_results)} errors:")
        for i, err in enumerate(error_results[:5]):  # Show first 5 errors
            print(f"  {err['unique_id']}-{err['sample_idx']}: {err['error']}")
        if len(error_results) > 5:
            print(f"  ... and {len(error_results) - 5} more errors")


async def run_tool_evaluation(
    min_idx: int,
    max_idx: int,
    samples: int,
    temperature: float,
    concurrency: int,
    dataset: str,
    output: Optional[str],
    max_iterations: int,
):
    """Main tool-based evaluation function"""
    from math_agent_runner import (
        solve_math_problem_async,
        create_math_tool_system_prompt,
    )

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Load dataset
    dataset_file = f"data/OlymMATH-{dataset}.jsonl"
    problems = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            problems.append(json.loads(line))

    # Select problems
    selected = problems[min_idx:max_idx]
    print(f"📊 Loaded {len(selected)} problems from index {min_idx} to {max_idx}")

    # Determine output file
    if output:
        output_file = output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"tool_eval_{timestamp}_{dataset}_{min_idx}-{max_idx}.jsonl"

    # Load completed attempts for resume functionality
    completed_attempts = load_completed_attempts(output_file)

    # Initialize
    is_chinese = dataset.startswith("ZH")

    # Create writer
    writer = AsyncJSONLWriter(output_file)
    await writer.start()

    # Create semaphore
    semaphore = asyncio.Semaphore(concurrency)

    # Create all tasks, skipping completed ones
    tasks = []
    skipped = 0
    for problem in selected:
        for sample_idx in range(samples):
            # Skip if already completed
            if (problem["unique_id"], sample_idx) in completed_attempts:
                print(
                    f"⏭️  Skipping {problem['unique_id']}-{sample_idx} (already completed)"
                )
                skipped += 1
                continue

            # Create messages for tool-based evaluation (English only)
            problem_text = problem["problem"]

            tool_instructions = """You MUST use the submit_math_answer tool to submit your answer. This is the ONLY way your answer will be evaluated.

Work through the problem step by step, then submit your final answer using the tool.

The submit_math_answer tool accepts your mathematical answer as a string.

CRITICAL: Text answers in your response are ignored. You MUST use the submit_math_answer tool.

Problem: """

            full_content = tool_instructions + problem_text + " /think"

            messages = [{"role": "user", "content": full_content}]

            task = solve_math_problem_async(
                unique_id=problem["unique_id"],
                sample_idx=sample_idx,
                messages=messages,
                ground_truth=problem["answer"],
                max_iterations=max_iterations,
                verbose=True,
                writer=writer,
                temperature=temperature,
                semaphore=semaphore,
                problem_data=problem,  # Pass full problem data
            )
            tasks.append(task)

    if skipped > 0:
        print(f"📊 Skipped {skipped} already completed attempts")

    results = []
    if tasks:
        print(f"🏃 Running {len(tasks)} tool-based evaluations...")

        # Run all tasks
        results = await asyncio.gather(*tasks)
        print(f"🏃 Completed {len(results)} evaluations")
    else:
        print("⚠️  No new tasks to run (all already completed)")

    # Stop writer
    await writer.stop()

    print(f"\n📄 Results saved to: {output_file}")
    return results


@app.command()
def evaluate_with_tools(
    min_idx: int = typer.Option(
        ..., "--min", help="Start index for problems (inclusive)"
    ),
    max_idx: int = typer.Option(
        ..., "--max", help="End index for problems (exclusive)"
    ),
    samples: int = typer.Option(10, "--samples", help="Number of samples per question"),
    temperature: float = typer.Option(
        0.6, "--temperature", help="Sampling temperature"
    ),
    concurrency: int = typer.Option(
        10, "--concurrency", help="Maximum concurrent API requests"
    ),
    dataset: str = typer.Option(
        "EN-EASY",
        "--dataset",
        help="Dataset to use (EN-EASY, EN-HARD, ZH-EASY, ZH-HARD)",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", help="Output JSONL file (default: auto-generated)"
    ),
    max_iterations: int = typer.Option(
        3, "--max-iterations", help="Maximum attempts per problem"
    ),
):
    """Run tool-based math evaluation with interactive feedback"""
    results = asyncio.run(
        run_tool_evaluation(
            min_idx,
            max_idx,
            samples,
            temperature,
            concurrency,
            dataset,
            output,
            max_iterations,
        )
    )


if __name__ == "__main__":
    app()
