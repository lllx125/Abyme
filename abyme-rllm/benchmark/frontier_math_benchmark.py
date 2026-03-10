"""
FrontierMath Dataset Loader

Purpose: Standardizes the FrontierMath problem format from Hugging Face into a clean usage object.
FrontierMath is a benchmark of original, expert-level mathematics problems designed to evaluate
AI models on advanced mathematical reasoning.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
from datasets import load_dataset
from abyme.model import Model, deepseek
from abyme.core import Abyme_DeepSeek
from run_benchmark import pass_at_k, run_all_benchmark


def normalize_frontier_math_data(dataset) -> List[Dict]:
    """
    Normalize FrontierMath dataset to standard format.

    Args:
        dataset: Loaded dataset object from Hugging Face

    Returns:
        List of dictionaries containing:
            - id: Problem ID
            - problem: The text of the math problem (with boxed instruction)
            - solution: The ground truth answer
    """
    normalized_data = []
    for idx, item in enumerate(dataset):
        # Normalize column names
        problem_text = item.get("problem") or item.get("question") or item.get("text")
        # FrontierMath may use "answer" or "solution" field
        answer = item.get("answer") or item.get("solution")
        problem_id = item.get("id") or item.get("problem_id") or f"problem_{idx}"

        if problem_text is None:
            raise ValueError(f"Could not find problem text in item {idx}. Available keys: {item.keys()}")

        if answer is None:
            raise ValueError(f"Could not find answer in item {idx}. Available keys: {item.keys()}")

        # Store the answer as-is (could be integer, string, or other format)
        # We'll handle normalization in the decider function
        if isinstance(answer, (int, float)):
            solution = answer
        else:
            solution = str(answer).strip()

        # Add explicit instruction to put answer in \boxed{} format
        problem_with_instruction = (
            f"{problem_text}\n\n"
            f"Please provide your final answer in the format \\boxed{{answer}}. "
            f"Ensure your answer is simplified and in the correct mathematical form."
        )

        normalized_data.append({
            "id": problem_id,
            "problem": problem_with_instruction,
            "solution": solution
        })

    return normalized_data


def save_frontier_math_data(
    dataset,
    output_jsonl_path: str,
    overwrite: bool = True
) -> int:
    """
    Save FrontierMath dataset to JSONL file.

    Args:
        dataset: Loaded dataset object from Hugging Face
        output_jsonl_path: Path to output JSONL file
        overwrite: If True, overwrite file; if False, append (default: True)

    Returns:
        Number of problems saved

    Example:
        dataset = load_dataset("epoch-ai/frontiermath-public", split="train")
        save_frontier_math_data(dataset, "data/frontiermath.jsonl", overwrite=True)
    """
    # Normalize data
    data = normalize_frontier_math_data(dataset)

    # Convert to Path object
    output_path = Path(output_jsonl_path)

    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine file mode
    mode = 'w' if overwrite else 'a'

    # Write to JSONL file
    with output_path.open(mode) as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

    return len(data)


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract the answer from LaTeX \\boxed{...} notation.

    Searches from the end of the string backwards to get the final answer,
    as Chain-of-Thought models might hallucinate boxed numbers in intermediate steps.

    Args:
        text: The raw model output

    Returns:
        The extracted content as a string, or None if not found
    """
    if not text:
        return None

    # Pattern to match \boxed{...} with potential nested braces
    # We'll use a simpler approach: find all \boxed{...} and take the last one
    pattern = r'\\boxed\{([^}]*)\}'

    matches = list(re.finditer(pattern, text))

    if not matches:
        return None

    # Take the last match (searching from the end backwards)
    last_match = matches[-1]
    content = last_match.group(1)

    # Handle nested braces if necessary
    if '{' in content or '}' in content:
        # Try to handle simple nested cases
        # Count braces to extract the full content
        start_pos = last_match.start()
        brace_count = 0
        result_chars = []
        i = start_pos + len('\\boxed{')

        while i < len(text):
            char = text[i]
            if char == '{':
                brace_count += 1
                result_chars.append(char)
            elif char == '}':
                if brace_count == 0:
                    break
                brace_count -= 1
                result_chars.append(char)
            else:
                result_chars.append(char)
            i += 1

        content = ''.join(result_chars)

    return content.strip()


def normalize_answer(extracted_text: str) -> Optional[str]:
    """
    Normalize the extracted answer text.

    For FrontierMath, answers can be integers, expressions, or other mathematical objects.
    This function performs basic normalization while preserving the answer format.

    Args:
        extracted_text: The extracted answer text

    Returns:
        The normalized string, or None if empty
    """
    if not extracted_text:
        return None

    # Strip whitespace
    text = extracted_text.strip()

    # Remove dollar signs (LaTeX math mode)
    text = text.replace('$', '')

    # Remove any remaining leading/trailing whitespace
    text = text.strip()

    return text if text else None


def verify_answer(model_output: str, ground_truth) -> bool:
    """
    Verify if the model output matches the ground truth.

    This is the deterministic verifier for FrontierMath.
    Handles both integer answers and string-based answers.

    Args:
        model_output: The extracted and normalized model output
        ground_truth: The ground truth answer (can be int, float, or string)

    Returns:
        True if the answer is correct, False otherwise
    """
    if not model_output:
        return False

    # Normalize both model output and ground truth
    model_answer = model_output.strip()

    # Try integer comparison first (most common case)
    try:
        model_int = int(model_answer.replace(',', ''))
        if isinstance(ground_truth, (int, float)):
            return model_int == int(ground_truth)
        else:
            truth_int = int(str(ground_truth).replace(',', ''))
            return model_int == truth_int
    except ValueError:
        pass

    # Try float comparison with tolerance
    try:
        import math
        model_float = float(model_answer.replace(',', ''))
        if isinstance(ground_truth, (int, float)):
            return math.isclose(model_float, float(ground_truth), rel_tol=1e-6, abs_tol=1e-9)
        else:
            truth_float = float(str(ground_truth).replace(',', ''))
            return math.isclose(model_float, truth_float, rel_tol=1e-6, abs_tol=1e-9)
    except ValueError:
        pass

    # Fall back to string comparison (case-insensitive, whitespace-normalized)
    model_normalized = ' '.join(model_answer.lower().split())
    truth_normalized = ' '.join(str(ground_truth).lower().split())

    return model_normalized == truth_normalized


def frontier_math_decider(model_output: str, json_data: Dict) -> bool:
    """
    Decider function for FrontierMath problems.

    Extracts the answer from \\boxed{...} notation and compares it to the ground truth
    using the deterministic verifier.

    Args:
        model_output: The raw model output string
        json_data: Dictionary containing 'solution' field with the ground truth

    Returns:
        True if the answer is correct, False otherwise
    """
    ground_truth = json_data['solution']

    # Extract answer from \boxed{...}
    extracted = extract_boxed_answer(model_output)
    if not extracted:
        return False

    # Normalize the extracted answer
    normalized = normalize_answer(extracted)
    if not normalized:
        return False

    # Verify using deterministic verifier
    return verify_answer(normalized, ground_truth)


def frontier_math_pass_at_k(n: int, k: int) -> Callable[[Model, Dict], Tuple[str, float]]:
    """
    Factory function that returns a scoring function for pass@k evaluation on FrontierMath problems.

    Uses the general pass_at_k function with a FrontierMath-specific decider function.

    Args:
        n: Total number of samples to generate
        k: Number of samples to evaluate (k <= n)

    Returns:
        A scoring function with signature (Model, Dict) -> (str, float)

    Example:
        scorer = frontier_math_pass_at_k(n=10, k=5)  # Generate 10, evaluate pass@5
        output, score = scorer(model, json_data)
    """
    return pass_at_k(n=n, k=k, decider_function=frontier_math_decider)


def download_frontier_math_dataset(output_jsonl_path: str = "data/frontiermath.jsonl") -> int:
    """
    Download and save FrontierMath public dataset from Hugging Face.

    Downloads the public evaluation suite from epoch-ai/frontiermath-public.
    Note: This is only the public subset, not the full 350-problem private set.

    Args:
        output_jsonl_path: Path to output JSONL file (default: "data/frontiermath.jsonl")

    Returns:
        Total number of problems saved

    Example:
        total_problems = download_frontier_math_dataset("data/frontiermath.jsonl")
        print(f"Downloaded {total_problems} FrontierMath problems")
    """
    print("Downloading FrontierMath public dataset from Hugging Face...")
    print("Note: This is the public evaluation suite, not the full private set.")

    # Load the public dataset from Hugging Face
    # The exact split name may vary - adjust as needed
    try:
        dataset = load_dataset("epoch-ai/frontiermath-public", split="train")
    except Exception as e:
        print(f"Error loading dataset with split='train': {e}")
        print("Trying to load without specifying split...")
        dataset = load_dataset("epoch-ai/frontiermath-public")
        # If it returns a DatasetDict, take the first split
        if hasattr(dataset, 'keys'):
            split_name = list(dataset.keys())[0]
            print(f"Using split: {split_name}")
            dataset = dataset[split_name]

    count = save_frontier_math_data(
        dataset=dataset,
        output_jsonl_path=output_jsonl_path,
        overwrite=True
    )
    print(f"Saved {count} problems from FrontierMath public dataset")
    print(f"Total: {count} problems saved to {output_jsonl_path}")

    return count


def run_frontier_math_benchmark(
    results_folder: str = "results/frontiermath_benchmark",
    data_path: str = "data/frontiermath.jsonl"
) -> Dict[str, List[float]]:
    """
    Run FrontierMath benchmark on all defined models with pass@1 scoring.

    This function evaluates the following models:
    - DeepSeek Base (non-reasoning)
    - DeepSeek Reasoning
    - Abyme DeepSeek Base (recursive, non-reasoning)
    - Abyme DeepSeek Reasoning (recursive, reasoning)

    All models are evaluated using pass@1 (n=1, k=1) scoring.

    Args:
        results_folder: Folder to save all benchmark results (default: "results/frontiermath_benchmark")
        data_path: Path to FrontierMath JSONL dataset (default: "data/frontiermath.jsonl")

    Returns:
        Dictionary mapping model names to their score lists

    Example:
        results = run_frontier_math_benchmark(results_folder="results/my_run")
        # Creates:
        # - results/my_run/deepseek_base.jsonl
        # - results/my_run/deepseek_reasoning.jsonl
        # - results/my_run/abyme_deepseek_base.jsonl
        # - results/my_run/abyme_deepseek_reasoning.jsonl
        # - results/my_run/summary.json
    """
    # Create models
    print("Initializing models...")
    DeepSeek_Base = deepseek(reasoning=False)
    DeepSeek_Reasoning = deepseek(reasoning=True)
    Abyme_DeepSeek_Base = Abyme_DeepSeek(
        reasoning=False,
        max_parallel_workers=6,
        max_depth=20,
        max_call=100
    )
    Abyme_DeepSeek_Reasoning = Abyme_DeepSeek(
        reasoning=True,
        max_parallel_workers=3,
        max_depth=20,
        max_call=100
    )

    # Create pass@1 scorer (n=1, k=1)
    scorer = frontier_math_pass_at_k(n=1, k=1)

    # Configure benchmarks for all models
    benchmark_configs = [
        {
            'input_jsonl_path': data_path,
            'scoring_function': scorer,
            'output_jsonl_path': 'deepseek_base.jsonl',
            'model': DeepSeek_Base,
            'model_name': 'DeepSeek-Base',
            'task_name': 'FrontierMath - DeepSeek Base',
            'overwrite': True
        },
        {
            'input_jsonl_path': data_path,
            'scoring_function': scorer,
            'output_jsonl_path': 'deepseek_reasoning.jsonl',
            'model': DeepSeek_Reasoning,
            'model_name': 'DeepSeek-Reasoning',
            'task_name': 'FrontierMath - DeepSeek Reasoning',
            'overwrite': True
        },
        {
            'input_jsonl_path': data_path,
            'scoring_function': scorer,
            'output_jsonl_path': 'abyme_deepseek_base.jsonl',
            'model': Abyme_DeepSeek_Base,
            'model_name': 'Abyme-DeepSeek-Base',
            'task_name': 'FrontierMath - Abyme DeepSeek Base',
            'overwrite': True
        },
        {
            'input_jsonl_path': data_path,
            'scoring_function': scorer,
            'output_jsonl_path': 'abyme_deepseek_reasoning.jsonl',
            'model': Abyme_DeepSeek_Reasoning,
            'model_name': 'Abyme-DeepSeek-Reasoning',
            'task_name': 'FrontierMath - Abyme DeepSeek Reasoning',
            'overwrite': True
        }
    ]

    # Run all benchmarks in parallel with summary generation
    print(f"\nRunning benchmarks on FrontierMath dataset: {data_path}")
    print(f"Results will be saved to: {results_folder}\n")

    results = run_all_benchmark(
        benchmark_configs=benchmark_configs,
        max_workers=4,  # Run up to 4 models in parallel
        results_folder=results_folder,
        generate_summary=True
    )

    print("\nBenchmark completed!")
    print(f"Results saved to {results_folder}")
    print(f"Summary file: {results_folder}/summary.json")

    return results


if __name__ == "__main__":
    # Download FrontierMath dataset if it doesn't exist
    data_path = "data/frontiermath.jsonl"
    if not Path(data_path).exists():
        print("FrontierMath dataset not found. Downloading...")
        download_frontier_math_dataset(data_path)
    else:
        print(f"FrontierMath dataset found at {data_path}")

    # Run the benchmark
    # print("\nStarting FrontierMath benchmark...")
    # run_frontier_math_benchmark(
    #     results_folder="results/frontiermath_benchmark",
    #     data_path=data_path
    # )
