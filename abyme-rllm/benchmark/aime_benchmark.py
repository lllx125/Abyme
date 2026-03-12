"""
AIME Dataset Loader

Purpose: Standardizes the problem format from Hugging Face into a clean usage object.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
from datasets import load_dataset
from abyme.model import Model, DeepSeekModel
from abyme.core import Abyme_DeepSeek
from run_benchmark import pass_at_k, run_all_benchmark, score_result

def normalize_aime_data(dataset) -> List[Dict]:
    """
    Normalize AIME dataset to standard format.

    Args:
        dataset: Loaded dataset object from Hugging Face

    Returns:
        List of dictionaries containing:
            - id: Problem ID (e.g., "2024-I-1")
            - problem: The text of the math problem
            - solution: The ground truth integer
    """
    normalized_data = []
    for idx, item in enumerate(dataset):
        # Normalize column names
        problem_text = item.get("problem") or item.get("question") or item.get("text")
        # Use "answer" field for numeric answer, fallback to "solution"
        answer = item.get("answer") or item.get("solution")
        problem_id = item.get("id") or f"problem_{idx}"

        if problem_text is None:
            raise ValueError(f"Could not find problem text in item {idx}. Available keys: {item.keys()}")

        if answer is None:
            raise ValueError(f"Could not find answer in item {idx}. Available keys: {item.keys()}")

        # Convert answer to integer if it's a string
        if isinstance(answer, str):
            answer = int(normalize_answer(answer) or answer)
        elif not isinstance(answer, int):
            answer = int(answer)

        # Add explicit instruction to put answer in \boxed{} format
        problem_with_instruction = (
            f"{problem_text}\n\n"
            f"Please provide your final answer in the format \\boxed{{answer}}, "
            f"where answer is an integer between 0 and 999."
        )

        normalized_data.append({
            "id": problem_id,
            "problem": problem_with_instruction,
            "solution": answer  # Store as "solution" for compatibility
        })

    return normalized_data



def save_aime_data(
    dataset,
    output_jsonl_path: str,
    overwrite: bool = True
) -> int:
    """
    Save AIME dataset to JSONL file.

    Args:
        dataset: Loaded dataset object from Hugging Face
        output_jsonl_path: Path to output JSONL file
        overwrite: If True, overwrite file; if False, append (default: True)

    Returns:
        Number of problems saved

    Example:
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        save_aime_data(dataset, "data/aime.jsonl", overwrite=True)
    """
    # Normalize data
    data = normalize_aime_data(dataset)

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
        The extracted number as a string, or None if not found
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
    # For now, we'll assume AIME answers are simple integers
    # If nested braces are detected, we need more sophisticated parsing
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


def normalize_answer(extracted_text: str) -> Optional[int]:
    """
    Normalize the extracted answer text to an integer.

    Args:
        extracted_text: The extracted answer text

    Returns:
        The normalized integer, or None if conversion fails
    """
    if not extracted_text:
        return None

    # Strip whitespace
    text = extracted_text.strip()

    # Remove commas (e.g., "1,024" -> "1024")
    text = text.replace(',', '')

    # Remove dollar signs (LaTeX math mode)
    text = text.replace('$', '')

    # Remove any remaining whitespace
    text = text.strip()

    # Try to convert to integer
    try:
        return int(text)
    except ValueError:
        # Try to extract just the numeric part
        # Look for sequences of digits
        numeric_match = re.search(r'-?\d+', text)
        if numeric_match:
            return int(numeric_match.group())
        return None


def aime_decider(model_output: str, json_data: Dict) -> bool:
    """
    Decider function for AIME problems.

    Extracts the answer from \\boxed{...} notation and compares it to the ground truth.

    Args:
        model_output: The raw model output string
        json_data: Dictionary containing 'solution' field with the ground truth

    Returns:
        True if the answer is correct, False otherwise
    """
    ground_truth = json_data['solution']

    # Extract and normalize answer
    extracted = extract_boxed_answer(model_output)
    if extracted:
        normalized = normalize_answer(extracted)
        if normalized is not None and normalized == ground_truth:
            return True

    return False


def aime_pass_at_k(n: int, k: int) -> Callable[[Model, Dict], Tuple[str, float]]:
    """
    Factory function that returns a scoring function for pass@k evaluation on AIME problems.

    Uses the general pass_at_k function with an AIME-specific decider function.

    Args:
        n: Total number of samples to generate
        k: Number of samples to evaluate (k <= n)

    Returns:
        A scoring function with signature (Model, Dict) -> (str, float)

    Example:
        scorer = aime_pass_at_k(n=10, k=5)  # Generate 10, evaluate pass@5
        output, score = scorer(model, json_data)
    """
    return pass_at_k(n=n, k=k, decider_function=aime_decider)


def download_aime_dataset(output_jsonl_path: str = "data/aime.jsonl") -> int:
    """
    Download and save AIME datasets from Hugging Face.

    Downloads AIME 2024, AIME 2025-I, and AIME 2025-II datasets and saves them
    to a single JSONL file.

    Args:
        output_jsonl_path: Path to output JSONL file (default: "data/aime.jsonl")

    Returns:
        Total number of problems saved

    Example:
        total_problems = download_aime_dataset("data/aime.jsonl")
        print(f"Downloaded {total_problems} AIME problems")
    """
    # Load and save 2024 AIME data (overwrite)
    dataset_2024 = load_dataset("HuggingFaceH4/aime_2024", split="train")
    count_2024 = save_aime_data(
        dataset=dataset_2024,
        output_jsonl_path=output_jsonl_path,
        overwrite=True
    )
    print(f"Saved {count_2024} problems from AIME 2024")

    # Load and append 2025 AIME I data
    dataset_2025_i = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
    count_2025_i = save_aime_data(
        dataset=dataset_2025_i,
        output_jsonl_path=output_jsonl_path,
        overwrite=False
    )
    print(f"Appended {count_2025_i} problems from AIME 2025-I")

    # Load and append 2025 AIME II data
    dataset_2025_ii = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")
    count_2025_ii = save_aime_data(
        dataset=dataset_2025_ii,
        output_jsonl_path=output_jsonl_path,
        overwrite=False
    )
    print(f"Appended {count_2025_ii} problems from AIME 2025-II")

    total = count_2024 + count_2025_i + count_2025_ii
    print(f"\nTotal: {total} problems saved to {output_jsonl_path}")

    return total

def run_aime_benchmark(
    results_folder: str = "results/aime_benchmark",
    data_path: str = "data/aime.jsonl"
) -> Dict[str, List[float]]:
    """
    Run AIME benchmark on all defined models with pass@1 scoring.

    This function evaluates the following models:
    - DeepSeek Base (non-reasoning)
    - DeepSeek Reasoning
    - Abyme DeepSeek Base (recursive, non-reasoning)
    - Abyme DeepSeek Reasoning (recursive, reasoning)

    All models are evaluated using pass@1 (n=1, k=1) scoring.

    Args:
        results_folder: Folder to save all benchmark results (default: "results/aime_benchmark")
        data_path: Path to AIME JSONL dataset (default: "data/aime.jsonl")

    Returns:
        Dictionary mapping model names to their score lists

    Example:
        results = run_aime_benchmark(results_folder="results/my_run")
        # Creates:
        # - results/my_run/deepseek_base.jsonl
        # - results/my_run/deepseek_reasoning.jsonl
        # - results/my_run/abyme_deepseek_base.jsonl
        # - results/my_run/abyme_deepseek_reasoning.jsonl
        # - results/my_run/summary.json
    """
    # Create models
    print("Initializing models...")
    DeepSeek_Base = DeepSeekModel(reasoning=False)
    DeepSeek_Reasoning = DeepSeekModel(reasoning=True)
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
    scorer = aime_pass_at_k(n=1, k=1)

    # Configure benchmarks for all models
    benchmark_configs = [
        {
            'input_jsonl_path': data_path,
            'scoring_function': scorer,
            'output_jsonl_path': 'deepseek_base.jsonl',
            'model': DeepSeek_Base,
            'model_name': 'DeepSeek-Base',
            'task_name': 'AIME - DeepSeek Base',
            'overwrite': True
        },
        {
            'input_jsonl_path': data_path,
            'scoring_function': scorer,
            'output_jsonl_path': 'deepseek_reasoning.jsonl',
            'model': DeepSeek_Reasoning,
            'model_name': 'DeepSeek-Reasoning',
            'task_name': 'AIME - DeepSeek Reasoning',
            'overwrite': True
        },
        {
            'input_jsonl_path': data_path,
            'scoring_function': scorer,
            'output_jsonl_path': 'abyme_deepseek_base.jsonl',
            'model': Abyme_DeepSeek_Base,
            'model_name': 'Abyme-DeepSeek-Base',
            'task_name': 'AIME - Abyme DeepSeek Base',
            'overwrite': True
        },
        {
            'input_jsonl_path': data_path,
            'scoring_function': scorer,
            'output_jsonl_path': 'abyme_deepseek_reasoning.jsonl',
            'model': Abyme_DeepSeek_Reasoning,
            'model_name': 'Abyme-DeepSeek-Reasoning',
            'task_name': 'AIME - Abyme DeepSeek Reasoning',
            'overwrite': True
        }
    ]

    # Run all benchmarks in parallel with summary generation
    print(f"\nRunning benchmarks on AIME dataset: {data_path}")
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
    # Download AIME dataset if it doesn't exist
    data_path = "data/aime.jsonl"
    if not Path(data_path).exists():
        print("AIME dataset not found. Downloading...")
        download_aime_dataset(data_path)
    else:
        print(f"AIME dataset found at {data_path}")

    # Run the benchmark
    # print("\nStarting AIME benchmark...")
    # run_aime_benchmark(
    #     results_folder="results/aime_benchmark",
    #     data_path=data_path
    # )
    
    score, score_list = score_result("results/aime_benchmark/deepseek_reasoning.jsonl")
    print(score)
    print(score_list)
    
    