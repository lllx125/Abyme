"""
Benchmark Runner

Purpose: Run benchmarks on JSONL datasets with any model and scoring function.
"""

import json
from pathlib import Path
from typing import Callable, Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from math import comb
from abyme.model import Model


def pass_at_k(
    n: int,
    k: int,
    decider_function: Callable[[str, Dict], bool]
) -> Callable[[Model, Dict], Tuple[str, float]]:
    """
    Factory function that returns a general scoring function for pass@k evaluation.

    This function creates a scorer that generates n samples and calculates the probability
    that at least 1 of k randomly selected samples is correct using the formula:
    pass@k = 1 - C(n-c, k) / C(n, k)
    where n is total samples generated, c is correct samples, k is samples evaluated.

    Args:
        n: Total number of samples to generate
        k: Number of samples to evaluate (k <= n)
        decider_function: Function with signature (model_output: str, json_data: dict) -> bool
                         Returns True if the model output is correct, False otherwise

    Returns:
        A scoring function with signature (Model, Dict) -> (str, float)

    Example:
        def my_decider(output: str, data: dict) -> bool:
            return extract_answer(output) == data['ground_truth']

        scorer = pass_at_k(n=10, k=5, decider_function=my_decider)
        output, score = scorer(model, json_data)
    """
    if k > n:
        raise ValueError(f"k ({k}) must be <= n ({n})")

    def scoring_function(model: Model, json_data: Dict) -> Tuple[str, float]:
        """
        Generate n solutions and calculate pass@k score.

        Args:
            model: Model instance to generate responses
            json_data: Dictionary containing problem data

        Returns:
            Tuple of (concatenated_outputs, pass@k_score)
            - concatenated_outputs: All n outputs concatenated
            - pass@k_score: Probability that at least 1 of k samples is correct
        """
        problem = json_data['problem']

        outputs = []
        c = 0  # Count of correct samples

        for i in range(n):
            # Generate output with error handling
            try:
                output = model.generate(problem, max_attempt=1)
                outputs.append(f"=== Sample {i+1}/{n} ===\n{output}")

                # Check if output is correct using decider function
                if decider_function(output, json_data):
                    c += 1
            except Exception as e:
                # If model throws an error, count it as wrong answer
                error_message = f"ERROR: {str(e)}"
                outputs.append(f"=== Sample {i+1}/{n} ===\n{error_message}")
                # c is not incremented, so this counts as a wrong answer

        # Concatenate all outputs
        combined_output = "\n\n".join(outputs)

        # Calculate pass@k score using the formula: pass@k = 1 - C(n-c, k) / C(n, k)
        if n - c < k:
            score = 1.0
        else:
            # Formula: 1 - (comb(n-c, k) / comb(n, k))
            # math.comb is efficient and handles large integers
            negative_space = comb(n - c, k)
            total_space = comb(n, k)
            score = 1.0 - (negative_space / total_space)

        return combined_output, float(score)

    return scoring_function


def run_benchmark(
    input_jsonl_path: str,
    scoring_function: Callable[[Model, Dict[str, Any]], Tuple[str,float]],
    output_jsonl_path: str,
    model: Model,
    overwrite: bool = True,
    task_name: Optional[str] = None,
    results_folder: Optional[str] = None
) -> List[float]:
    """
    Run a benchmark on a JSONL dataset with progress bar.

    Args:
        input_jsonl_path: Path to input JSONL file containing problems
        scoring_function: Function that takes (model: Model, json_data: dict) -> (output: str, score: float)
        output_jsonl_path: Path to output JSONL file to write results
        model: Model instance to generate responses
        overwrite: If True, overwrite; if False, append (default: True)
        task_name: Optional name for the progress bar (default: uses input filename)
        results_folder: Optional folder to place results in. If provided, output_jsonl_path
                       will be placed inside this folder (default: None)

    Returns:
        List of scores for each problem

    Raises:
        FileNotFoundError: If input file does not exist
        ValueError: If JSON line is missing 'problem' field
    """
    # Convert to Path objects
    input_path = Path(input_jsonl_path)

    # If results_folder is specified, put output inside it
    if results_folder:
        output_path = Path(results_folder) / output_jsonl_path
    else:
        output_path = Path(output_jsonl_path)

    # Check if input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Count total lines for progress bar
    with input_path.open('r') as f:
        total_lines = sum(1 for line in f if line.strip())

    scores = []

    # Determine file mode: overwrite or append
    mode = 'w' if overwrite else 'a'

    # Progress bar description
    desc = task_name or f"Benchmark: {input_path.name}"

    # Open output file for writing
    with output_path.open(mode) as outfile:
        # Read input file line by line with progress bar
        with input_path.open('r') as infile:
            for line in tqdm(infile, total=total_lines, desc=desc):
                # Skip empty lines
                if not line.strip():
                    continue

                # Parse JSON
                json_data = json.loads(line.strip())

                # Score the output
                output, score = scoring_function(model, json_data)
                scores.append(score)

                # Prepare output record
                output_record = {
                    **json_data,  # Include all original fields
                    'model_output': output,
                    'score': score
                }

                # Write to output file immediately
                outfile.write(json.dumps(output_record) + '\n')
                outfile.flush()  # Ensure it's written immediately

    return scores


def score_result(result_file_path: str) -> Tuple[float, List[float]]:
    """
    Read scores from a result JSONL file and calculate average score.

    Args:
        result_file_path: Path to the result JSONL file containing 'score' fields

    Returns:
        Tuple of (average_score, score_list)
        - average_score: Mean of all scores in the file
        - score_list: List of all individual scores

    Raises:
        FileNotFoundError: If result file does not exist
        ValueError: If file is empty or contains no valid scores
    """
    result_path = Path(result_file_path)

    if not result_path.exists():
        raise FileNotFoundError(f"Result file does not exist: {result_path}")

    scores = []

    with result_path.open('r') as f:
        for line in f:
            if not line.strip():
                continue

            json_data = json.loads(line.strip())
            if 'score' in json_data:
                scores.append(json_data['score'])

    if not scores:
        raise ValueError(f"No valid scores found in {result_path}")

    average_score = sum(scores) / len(scores)
    return average_score, scores


def generate_summary_file(
    results: Dict[str, List[float]],
    benchmark_configs: List[Dict[str, Any]],
    results_folder: str
) -> None:
    """
    Generate a summary JSON file with average scores and score lists for each model.

    Args:
        results: Dictionary mapping task names to their score lists
        benchmark_configs: List of benchmark configurations
        results_folder: Folder to save the summary file

    Creates a file named 'summary.json' in the results folder with the following structure:
    {
        "model_name": {
            "average_score": 0.85,
            "scores": [0.8, 0.9, 0.85],
            "num_problems": 3
        },
        ...
    }
    """
    # Create results folder if it doesn't exist
    results_path = Path(results_folder)
    results_path.mkdir(parents=True, exist_ok=True)

    # Group results by model name
    model_summaries = {}

    for config in benchmark_configs:
        task_name = config.get('task_name') or config['input_jsonl_path']
        model_name = config.get('model_name', 'unknown_model')

        if task_name in results and results[task_name]:
            scores = results[task_name]

            if model_name not in model_summaries:
                model_summaries[model_name] = {
                    'scores': [],
                    'tasks': []
                }

            model_summaries[model_name]['scores'].extend(scores)
            model_summaries[model_name]['tasks'].append({
                'task_name': task_name,
                'average_score': sum(scores) / len(scores) if scores else 0.0,
                'num_problems': len(scores)
            })

    # Calculate overall averages for each model
    summary = {}
    for model_name, data in model_summaries.items():
        all_scores = data['scores']
        summary[model_name] = {
            'average_score': sum(all_scores) / len(all_scores) if all_scores else 0.0,
            'scores': all_scores,
            'num_problems': len(all_scores),
            'tasks': data['tasks']
        }

    # Write summary to file
    summary_path = results_path / 'summary.json'
    with summary_path.open('w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {summary_path}")


def run_all_benchmark(
    benchmark_configs: List[Dict[str, Any]],
    max_workers: Optional[int] = None,
    results_folder: Optional[str] = None,
    generate_summary: bool = True
) -> Dict[str, List[float]]:
    """
    Run multiple benchmarks in parallel with progress bars for each task.

    Args:
        benchmark_configs: List of dictionaries, each containing:
            - input_jsonl_path: str
            - scoring_function: Callable[[Model, Dict[str, Any]], Tuple[str, float]]
            - output_jsonl_path: str
            - model: Model
            - model_name: str (optional, used for summary file)
            - overwrite: bool (optional, default: True)
            - task_name: str (optional, used for progress bar description)
        max_workers: Maximum number of parallel workers (default: None, uses ThreadPoolExecutor default)
        results_folder: Optional folder to place all results in (default: None)
        generate_summary: If True, generate a summary file in the results folder (default: True)

    Returns:
        Dictionary mapping task names to their score lists

    Example:
        configs = [
            {
                'input_jsonl_path': 'data/test1.jsonl',
                'scoring_function': my_scorer,
                'output_jsonl_path': 'test1_out.jsonl',
                'model': model1,
                'model_name': 'gpt-4',
                'task_name': 'Test 1'
            },
            {
                'input_jsonl_path': 'data/test2.jsonl',
                'scoring_function': my_scorer,
                'output_jsonl_path': 'test2_out.jsonl',
                'model': model2,
                'model_name': 'claude',
                'task_name': 'Test 2'
            }
        ]
        results = run_all_benchmark(configs, max_workers=2, results_folder='results/run1')
    """
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_config = {
            executor.submit(
                run_benchmark,
                config['input_jsonl_path'],
                config['scoring_function'],
                config['output_jsonl_path'],
                config['model'],
                config.get('overwrite', True),
                config.get('task_name'),
                results_folder
            ): config
            for config in benchmark_configs
        }

        # Wait for all tasks to complete
        for future in as_completed(future_to_config):
            config = future_to_config[future]
            task_name = config.get('task_name') or config['input_jsonl_path']

            try:
                # Wait for benchmark to complete
                future.result()

                # Read scores from the result file using score_result function
                if results_folder:
                    result_file = Path(results_folder) / config['output_jsonl_path']
                else:
                    result_file = Path(config['output_jsonl_path'])

                avg_score, scores = score_result(str(result_file))
                results[task_name] = scores
                print(f"\n{task_name}: Average score = {avg_score:.4f} ({len(scores)} problems)")

            except Exception as e:
                print(f"\nError in task '{task_name}': {e}")
                results[task_name] = []

    # Generate summary file if requested and results_folder is provided
    if generate_summary and results_folder:
        generate_summary_file(results, benchmark_configs, results_folder)

    return results
