"""
Supervised Fine-Tuning Data Generator for Abyme

This script uses the DeepSeek API to generate recursive problem-solving traces
for training. It simulates the recursive behavior by pausing at <elaborate> tags,
recursively solving sub-problems, and injecting results back into the trace.
"""

import os
import json
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from tqdm import tqdm
import threading

# Load environment variables from .env file
load_dotenv()

from abyme.core import Abyme_DeepSeek
from abyme.tree_trace import flatten_trace


def call_teacher_model(problem: str, max_depth: int = 10, max_call: int = 10) -> List[Dict[str, str]]:
    """
    Generate training examples from a single problem using Abyme_DeepSeek.

    This function:
    1. Creates an Abyme_DeepSeek recursive model instance
    2. Generates a solution with recursive elaborations
    3. Extracts the trace tree from the generation
    4. Flattens the trace into a list using DFS traversal
    5. Converts each node to a training example with prompt, context, and output

    Args:
        problem: The seed problem to solve
        max_depth: Maximum recursion depth for the model (default: 10)
        max_call: Maximum number of API calls for the model (default: 10)

    Returns:
        List of training examples, each with:
        - prompt: The problem/instruction at this step
        - context: The previously generated context
        - output: The raw output generated at this step

    Raises:
        Exception: If generation fails or if the trace is invalid

    Example:
        >>> examples = call_teacher_model("Solve x^2 + 5x + 6 = 0", max_depth=5, max_call=15)
        >>> # Returns: [
        >>> #   {"prompt": "Solve x^2 + 5x + 6 = 0", "context": "", "output": "..."},
        >>> #   {"prompt": "x^2 + 5x + 6", "context": "", "output": "..."},
        >>> #   ...
        >>> # ]
    """
    # Create Abyme_DeepSeek instance
    model = Abyme_DeepSeek(
        reasoning=False,
        max_depth=max_depth,
        max_call=max_call,
        max_parallel_workers=1
    )

    # Generate solution (with retry mechanism)
    try:
        model.generate(problem, max_attempt=1)
    except Exception as e:
        raise Exception(f"Failed to generate solution for problem '{problem}': {e}")

    # Get the trace from the last generation
    trace = model.trace
    if trace is None:
        raise Exception("No trace available from generation")

    # Flatten the trace using DFS traversal
    flattened_nodes = flatten_trace(trace)

    # Convert each node to a training example
    training_examples = []
    for node in flattened_nodes:
        # Only include nodes that have been generated
        if node.is_generated:
            example = {
                "prompt": node.prompt,
                "context": node.context,
                "output": node.output
            }
            training_examples.append(example)

    return training_examples


def generate_training_dataset(
    seed_file: str = "seed_problems.jsonl",
    output_file: str = "training_data.jsonl",
    overwrite: bool = False,
    num_threads: int = 5,
    max_depth: int = 10,
    max_call: int = 10,
    num_samples: int = 8
):
    """
    Generate training dataset from seed problems using multi-threaded API calls.

    This function:
    1. Reads problems from a JSONL file (each line has a "problem" field)
    2. For each problem, generates num_samples independent samples
    3. Calls call_teacher_model() directly in parallel threads for each sample
    4. If a sample generation fails with an error, it is skipped (not retried)
    5. Writes training examples incrementally to output JSONL file
    6. Shows progress bar during generation
    7. Handles file overwriting based on overwrite flag

    **Process Flow:**
    - Load all problems from seed_file
    - Initialize output file (clear if overwrite=True, append if overwrite=False)
    - Create thread pool with num_threads workers
    - Submit each problem num_samples times to thread pool
    - As each sample completes:
      * If successful: Write all generated examples to output file immediately
      * If failed: Skip and continue (error logged but sample not retried)
      * Update progress bar
    - Display final statistics (including success/failure counts)

    **Thread Safety:**
    - Uses a lock to ensure only one thread writes to the file at a time
    - Each sample is processed independently in its own thread

    **Error Handling:**
    - If a sample fails during generation, it is discarded
    - No retries are attempted for failed samples
    - Total attempts = len(problems) × num_samples (regardless of failures)

    Args:
        seed_file: Path to input JSONL file containing seed problems
                  Each line should be JSON with required "problem" field
                  Example: {"problem": "Solve x+5=10"}
        output_file: Path to output JSONL file for training data
                    Each line will be a training example with:
                    {"instruction": str, "context": str, "output": str}
        overwrite: If True, clear output_file before starting
                  If False, append to existing file (default: False)
        num_threads: Number of parallel threads for API calls (default: 5)
        max_depth: Maximum recursion depth per sample (default: 10)
        max_call: Maximum total API calls per sample (default: 10)
        num_samples: Number of independent samples to generate per problem (default: 8)

    Raises:
        FileNotFoundError: If seed_file doesn't exist
        ValueError: If seed_file is empty or malformed

    Example:
        >>> # Input file (math_problems.jsonl):
        >>> # {"problem": "Solve x+5=10"}
        >>> # {"problem": "Find area of circle with r=5"}
        >>>
        >>> generate_training_dataset(
        ...     seed_file="math_problems.jsonl",
        ...     output_file="math_training.jsonl",
        ...     overwrite=True,
        ...     num_threads=10,
        ...     max_call=15,
        ...     num_samples=5
        ... )
        Loaded 2 problems from math_problems.jsonl
        Processing 2 problems with 5 samples each (10 total attempts)
        Using 10 threads...
        100%|██████████| 10/10 [05:23<00:00,  32.3s/sample]

        ============================================================
        Generation Complete!
        ============================================================
        Total problems: 2
        Samples per problem: 5
        Total attempts: 10
        Successful samples: 9
        Failed samples: 1
        Success rate: 90.0%
        Total training examples: 1,247
        Average examples per successful sample: 138.56
        Output saved to: math_training.jsonl
        ============================================================

        >>> # Output file will contain training examples from successful samples:
        >>> # {"instruction": "Solve x+5=10", "context": "", "output": "..."}
        >>> # {"instruction": "x+5", "context": "...", "output": "..."}
        >>> # ...
    """
    # Load seed problems
    if not os.path.exists(seed_file):
        raise FileNotFoundError(f"Seed file not found: {seed_file}")

    problems = []
    with open(seed_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if 'problem' not in data:
                    print(f"Warning: Line {line_num} missing 'problem' field, skipping")
                    continue
                problems.append(data['problem'])
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} is not valid JSON, skipping. Error: {e}")
                continue

    if not problems:
        raise ValueError(f"No valid problems found in {seed_file}")

    print(f"Loaded {len(problems)} problems from {seed_file}")

    # Handle output file
    if overwrite and os.path.exists(output_file):
        print(f"Overwriting existing file: {output_file}")
        open(output_file, 'w').close()  # Clear the file
    elif os.path.exists(output_file):
        print(f"Appending to existing file: {output_file}")
    else:
        print(f"Creating new file: {output_file}")

    # Thread-safe file writing
    write_lock = threading.Lock()
    total_examples = 0

    def write_examples(examples: List[Dict[str, str]]) -> int:
        """Write examples to file with thread safety."""
        nonlocal total_examples
        if not examples:
            return 0

        with write_lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                for example in examples:
                    json.dump(example, f, ensure_ascii=False)
                    f.write('\n')
            count = len(examples)
            total_examples += count
            return count

    # Process problems with thread pool
    total_tasks = len(problems) * num_samples
    print(f"\nProcessing {len(problems)} problems with {num_samples} samples each ({total_tasks} total attempts)")
    print(f"Using {num_threads} threads...")
    print(f"Max recursion depth: {max_depth}")
    print(f"Max API calls per problem: {max_call}")
    print(f"Output: {output_file}\n")

    failed_count = 0
    success_count = 0

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit each problem num_samples times
        future_to_problem = {}
        for problem in problems:
            for sample_idx in range(num_samples):
                # Submit task to generate training examples using Abyme_DeepSeek
                future = executor.submit(
                    call_teacher_model,
                    problem=problem,
                    max_depth=max_depth,
                    max_call=max_call
                )
                future_to_problem[future] = (problem, sample_idx)


        # Process results as they complete with progress bar
        with tqdm(total=total_tasks, desc="Generating", unit="sample") as pbar:
            for future in as_completed(future_to_problem):
                problem, sample_idx = future_to_problem[future]
                try:
                    # call_teacher_model returns list of training examples
                    examples = future.result()
                    write_examples(examples)
                    success_count += 1
                except Exception as e:
                    failed_count += 1
                    print(f"\n[Sample {sample_idx+1}/{num_samples}] Failed, skipping: {problem[:50]}... Error: {str(e)}")
                finally:
                    pbar.update(1)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Generation Complete!")
    print(f"{'='*60}")
    print(f"Total problems: {len(problems)}")
    print(f"Samples per problem: {num_samples}")
    print(f"Total attempts: {total_tasks}")
    print(f"Successful samples: {success_count}")
    print(f"Failed samples: {failed_count}")
    print(f"Success rate: {success_count/total_tasks*100:.1f}%")
    print(f"Total training examples: {total_examples}")
    if success_count > 0:
        print(f"Average examples per successful sample: {total_examples/success_count:.2f}")
    print(f"Output saved to: {output_file}")
    print(f"{'='*60}\n")

def main():
    examples = call_teacher_model(
        "Find the derivative of f(x) = (x^2 + 1)^(x^3) using logarithmic differentiation, then evaluate f'(1). don't simplify ln()",
        max_depth=10,
        max_call=20
    )
    print(f"\nGenerated {len(examples)} training examples")

if __name__ == "__main__":
    main()