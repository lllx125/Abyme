"""
Supervised Fine-Tuning Data Generator for Abyme

This script uses the DeepSeek API to generate recursive problem-solving traces
for training. It simulates the recursive behavior by pausing at <elaborate> tags,
recursively solving sub-problems, and injecting results back into the trace.
"""

import os
import re
import json
import argparse
from typing import Dict, List, Tuple
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from dotenv import load_dotenv
from tqdm import tqdm
import threading

# Load environment variables from .env file
load_dotenv()


from abyme.utils import extract_elaborations, replace_elaborations_with_responses, format_output, verify_format

# Initialize OpenAI client for DeepSeek API
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

symstem_prompt = '''The user will provide a partial solution.
Your goal is to continue to break the problem into immediate next step sub-problems and put them inside <elaborate> tags.

# ARCHITECTURE RULES
- Plan for immediate next steps and put subproblems and all relevant context inside <elaborate> tags. DO NOT solve the subproblems yourself. 
- End with a </run> tag if there are any subproblems to solve.
- The format for delegation is:
   [Your current thought process and reasoning]
   Elaborations:
   1. [The specific sub-problem 1 query]
   2. [The specific sub-problem 2 query]
   ...(many other elaborations)
    <elaborate>
   [The specific sub-problem 1 query]
   </elaborate>
   <elaborate>
   [The specific sub-problem 2 query]
   </elaborate>
   ...(many other elaborations)
   </run>
- Otherwise, if no elaboration is needed, denote your final concise answer with a </think> tag
   [Your current thought process and reasoning]
   </think>
   [The final answer here]
- The sub-problems in the tag will be parsed and give to external AI, so please put all relevent context in the tag. DO NOT use refer to anything outside the tag. Make sure the problem is solvable with the information inside the tag.


# BEHAVIOR CHECKLIST
- DO NOT elaborate if the problem is trivial and can be solved directly.
- DO NOT elaborate if the answer is provided in user partial solution.
- DO NOT elaborate the same original problem.
- DO NOT elaborate on problems that depends on previous elaborations that have not been solved yet
- Provide enough context inside the <elaboration> tags so that they can be solved without ambiguity. 
- Maintain a list to keep track of your queries concisely but don't repeat the context
- Break down as many immediate next steps as possible.
- keep the final answer after the </think> tag, and keep it concise.
- keep all output clean and concise.
- ONLY elaborate the next step of the problem, do not jump into future steps that requires the solution of the current step.
- DO NOT refer to anything outside the <elaborate> tags: <elaborate> use previous output to solve...</elaborate> **THIS IS NOT ALLOWED**.

# EXAMPLES
## Example 1:
Input:
Differentiate g(x) = (e^x sin(x))/2 with respect to x.

Output:
We need to differentiate g(x) = (e^x sin(x))/2 with respect to x.
g(x) = (1/2) * e^x * sin(x). This is a product of two functions: e^x and sin(x), multiplied by constant 1/2.
We can use product rule: d/dx [u*v] = u' v + u v', where u = e^x, v = sin(x). Then multiply by 1/2.
So derivative: g'(x) = (1/2) * [ d/dx(e^x) * sin(x) + e^x * d/dx(sin(x)) ].
We need to compute d/dx(e^x) and d/dx(sin(x)).
Elaborations:
1. d/dx(e^x)
2. d/dx(sin(x))
<elaborate>
Differentiate e^x with respect to x.
</elaborate> 
<elaborate>
Differentiate sin(x) with respect to x.
</elaborate>
</run>

## Example 2 (No need to elaborate):
Input:
Differentiate e^x with respect to x.

Output:
we can directly differentiate e^x=e^x
</think>
e^x

## Example 3 (Don't skip to future steps):
Input:
Find second derivative of e^{x^2} with respect to x.

Output:
We need to find the second derivative of e^{x^2} with respect to x.
Lets first find the first derivative. 
Elaboration:
1. d/dx(e^{x^2})
<elaborate>
Differentiate e^{x^2} with respect to x.
</elaborate>
</run>

## Example 4 (Provide all necessary context in elaboration):
Input:
Problem:
Differentiate g(x) = (e^x sin(x))/2 with respect to x.
User partial solution (If any):
We need to differentiate g(x) = (e^x sin(x))/2 with respect to x.
g(x) = (1/2) * e^x * sin(x). This is a product of two functions: e^x and sin(x), multiplied by constant 1/2.
We can use product rule: d/dx [u*v] = u' v + u v', where u = e^x, v = sin(x). Then multiply by 1/2.
So derivative: g'(x) = (1/2) * [ d/dx(e^x) * sin(x) + e^x * d/dx(sin(x)) ].
We need to compute d/dx(e^x) and d/dx(sin(x)).
Elaborations:
1. d/dx(e^x)
2. d/dx(sin(x))
<response>
e^x
</response> 
<response>
cos
</response>

Output:
Next, use the product rule to combine the results:
Elaboration:
1. Combine the derivatives using product rule
<elaborate>
Differentiate e^x * sin(x) with respect to x using the product rule, given that d/dx(e^x) = e^x and d/dx(sin(x)) = cos(x).
</elaborate>

'''

def call_teacher_model(problem: str, context: str = "", max_call: int = 10) -> Tuple[str, List[Dict[str, str]], int]:
    """
    Recursively call DeepSeek API to generate training data with elaborations.

    This function implements the recursive teacher model generation process:

    **Generation Process:**
    1. **Initial Generation**: Calls DeepSeek API with problem + context
       - Uses generate_with_check() which validates format and auto-adds </run> if <elaborate> tags exist
       - Retries up to 3 times if format validation fails

    2. **Extract Elaborations**: Checks output for <elaborate>...</elaborate> tags
       - If no elaborations OR no </run> tag: Returns formatted output immediately
       - If elaborations exist: Proceeds to recursive processing

    3. **Recursive Sub-problem Processing**:
       - For each elaboration content, recursively calls call_teacher_model()
       - Passes the elaboration as new problem with context=current_context+current_output
       - Properly tracks remaining API call budget across all recursive calls
       - Tracks actual consumption of API calls through recursion
       - Collects all sub-responses and their training examples

    4. **Replace & Continue**:
       - Replaces all <elaborate>...</elaborate> tags with <response>sub_response</response>
       - Removes the final </run> tag
       - Makes another recursive call to continue generation with updated context
       - This allows the model to see previous responses and generate further elaborations if needed

    5. **Training Data Collection**:
       - Each generation step creates a training example with:
         * instruction: The problem being solved
         * context: Previously generated text (empty for initial call)
         * output: The newly generated text
       - Sub-problems also create their own training examples
       - All examples are collected and returned in result_list

    **Example Flow:**
    ```
    Problem: "Solve 5*5 + 3*3"

    Call 1: call_teacher_model("Solve 5*5 + 3*3", "", 10)
      -> API Call #1, 9 remaining
      -> Generates: "<think>Break into parts<elaborate>5*5</elaborate><elaborate>3*3</elaborate></run>"
      -> Adds to result_list: {"instruction": "Solve 5*5 + 3*3", "context": "", "output": "..."}
      -> Extracts elaborations: ["5*5", "3*3"]

    Call 2a: call_teacher_model("5*5", "...previous output...", 9)
      -> API Call #2, 8 remaining
      -> Generates: "25"
      -> Adds: {"instruction": "5*5", "context": "...", "output": "25"}
      -> Returns: consumed 1 call

    Call 2b: call_teacher_model("3*3", "...previous output...", 8)
      -> API Call #3, 7 remaining
      -> Generates: "9"
      -> Adds: {"instruction": "3*3", "context": "...", "output": "9"}
      -> Returns: consumed 1 call

    Back to Call 1:
      -> Replaces elaborations with responses
      -> Output becomes: "<think>Break into parts<response>25</response><response>9</response>"
      -> Removes </run>
      -> 7 calls remaining after subproblems

    Call 3: call_teacher_model("Solve 5*5 + 3*3", "...output with responses...", 7)
      -> API Call #4, 6 remaining
      -> Generates: "25 + 9 = 34</think>\nThe answer is 34"
      -> Adds: {"instruction": "Solve 5*5 + 3*3", "context": "...", "output": "..."}
      -> Returns: consumed 1 call

    Total calls consumed: 4
    Returns: ("The answer is 34", [all training examples], 4)
    ```

    Args:
        problem: The problem/instruction to solve at this recursion level
        context: Previously generated text to provide as context (empty string for initial call)
        max_call: Maximum remaining API calls to prevent infinite recursion

    Returns:
        Tuple containing:
        - str: Final formatted output (content after final </think> tag, with EOS token removed)
        - List[Dict[str, str]]: Training examples, each with keys:
            * "instruction": The problem being solved
            * "context": Previously generated context
            * "output": The generated output for this step
        - int: Number of API calls actually consumed in this invocation and all recursive calls

    Raises:
        ValueError: If format validation fails after max_attempts (3) retries
        ValueError: If API returns None response

    Notes:
        - Requires DEEPSEEK_API_KEY environment variable
        - Uses verify_format() for strict XML tag validation
        - Automatically appends </run> when <elaborate> tags are detected
        - Properly tracks actual API call consumption through all recursion levels
        - Guarantees total calls will never exceed max_call parameter
    """

    # Check if we have any calls remaining
    if max_call <= 0:
        return context, [], 0

    result_list = []
    total_calls_consumed = 0

    # Make the current API call
    try:
        output = generate_with_check(problem, context)
    except Exception as e:
        raise ValueError(f"Failed to generate valid output for problem: {problem[:50]}... Error: {str(e)}")
        
        
    total_calls_consumed += 1  # Count this API call
    remaining_calls = max_call - total_calls_consumed

    # Store the current problem, context, and output as a training example
    result_list.append({
        "instruction": problem,
        "context": context,
        "output": output
    })

    # Extract elaborations and recursively solve them
    subproblems = extract_elaborations(output)
    if len(subproblems) == 0 or '</run>' not in output:
        return format_output(context + output), result_list, total_calls_consumed

    sub_responses: List[str] = []

    # Process each subproblem sequentially until we run out of calls
    for sub in subproblems:
        if remaining_calls <= 0:
            # No more calls available, use empty response
            sub_responses.append("")
            continue

        # Recursively solve the subproblem with all remaining budget
        sub_response, sub_results, sub_calls = call_teacher_model(sub, "", remaining_calls)
        sub_responses.append(sub_response)
        result_list.extend(sub_results)

        # Update remaining calls based on actual consumption
        total_calls_consumed += sub_calls
        remaining_calls = max_call - total_calls_consumed

    # Continue generating the current level output
    # Replace elaborations with their responses
    output = replace_elaborations_with_responses(output, sub_responses)

    # Remove the end </run> tag
    if output.endswith('</run>'):
        output = output[:-6]

    # Continue generation with previous output as context
    # Use remaining call budget for continuation
    if remaining_calls > 0:
        output, future_results, future_calls = call_teacher_model(problem, output, remaining_calls)
        result_list.extend(future_results)
        total_calls_consumed += future_calls

    return output, result_list, total_calls_consumed

def generate_with_check(problem: str, context: str, attempt: int = 0, max_attempts: int = 3) -> str:
    """
    Generate text using DeepSeek API with automatic format validation and retry logic.

    This function:
    1. Calls DeepSeek API with the input text as prompt
    2. Automatically appends </run> if <elaborate> tags are detected in output
    3. Validates the output format using verify_format()
    4. Retries generation if validation fails (up to max_attempts)

    The function ensures that all generated outputs have valid XML-like tag structure
    according to the rules defined in verify_format().

    Args:
        problem: The problem text to send to the API
        context: The context text to send to the API
        attempt: Current attempt number (used internally for recursion, default: 0)
        max_attempts: Maximum number of retry attempts if format is invalid (default: 3)

    Returns:
        str: Generated text with valid format, potentially with </run> appended

    Raises:
        ValueError: If format validation fails after max_attempts retries
        ValueError: If API returns None response

    Notes:
        - Uses DeepSeek API via OpenAI-compatible client
        - Temperature set to 0.7 for balanced creativity
        - Automatically adds </run> when elaborations are detected to trigger parallel processing
        - Format validation ensures proper XML tag nesting and closure
    """
    if attempt >= max_attempts:
        raise ValueError(f"Failed to generate valid format after {max_attempts} attempts")

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages = [
            # 1. System Instruction
            {"role": "system", "content": symstem_prompt}, # Split rules from problem

            # 2. The Original User Problem
            {"role": "user", "content": f"now solve this problem: {problem} \n User partial solution: {context}"},
        ],
        temperature=0.7,
    )

    output = response.choices[0].message.content
    
    print(f"problem:\n{problem}\ncontext:\n{context}\noutput:\n{output}\n{'='*50}")

    if output is None:
        raise ValueError("API returned None response")

    # # Check format, make sure the new generated output does not contain <response> tag
    # if not verify_format(context + output) and ("<response>" not in output or "</response>" not in output):
    #     # Regenerate if format is invalid
    #     return generate_with_check(problem, context, attempt + 1, max_attempts)

    return output

def generate_training_dataset(
    seed_file: str = "seed_problems.jsonl",
    output_file: str = "training_data.jsonl",
    overwrite: bool = False,
    num_threads: int = 5,
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
        max_call: Maximum recursive API calls per sample (default: 10)
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
    print(f"Max recursive calls per problem: {max_call}")
    print(f"Output: {output_file}\n")

    failed_count = 0
    success_count = 0

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit each problem num_samples times to call_teacher_model
        future_to_problem = {}
        for problem in problems:
            for sample_idx in range(num_samples):
                future = executor.submit(call_teacher_model, problem, "", max_call)
                future_to_problem[future] = (problem, sample_idx)

        # Process results as they complete with progress bar
        with tqdm(total=total_tasks, desc="Generating", unit="sample") as pbar:
            for future in as_completed(future_to_problem):
                problem, sample_idx = future_to_problem[future]
                try:
                    # call_teacher_model returns (final_output, result_list, calls_consumed)
                    # We only need result_list for training data
                    _, examples, calls_consumed = future.result()
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
    call_teacher_model("Find the derivative of f(x) = (x^2 + 1)^(x^3) using logarithmic differentiation, then evaluate f'(1). don't simplify ln()", "", 20)

if __name__ == "__main__":
    main()