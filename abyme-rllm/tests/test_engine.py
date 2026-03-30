"""
test_engine.py
Tests the core RecursiveEngine, continue_from_node, and analyzes trace lengths.
"""
import os
import json
import statistics
from collections import Counter
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))
from abyme.vllm_model import LocalVLLMModel
from abyme.recursive_engine import RecursiveEngine
from abyme.tree_trace import dict_to_node, length

BASE_MODEL = "Lixing-Li/Abyme-Qwen3.5-9B-SFT"

def analyze_length_distribution(jsonl_path: str):
    """Analyzes and prints the distribution of trace lengths from a batch run."""
    lengths = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                lengths.append(data["metrics"]["length"])
    
    if not lengths:
        print("No traces found to analyze.")
        return

    print(f"\n--- Trace Length Distribution ({len(lengths)} traces) ---")
    print(f"Min Length : {min(lengths)} chars")
    print(f"Max Length : {max(lengths)} chars")
    print(f"Mean Length: {statistics.mean(lengths):.2f} chars")
    if len(lengths) > 1:
        print(f"Std Dev    : {statistics.stdev(lengths):.2f} chars")
    print("-" * 50)

    freq = Counter(lengths)
    xs = sorted(freq.keys())
    ys = [freq[x] for x in xs]

    plt.figure(figsize=(10, 5))
    plt.bar(xs, ys, width=max(1, (xs[-1] - xs[0]) / (len(xs) * 2)))
    plt.xlabel("Length (chars)")
    plt.ylabel("Frequency")
    plt.title(f"Trace Length Distribution ({len(lengths)} traces)")
    plt.tight_layout()
    plot_path = jsonl_path.replace(".jsonl", "_lengths.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Bar chart saved to {plot_path}")

if __name__ == "__main__":
    print("Loading model for Engine Test...")
    model = LocalVLLMModel(model_path=BASE_MODEL)
    
    engine = RecursiveEngine(
        base_model=model,
        max_workers=64,
        max_depth=5,        
        max_call=50
    )

    test_prompt = "Calculate the second derivative of x^2 * sin(x) step by step."
    batch_output = "results/test_batch.jsonl"
    cont_output = "results/test_continuation.jsonl"

    # ---------------------------------------------------------
    # TEST 1: Parallel Batch Generation
    # ---------------------------------------------------------
    print("\n[TEST 1] Running Parallel Batch Generation...")
    prompts = [test_prompt] * 64  # Generate 4 parallel traces
    engine.process_batch(prompts, output_jsonl_path=batch_output)
    
    # ---------------------------------------------------------
    # TEST 2: Trace Length Distribution
    # ---------------------------------------------------------
    print("\n[TEST 2] Analyzing Length Distribution...")
    analyze_length_distribution(batch_output)

    # ---------------------------------------------------------
    # TEST 3: Continuation From Node
    # ---------------------------------------------------------
    print("\n[TEST 3] Testing Continuation From Node...")
    # Read the first successful trace
    with open(batch_output, 'r') as f:
        first_result = json.loads(f.readline())
    
    root_node = dict_to_node(first_result["trace_tree"])
    
    # Find a subproblem to continue from (or just use root if none generated)
    target_node = root_node
    if root_node.subproblems:
        target_node = root_node.subproblems[0]
        print(f"Continuing from subproblem: {target_node.prompt[:50]}...")
    else:
        print("No subproblems found, continuing from root...")

    # Generate 2 continuations from this specific node
    engine.continue_from_node(
        source_nodes=[target_node] * 2,
        output_jsonl_path=cont_output,
        group_size=2
    )

    print("\n✅ Engine Tests Completed Successfully!")
    model.shutdown()