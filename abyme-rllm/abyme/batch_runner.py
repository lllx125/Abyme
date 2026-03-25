"""
batch_runner.py

Orchestrates multiple RecursiveModel trees concurrently.
Optimized for A100 vLLM backend with immediate, thread-safe JSONL saving.
"""

import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any

# Assuming these are imported from your project structure
from abyme.core import RecursiveModel
from abyme.tree_trace import (
    total_calls, max_depth, max_subproblems, to_dict,
    max_output_character, parallel_latency
)
from abyme.vllm_model import LocalVLLMModel, APIModel, Model

class ParallelTreeOrchestrator:
    def __init__(self, 
                 base_model: LocalVLLMModel, 
                 output_jsonl_path: str = "results/results.jsonl",
                 max_concurrent_trees: int = 20,  # 20 trees * 3 workers/tree = 60 concurrent vLLM requests
                 **recursive_kwargs):
        """
        Initialize the Batch Runner.
        
        Args:
            base_model: The shared LocalVLLMModel or APIModel.
            output_jsonl_path: Where to stream results immediately.
            max_concurrent_trees: How many main problems to process simultaneously.
            **recursive_kwargs: Arguments passed directly to the RecursiveModel (e.g., max_depth, max_parallel_workers).
        """
        self.base_model = base_model
        self.output_jsonl_path = output_jsonl_path
        self.max_concurrent_trees = max_concurrent_trees
        self.recursive_kwargs = recursive_kwargs
        
        # Thread-safe lock for writing to the JSONL file
        self.file_write_lock = threading.Lock()

        # Set default RecursiveModel args optimal for A100 if not provided
        if 'max_parallel_workers' not in self.recursive_kwargs:
            self.recursive_kwargs['max_parallel_workers'] = 3
        if 'print_progress' not in self.recursive_kwargs:
            self.recursive_kwargs['print_progress'] = False

    def _run_single_tree(self, prompt: str, index: int) -> Dict[str, Any]:
        """Worker function that executes a single complete recursive tree."""
        # 1. Instantiate a fresh tree model for this specific prompt
        tree_model = RecursiveModel(
            base_model=self.base_model,
            **self.recursive_kwargs
        )
        
        start_time = time.time()
        result_data = {
            "index": index,
            "prompt": prompt,
            "status": "PENDING",
            "output": "",
            "error": None,
            "metrics": {}
        }

        # 2. Execute Generation
        try:
            final_output = tree_model.generate(prompt)
            result_data["status"] = "SUCCESS"
            result_data["output"] = final_output
        except Exception as e:
            result_data["status"] = "FAILED"
            result_data["error"] = str(e)

        # 3. Collect Metrics (if trace exists)
        actual_latency = time.time() - start_time
        result_data["metrics"]["actual_latency_seconds"] = actual_latency
        
        if tree_model.trace:
            try:
                result_data["metrics"].update({
                    "total_llm_calls": total_calls(tree_model.trace),
                    "max_tree_depth": max_depth(tree_model.trace),
                    "max_subproblems": max_subproblems(tree_model.trace),
                    "max_output_chars": max_output_character(tree_model.trace),
                    "theoretical_parallel_latency": parallel_latency(tree_model.trace)
                })
                
                # ---> NEW: Save the entire hierarchical tree structure <---
                # Make sure to import to_dict from tree_trace at the top of the file!
                result_data["trace_tree"] = to_dict(tree_model.trace)
                
            except Exception as metric_error:
                result_data["metrics"]["metric_extraction_error"] = str(metric_error)

        # 4. Immediate, Thread-Safe JSONL Saving
        self._append_to_jsonl(result_data)
        
        return result_data

    def _append_to_jsonl(self, data: dict):
        """Thread-safe file append."""
        with self.file_write_lock:
            with open(self.output_jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

    def run_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """
        Executes the batch of prompts concurrently.
        """
        print(f"Starting batch run of {len(prompts)} prompts...")
        print(f"Outputting real-time results to: {self.output_jsonl_path}")
        print(f"Concurrency: {self.max_concurrent_trees} trees (Max {self.max_concurrent_trees * self.recursive_kwargs['max_parallel_workers']} vLLM requests)")
        
        results = []
        # Clear/Create the output file
        with open(self.output_jsonl_path, 'w', encoding='utf-8') as f:
            pass 

        with ThreadPoolExecutor(max_workers=self.max_concurrent_trees) as executor:
            # Submit all tasks and map futures to their index
            future_to_index = {
                executor.submit(self._run_single_tree, prompt, i): i 
                for i, prompt in enumerate(prompts)
            }
            
            # Process results as they complete
            completed_count = 0
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    res = future.result()
                    results.append(res)
                    completed_count += 1
                    status = "✅" if res["status"] == "SUCCESS" else "❌"
                    print(f"[{completed_count}/{len(prompts)}] {status} Tree {idx} finished in {res['metrics'].get('actual_latency_seconds', 0):.1f}s")
                except Exception as exc:
                    print(f"Tree {idx} generated a fatal framework exception: {exc}")

        print("Batch run complete!")
        return results

# ==========================================
# USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":
    
    # 1. Initialize the shared vLLM Model (Optimized for 8B on 40GB A100)
    base_model = LocalVLLMModel(
        model_path="Lixing-Li/Abyme-Qwen3.5-9B-SFT",
        tensor_parallel_size=1
    )
    

    # 2. Initialize the Orchestrator
    orchestrator = ParallelTreeOrchestrator(
        base_model=base_model,
        output_jsonl_path="./a100_run_results.jsonl",
        
        # --- Batch Tuning ---
        max_concurrent_trees=10, 
        
        # --- **kwargs passed directly to RecursiveModel ---
        max_parallel_workers=5,   # Subproblem workers per tree
        max_depth=1,
        max_call=10,
        max_chain_length=5,
        proceed_when_fail=True,
        print_progress=True      # Keep False to avoid terminal spam during batching
    )

    # 3. Load your prompts
    test_prompts = [
        "what is 1+1?",
        "what is 2+2?",
        "what is 3+3?",
        "what is 4+4?",
        "what is 5+5?"
        # ... load thousands more from a dataset
    ]

    # 4. Fire the batch
    results = orchestrator.run_batch(test_prompts)

    base_model.shutdown()