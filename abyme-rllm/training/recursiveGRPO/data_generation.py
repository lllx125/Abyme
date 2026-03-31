"""
data_generation.py
Recursive GRPO Data Generator for a SINGLE problem.
"""

import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import threading
import statistics
import os

from abyme.magic import magic_formatter
from abyme.recursive_engine import RecursiveEngine
from abyme.tree_trace import TreeTraceNode, dict_to_node, length, future_length
from abyme.utils import verify_output_format_strict
from abyme.vllm_model import LocalVLLMModel
from benchmark.math_full_benchmark import MATHFullBenchmark

PROBLEM_PER_PHASE = 100

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_verify_format(text: str) -> bool:
    """Safeguard wrapper to prevent regex catastrophic backtracking on massive strings."""
    if not text:
        return False
    if len(text) > 4000:
        return False
    return verify_output_format_strict(text)

def _compute_advantages(final_roots: List[TreeTraceNode], first_nodes: List[TreeTraceNode], correctness: List[bool], alpha: float) -> List[float]:
    print("      [Debug] Starting _compute_advantages...")
    sys.stdout.flush()
    
    f = []
    for i, n in enumerate(first_nodes):
        # Verbose print so you can see exactly where it freezes if it ever happens again
        print(f"        [Debug] Advantage Format Check Trace {i} (len: {len(n.output)})... ", end="")
        sys.stdout.flush()
        
        is_valid = _safe_verify_format(n.output)
        f.append(1.0 if is_valid else 0.0)
        
        print("Done.")
        sys.stdout.flush()

    lengths = [future_length(end_n, fn) for end_n, fn in zip(final_roots, first_nodes)]
    
    correct_valid_lengths = [l for l, c, fi in zip(lengths, correctness, f) if c and fi == 1.0]
    mu_l = (sum(correct_valid_lengths) / len(correct_valid_lengths)) if correct_valid_lengths else (
        sum(lengths) / len(lengths) if lengths else 1.0
    )

    raw_rewards = []
    for fi, li, ci in zip(f, lengths, correctness):
        ci_val = 1.0 if ci else 0.0
        excess = max(0.0, (li - mu_l) / mu_l) if mu_l > 0 else 0.0
        r = fi * ci_val * math.exp(-alpha * excess ** 2)
        raw_rewards.append(r)

    if len(raw_rewards) > 1:
        mean_r = statistics.mean(raw_rewards)
        std_r = statistics.stdev(raw_rewards) + 1e-8
        advantages = [(r - mean_r) / std_r for r in raw_rewards]
    else:
        advantages = [0.0]

    return advantages

def _pick_reference_index(final_roots: List[TreeTraceNode], first_nodes: List[TreeTraceNode], correctness: List[bool], temperature: float) -> int:
    print("      [Debug] Starting _pick_reference_index...")
    sys.stdout.flush()
    
    f = []
    for i, n in enumerate(first_nodes):
        is_valid = _safe_verify_format(n.output)
        f.append(1.0 if is_valid else 0.0)
        
    lengths = [future_length(end_n, fn) for end_n, fn in zip(final_roots, first_nodes)]

    candidates = [(i, lengths[i]) for i in range(len(final_roots)) if correctness[i] and f[i] == 1.0]
    
    if not candidates:
        candidates = [(i, lengths[i]) for i in range(len(final_roots)) if first_nodes[i].output]
        
    if temperature <= 0.0 or len(candidates) <= 1:
        if not candidates: return 0
        return min(candidates, key=lambda x: x[1])[0]

    cand_lengths = [l for _, l in candidates]
    
    # THE FIX: Shift by the minimum length to prevent Math Overflow
    min_l = min(cand_lengths) 
    
    # Now the shortest length gets exp(0) = 1.0. 
    # Longer lengths get exp(-large_number), which safely underflows to 0.0.
    weights = [math.exp(-(l - min_l) / temperature) for l in cand_lengths]
    
    total = sum(weights)
    weights = [w / total for w in weights]

    r = random.random()
    cumulative = 0.0
    for (idx, _), w in zip(candidates, weights):
        cumulative += w
        if r <= cumulative: return idx
    return candidates[-1][0]

def _ordered_nodes(root: TreeTraceNode) -> List[TreeTraceNode]:
    result: List[TreeTraceNode] = []
    def _visit(node: TreeTraceNode):
        if not node.output: return
        result.append(node)
        for child in node.subproblems:
            _visit(child)
    for temporal_node in list(root.past) + [root]:
        _visit(temporal_node)
    return result

# ---------------------------------------------------------------------------
# DataManager
# ---------------------------------------------------------------------------

class DataManager(MATHFullBenchmark):
    def __init__(self, group_size: int = 64, temperature: float = 1.0, alpha: float = 1.0):
        super().__init__()
        self.group_size = group_size
        self.temperature = temperature
        self.alpha = alpha
        self.train_file = Path("data/grpo_train_curriculum.jsonl")
        self.test_file = Path("data/grpo_test_set.jsonl")

    # ==========================================
    # CURRICULUM MANAGEMENT
    # ==========================================
    def prepare_datasets(self, raw_data_file: str = "data/math_full.jsonl", test_samples_per_level: int = 50):
        if self.train_file.exists() and self.test_file.exists():
            print(f"    [Info] Curriculum already exists at {self.train_file}. Using existing split.")
            return

        print("    [Info] Building new curriculum and test sets...")
        if not os.path.exists(raw_data_file):
            print("    [Info] Downloading MATH dataset...")
            self.download()

        levels_train = {1: [], 2: [], 3: [], 4: [], 5: []}
        levels_test = {1: [], 2: [], 3: [], 4: [], 5: []}
        
        with open(raw_data_file, "r") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    lvl = item.get("level_num", 1)
                    if item.get("selected_split") == "test" or item.get("training") is False:
                         levels_test[lvl].append(item)
                    else:
                         levels_train[lvl].append(item)
        
        test_data = []
        for l in range(1, 6):
            missing = test_samples_per_level - len(levels_test[l])
            if missing > 0:
                random.shuffle(levels_train[l])
                levels_test[l].extend(levels_train[l][:missing])
                levels_train[l] = levels_train[l][missing:]
            
            levels_test[l] = levels_test[l][:test_samples_per_level]
            test_data.extend(levels_test[l])

        curriculum = self._build_progressive_curriculum(levels_train)

        self.train_file.parent.mkdir(parents=True, exist_ok=True)
        with self.train_file.open("w") as f:
            for item in curriculum:
                f.write(json.dumps(item) + "\n")
                
        with self.test_file.open("w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")
                
        print(f"    [Success] Saved {len(curriculum)} training and {len(test_data)} test problems.")

    def _build_progressive_curriculum(self, levels_train: Dict[int, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
            for l in levels_train.values():
                random.shuffle(l)
            
            curriculum = []
            phases = [
                (0.7, 0.2, 0.1, 0.0, 0.0), 
                (0.3, 0.4, 0.2, 0.1, 0.0), 
                (0.1, 0.2, 0.4, 0.2, 0.1), 
                (0.0, 0.1, 0.3, 0.4, 0.2), 
                (0.0, 0.0, 0.1, 0.4, 0.5)  
            ]
            
            total_problems = PROBLEM_PER_PHASE * len(phases)

            for step in range(total_problems):
                phase_idx = step // PROBLEM_PER_PHASE
                weights = phases[phase_idx]
                
                available_levels = [l for l in range(1, 6) if levels_train.get(l)]
                if not available_levels: 
                    break
                
                active_weights = [weights[l-1] for l in available_levels]
                weight_sum = sum(active_weights)
                active_weights = [w / weight_sum for w in active_weights] if weight_sum > 0 else [1.0 / len(available_levels)] * len(available_levels)

                chosen_level = random.choices(available_levels, weights=active_weights, k=1)[0]
                curriculum.append(levels_train[chosen_level].pop())

            return curriculum

    def load_curriculum(self) -> List[Dict[str, Any]]:
        with self.train_file.open("r") as f:
            return [json.loads(line) for line in f if line.strip()]
            
    def load_test_set(self) -> List[Dict[str, Any]]:
        with self.test_file.open("r") as f:
            return [json.loads(line) for line in f if line.strip()]

    def score(self, model_output: str, input: Dict[str, Any]) -> float:
        input_with_answer = {**input, "answer": input.get("ground_truth", input.get("answer", ""))}
        return super().score(model_output, input_with_answer)

    # ==========================================
    # GRPO GENERATION LOGIC
    # ==========================================
    def generate(self, problem: Dict[str, Any], engine: RecursiveEngine, output_path: Path):
        file_lock = threading.Lock()
        prompt = problem["problem"]

        print(f"\n    >>> Generating Initial {self.group_size} Traces from Root...")
        sys.stdout.flush()
        
        prompts = [prompt] * self.group_size
        tmp_path = output_path.parent / f"_tmp_grpo_{id(threading.current_thread())}.jsonl"
        try:
            results = engine.process_batch(prompts, str(tmp_path))
        finally:
            if tmp_path.exists(): 
                try:
                    tmp_path.unlink()
                except Exception as e:
                    print(f"    [Warning] Could not delete temp file: {e}")

        print("    [Debug] process_batch completed. Parsing results & Scoring...")
        sys.stdout.flush()
        
        final_roots, first_nodes, correctness = [], [], []
        
        for i, res in enumerate(results):
            final_root = dict_to_node(res["trace_tree"])
            first_node = dict_to_node(res["first_generated_node"])
            final_roots.append(final_root)
            first_nodes.append(first_node)
            
            output_text = res.get("output", "")
            status = res.get("status")
            
            if status == "SUCCESS" and len(output_text) < 8000:
                try:
                    is_correct = (self.score(output_text, problem) == 1.0)
                except Exception as e:
                    print(f"    [Warning] Math Scorer crashed on trace {i}: {e}")
                    is_correct = False
            else:
                is_correct = False
                
            correctness.append(is_correct)

        print("    [Debug] Scoring complete. Calculating lengths...")
        sys.stdout.flush()

        valid_lengths = []
        for end_n, fn, c in zip(final_roots, first_nodes, correctness):
            if c and _safe_verify_format(fn.output):
                valid_lengths.append(future_length(end_n, fn))

        if valid_lengths:
            avg_len = sum(valid_lengths) / len(valid_lengths)
            print(f"    [Root Generation] Valid Future Lengths -> Max: {max(valid_lengths):.1f} | Min: {min(valid_lengths):.1f} | Avg: {avg_len:.2f}")
        else:
            print("    [Root Generation] No valid (correct + correct format) traces found for length tracking.")

        any_success = any(c and _safe_verify_format(fn.output) for c, fn in zip(correctness, first_nodes))
        if not any_success:
            print("    [Skip] All root generations failed (wrong answer or invalid format). Skipping this problem entirely.")
            return

        self._write_group(final_roots, first_nodes, correctness, output_path, file_lock)

        ref_idx = _pick_reference_index(final_roots, first_nodes, correctness, self.temperature)
        ref_final_root = final_roots[ref_idx]
        ref_first_node = first_nodes[ref_idx]
        
        has_subprobs = len(_ordered_nodes(ref_final_root)) > 1
        ref_len = future_length(ref_final_root, ref_first_node)
        
        print(f"    [Reference] Selected root index: {ref_idx} (correct: {correctness[ref_idx]}, length: {ref_len:.1f}, has_subproblems: {has_subprobs})")

        if not correctness[ref_idx] or not _safe_verify_format(ref_first_node.output):
            print("    [Early Stop] Reference trace is incorrect or has invalid format. Aborting recursion.")
            return

        self._recurse_reference(ref_final_root, problem, engine, output_path, file_lock)


    def _recurse_reference(self, ref_final_root: TreeTraceNode, problem: Dict[str, Any], engine: RecursiveEngine, output_path: Path, file_lock):
        nodes_to_process = _ordered_nodes(ref_final_root)
        print(f"    [Recursion] Ordered nodes to process: {len(nodes_to_process)} (will continue from {len(nodes_to_process) - 1} non-root nodes)")

        for list_idx, node in enumerate(nodes_to_process[1:], start=1):
            
            future_len = future_length(ref_final_root, node)
            
            print(f"\n    {'-'*60}")
            print(f"    >>> Continuing from Node #{list_idx}/{len(nodes_to_process)-1} (Depth: {node.depth}, Index: {node.index})")
            print(f"    Main Problem  : {node.main_problem[:100]}..." if len(node.main_problem) > 100 else f"    Main Problem  : {node.main_problem}")
            print(f"    Parent Problem: {node.parent_problem[:100]}..." if len(node.parent_problem) > 100 else f"    Parent Problem: {node.parent_problem}")
            print(f"    Prompt        : {node.prompt[:100]}..." if len(node.prompt) > 100 else f"    Prompt        : {node.prompt}")
            print(f"    Fragment      : {node.fragment[:100]}..." if len(node.fragment) > 100 else f"    Fragment      : {node.fragment}")
            print(f"    Future Length : {future_len:.1f}")
            print(f"    {'-'*60}")

            if not _safe_verify_format(node.output):
                print(f"    [Early Stop] Reference node at depth {node.depth} has invalid format or is too large. Skipping continuations.")
                continue
                
            self._generate_and_write_node_group(node, ref_final_root, problem, engine, output_path, file_lock)


    def _generate_and_write_node_group(self, node: TreeTraceNode, ref_final_root: TreeTraceNode, problem: Dict[str, Any], engine: RecursiveEngine, output_path: Path, file_lock):
        num_to_generate = self.group_size - 1
        source_nodes = [node] * num_to_generate
        tmp_path = output_path.parent / f"_tmp_grpo_node_{id(threading.current_thread())}.jsonl"
        
        try:
            results = engine.continue_from_node(source_nodes, str(tmp_path), group_size=num_to_generate)
        finally:
            if tmp_path.exists(): 
                try:
                    tmp_path.unlink()
                except Exception as e:
                    pass

        final_roots, first_nodes, correctness = [], [], []
        for i, res in enumerate(results):
            final_root = dict_to_node(res["trace_tree"])
            first_node = dict_to_node(res["first_generated_node"])
            final_roots.append(final_root)
            first_nodes.append(first_node)
            
            output_text = res.get("output", "")
            status = res.get("status")
            
            if status == "SUCCESS" and len(output_text) < 8000:
                try:
                    is_correct = (self.score(output_text, problem) == 1.0)
                except Exception:
                    is_correct = False
            else:
                is_correct = False
                
            correctness.append(is_correct)

        # Inject original reference trace
        final_roots.append(ref_final_root)
        first_nodes.append(node)
        correctness.append(True) 
        
        valid_lengths = []
        for end_n, fn, c in zip(final_roots, first_nodes, correctness):
            if c and _safe_verify_format(fn.output):
                valid_lengths.append(future_length(end_n, fn))

        if valid_lengths:
            avg_len = sum(valid_lengths) / len(valid_lengths)
            print(f"    [Node Generation] Valid Future Lengths -> Max: {max(valid_lengths):.1f} | Min: {min(valid_lengths):.1f} | Avg: {avg_len:.2f}")
        else:
            print("    [Node Generation] No valid (correct + format) traces found for length tracking.")

        self._write_group(final_roots, first_nodes, correctness, output_path, file_lock)


    def _write_group(self, final_roots: List[TreeTraceNode], first_nodes: List[TreeTraceNode], correctness: List[bool], output_path: Path, file_lock):
        print("    [Debug] calling _compute_advantages...")
        sys.stdout.flush()
        advantages = _compute_advantages(final_roots, first_nodes, correctness, self.alpha)
        
        print("    [Debug] Writing group to disk...")
        sys.stdout.flush()
        records = []
        for fn, advantage in zip(first_nodes, advantages):
            if not fn.output: continue
            record = {
                "input": magic_formatter(fn.prompt, fn.main_problem, fn.parent_problem, fn.fragment),
                "output": fn.output, 
                "advantage": advantage,
            }
            records.append(record)
            
        with file_lock:
            with output_path.open("a") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")
        print("    [Debug] File write complete.")
        sys.stdout.flush()

    def score_all(self, model: LocalVLLMModel, test_data: List[Dict[str, Any]], iteration: int, max_workers: int = 60, **recursive_kwargs):
        from tqdm import tqdm
        output_path = Path(f"results/grpo_{iteration}_test_results.jsonl")
        scored_path = Path(f"results/grpo_{iteration}_test_scored.jsonl")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        engine = RecursiveEngine(base_model=model, max_workers=max_workers, **recursive_kwargs)
        prompts = [item["problem"] for item in test_data]
        results = engine.process_batch(prompts, str(output_path))

        augmented = []
        for res in results:
            i = res["index"]
            original = test_data[i]
            augmented.append({
                **res, "problem_index": i, "level_num": original["level_num"],
                "type": original["type"], "ground_truth": original.get("ground_truth", "")
            })
        augmented.sort(key=lambda r: r["problem_index"])
        
        with output_path.open("w") as f:
            for rec in augmented: f.write(json.dumps(rec) + "\n")

        total = 0.0
        scores = []
        with output_path.open("r") as infile, scored_path.open("w") as outfile:
            for line in tqdm([l for l in infile if l.strip()], desc=f"Scoring Iteration {iteration}"):
                record = json.loads(line)
                s = self.score(record.get("output", ""), record)
                total += s
                scores.append(s)
                outfile.write(json.dumps({**record, "score": s}) + "\n")

        avg = total / len(scores) if scores else 0.0
        print(f"Average score for Iteration {iteration}: {avg:.4f} ({int(total)}/{len(scores)} correct)")
        return avg, scores