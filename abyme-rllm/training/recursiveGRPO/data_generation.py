"""
Data Generation for Recursive GRPO Training

Generates grouped training data for Policy Gradient with KL penalty.
For each problem, recursively walks the reference trace and generates
a group of `group_size` outputs per node, scored by the GRPO reward.

Reward formula:
    R_i = f_i * c_i * exp(-alpha * max(0, (l_i - mu_l) / mu_l)^2)

    f_i  = 1 if output has valid format, else 0
    c_i  = 1 if final answer is correct, else 0
    l_i  = future_length(node)
    mu_l = mean future_length of correct traces in the group
"""

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from abyme.magic import magic_formatter
from abyme.recursive_engine import RecursiveEngine
from abyme.tree_trace import (
    TreeTraceNode,
    dict_to_node,
    future_length,
)
from abyme.utils import verify_output_format_strict
from abyme.vllm_model import LocalVLLMModel
from benchmark.math_full_benchmark import MATHFullBenchmark

ALPHA = 1.0  # Penalty strength for above-average length in GRPO reward
TEMPERATURE = 1.0  # Softmax temperature for reference trace selection (lower = greedier)
GROUP_SIZE = 64  # Number of traces generated per node in the reference trace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_advantages(
    nodes: List[TreeTraceNode],
    correctness: List[bool],
    alpha: float,
) -> List[float]:
    """
    Compute normalized GRPO advantages for a group of nodes.
    """
    import statistics
    
    f = [1.0 if verify_output_format_strict(n.output) else 0.0 for n in nodes]
    lengths = [future_length(n) for n in nodes]

    correct_lengths = [l for l, c in zip(lengths, correctness) if c]
    mu_l = (sum(correct_lengths) / len(correct_lengths)) if correct_lengths else (
        sum(lengths) / len(lengths) if lengths else 1.0
    )

    # 1. Calculate Raw Rewards (R_i)
    raw_rewards = []
    for fi, li, ci in zip(f, lengths, correctness):
        ci_val = 1.0 if ci else 0.0
        excess = max(0.0, (li - mu_l) / mu_l) if mu_l > 0 else 0.0
        r = fi * ci_val * math.exp(-alpha * excess ** 2)
        raw_rewards.append(r)

    # 2. Normalize to get Advantages (A_i)
    if len(raw_rewards) > 1:
        mean_r = statistics.mean(raw_rewards)
        std_r = statistics.stdev(raw_rewards) + 1e-8
        advantages = [(r - mean_r) / std_r for r in raw_rewards]
    else:
        advantages = [0.0] # Fallback if group size is 1

    return advantages


def _pick_reference_index(
    roots: List[TreeTraceNode],
    correctness: List[bool],
    temperature: float,
) -> int:
    """
    Pick the index of the reference trace using temperature-weighted sampling.

    Shorter future_length → higher probability. Falls back to all traces if
    none are correct.

    Args:
        roots:       Root nodes of all traces in the group.
        correctness: Per-trace correctness flags.
        temperature: Softmax temperature (lower → greedier selection).

    Returns:
        Index into roots/correctness of the chosen reference trace.
    """
    candidates = [(i, future_length(r)) for i, r in enumerate(roots) if correctness[i]]
    if not candidates:
        candidates = [(i, future_length(r)) for i, r in enumerate(roots)]

    if temperature <= 0.0 or len(candidates) == 1:
        return min(candidates, key=lambda x: x[1])[0]

    # Softmax on negative length / temperature  →  shorter = higher weight
    lengths = [l for _, l in candidates]
    max_l = max(lengths)
    weights = [math.exp(-(l - max_l) / temperature) for l in lengths]
    total = sum(weights)
    weights = [w / total for w in weights]

    r = random.random()
    cumulative = 0.0
    for (idx, _), w in zip(candidates, weights):
        cumulative += w
        if r <= cumulative:
            return idx
    return candidates[-1][0]


def _ordered_nodes(root: TreeTraceNode) -> List[TreeTraceNode]:
    """
    Return nodes from the reference trace in the order we should generate groups.

    Traversal: for each temporal step (oldest past → current), process all spatial
    children recursively before moving to the next temporal step.

    Concretely, for a stored trace root:
        root.past = [wait_sub_state_0, wait_sub_state_1, ...]
        root itself = final temporal state (ANSWER or last continuation)

    The returned order is:
        past[0], past[0].children recursively,
        past[1], past[1].children recursively,
        ...,
        root, root.children recursively

    Only nodes that have non-empty output are included (skips empty/failed states).
    """
    result: List[TreeTraceNode] = []

    def _visit(node: TreeTraceNode):
        if not node.output:
            return
        result.append(node)
        for child in node.subproblems:
            _visit(child)

    # Temporal chain: oldest past first, then current node
    temporal_chain = list(root.past) + [root]
    for temporal_node in temporal_chain:
        _visit(temporal_node)

    return result


# ---------------------------------------------------------------------------
# DataManager
# ---------------------------------------------------------------------------

class DataManager(MATHFullBenchmark):
    """
    Data manager for Recursive GRPO training on the MATH dataset.

    For each training problem, generates groups of `group_size` outputs per
    node in the reference trace and writes {input, output, advantage} records
    to a JSONL file.
    """

    def __init__(
        self,
        samples: Tuple[int, int, int, int, int],
        tests: Tuple[int, int, int, int, int],
        iteration: int,
        group_size: int = GROUP_SIZE,
        temperature: float = TEMPERATURE,
        alpha: float = ALPHA,
        seed: int = 42,
    ):
        """
        Args:
            samples:    5-tuple of training samples per level (levels 1-5).
            tests:      5-tuple of test samples per level (levels 1-5).
            iteration:  Iteration number used in file naming.
            group_size: Number of traces generated per node (default 64).
            temperature: Softmax temperature for reference trace selection.
            alpha:      Penalty exponent in the reward formula.
            seed:       Random seed for reproducibility.
        """
        super().__init__()
        self.samples = samples
        self.tests = tests
        self.iteration = iteration
        self.group_size = group_size
        self.temperature = temperature
        self.alpha = alpha
        self.seed = seed

        self.train_data: List[Dict[str, Any]] = []
        self.test_data: List[Dict[str, Any]] = []

        self.train_data_path = Path(f"data/grpo_{self.iteration}.jsonl")
        self.train_output_path = Path(f"results/grpo_{self.iteration}_train.jsonl")
        self.test_output_path = Path(f"results/grpo_{self.iteration}_test_results.jsonl")
        self.test_scored_output_path = Path(f"results/grpo_{self.iteration}_test_scored.jsonl")

        self._load_and_select_data()

    # ------------------------------------------------------------------
    # Data selection (mirrors restKTO DataManager)
    # ------------------------------------------------------------------

    def _load_and_select_data(self):
        """Load MATH full dataset and select training/test subsets."""
        from collections import defaultdict

        saved_path = self.train_data_path
        if saved_path.exists():
            print(f"Loading previously selected data from {saved_path}")
            with saved_path.open("r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line.strip())
                    if item.get("selected_split") == "train":
                        self.train_data.append(item)
                    elif item.get("selected_split") == "test":
                        self.test_data.append(item)
            print(f"Loaded {len(self.train_data)} training, {len(self.test_data)} test samples")
            return

        data_path = Path("data/math_full.jsonl")
        if not data_path.exists():
            self.download()

        all_data: List[Dict[str, Any]] = []
        with data_path.open("r") as f:
            for line in f:
                if line.strip():
                    all_data.append(json.loads(line.strip()))

        grouped = defaultdict(list)
        for item in all_data:
            key = (item["level_num"], item["type"], item["training"])
            grouped[key].append(item)

        random.seed(self.seed)

        def _select(level: int, is_train: bool, n_total: int) -> List[Dict[str, Any]]:
            subjects = {k[1] for k in grouped if k[0] == level and k[2] == is_train}
            per_subject = max(1, n_total // len(subjects)) if subjects else 0
            selected: List[Dict[str, Any]] = []
            for subj in sorted(subjects):
                available = grouped[(level, subj, is_train)]
                n = min(per_subject, len(available))
                selected.extend(random.sample(available, n))
            if len(selected) < n_total:
                remaining = [
                    item for subj in subjects
                    for item in grouped[(level, subj, is_train)]
                    if item not in selected
                ]
                extra = min(n_total - len(selected), len(remaining))
                selected.extend(random.sample(remaining, extra))
            return selected[:n_total]

        print(f"Selecting data for iteration {self.iteration}")
        for level in range(1, 6):
            train_items = _select(level, True, self.samples[level - 1])
            self.train_data.extend(train_items)
            test_items = _select(level, False, self.tests[level - 1])
            self.test_data.extend(test_items)
            print(f"  Level {level}: {len(train_items)} train, {len(test_items)} test")

        print(f"Total: {len(self.train_data)} train, {len(self.test_data)} test")

        saved_path.parent.mkdir(parents=True, exist_ok=True)
        with saved_path.open("w") as f:
            for item in self.train_data:
                item["selected_split"] = "train"
                f.write(json.dumps(item) + "\n")
            for item in self.test_data:
                item["selected_split"] = "test"
                f.write(json.dumps(item) + "\n")

    # ------------------------------------------------------------------
    # Score override (use ground_truth field from MATH full dataset)
    # ------------------------------------------------------------------

    def score(self, model_output: str, input: Dict[str, Any]) -> float:
        """Score using ground_truth field."""
        input_with_answer = {**input, "answer": input.get("ground_truth", input.get("answer", ""))}
        return super().score(model_output, input_with_answer)

    # ------------------------------------------------------------------
    # Core group generation
    # ------------------------------------------------------------------

    def generate_group(
        self,
        problem: Dict[str, Any],
        engine: RecursiveEngine,
        output_path: Path,
        file_lock,
    ):
        """

        Generate GRPO training data for a single problem.
        Generates `group_size` traces from the root, scores them, picks the
        reference trace (shortest correct), then recursively generates groups
        for each node in the reference trace using continue_from_node.
        Each written record:
        {"input": <magic_formatter output>, "output": <node.output>, "advantage": <float>}
        Args:
        problem: Dict with 'problem' (prompt) and 'ground_truth' fields.
        engine: RecursiveEngine instance (shared across calls).
        output_path: JSONL file to append records to.
        file_lock: threading.Lock for safe concurrent writes.
        """
        import threading
        from abyme.utils import verify_output_format_strict

        prompt = problem["problem"]

        # --- Step 1: Generate group_size traces from root ---
        prompts = [prompt] * self.group_size
        tmp_path = output_path.parent / f"_tmp_grpo_{id(threading.current_thread())}.jsonl"
        try:
            results = engine.process_batch(prompts, str(tmp_path))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        # Reconstruct root nodes and check correctness
        roots: List[TreeTraceNode] = []
        correctness: List[bool] = []
        for res in results:
            root_node = dict_to_node(res["trace_tree"])
            roots.append(root_node)
            is_correct = (
                res.get("status") == "SUCCESS"
                and self.score(res.get("output", ""), problem) == 1.0
            )
            correctness.append(is_correct)

        # --- Step 2: Score and write group for root ---
        self._write_group(roots, correctness, output_path, file_lock)

        # --- Step 3: Pick reference trace ---
        ref_idx = _pick_reference_index(roots, correctness, self.temperature)
        ref_root = roots[ref_idx]

        # =====================================================================
        # EARLY STOP (ROOT LEVEL):
        # If the chosen reference trace has a broken format, do not recurse.
        # Continuing a tree based on bad formatting will only poison the dataset.
        # =====================================================================
        if not verify_output_format_strict(ref_root.output):
            print("    [Early Stop] Reference root has invalid format. Aborting recursion for this problem.")
            return

        # --- Step 4: Walk reference trace and generate groups for each node ---
        self._recurse_reference(ref_root, problem, engine, output_path, file_lock)


    def _recurse_reference(
        self,
        ref_root: TreeTraceNode,
        problem: Dict[str, Any],
        engine: RecursiveEngine,
        output_path: Path,
        file_lock,
    ):
        """
        Walk the reference trace in order and generate continuations, 
        skipping any nodes that have formatting errors.
        """
        from abyme.utils import verify_output_format_strict

        nodes_to_process = _ordered_nodes(ref_root)

        # Skip the very first node — already covered by generate_group's initial batch
        for node in nodes_to_process[1:]:
            
            # =================================================================
            # EARLY STOP (NODE LEVEL):
            # Even if the root was okay, an intermediate node might be malformed.
            # Don't waste 63 LLM calls branching off a broken state.
            # =================================================================
            if not verify_output_format_strict(node.output):
                print(f"    [Early Stop] Node at depth {node.depth} has invalid format. Skipping its continuations.")
                continue
                
            self._generate_and_write_node_group(node, problem, engine, output_path, file_lock)

    def _generate_and_write_node_group(
        self,
        node: TreeTraceNode,
        problem: Dict[str, Any],
        engine: RecursiveEngine,
        output_path: Path,
        file_lock,
    ):
        import threading
        from abyme.tree_trace import collect_all_nodes

        # FIX 1: Generate 63 traces instead of 64
        num_to_generate = self.group_size - 1
        source_nodes = [node] * num_to_generate
        
        tmp_path = output_path.parent / f"_tmp_grpo_node_{id(threading.current_thread())}.jsonl"
        try:
            results = engine.continue_from_node(
                source_nodes,
                str(tmp_path),
                group_size=num_to_generate,
            )
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        regen_nodes: List[TreeTraceNode] = []
        correctness: List[bool] = []

        for res in results:
            regen_root = dict_to_node(res["trace_tree"])
            
            # FIX 2: Use the robust state-matching search
            regen_node = self._find_node_in_tree(regen_root, node)
            if regen_node is None or not regen_node.output:
                regen_node = regen_root # Fallback

            regen_nodes.append(regen_node)
            is_correct = (
                res.get("status") == "SUCCESS"
                and self.score(res.get("output", ""), problem) == 1.0
            )
            correctness.append(is_correct)

        # FIX 1 (Continued): Manually inject the known-good reference trace (+1)
        regen_nodes.append(node)
        correctness.append(True) # The reference trace was selected because it was correct

        self._write_group(regen_nodes, correctness, output_path, file_lock)

    def _find_node_in_tree(
        self,
        regen_root: TreeTraceNode,
        original_node: TreeTraceNode,
    ) -> Optional[TreeTraceNode]:
        """
        Robustly locate the regenerated equivalent of `original_node` inside `regen_root`
        by matching the exact state signature (depth, prompt, fragment).
        """
        from abyme.tree_trace import collect_all_nodes
        
        all_nodes = collect_all_nodes(regen_root)
        for n in all_nodes:
            if (n.depth == original_node.depth and 
                n.prompt == original_node.prompt and 
                n.fragment == original_node.fragment):
                return n
                
        return None

    def _write_group(
        self,
        nodes: List[TreeTraceNode],
        correctness: List[bool],
        output_path: Path,
        file_lock,
    ):
        """Compute advantages and write {input, output, advantage} records."""
        advantages = _compute_advantages(nodes, correctness, self.alpha)

        records = []
        for node, advantage in zip(nodes, advantages):
            if not node.output:
                continue
            record = {
                "input": magic_formatter(
                    node.prompt,
                    node.main_problem,
                    node.parent_problem,
                    node.fragment,
                ),
                "output": node.output,
                "advantage": advantage,
            }
            records.append(record)

        with file_lock:
            with output_path.open("a") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------
    # generate_all
    # ------------------------------------------------------------------

    def generate_all(
        self,
        model: LocalVLLMModel,
        max_workers: int = 60,
        **recursive_kwargs,
    ):
        """
        Generate GRPO training data for all training problems.

        Results are appended to results/grpo_{iteration}_train.jsonl as each
        problem is processed.

        Args:
            model:          LocalVLLMModel to use for generation.
            max_workers:    Shared thread-pool size for RecursiveEngine.
            **recursive_kwargs: Passed to RecursiveEngine (max_depth, max_call, ...).
        """
        import threading

        output_path = self.train_output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("")  # clear

        file_lock = threading.Lock()

        engine = RecursiveEngine(
            base_model=model,
            max_workers=max_workers,
            **recursive_kwargs,
        )

        print(f"\n{'='*60}")
        print(f"GRPO GENERATION: iteration {self.iteration}")
        print(f"{'='*60}")
        print(f"Training problems : {len(self.train_data)}")
        print(f"Group size        : {self.group_size}")
        print(f"Output            : {output_path}")

        for i, problem in enumerate(self.train_data):
            print(f"  [{i+1}/{len(self.train_data)}] {problem.get('type','')} Level {problem.get('level_num','')}")
            try:
                self.generate_group(problem, engine, output_path, file_lock)
            except Exception as e:
                print(f"    ERROR: {e}")

        print(f"\nGeneration complete. Saved to {output_path}")

    # ------------------------------------------------------------------
    # score_all
    # ------------------------------------------------------------------

    def score_all(
        self,
        model: LocalVLLMModel,
        max_workers: int = 60,
        **recursive_kwargs,
    ) -> Tuple[float, List[float]]:
        """
        Run model on test set, score results, and save to JSONL.

        Args:
            model:       LocalVLLMModel to use.
            max_workers: Shared thread-pool size.

        Returns:
            (avg_score, list_of_scores)
        """
        from tqdm import tqdm

        output_path = self.test_output_path
        scored_path = self.test_scored_output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"GRPO TESTING: iteration {self.iteration}")
        print(f"{'='*60}")
        print(f"Test samples : {len(self.test_data)}")

        engine = RecursiveEngine(
            base_model=model,
            max_workers=max_workers,
            **recursive_kwargs,
        )

        prompts = [item["problem"] for item in self.test_data]
        results = engine.process_batch(prompts, str(output_path))

        # Augment with original problem metadata
        augmented: List[Dict[str, Any]] = []
        for res in results:
            i = res["index"]
            original = self.test_data[i]
            augmented.append({
                **res,
                "problem_index": i,
                "level_num": original["level_num"],
                "type": original["type"],
                "ground_truth": original.get("ground_truth", ""),
            })

        augmented.sort(key=lambda r: r["problem_index"])
        with output_path.open("w") as f:
            for rec in augmented:
                f.write(json.dumps(rec) + "\n")

        # Score
        total = 0.0
        scores: List[float] = []
        with output_path.open("r") as infile, scored_path.open("w") as outfile:
            lines = [l for l in infile if l.strip()]
            for line in tqdm(lines, desc=f"Scoring iteration {self.iteration}"):
                record = json.loads(line)
                s = self.score(record.get("output", ""), record)
                total += s
                scores.append(s)
                outfile.write(json.dumps({**record, "score": s}) + "\n")

        avg = total / len(scores) if scores else 0.0
        print(f"Average score: {avg:.4f} ({int(total)}/{len(scores)} correct)")
        return avg, scores

    def check_scores_by_level(self) -> Tuple[float, float, float, float, float, float]:
        """Print and return per-level scores from the scored test output."""
        return self._check_scores_from_path(
            self.test_scored_output_path,
            f"grpo_iteration_{self.iteration}",
            level_field="level_num",
        )


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    base_model = "Lixing-Li/Abyme-Qwen3.5-9B-SFT"
    iteration = 0
    samples = (5, 5, 5, 5, 5)
    tests = (5, 5, 5, 5, 5)

    data_manager = DataManager(
        iteration=iteration,
        samples=samples,
        tests=tests,
        group_size=4,   # small for smoke-test
        temperature=1.0,
        alpha=1.0,
    )

    model = LocalVLLMModel(model_path=base_model)
    print("Model loaded.")

    data_manager.generate_all(
        model=model,
        max_workers=8,
        max_depth=5,
        max_call=50,
        max_chain_length=5,
    )

    avg, _ = data_manager.score_all(
        model=model,
        max_workers=8,
        max_depth=5,
        max_call=50,
        max_chain_length=5,
    )
    data_manager.check_scores_by_level()
