"""
Recursive Engine - Unified API for recursive generation.

Replaces both RecursiveModel (single-tree) and ParallelTreeOrchestrator (multi-tree)
with a shared worker pool for maximum GPU utilization.

Interface:
    engine = RecursiveEngine(base_model, max_workers=60, ...)

    # Single prompt (backwards compatible with RecursiveModel)
    answer = engine.generate("Solve this problem.")

    # Batch: shared workers across all trees, streams results to JSONL
    engine.process_batch(prompts, output_jsonl_path="results.jsonl")

    # Restart generation from an existing trace node
    engine.continue_from_node(some_node)
"""

import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Callable, List, Dict, Any, Union

from abyme.utils import verify_format
from abyme.tree_trace import TreeTraceNode, length, to_dict, deep_clone_subtree, clone_trace_for_continuation
from abyme.tree_trace import total_calls, max_depth, max_subproblems, max_output_character, parallel_latency
from abyme.global_task_manager import GlobalTaskManager
from abyme.model import Model, ErrorGuardModel
from abyme.magic import magic_formatter


class RecursiveEngine(Model):
    """
    Unified recursive generation engine with shared worker pool.

    A single pool of workers dynamically serves all active trees, so idle workers
    from one tree can immediately pick up tasks from another.
    """

    def __init__(
        self,
        base_model: Model,
        guard_model: Optional[Model] = None,
        max_workers: int = 60,
        max_depth: int = 20,
        max_call: int = 1000,
        max_subproblem_retry: int = 2,
        max_chain_length: int = 5,
        proceed_when_fail: bool = True,
        formatter: Callable[[str, str, str, str], str] = magic_formatter,
        print_progress: bool = False
    ):
        """
        Args:
            base_model: Model used for generation
            guard_model: Fallback model when max_depth/max_chain_length is reached
            max_workers: Number of worker threads shared across all trees
            max_depth: Maximum recursion depth per node
            max_call: Maximum total LLM calls across all trees in one run
            max_subproblem_retry: Retry attempts per node on failure
            max_chain_length: Maximum temporal continuation steps per chain
            proceed_when_fail: Continue generation even when subproblems fail
            formatter: Prompt formatter (prompt, main, boss, fragment) -> str
            print_progress: Print per-node debug lines
        """
        self.base_model = base_model
        self.guard_model = guard_model if guard_model is not None else ErrorGuardModel()
        self.max_workers = max_workers
        self.max_depth = max_depth
        self.max_call = max_call
        self.max_subproblem_retry = max_subproblem_retry
        self.max_chain_length = max_chain_length
        self.proceed_when_fail = proceed_when_fail
        self.formatter = formatter
        self.print_progress = print_progress
        self.print_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_attempt: int = 1) -> str:
        """
        Generate a response for a single prompt.

        Backwards-compatible with RecursiveModel.generate().

        Args:
            prompt: Input prompt
            max_attempt: Retry attempts per node

        Returns:
            Final answer string (content after </think>)
        """
        tree_id = "single"
        root = TreeTraceNode(prompt=prompt, fragment="", index=0)

        manager = GlobalTaskManager(
            max_call=self.max_call,
            proceed_when_fail=self.proceed_when_fail
        )
        manager.register_tree(tree_id, root)

        self._run_pool(manager, max_attempt, wait_for=tree_id)

        final_root = manager.get_tree_root(tree_id)
        if final_root.status == "FINAL":
            return final_root.get_final_output()
        raise Exception(f"Generation failed: {final_root.error_message}")

    def process_batch(
        self,
        prompts: List[str],
        output_jsonl_path: str,
        max_attempt: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Process a list of prompts with a shared worker pool.

        Results are written to output_jsonl_path as each tree finishes (streaming).
        Each line contains: index, prompt, status, output, metrics, trace_tree.

        Args:
            prompts: Input prompts
            output_jsonl_path: Path to write JSONL results
            max_attempt: Retry attempts per node

        Returns:
            List of result dicts (same content as the JSONL file)
        """
        from pathlib import Path

        output_path = Path(output_jsonl_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("")  # clear file

        file_lock = threading.Lock()

        print(f"Batch: {len(prompts)} prompts | {self.max_workers} shared workers | max_call={self.max_call}")

        manager = GlobalTaskManager(
            max_call=self.max_call,
            proceed_when_fail=self.proceed_when_fail
        )

        for i, prompt in enumerate(prompts):
            root = TreeTraceNode(prompt=prompt, fragment="", index=0)
            manager.register_tree(f"tree_{i}", root)

        # Launch worker pool in background threads
        completed_count = [0]  # mutable int in closure

        def on_tree_done(tree_id: str, idx: int):
            root = manager.get_tree_root(tree_id)
            result = self._build_result(idx, prompts[idx], root)
            with file_lock:
                with output_path.open('a') as f:
                    f.write(json.dumps(result) + '\n')
            completed_count[0] += 1
            status = "OK" if root.status == "FINAL" else "FAIL"
            print(f"  [{completed_count[0]}/{len(prompts)}] {status} tree_{idx}")

        # Watcher thread: fires on_tree_done as each tree's Event becomes set
        def watcher():
            pending = list(range(len(prompts)))
            while pending:
                still_pending = []
                for idx in pending:
                    tid = f"tree_{idx}"
                    if manager.is_tree_finished(tid):
                        on_tree_done(tid, idx)
                    else:
                        still_pending.append(idx)
                pending = still_pending
                if pending:
                    # Wait for any notification rather than busy-spinning
                    with manager.global_lock:
                        manager.worker_condition.wait(timeout=0.2)

        watcher_thread = threading.Thread(target=watcher, daemon=True)
        watcher_thread.start()

        self._run_pool(manager, max_attempt, wait_for=None)
        manager.wait_for_all_trees()
        watcher_thread.join()

        # Read back results in order
        results = []
        with output_path.open('r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        results.sort(key=lambda r: r['index'])
        return results

    def continue_from_node(
        self,
        source_nodes: Union[TreeTraceNode, List[TreeTraceNode]],
        output_jsonl_path: str,
        max_attempt: int = 1,
        group_size: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Restart generation from one or more nodes in existing traces and save results.

        Each source_node is the start of a subtree to regenerate.  The full
        ancestor chain is cloned so the regenerated subtree bubbles its result
        up through its parents (which already have their sibling subtrees
        available) until the root reaches FINAL.  The original traces are
        never modified.

        General case (source_node with non-empty past):
            The node's prompt, fragment, and past chain are preserved so the
            model sees its full temporal context.  Its output and subproblems
            are cleared.  Ancestor nodes are restored to WAIT_SUB with their
            original outputs intact (needed to reconstruct sibling subproblem
            prompts).  Completed siblings are deep-cloned as-is so their
            outputs are injected when the parent continues.

        Batch processing:
            source_nodes are processed in groups of group_size, each group
            sharing one worker pool.  Results stream to output_jsonl_path as
            each tree finishes.

        Args:
            source_nodes:       Single node or list of nodes to restart from
            output_jsonl_path:  Path to write JSONL results (one line per node)
            max_attempt:        Retry attempts per generation step
            group_size:         Number of nodes to process in parallel per batch

        Returns:
            List of result dicts sorted by index (same content as the JSONL)
        """
        from pathlib import Path

        if isinstance(source_nodes, TreeTraceNode):
            source_nodes = [source_nodes]

        output_path = Path(output_jsonl_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("")

        file_lock = threading.Lock()
        all_results: List[Dict[str, Any]] = []
        completed_count = [0]

        print(f"Continuing {len(source_nodes)} node(s) | group_size={group_size} | {self.max_workers} workers")

        for group_start in range(0, len(source_nodes), group_size):
            group = source_nodes[group_start : group_start + group_size]

            manager = GlobalTaskManager(
                max_call=self.max_call,
                proceed_when_fail=self.proceed_when_fail,
            )

            # Build (tree_id -> original source_node) map for this group
            source_map: Dict[str, TreeTraceNode] = {}
            for local_idx, src in enumerate(group):
                global_idx = group_start + local_idx
                tree_id = f"continue_{global_idx}"
                cloned_root, _ = clone_trace_for_continuation(src)
                manager.register_tree(tree_id, cloned_root)
                source_map[tree_id] = src

            def on_tree_done(tree_id: str, global_idx: int, src: TreeTraceNode):
                root = manager.get_tree_root(tree_id)
                result = self._build_continuation_result(global_idx, src, root)
                with file_lock:
                    with output_path.open("a") as f:
                        f.write(json.dumps(result) + "\n")
                    all_results.append(result)
                completed_count[0] += 1
                status = "OK" if root.status == "FINAL" else "FAIL"
                print(f"  [{completed_count[0]}/{len(source_nodes)}] {status} continue_{global_idx}")

            def watcher():
                pending = list(range(len(group)))
                while pending:
                    still_pending = []
                    for local_idx in pending:
                        global_idx = group_start + local_idx
                        tree_id = f"continue_{global_idx}"
                        if manager.is_tree_finished(tree_id):
                            on_tree_done(tree_id, global_idx, source_map[tree_id])
                        else:
                            still_pending.append(local_idx)
                    pending = still_pending
                    if pending:
                        with manager.global_lock:
                            manager.worker_condition.wait(timeout=0.2)

            watcher_thread = threading.Thread(target=watcher, daemon=True)
            watcher_thread.start()

            self._run_pool(manager, max_attempt, wait_for=None)
            manager.wait_for_all_trees()
            watcher_thread.join()

        all_results.sort(key=lambda r: r["index"])
        return all_results

    def _build_continuation_result(
        self, idx: int, source_node: TreeTraceNode, root: TreeTraceNode
    ) -> Dict[str, Any]:
        """Build the JSONL result dict for one completed continuation."""
        is_success = root.status == "FINAL"
        return {
            "index": idx,
            "source_node": to_dict(source_node),
            "status": "SUCCESS" if is_success else "FAILED",
            "output": root.get_final_output() if is_success else "",
            "error": None if is_success else root.error_message,
            "metrics": {
                "total_llm_calls": total_calls(root),
                "max_tree_depth": max_depth(root),
                "max_subproblems": max_subproblems(root),
                "max_output_chars": max_output_character(root),
                "theoretical_parallel_latency": parallel_latency(root),
                "length": length(root),
            },
            "trace_tree": to_dict(root),
        }

    # ------------------------------------------------------------------
    # Internal machinery
    # ------------------------------------------------------------------

    def _run_pool(
        self,
        manager: GlobalTaskManager,
        max_attempt: int,
        wait_for: Optional[str]
    ):
        """
        Launch the shared ThreadPoolExecutor and block until done.

        Args:
            manager: GlobalTaskManager that owns all trees
            max_attempt: Retry attempts passed to _guarded_generate
            wait_for: tree_id to wait for (None = wait for all trees)
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for _ in range(self.max_workers):
                executor.submit(self._worker_loop, manager, max_attempt)

            if wait_for is not None:
                manager.wait_for_tree(wait_for)
            else:
                manager.wait_for_all_trees()

            # Wake sleeping workers so they exit the loop cleanly
            with manager.global_lock:
                manager.worker_condition.notify_all()

    def _worker_loop(self, manager: GlobalTaskManager, max_attempt: int):
        """
        Continuous loop: pull tasks from the global queue and generate.
        """
        while True:
            with manager.global_lock:
                if manager.all_finished:
                    break

            task_info = manager.get_next_task()

            if task_info is None:
                with manager.global_lock:
                    if manager.all_finished:
                        break
                    manager.worker_condition.wait(timeout=0.5)
                continue

            tree_id, node = task_info

            if node.is_cancelled:
                continue

            try:
                output, latency = self._guarded_generate(node, max_attempt)
                manager.report_success(tree_id, node, output, latency)
            except Exception as e:
                manager.report_failure(tree_id, node, str(e))

    def _guarded_generate(
        self,
        node: TreeTraceNode,
        max_attempt: int
    ):
        """
        Format prompt, enforce depth limits, call model, validate output.

        Returns:
            (output_str, latency_seconds)

        Raises:
            Exception if all attempts fail
        """
        active_model = self.base_model
        if node.depth >= self.max_depth or len(node.past) >= self.max_chain_length:
            active_model = self.guard_model

        boss_problem = node.parent.prompt if node.parent else "None"
        formatted_prompt = self.formatter(
            node.prompt,
            node.main_problem,
            boss_problem,
            node.fragment
        )

        last_error = None
        for _ in range(max_attempt):
            try:
                start = time.time()
                response = active_model.generate(formatted_prompt, max_attempt=1)
                latency = time.time() - start

                if not verify_format(response):
                    raise ValueError("Invalid format in response")

                if self.print_progress:
                    with self.print_lock:
                        print(f"  depth={node.depth} lat={latency:.2f}s | {node.prompt[:80]}")

                return response, latency

            except Exception as e:
                last_error = e

        raise Exception(f"All {max_attempt} attempts failed. Last: {last_error}")

    def _build_result(self, idx: int, prompt: str, root: TreeTraceNode) -> Dict[str, Any]:
        """Build the JSONL result dict for one completed tree."""
        is_success = root.status == "FINAL"
        return {
            "index": idx,
            "prompt": prompt,
            "status": "SUCCESS" if is_success else "FAILED",
            "output": root.get_final_output() if is_success else "",
            "error": None if is_success else root.error_message,
            "metrics": {
                "total_llm_calls": total_calls(root),
                "max_tree_depth": max_depth(root),
                "max_subproblems": max_subproblems(root),
                "max_output_chars": max_output_character(root),
                "theoretical_parallel_latency": parallel_latency(root),
                "length": length(root)
            },
            "trace_tree": to_dict(root)
        }
