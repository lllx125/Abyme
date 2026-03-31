"""
Global Task Manager Module

Centralized task scheduler that coordinates multiple RecursiveModel trees simultaneously.
Uses a single shared RLock across all TreeManagers to enable safe cross-tree task scheduling
with zero deadlock risk and maximum worker utilization.
"""

import threading
from typing import Optional, Dict, Tuple, List
from abyme.tree_trace import TreeTraceNode
from abyme.tree_manager import TreeManager


class GlobalTaskManager:
    """
    Thread-safe orchestrator for multiple RecursiveModel generation trees.

    Manages a shared worker pool that dynamically balances work across all active trees.
    All TreeManagers share the same RLock so cross-tree scheduling is atomic.
    """

    def __init__(self, proceed_when_fail: bool = True):
        """
        Args:
            proceed_when_fail: Whether to continue generation when subproblems fail
        """
        self.proceed_when_fail = proceed_when_fail

        # Single shared RLock used by this manager AND all TreeManagers.
        # RLock allows the same thread to re-enter it (e.g., GlobalTaskManager.get_next_task
        # holds the lock and calls TreeManager.get_next_task which also acquires it).
        self.global_lock = threading.RLock()
        self.worker_condition = threading.Condition(self.global_lock)

        # Tree registry: tree_id -> TreeManager
        self.trees: Dict[str, TreeManager] = {}

        # Completion events: tree_id -> threading.Event, set when that tree finishes
        self.completion_events: Dict[str, threading.Event] = {}

        # Set to True once all trees are finished
        self.all_finished = False

        # Pulse event for the watcher thread: set whenever any tree finishes.
        # The watcher clears this before polling so it never misses a completion.
        self.tree_done_event = threading.Event()

    def register_tree(self, tree_id: str, root: TreeTraceNode, max_call: int = 1000) -> TreeManager:
        """
        Register a new tree for processing.

        Args:
            tree_id: Unique identifier for this tree
            root: Root node of the tree
            max_call: Maximum LLM calls allowed for this tree

        Returns:
            TreeManager instance for this tree
        """
        with self.global_lock:
            if tree_id in self.trees:
                raise ValueError(f"Tree ID '{tree_id}' already registered")

            # Pass the shared lock so TreeManager operations are atomic with respect
            # to the global manager — no separate per-tree lock, just one shared RLock.
            manager = TreeManager(
                root=root,
                proceed_when_fail=self.proceed_when_fail,
                shared_lock=self.global_lock,
                shared_condition=self.worker_condition,
                max_call=max_call
            )
            self.trees[tree_id] = manager
            self.completion_events[tree_id] = threading.Event()
            return manager

    def get_next_task(self) -> Optional[Tuple[str, TreeTraceNode]]:
        """
        Get the next optimal task across ALL trees using Greedy DFS.

        Atomically marks the chosen node as GENERATING. If the best candidate's
        tree has exceeded its per-tree max_call limit, the node is instantly failed
        and the search continues for the next available tree.

        Returns:
            (tree_id, node) for the easiest pending task, or None if nothing is ready
        """
        with self.global_lock:
            while True:
                best_node = None
                best_tree_id = None
                best_manager = None
                best_difficulty = float('inf')

                for tree_id, manager in self.trees.items():
                    if manager.is_finished:
                        continue
                    # _dfs_find_best does not mark status — we do it below after picking the winner
                    candidate = manager._dfs_find_best(manager.root)
                    if candidate is not None and candidate.difficulty < best_difficulty:
                        best_difficulty = candidate.difficulty
                        best_node = candidate
                        best_tree_id = tree_id
                        best_manager = manager

                if best_node is None:
                    return None

                # Enforce per-tree max_call limit
                if best_manager.call_count >= best_manager.max_call:
                    # HARD KILL: Abort the entire tree instantly instead of bubbling failures
                    best_manager.root.cancel_tree()  # Cancels all pending subproblems
                    best_manager.root.status = "FAILED" # Override root status
                    best_manager.root.error_message = f"Maximum per-tree call count ({best_manager.max_call}) exceeded."
                    best_manager.is_finished = True
                    
                    self.completion_events[best_tree_id].set()
                    self.tree_done_event.set()
                    if self._all_trees_done():
                        self.all_finished = True
                        
                    self.worker_condition.notify_all()
                    continue  # Loop again to find the next best task from other trees

                best_node.status = "GENERATING"
                best_manager.call_count += 1
                return (best_tree_id, best_node)

    def report_success(self, tree_id: str, node: TreeTraceNode, output: str, latency: float):
        """
        Report successful generation for a node.

        Args:
            tree_id: ID of the tree this node belongs to
            node: The node that was successfully generated
            output: The generated output text
            latency: Wall-clock time taken for generation in seconds
        """
        with self.global_lock:
            if tree_id not in self.trees:
                return

            node.record_generation(output, latency)
            manager = self.trees[tree_id]
            # report_success internally acquires manager.lock (== global_lock, re-entrant OK)
            manager.report_success(node, output)

            if manager.is_finished:
                self.completion_events[tree_id].set()
                self.tree_done_event.set()
                if self._all_trees_done():
                    self.all_finished = True

            self.worker_condition.notify_all()

    def report_failure(self, tree_id: str, node: TreeTraceNode, error: str):
        """
        Report generation failure for a node.

        Args:
            tree_id: ID of the tree this node belongs to
            node: The node that failed
            error: Error message
        """
        with self.global_lock:
            if tree_id not in self.trees:
                return

            manager = self.trees[tree_id]
            manager.report_failure(node, error)

            if manager.is_finished:
                self.completion_events[tree_id].set()
                self.tree_done_event.set()
                if self._all_trees_done():
                    self.all_finished = True

            self.worker_condition.notify_all()

    def wait_for_tree(self, tree_id: str):
        """Block until a specific tree completes."""
        if tree_id not in self.completion_events:
            raise ValueError(f"Unknown tree ID: {tree_id}")
        self.completion_events[tree_id].wait()

    def wait_for_all_trees(self):
        """Block until all registered trees complete."""
        for event in self.completion_events.values():
            event.wait()

    def is_tree_finished(self, tree_id: str) -> bool:
        """Return True if the given tree has finished."""
        with self.global_lock:
            if tree_id not in self.trees:
                return False
            return self.trees[tree_id].is_finished

    def wait_for_any_tree_done(self, timeout: float = 0.5):
        """Block until any tree finishes (or timeout). Must be called after clearing tree_done_event."""
        self.tree_done_event.wait(timeout=timeout)

    def get_tree_root(self, tree_id: str) -> Optional[TreeTraceNode]:
        """Return the (possibly updated) root node of a tree."""
        with self.global_lock:
            if tree_id not in self.trees:
                return None
            return self.trees[tree_id].root

    def _all_trees_done(self) -> bool:
        """Return True if every registered tree has finished (called under lock)."""
        return bool(self.trees) and all(m.is_finished for m in self.trees.values())
