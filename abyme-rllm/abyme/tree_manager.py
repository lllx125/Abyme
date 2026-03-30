"""
Tree Manager Module for Agent 4.0 Architecture
Handles thread-safe execution, Greedy DFS task selection, state resolution, and failure injection.
"""
import threading
from typing import Optional, List

# Assuming these are imported from your project structure
from .utils import AND, OR, LEAF, THINK, extract_delegations, replace_delegations_with_responses
from .tree_trace import TreeTraceNode

class TreeManager:
    """
    Thread-safe orchestrator for the RecursiveModel generation tree.
    """

    def __init__(
        self,
        root: TreeTraceNode,
        proceed_when_fail: bool = True,
        shared_lock: Optional[threading.RLock] = None,
        shared_condition: Optional[threading.Condition] = None,
        max_call: int = 1000
    ):
        self.root: TreeTraceNode = root
        self.proceed_when_fail: bool = proceed_when_fail

        self.max_call: int = max_call
        self.call_count: int = 0

        # Use provided shared lock (from GlobalTaskManager) or create a private one.
        # Sharing the lock with GlobalTaskManager avoids cross-lock deadlocks.
        self.lock = shared_lock if shared_lock is not None else threading.RLock()
        self.worker_condition = shared_condition if shared_condition is not None else threading.Condition(self.lock)

        # Execution State
        self.is_finished: bool = False

    def get_next_task(self) -> Optional[TreeTraceNode]:
        """
        Thread-Safe Greedy DFS Traversal.
        Returns the next optimal node in 'WAIT_GEN' state.
        """
        with self.lock:
            task = self._dfs_find_best(self.root)
            if task:
                task.status = "GENERATING"
            return task

    def _dfs_find_best(self, node: TreeTraceNode) -> Optional[TreeTraceNode]:
        """Recursive helper for Greedy DFS."""
        if node.is_cancelled or node.status in ["COMPLETED", "FINAL", "FAILED"]:
            return None

        if node.status == "WAIT_GEN":
            return node

        if node.status == "WAIT_SUB":
            # Filter valid children
            valid_children = [
                c for c in node.subproblems
                if not c.is_cancelled and c.status not in ["COMPLETED", "FINAL", "FAILED"]
            ]

            if not valid_children:
                return None

            # Greedy Selection: Evaluate the easiest path first
            valid_children.sort(key=lambda c: c.difficulty)

            for child in valid_children:
                best_leaf = self._dfs_find_best(child)
                if best_leaf:
                    return best_leaf

        return None

    def get_all_pending_tasks(self) -> List[TreeTraceNode]:
        """
        Get all nodes currently in WAIT_GEN state.

        Returns:
            List of all pending task nodes
        """
        with self.lock:
            return self._collect_pending(self.root)

    def _collect_pending(self, node: TreeTraceNode) -> List[TreeTraceNode]:
        """Recursively collect all WAIT_GEN nodes."""
        if node.is_cancelled or node.status in ["COMPLETED", "FINAL", "FAILED"]:
            return []

        if node.status == "WAIT_GEN":
            return [node]

        pending = []
        for child in node.subproblems:
            pending.extend(self._collect_pending(child))

        return pending

    def get_best_pending_task(self) -> Optional[TreeTraceNode]:
        """
        Get the easiest pending task without marking it as GENERATING.

        Used by GlobalTaskManager for cross-tree scheduling.

        Returns:
            Easiest pending task or None
        """
        with self.lock:
            return self._dfs_find_best(self.root)

    def report_success(self, node: TreeTraceNode, output: str):
        """
        Called by a worker thread when the base model successfully generates an output.
        """
        with self.lock:
            if node.is_cancelled:
                return

            node.output = output

            # Base Case
            if f"</{THINK}>" in output:
                node.type = LEAF
                node.status = "FINAL"
                self._resolve_parent(node)

            # OR Node
            elif f"<{OR}>" in output:
                node.type = OR
                node.status = "WAIT_SUB"
                sub_prompts = extract_delegations(output, tag=OR)
                node.add_subproblems(sub_prompts)
                
            # AND Node
            elif f"<{AND}>" in output:
                node.type = AND
                node.status = "WAIT_SUB"
                sub_prompts = extract_delegations(output, tag=AND)
                node.add_subproblems(sub_prompts)
                
            # Linear Continuation
            else:
                node.type = LEAF
                node.status = "COMPLETED"
                new_node = node.continue_generation(node.fragment + "\n" + output)
                if node == self.root:
                    self.root = new_node
                    
            self.worker_condition.notify_all()

    def report_failure(self, node: TreeTraceNode, error: str):
        """
        Called by a worker thread when the generation fails.
        """
        with self.lock:
            if node.is_cancelled:
                return
                
            node.status = "FAILED"
            node.error_message = error
            self._resolve_parent(node)
            self.worker_condition.notify_all()

    def wait_for_completion(self):
        """
        Blocks the main application thread until the tree successfully completes or fails.
        """
        with self.lock:
            while not self.is_finished:
                self.worker_condition.wait()

    def _resolve_parent(self, child_node: TreeTraceNode):
        parent = child_node.parent
        if parent is None:
            self.is_finished = True
            return

        from .utils import AND, OR, replace_delegations_with_responses

        # --- 1. EARLY KILL SWITCHES ---
        if child_node.status == "FAILED" and parent.type == AND:
            # One failure dooms the AND node. Cancel others.
            for sub in parent.subproblems:
                if sub != child_node and sub.status not in ["FINAL", "COMPLETED"]:
                    sub.cancel_tree()
                    
        elif child_node.status == "FINAL" and parent.type == OR:
            # One success fulfills the OR node. Cancel others.
            for sub in parent.subproblems:
                if sub != child_node and sub.status not in ["FINAL", "COMPLETED"]:
                    sub.cancel_tree()

        # --- 2. WAIT FOR ALL SIBLINGS ---
        active_siblings = any(
            sub.status in ["WAIT_GEN", "GENERATING", "WAIT_SUB"] 
            for sub in parent.subproblems
        )
        
        if active_siblings:
            # Wait for the other siblings to finish or be cancelled
            return

        # --- 3. RESOLVE PARENT OUTCOME ---
        if parent.type == AND:
            if any(sub.status == "FAILED" for sub in parent.subproblems) or any(sub.status == "CANCELLED" for sub in parent.subproblems):
                if self.proceed_when_fail:
                    parent.status = "COMPLETED"
                    responses = []
                    for sub in parent.subproblems:
                        if sub.status == "FINAL": responses.append(sub.get_final_output())
                        elif sub.status == "FAILED": responses.append(f"FAILED\n{sub.output}")
                        else: responses.append("CANCELLED")
                    reconstructed = replace_delegations_with_responses(parent.output, responses, tag=AND)
                    new_node = parent.continue_generation(parent.fragment + "\n\n" + reconstructed)
                    if parent == self.root: self.root = new_node
                else:
                    parent.status = "FAILED"
                    parent.error_message = f"A subproblem failed or was cancelled."
                    self._resolve_parent(parent)
            else:
                # All must be FINAL
                parent.status = "COMPLETED"
                responses = [sub.get_final_output() for sub in parent.subproblems]
                reconstructed = replace_delegations_with_responses(parent.output, responses, tag=AND)
                new_node = parent.continue_generation(parent.fragment + "\n\n" + reconstructed)
                if parent == self.root: self.root = new_node
                
        elif parent.type == OR:
            if any(sub.status == "FINAL" for sub in parent.subproblems):
                parent.status = "COMPLETED"
                responses = []
                for sub in parent.subproblems:
                    if sub.status == "FINAL": responses.append(sub.get_final_output())
                    elif sub.status == "FAILED": responses.append(f"FAILED\n{sub.output}")
                    else: responses.append("CANCELLED")
                reconstructed = replace_delegations_with_responses(parent.output, responses, tag=OR)
                new_node = parent.continue_generation(parent.fragment + "\n\n" + reconstructed)
                if parent == self.root: self.root = new_node
            else:
                # All failed or cancelled
                if self.proceed_when_fail:
                    parent.status = "COMPLETED"
                    responses = []
                    for sub in parent.subproblems:
                        if sub.status == "FAILED": responses.append(f"FAILED\n{sub.output}")
                        else: responses.append("CANCELLED")
                    reconstructed = replace_delegations_with_responses(parent.output, responses, tag=OR)
                    new_node = parent.continue_generation(parent.fragment + "\n\n" + reconstructed)
                    if parent == self.root: self.root = new_node
                else:
                    parent.status = "FAILED"
                    parent.error_message = "All attempted approaches failed."
                    self._resolve_parent(parent)