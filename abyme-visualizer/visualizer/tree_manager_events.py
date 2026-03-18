"""
VisualizerTreeManager - TreeManager subclass with event emission for real-time visualization

This module extends the base TreeManager to emit events whenever node states change,
enabling real-time WebSocket updates to the frontend without modifying the original abyme code.
"""

import sys
sys.path.append('/home/lilixing/Abyme/abyme-rllm')

from abyme.tree_manager import TreeManager
from abyme.tree_trace import TreeTraceNode
from typing import Optional, Callable


class VisualizerTreeManager(TreeManager):
    """
    TreeManager subclass that emits events for visualization.

    Maintains all thread-safety guarantees from the base class while adding
    event callbacks fired after state changes (inside lock protection).
    """

    def __init__(
        self,
        root: TreeTraceNode,
        proceed_when_fail: bool = True,
        event_callback: Optional[Callable] = None
    ):
        """
        Initialize the visualizer tree manager.

        Args:
            root: Root TreeTraceNode for the generation tree
            proceed_when_fail: Whether to continue generation when subproblems fail
            event_callback: Callable(event_type, node, root, call_count) to invoke on state changes
        """
        super().__init__(root, proceed_when_fail)
        self.event_callback = event_callback
        self.call_count_ref = None  # Will be set to reference RecursiveModel.call_count

    def set_call_count_ref(self, call_count_ref):
        """
        Set reference to the RecursiveModel's call_count for metrics.

        Args:
            call_count_ref: Reference to RecursiveModel instance for accessing call_count
        """
        self.call_count_ref = call_count_ref

    def _emit_event(self, event_type: str, node: TreeTraceNode):
        """
        Emit event to the callback if registered.

        This is called INSIDE the lock, so the tree state is consistent.

        Args:
            event_type: Type of event ("node_updated", "node_failed", etc.)
            node: The node that changed
        """
        if self.event_callback:
            try:
                # Get call_count from the model reference if available
                call_count = 0
                if self.call_count_ref is not None:
                    if hasattr(self.call_count_ref, 'call_count'):
                        call_count = self.call_count_ref.call_count

                # Event callback receives: event_type, node, current root, call_count
                self.event_callback(event_type, node, self.root, call_count)
            except Exception as e:
                # Don't let callback errors crash the generation
                print(f"[ERROR] Event callback failed: {e}")

    def report_success(self, node: TreeTraceNode, output: str):
        """
        Override to emit event after successful generation.

        Args:
            node: Node that successfully generated output
            output: The generated output string
        """
        # Call parent implementation (handles all state logic)
        super().report_success(node, output)

        # Emit event (we're still inside the lock here)
        self._emit_event("node_updated", node)

    def report_failure(self, node: TreeTraceNode, error: str):
        """
        Override to emit event after generation failure.

        Args:
            node: Node that failed to generate
            error: Error message
        """
        # Call parent implementation
        super().report_failure(node, error)

        # Emit event
        self._emit_event("node_failed", node)
