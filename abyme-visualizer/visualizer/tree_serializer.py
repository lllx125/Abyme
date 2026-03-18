"""
Thread-Safe Tree Serialization

Provides lock-protected serialization of TreeTraceNode graphs using the draw_tree() function.
Critical for preventing race conditions when multiple workers are modifying the tree concurrently.
"""

import sys
sys.path.append('/home/lilixing/Abyme/abyme-rllm')

import threading
from typing import Dict, List, Tuple, Any
from abyme.tree_trace import TreeTraceNode, draw_tree


class ThreadSafeTreeSerializer:
    """
    Thread-safe tree serialization with locking.

    Uses the TreeManager's RLock to protect tree traversal during draw_tree() calls.
    """

    @staticmethod
    def serialize_tree_safe(
        root: TreeTraceNode,
        manager_lock: threading.RLock
    ) -> Dict[str, Any]:
        """
        Safely serialize tree while holding the manager's lock.

        Args:
            root: Root TreeTraceNode to serialize
            manager_lock: The TreeManager's RLock (must acquire this during traversal)

        Returns:
            Serialized tree data with nodes and edges in JSON-serializable format
        """
        with manager_lock:
            # Call draw_tree while holding the lock to prevent concurrent modifications
            nodes_list, edges_list = draw_tree(
                root,
                right_spacing=150.0,   # Horizontal spacing between sibling subproblems
                down_spacing=100.0,    # Vertical spacing for hierarchy
                node_radius=30.0       # Node radius for bounding box calculations
            )

            # Make deep copies of the results while still holding the lock
            # This ensures we capture a consistent snapshot
            nodes_snapshot = [dict(node) for node in nodes_list]
            edges_snapshot = list(edges_list)

        # Now release lock and convert edges to JSON-serializable format
        return {
            'nodes': nodes_snapshot,
            'edges': ThreadSafeTreeSerializer._serialize_edges(edges_snapshot)
        }

    @staticmethod
    def _serialize_edges(edges_list: List[Tuple]) -> List[Dict]:
        """
        Convert edge tuples to JSON-serializable dictionaries.

        Args:
            edges_list: List of (edge_type, (x1, y1), (x2, y2)) tuples from draw_tree()

        Returns:
            List of edge dictionaries with type and coordinate objects
        """
        serialized_edges = []

        for edge_type, (x1, y1), (x2, y2) in edges_list:
            serialized_edges.append({
                'type': edge_type,
                'from': {'x': float(x1), 'y': float(y1)},
                'to': {'x': float(x2), 'y': float(y2)}
            })

        return serialized_edges
