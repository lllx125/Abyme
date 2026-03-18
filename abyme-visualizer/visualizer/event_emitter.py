"""
Throttled Event Emitter for WebSocket Communication

Prevents overwhelming clients with rapid tree updates by throttling and coalescing events.
Ensures smooth real-time visualization without performance degradation.
"""

import time
import threading
from typing import Optional, Tuple
from .tree_serializer import ThreadSafeTreeSerializer


class ThrottledEventEmitter:
    """
    Thread-safe event emitter with throttling and coalescing.

    Limits WebSocket emissions to a maximum frequency (default: 10/second),
    and coalesces multiple rapid updates into a single emission with the latest state.
    """

    def __init__(self, socketio, manager_lock: threading.RLock, throttle_interval: float = 0.1):
        """
        Initialize the throttled event emitter.

        Args:
            socketio: Flask-SocketIO instance for emitting events
            manager_lock: TreeManager's RLock for thread-safe serialization
            throttle_interval: Minimum time between emissions in seconds (default: 0.1 = 100ms)
        """
        self.socketio = socketio
        self.manager_lock = manager_lock
        self.throttle_interval = throttle_interval

        # Threading primitives
        self.emit_lock = threading.Lock()
        self.emit_timer: Optional[threading.Timer] = None

        # State tracking
        self.last_emit_time = 0.0
        self.pending_event: Optional[Tuple] = None

    def emit_tree_update(self, event_type: str, node, root, call_count: int):
        """
        Emit a tree update event with throttling and coalescing.

        If called too frequently, will schedule a delayed emission instead.
        Only the latest pending event is kept (coalescing).

        Args:
            event_type: Type of event ("node_updated", "node_failed", etc.)
            node: The node that changed (not used in emission, just for context)
            root: Current root of the tree
            call_count: Current call count from RecursiveModel
        """
        with self.emit_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_emit_time

            # Store the latest event (coalescing)
            self.pending_event = (event_type, root, call_count)

            if time_since_last >= self.throttle_interval:
                # Enough time has passed - emit immediately
                self._do_emit()
            elif self.emit_timer is None:
                # Schedule emission for later
                delay = self.throttle_interval - time_since_last
                self.emit_timer = threading.Timer(delay, self._do_emit)
                self.emit_timer.start()
            # else: timer already running, event will be coalesced

    def _do_emit(self):
        """
        Actually emit the pending event.

        This is called either immediately or from the timer callback.
        """
        with self.emit_lock:
            if self.pending_event is None:
                # No pending event (shouldn't happen, but be safe)
                self.emit_timer = None
                return

            event_type, root, call_count = self.pending_event
            self.pending_event = None
            self.emit_timer = None
            self.last_emit_time = time.time()

        # Serialize tree OUTSIDE the emit_lock to avoid holding it during slow operation
        try:
            tree_data = ThreadSafeTreeSerializer.serialize_tree_safe(root, self.manager_lock)

            # Check if generation is finished
            is_finished = root.status in ["FINAL", "FAILED"]

            # Emit to all connected clients
            self.socketio.emit('tree_update', {
                'event_type': event_type,
                'tree': tree_data,
                'metrics': {
                    'call_count': call_count,
                    'is_finished': is_finished,
                    'node_count': len(tree_data['nodes'])
                }
            }, namespace='/visualizer')

            print(f"[INFO] Emitted tree_update: {event_type}, nodes={len(tree_data['nodes'])}, calls={call_count}")

        except Exception as e:
            print(f"[ERROR] Failed to emit tree update: {e}")
            import traceback
            traceback.print_exc()

    def force_emit(self, event_type: str, root, call_count: int):
        """
        Force immediate emission bypassing throttle (used for final state).

        Args:
            event_type: Type of event
            root: Current root of the tree
            call_count: Current call count
        """
        try:
            tree_data = ThreadSafeTreeSerializer.serialize_tree_safe(root, self.manager_lock)

            is_finished = root.status in ["FINAL", "FAILED"]

            self.socketio.emit('tree_update', {
                'event_type': event_type,
                'tree': tree_data,
                'metrics': {
                    'call_count': call_count,
                    'is_finished': is_finished,
                    'node_count': len(tree_data['nodes'])
                }
            }, namespace='/visualizer')

            print(f"[INFO] Force-emitted tree_update: {event_type}, nodes={len(tree_data['nodes'])}")

        except Exception as e:
            print(f"[ERROR] Failed to force-emit tree update: {e}")
