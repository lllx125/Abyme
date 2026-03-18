"""
Stoppable Recursive Model

Extends RecursiveModel with graceful stop capability for user-initiated cancellation.
Workers cooperatively check a stop signal and exit cleanly.
"""

import sys
sys.path.append('/home/lilixing/Abyme/abyme-rllm')

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Callable

from abyme.core import RecursiveModel
from abyme.tree_trace import TreeTraceNode
from abyme.model import Model
from .tree_manager_events import VisualizerTreeManager


class StoppableRecursiveModel(RecursiveModel):
    """
    RecursiveModel with stop capability for visualization.

    Adds a stop_requested Event that workers check cooperatively,
    allowing graceful shutdown when the user clicks the stop button.
    """

    def __init__(self, *args, event_callback: Optional[Callable] = None, **kwargs):
        """
        Initialize the stoppable recursive model.

        Args:
            *args: Arguments for RecursiveModel
            event_callback: Callback for tree state changes
            **kwargs: Keyword arguments for RecursiveModel
        """
        super().__init__(*args, **kwargs)
        self.stop_requested = threading.Event()
        self.event_callback = event_callback

    def generate(self, prompt: str, max_attempt: int = 1) -> str:
        """
        Override generate to use VisualizerTreeManager and handle stop signals.

        Args:
            prompt: The input prompt
            max_attempt: Maximum retry attempts (default: 1)

        Returns:
            Generated output string

        Raises:
            Exception: If generation fails or is stopped
        """
        # Clear stop signal for new generation
        self.stop_requested.clear()

        # Reset execution state for a fresh run
        with self.call_lock:
            self.call_count = 0

        self.trace = TreeTraceNode(prompt=prompt, fragment="", index=0)

        # Use VisualizerTreeManager instead of base TreeManager
        manager = VisualizerTreeManager(
            self.trace,
            proceed_when_fail=self.proceed_when_fail,
            event_callback=None  # Will be set after passing manager to callback
        )

        # Set reference to this model so manager can access call_count
        manager.set_call_count_ref(self)

        # Now call the user's event_callback with the manager so they can set up emitter
        if self.event_callback:
            # Pass the manager to the callback so it can access manager.lock
            manager.event_callback = lambda event_type, node, root, call_count: \
                self.event_callback(event_type, node, root, call_count, manager)

        # Spawn the thread pool with stoppable workers
        with ThreadPoolExecutor(max_workers=self.max_parallel_workers) as executor:
            # Submit N worker loops to the pool
            futures = []
            for _ in range(self.max_parallel_workers):
                future = executor.submit(
                    self._worker_loop_stoppable,
                    manager,
                    self.max_subproblem_retry,
                    prompt
                )
                futures.append(future)

            # Block the main thread until the tree resolves to FINAL/FAILED or stop is requested
            while not manager.is_finished and not self.stop_requested.is_set():
                with manager.lock:
                    if manager.is_finished or self.stop_requested.is_set():
                        break
                    manager.worker_condition.wait(timeout=0.5)

            # Wake up any workers currently sleeping so they can exit gracefully
            with manager.lock:
                manager.is_finished = True  # Ensure workers see finish signal
                manager.worker_condition.notify_all()

        # Update self.trace to point to the resolved root
        self.trace = manager.root

        # Check if user stopped generation
        if self.stop_requested.is_set():
            raise Exception("Generation stopped by user")

        # Evaluate the final state
        if self.trace.status == "FINAL":
            return self.trace.get_final_output()
        else:
            raise Exception(f"Tree generation failed. Reason: {self.trace.error_message}")

    def _worker_loop_stoppable(self, manager: VisualizerTreeManager, max_attempt: int, main_problem: str):
        """
        The continuous execution loop run by each thread (with stop signal checking).

        Args:
            manager: VisualizerTreeManager instance
            max_attempt: Maximum retry attempts for individual tasks
            main_problem: The root problem prompt
        """
        while True:
            # Check stop signal FIRST
            if self.stop_requested.is_set():
                print("[INFO] Worker exiting due to stop signal")
                break

            # Check if the overall generation has finished
            with manager.lock:
                if manager.is_finished:
                    break

            # Ask TreeManager for the next best node based on Greedy DFS
            task = manager.get_next_task()

            # If no tasks are currently ready, wait cooperatively until notified
            if task is None:
                with manager.lock:
                    if manager.is_finished or self.stop_requested.is_set():
                        break
                    # Wait for up to 0.5s or until another thread wakes us up
                    manager.worker_condition.wait(timeout=0.5)
                continue

            # Cooperative Cancellation Check: Skip if an OR-sibling already won or stopped
            if task.is_cancelled or self.stop_requested.is_set():
                continue

            # Execute the API call inside a broad try/catch to ensure we report back
            try:
                output = self._guarded_generate_with_formatter(task, max_attempt, main_problem)

                # Check stop signal before reporting (avoid reporting if stopped)
                if not self.stop_requested.is_set():
                    manager.report_success(task, output)
            except Exception as e:
                # Check stop signal before reporting failure
                if not self.stop_requested.is_set():
                    manager.report_failure(task, str(e))

    def stop(self):
        """
        Stop the generation gracefully.

        Sets the stop_requested Event, which all workers check cooperatively.
        Workers will exit their loops and the ThreadPoolExecutor will clean up.
        """
        print("[INFO] Stop requested - signaling all workers")
        self.stop_requested.set()
