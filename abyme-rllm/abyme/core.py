"""
Recursive engine for processing elaborate/response cycles using RecursiveModel pattern.

DEPRECATED: This module is kept for backwards compatibility.
Please use RecursiveEngine from abyme.recursive_engine instead.
"""
import time
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Callable, TYPE_CHECKING

# Assuming these are imported from your project structure
from .utils import verify_format
from .tree_trace import TreeTraceNode
from .tree_manager import TreeManager
from .model import Model, DeepSeekModel, GPTModel, ErrorGuardModel
from .magic import *


class RecursiveModel(Model):
    """
    A multi-threaded, recursive model wrapper that implements delegation-based problem decomposition.
    
    This class manages a pool of worker threads that continuously poll a thread-safe
    TreeManager for the next optimal subproblem (using Greedy DFS). It enforces global
    safety constraints such as max_call limits and max_depth boundaries.
    """

    def __init__(self,
                 base_model: Model,
                 guard_model: Optional[Model] = None,
                 max_depth: int = 20,
                 max_call: int = 50,
                 max_parallel_workers: int = 1,
                 max_subproblem_retry: int = 2,
                 max_chain_length: int = 5,
                 proceed_when_fail: bool = True,
                 formatter: Callable[[str, str, str, str], str] = magic_formatter,
                 print_progress: bool = False):
        """
        Initialize the multi-threaded recursive model.

        DEPRECATED: RecursiveModel is deprecated. Use RecursiveEngine instead:
            from abyme.recursive_engine import RecursiveEngine
            engine = RecursiveEngine(base_model=base_model, max_workers=max_parallel_workers, ...)

        Args:
            base_model: The underlying Model to use for generation.
            guard_model: Model to use when max recursion depth is reached (forces a leaf response).
            max_depth: Maximum recursion depth allowed. Prevents infinite nested delegations.
            max_call: Maximum total number of model calls allowed across the entire tree.
            max_parallel_workers: Number of threads to spawn for parallel subproblem execution.
            max_subproblem_retry: Retry limit for individual network/parsing failures.
            formatter: Function to format prompt context for the LLM.
            print_progress: If True, prints thread-safe debug logs to the console.
        """
        warnings.warn(
            "RecursiveModel is deprecated and will be removed in a future version. "
            "Use RecursiveEngine instead: from abyme.recursive_engine import RecursiveEngine",
            DeprecationWarning,
            stacklevel=2
        )
        self.base_model = base_model
        self.guard_model = guard_model
        self.max_depth = max_depth
        self.max_call = max_call
        self.max_parallel_workers = max_parallel_workers
        self.max_subproblem_retry = max_subproblem_retry
        self.formatter = formatter
        self.print_progress = print_progress
        self.trace: TreeTraceNode = TreeTraceNode(prompt="", fragment="")  # Placeholder root node for the trace
        # Thread-safe global call tracking
        self.call_count = 0
        self.call_lock = threading.Lock()
        self.print_lock = threading.Lock()
        self.max_chain_length = max_chain_length
        self.proceed_when_fail = proceed_when_fail

    def generate(self, prompt: str, max_attempt: int = 1) -> str:
        """
        Main entry point. Initializes the tree, starts the thread pool, and waits for completion.
        """
        # Reset execution state for a fresh run
        with self.call_lock:
            self.call_count = 0

        self.trace = TreeTraceNode(prompt=prompt, fragment="", index=0)
        manager = TreeManager(self.trace)

        # Spawn the thread pool
        with ThreadPoolExecutor(max_workers=self.max_parallel_workers) as executor:
            # Submit N worker loops to the pool
            futures = []
            for _ in range(self.max_parallel_workers):
                future = executor.submit(self._worker_loop, manager, self.max_subproblem_retry, prompt)
                futures.append(future)

            # Block the main thread until the tree resolves to FINAL or FAILED
            manager.wait_for_completion()
            
            # Wake up any workers currently sleeping so they can exit gracefully
            with manager.lock:
                manager.worker_condition.notify_all()

        # Update self.trace to point to the resolved root (in case manager replaced it during continuation)
        self.trace = manager.root

        # Evaluate the final state
        if self.trace.status == "FINAL":
            return self.trace.get_final_output()
        else:
            raise Exception(f"Tree generation failed. Reason: {self.trace.error_message}")

    def _worker_loop(self, manager: TreeManager, max_attempt: int, main_problem: str):
        """
        The continuous execution loop run by each thread.
        Asks the TreeManager for tasks, executes LLM calls, and reports back.
        """
        while True:
            # Check if the overall generation has finished (successful or failed)
            with manager.lock:
                if manager.is_finished:
                    break

            # Ask TreeManager for the next best node based on Greedy DFS
            task = manager.get_next_task()

            # If no tasks are currently ready, wait cooperatively until notified
            if task is None:
                with manager.lock:
                    if manager.is_finished:
                        break
                    # Wait for up to 0.5s or until another thread wakes us up with notify_all()
                    manager.worker_condition.wait(timeout=0.5)
                continue

            # Cooperative Cancellation Check: Skip if an OR-sibling already won
            if task.is_cancelled:
                continue

            # Execute the API call inside a broad try/catch to ensure we report back
            try:
                output = self._guarded_generate_with_formatter(task, max_attempt, main_problem)
                manager.report_success(task, output)
            except Exception as e:
                manager.report_failure(task, str(e))

    def _guarded_generate_with_formatter(self, node: TreeTraceNode, max_attempt: int, main_problem: str) -> str:
        """
        Handles formatting, API retries, format validation, and global limits.
        Raises exceptions on failure, which are caught by the _worker_loop.
        """
        # 1. Enforce Global Call Limits (Thread-Safe)
        with self.call_lock:
            if self.call_count >= self.max_call:
                raise Exception(f"Maximum global call count ({self.max_call}) exceeded.")
            self.call_count += 1

        # 2. Enforce Depth Limits
        active_model = self.base_model
        if node.depth >= self.max_depth or len(node.past) >= self.max_chain_length:
            if self.guard_model:
                active_model = self.guard_model
            else:
                raise Exception(f"Maximum recursion depth ({self.max_depth}) reached.")


        # 3. Format the Prompt Context
        boss_problem = node.parent.prompt if node.parent else "None"
        formatted_prompt = self.formatter(
            node.prompt,
            main_problem,
            boss_problem,
            node.fragment
        )

        last_error = None

        # 4. Retry Loop for the actual API generation
        for attempt in range(max_attempt):
            try:
                start_time = time.time()
                
                # Make the blocking network call
                response = active_model.generate(formatted_prompt, max_attempt=1)
                latency = time.time() - start_time

                # Validate the XML-like tags using your utility
                if not verify_format(response):
                    raise ValueError(f"Invalid format detected in response. Output: {response}")

                # Save metrics to the node directly
                node.record_generation(response, latency)

                # Thread-safe console logging
                if self.print_progress:
                    with self.print_lock:
                        print(f"{'{'}\n\"prompt\":\"{node.prompt}\"\n\"context\":\"{boss_problem}\"\n\"fragment\":\"{node.fragment}\"\n\"output\":\"{response}\"\n{'}'}\ndepth: {node.depth}\nlatency: {latency:.2f}s\ncall: {self.call_count}\n{'='*50}")

                return response

            except Exception as e:
                last_error = e
                # Only retry if we haven't exhausted attempts
                continue

        # If we broke out of the loop, all attempts failed
        raise Exception(f"All {max_attempt} attempts failed. Last error: {last_error}")

def Abyme_API_Model(model: str, **args) -> RecursiveModel:
    
    if model == "deepseek":
        base_model = DeepSeekModel(system_prompt=magic_prompt)
    elif model == "gpt":
        base_model = GPTModel(system_prompt=magic_prompt)
    elif model == "deepseek-r":
        base_model = DeepSeekModel(reasoning= True, system_prompt=magic_prompt)
    else:
        raise ValueError(f"Unsupported model type: {model}")
    
    guard_model = ErrorGuardModel()
    return RecursiveModel(
        base_model=base_model,
        guard_model=guard_model,
        formatter=magic_formatter,
        **args
    )