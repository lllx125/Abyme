"""
Abyme: Recursive reasoning engine with XML-based delegation.

A model can break a problem into subproblems (<do> / <try> tags), dispatch them
to sub-agents in parallel, and reconstruct the answer from their responses.

Recommended usage:
    from abyme import RecursiveEngine
    from abyme.vllm_model import LocalVLLMModel, APIModel

    model = LocalVLLMModel("path/to/model")
    engine = RecursiveEngine(base_model=model, max_workers=60)

    # Single prompt
    answer = engine.generate("Differentiate e^x sin(x).")

    # Batch — writes JSONL with full trace trees
    engine.process_batch(prompts, output_jsonl_path="results.jsonl")

    # Restart from a node in a previous trace
    engine.continue_from_node(some_node)
"""

from .recursive_engine import RecursiveEngine
from .global_task_manager import GlobalTaskManager
from .tree_trace import TreeTraceNode, to_dict, dict_to_node, clone_node_with_new_parent
from .model import Model, ErrorGuardModel, DeepSeekModel, GPTModel
from .magic import magic_formatter, magic_prompt, abyme_system_prompt

# Deprecated — kept for backwards compatibility
from .core import RecursiveModel, Abyme_API_Model
from .batch_runner import ParallelTreeOrchestrator

__all__ = [
    # Current API
    "RecursiveEngine",
    "GlobalTaskManager",
    "TreeTraceNode",
    "to_dict",
    "dict_to_node",
    "clone_node_with_new_parent",
    "Model",
    "ErrorGuardModel",
    "DeepSeekModel",
    "GPTModel",
    "magic_formatter",
    "magic_prompt",
    "abyme_system_prompt",
    # Deprecated
    "RecursiveModel",
    "Abyme_API_Model",
    "ParallelTreeOrchestrator",
]
