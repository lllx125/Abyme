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

    # Abort all in-progress generations from another thread
    engine.abort()
"""

from abyme.recursive_engine import RecursiveEngine
from abyme.tree_trace import CancelToken
