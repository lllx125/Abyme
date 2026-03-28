# abyme

Core package for the Abyme recursive reasoning engine.

A model solves problems by recursively delegating subproblems to sub-agents using XML tags. Sub-agents run in parallel; results are injected back and generation continues.

---

## Recursion Mechanism

Every generation produces exactly one of three outputs:

**AND node** — break into independent parallel subproblems (all must succeed):
```
## DO 1
> Differentiate e^x
<do>
Differentiate e^x with respect to x.
</do>
## DO 2
> Differentiate sin(x)
<do>
Differentiate sin(x) with respect to x.
</do>
```

**OR node** — try multiple approaches in parallel (first success wins):
```
## TRY 1
> Try induction
<try>
Determine if induction proves the square partition problem.
</try>
## TRY 2
> Try Sperner's Lemma
<try>
Determine if Sperner's Lemma proves the square partition problem.
</try>
```

**Base case** — answer directly:
```
we can directly differentiate e^x = e^x
</think>
e^x
```

When subproblems finish, their answers (the text after `</think>`) are injected as `<response>` tags and the parent continues generating.

Each child agent sees its full task hierarchy:
```
Main Task:  Can a square be partitioned into an odd number of equal-area triangles?
Boss Task:  Try mathematical induction.
Your Task:  Find the base case.
```

---

## Files

| File | Purpose |
|------|---------|
| `recursive_engine.py` | Main API — `RecursiveEngine` |
| `global_task_manager.py` | Shared worker pool coordinator across all trees |
| `tree_manager.py` | Per-tree state machine (DFS scheduling, resolution logic) |
| `tree_trace.py` | Node data structure + tree utility functions |
| `model.py` | `Model` ABC + `DeepSeekModel`, `GPTModel`, `HuggingFaceModel` |
| `vllm_model.py` | `LocalVLLMModel` (vLLM async engine) + `APIModel` (OpenAI-compatible) |
| `magic.py` | System prompt + `magic_formatter` |
| `utils.py` | Tag extraction, format validation |
| `core.py` | *Deprecated* — `RecursiveModel` (use `RecursiveEngine`) |
| `batch_runner.py` | *Deprecated* — `ParallelTreeOrchestrator` (use `RecursiveEngine.process_batch`) |

---

## Quick Start

### Single prompt

```python
from abyme import RecursiveEngine
from abyme.vllm_model import LocalVLLMModel

model = LocalVLLMModel("Lixing-Li/Abyme-Qwen3.5-9B-SFT")
engine = RecursiveEngine(base_model=model, max_workers=10)

answer = engine.generate("Differentiate g(x) = (e^x sin(x)) / 2")
print(answer)
```

With an API model:

```python
from abyme import RecursiveEngine
from abyme.vllm_model import APIModel

model = APIModel(
    model_name="gpt-4o",
    api_key="sk-...",
    base_url=None  # omit for OpenAI, set for vLLM server or DeepSeek
)
engine = RecursiveEngine(base_model=model, max_workers=20)
answer = engine.generate("Can a square be partitioned into an odd number of equal-area triangles?")
```

### Batch processing

Workers are shared across all trees — a tree with few pending tasks never blocks workers from picking up tasks from other trees.

```python
engine = RecursiveEngine(
    base_model=model,
    max_workers=60,     # shared pool for ALL trees
    max_depth=5,
    max_call=1000,
    max_chain_length=5,
)

prompts = ["Problem 1", "Problem 2", ..., "Problem 500"]
results = engine.process_batch(prompts, output_jsonl_path="results/run1.jsonl")
# Results stream to disk as each tree finishes
```

### Restart from a node (`continue_from_node`)

Reload a previous trace from JSONL and restart generation from any node — useful for retrying a failed branch or exploring an alternative approach.  The entire ancestor chain is cloned so the regenerated subtree bubbles its result up through its parents until the root reaches FINAL.  The original trace is never modified.

```python
import json
from abyme.tree_trace import dict_to_node

# Load a previous trace
with open("results/run1.jsonl") as f:
    record = json.loads(f.readline())

root = dict_to_node(record["trace_tree"])

# Pick any node to restart from (e.g. a failed subproblem)
failed_node = root.subproblems[2]

# Single node — streams result to JSONL, returns list of result dicts
results = engine.continue_from_node(failed_node, output_jsonl_path="results/retry.jsonl")
print(results[0]["output"])
```

Batch restart with shared workers:

```python
# Collect failed nodes across many traces
nodes_to_retry = [root.subproblems[i] for root in loaded_roots if ...]

results = engine.continue_from_node(
    nodes_to_retry,
    output_jsonl_path="results/retry.jsonl",
    group_size=10,   # process 10 trees in parallel per batch
    max_attempt=2,
)
```

`continue_from_node` preserves each node's `prompt`, `fragment`, and `past` chain (temporal context) and regenerates output and subproblems from scratch.  Completed sibling subtrees are cloned intact so their outputs are injected when the parent continues generating.

---

## API Reference

### `RecursiveEngine`

```python
RecursiveEngine(
    base_model: Model,
    guard_model: Model = None,   # fallback when depth/chain limit hit; defaults to ErrorGuardModel
    max_workers: int = 60,       # shared thread pool size
    max_depth: int = 20,         # max recursion depth before switching to guard_model
    max_call: int = 1000,        # max total LLM calls per run
    max_subproblem_retry: int = 2,
    max_chain_length: int = 5,   # max temporal continuation steps per chain
    proceed_when_fail: bool = True,
    formatter: Callable = magic_formatter,
    print_progress: bool = False,
)
```

| Method | Description |
|--------|-------------|
| `generate(prompt, max_attempt=1) -> str` | Single prompt. Returns final answer. |
| `process_batch(prompts, output_jsonl_path, max_attempt=1) -> List[dict]` | Batch. Streams JSONL. Returns list of result dicts. |
| `continue_from_node(source_nodes, output_jsonl_path, max_attempt=1, group_size=1) -> List[dict]` | Restart from one or more nodes in existing traces. Streams JSONL. |

### Result dict schema — `process_batch` (JSONL)

```json
{
  "index": 0,
  "prompt": "...",
  "status": "SUCCESS" | "FAILED",
  "output": "...",
  "error": null,
  "metrics": {
    "total_llm_calls": 7,
    "max_tree_depth": 3,
    "max_subproblems": 2,
    "max_output_chars": 1420,
    "theoretical_parallel_latency": 4.2,
    "length": 3800
  },
  "trace_tree": { ... }
}
```

### Result dict schema — `continue_from_node` (JSONL)

```json
{
  "index": 0,
  "source_node": { ... },
  "status": "SUCCESS" | "FAILED",
  "output": "...",
  "error": null,
  "metrics": {
    "total_llm_calls": 5,
    "max_tree_depth": 3,
    "max_subproblems": 2,
    "max_output_chars": 980,
    "theoretical_parallel_latency": 3.1,
    "length": 2400
  },
  "trace_tree": { ... }
}
```

`source_node` is a serialized snapshot of the original node before regeneration.
`trace_tree` is the completed cloned tree rooted at the top-level ancestor.

### `TreeTraceNode`

Key fields on each node:

| Field | Type | Description |
|-------|------|-------------|
| `prompt` | str | This node's task |
| `main_problem` | str | Root problem (global context) |
| `parent_problem` | str | Parent's task |
| `fragment` | str | Accumulated responses injected so far |
| `output` | str | Raw model output for this generation step |
| `past` | List[TreeTraceNode] | Previous temporal states (continuation chain) |
| `subproblems` | List[TreeTraceNode] | Child nodes spawned by `<do>`/`<try>` |
| `type` | "do" \| "try" \| "leaf" | AND / OR / base case |
| `status` | NodeStatus | WAIT_GEN → GENERATING → WAIT_SUB → FINAL/FAILED |
| `depth` | int | Depth in the recursion tree |
| `latency` | float | Wall-clock seconds for this generation |

### Tree utility functions

```python
from abyme.tree_trace import (
    to_dict,                      # TreeTraceNode -> JSON-serializable dict
    dict_to_node,                 # dict -> TreeTraceNode (reconstruct from JSONL)
    deep_clone_subtree,           # full deep clone preserving all state
    clone_trace_for_continuation, # clone trace, reset one node for regeneration
    total_calls,                  # total LLM calls in tree
    max_depth,                    # maximum recursion depth reached
    parallel_latency,             # minimum latency assuming infinite workers
    length,                       # total output character length
    fold,                         # generic catamorphism for custom traversal
)
```

### Model classes

```python
from abyme.vllm_model import LocalVLLMModel, APIModel
from abyme.model import DeepSeekModel, GPTModel, HuggingFaceModel

# Local vLLM (A100 optimized, async batching)
LocalVLLMModel(model_path, tensor_parallel_size=1, max_model_len=8192, ...)

# OpenAI-compatible API (OpenAI, DeepSeek, vLLM server)
APIModel(model_name, api_key, base_url=None, temperature=0.7, max_tokens=2048, ...)

# DeepSeek API
DeepSeekModel(reasoning=False, temperature=1.0, ...)

# HuggingFace local
HuggingFaceModel(model_name, load_in_4bit=True, ...)
```

---

## Parallelism Design

**Old design** — fixed workers per tree, idle workers cannot help other trees:
```
Tree A: [worker][worker][worker]   (3 tasks pending — all busy)
Tree B: [worker][worker][worker]   (1 task pending — 2 idle)
```

**New design** — single shared pool, workers dynamically serve any tree:
```
Shared pool: [w][w][w][w][w][w]
             pulls from global priority queue
             Tree A tasks | Tree B tasks | Tree C tasks ...
```

The `GlobalTaskManager` uses a single shared `RLock` across all `TreeManager` instances. `get_next_task()` atomically selects the easiest pending node across all trees (Greedy DFS by difficulty) and marks it `GENERATING` in one critical section — no deadlock risk.

---

## Deprecated

These classes still work but emit `DeprecationWarning`:

```python
# Old                              # New equivalent
RecursiveModel(base_model, max_parallel_workers=5, ...)
# -> RecursiveEngine(base_model, max_workers=5, ...)

ParallelTreeOrchestrator(base_model, output_jsonl_path, max_concurrent_trees=10, max_parallel_workers=5, ...)
# -> RecursiveEngine(base_model, max_workers=50, ...) + engine.process_batch(prompts, output_jsonl_path)
```
