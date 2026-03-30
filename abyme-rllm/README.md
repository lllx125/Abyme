# Abyme - Recursive Language Model

Recursive Language Model with XML-based elaboration system for advanced reasoning.

## Overview

Abyme is a recursive model architecture that enables language models to decompose complex problems into subproblems using XML-based delegation tags. It supports both AND (parallel) and OR (branching) decomposition strategies, with automatic backtracking and failure handling.

## Installation

### Basic Installation (API models only)
For using OpenAI/DeepSeek API models without PyTorch:
```bash
pip install -e .
```

### With PyTorch Support (for local HuggingFace models)
```bash
pip install -e ".[pytorch]"
```

### With SFT Training Support
For supervised fine-tuning with all training dependencies:
```bash
pip install -e ".[sft]"
```

### Development Installation
For development with testing and linting tools:
```bash
pip install -e ".[dev]"
```

### Full Installation
Install everything:
```bash
pip install -e ".[all]"
```

## Quick Start

```python
from abyme.core import Abyme_API_Models

# Create a model instance
model = Abyme_API_Models(
    model="deepseek",  # or "gpt", "deepseek-r"
    max_depth=5,
    max_call=3000,
    max_parallel_workers=10
)

# Generate a response
result = model.generate("Your complex problem here")
print(result)
```

## Features

- **Recursive Decomposition**: Break complex problems into manageable subproblems
- **AND/OR Nodes**: Parallel execution (AND) and branching strategies (OR)
- **Tree Tracing**: Complete tree structure tracking with state history
- **Thread-Safe Execution**: Multi-threaded worker pool with greedy DFS scheduling
- **Model Support**: OpenAI GPT, DeepSeek, and HuggingFace transformers
- **Visualization**: Real-time tree visualization with the Abyme Visualizer

## Components

- **tree_trace.py**: Tree structure and state tracking
- **tree_manager.py**: Thread-safe execution orchestration
- **core.py**: Main RecursiveModel implementation
- **model.py**: Base model interfaces
- **utils.py**: XML parsing and formatting utilities
- **magic.py**: Prompt templates and formatters

## Tree Visualizer

Run the interactive web-based tree visualizer:

```bash
cd ..
./visualize.sh
```

Then open http://localhost:5000 in your browser.

## Cancellation & Abort

### Why it exists

When an OR-node tree completes (one child succeeds and siblings are cancelled), worker
threads that are already mid-LLM-call for those cancelled siblings would previously
block until the full response arrived before exiting.  With 64 workers and a 9 B model
this caused a long stall between the last `✅ tree_N` print and `process_batch` returning.

The fix adds a `CancelToken` that is attached to each node while it is being generated.
`cancel_tree()` now fires the token, which calls `engine.abort(request_id)` on the vLLM
side to kill the request mid-stream.  Workers unblock immediately.

### Per-node cancellation (automatic)

No user action required.  Whenever `cancel_tree()` is called (OR kill-switch, max_call
enforcement, etc.) the in-progress vLLM request for that node is aborted automatically.

### External abort (manual)

Call `engine.abort()` from any thread to stop an entire batch immediately:

```python
import threading
from abyme import RecursiveEngine
from abyme.vllm_model import LocalVLLMModel

model = LocalVLLMModel("path/to/model")
engine = RecursiveEngine(base_model=model, max_workers=64)

# Start a batch in a background thread
t = threading.Thread(target=engine.process_batch, args=(prompts, "out.jsonl"))
t.start()

# ...later, abort everything:
engine.abort()
t.join()

# If you want to reuse the engine after aborting:
engine.reset()
```

`abort()` does three things:
1. Sets an internal shutdown event — workers exit at the next loop boundary.
2. Fires every active `CancelToken` — any mid-stream vLLM request is aborted via
   `engine.abort(request_id)`.
3. Is idempotent and thread-safe.

### Using CancelToken directly (advanced)

`CancelToken` is a plain object you can create and pass to `LocalVLLMModel.generate`
if you need fine-grained control outside the engine:

```python
from abyme import CancelToken
from abyme.vllm_model import LocalVLLMModel

model = LocalVLLMModel("path/to/model")
token = CancelToken()

# generate in a thread, cancel from the main thread
import threading
result = []
t = threading.Thread(target=lambda: result.append(
    model.generate("some prompt", cancel_token=token)
))
t.start()

token.cancel()   # aborts the vLLM request immediately
t.join()
```

## License

MIT License
