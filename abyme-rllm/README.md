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

## License

MIT License
