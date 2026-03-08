# Abyme - Recursive Reasoning LLM Framework

A 28-day project to build and train a recursive reasoning LLM based on DeepSeek-R1-Distill-Llama-8B.

## Overview

Abyme is a framework that enables Large Language Models to perform recursive reasoning through special XML tags. Unlike traditional recursive LLMs that require the model to learn Python code, Abyme uses a natural language control flow with XML tags.

## Key Features

- **Recursive Reasoning**: Models can break down complex problems into sub-problems
- **Parallel Execution**: Multiple elaborations execute in parallel after `</run>` token
- **XML-based Control Flow**: Natural language tags for recursion control
- **Turing Complete**: Supports conditions, loops, and arbitrary recursion depth

## Special Tags

- `<think></think>`: Internal reasoning (not visible to user or parent calls)
- `<elaborate></elaborate>`: Recursive call to solve sub-problems
- `<response></response>`: Results from recursive calls (auto-inserted)
- `</run>`: Execute pending elaborations in parallel

## Example

```
Input: Write code to process a dataset

<think>
We break this into A, B, C
<elaborate>
Write code for data loading
</elaborate>
<elaborate>
Write code for preprocessing
</elaborate>
<elaborate>
Write code for analysis
</elaborate>
</run>
<response>
[Data loading code]
</response>
<response>
[Preprocessing code]
</response>
<response>
[Analysis code]
</response>
</run>
</think>
Here is the complete code: ...
```

## Project Status

**Day 1 Complete** ✅
- Controller with XML parser and recursion stack
- Tokenizer setup with special tags
- Test framework for model validation

See [PLAN.md](PLAN.md) for the complete 28-day roadmap.

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (RTX 4080 12GB or better recommended)
- PyTorch with CUDA support

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/abyme.git
cd abyme
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Or use existing environment:
```bash
# Activate your pytorch-env
source ~/venvs/pytorch-env/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

### Optional: Install Unsloth for faster inference
```bash
pip install unsloth
```

## Usage

### Quick Start

```python
from abyme.controller import Controller
from abyme.tokenizer_setup import AbymeTokenizer

# Initialize
controller = Controller(max_depth=10)
abyme_tokenizer = AbymeTokenizer()
tokenizer = abyme_tokenizer.load_tokenizer()

# Parse model output
output = controller.parse_output(model_output_text)
if output.is_complete:
    final = controller.extract_final_output(output.text)
```

### Run Day 1 Tests

```bash
cd abyme
python day1_test.py
```

## Project Structure

```
Abyme/
├── abyme/                    # Main package
│   ├── __init__.py          # Package initialization
│   ├── controller.py        # XML parser & recursion controller
│   ├── tokenizer_setup.py   # Special token setup
│   ├── day1_test.py         # Day 1 test suite
│   └── README.md            # Package documentation
├── PLAN.md                   # 28-day project plan
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
└── pyproject.toml           # Modern Python project config
```

## Development Roadmap

- **Phase 1 (Days 1-5)**: Local Lab - Infrastructure & Verification
  - ✅ Day 1: Controller & Tokenizer
  - ⏳ Day 2: Math Verifier
  - ⏳ Day 3: SFT Dataset Generator
  - ⏳ Day 4: Dry Run Training

- **Phase 2 (Days 6-7)**: Data Generation
- **Phase 3 (Days 8-9)**: Supervised Fine-Tuning
- **Phase 4 (Days 10-14)**: GRPO Training
- **Phase 5 (Days 15-28)**: Evaluation & Paper

See [PLAN.md](PLAN.md) for detailed breakdown.

## Requirements

Core dependencies:
- PyTorch 2.9+
- Transformers 4.57+
- BitsAndBytes 0.49+ (4-bit quantization)
- NumPy, Pandas, Matplotlib

See [requirements.txt](requirements.txt) for complete list.

## Contributing

This is a research project following a structured 28-day plan. Contributions are welcome after the initial implementation phase.

## License

MIT License (to be added)

## Acknowledgments

- Inspired by recursive LLM research (rLLM)
- Based on DeepSeek-R1-Distill-Llama-8B
- Uses techniques from RLM and ReCAP papers
