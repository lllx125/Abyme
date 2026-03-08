"""
Abyme: A Recursive Language Model with XML-based elaboration system.

This package implements a recursive reasoning system where language models can:
- Plan their responses using <think> tags
- Recursively elaborate on sub-problems using <elaborate> tags
- Receive responses in <response> tags
- Control execution flow with </run> tokens

Example:
    >>> from abyme import RecursiveEngine
    >>>
    >>> # User can pass any HF arguments (quantization, device_map, etc.)
    >>> # Default settings are safe (CPU/Auto), not hardcoded to specific GPU.
    >>> engine = RecursiveEngine(
    ...     model_name="Abyme/Abyme-V1",
    ...     device_map="auto",
    ...     load_in_4bit=True,
    ...     trust_remote_code=True
    ... )
    >>>
    >>> # The engine handles the recursive loop hidden from the user
    >>> output = engine.generate("Solve this math problem...", max_depth=3)
"""

try:
    from .engine import RLLMEngine
    from .tokenization import (
        setup_model_and_tokenizer,
        inject_special_tokens,
        get_stopping_token_id
    )
    _engine_available = True
except ImportError:
    _engine_available = False

# Always available: verifiers module (no external dependencies)
from . import verifiers

# Data module (requires datasets, pandas, scipy)
try:
    from . import data
    _data_available = True
except ImportError:
    _data_available = False

__version__ = "0.1.0"

__all__ = [
    "verifiers",
]

# Conditionally add data components
if _data_available:
    __all__.append("data")

# Conditionally add engine components
if _engine_available:
    __all__.extend([
        "RLLMEngine",
        "setup_model_and_tokenizer",
        "inject_special_tokens",
        "get_stopping_token_id",
    ])
