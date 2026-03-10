"""
Abyme: A Recursive Language Model with XML-based elaboration system.

This package implements a recursive reasoning system where language models can:
- Plan their responses using <think> tags
- Recursively elaborate on sub-problems using <elaborate> tags
- Receive responses in <response> tags
- Control execution flow with </run> tokens

Example:
    >>> from abyme import RLLMEngine, create_recursive_model
    >>>
    >>> # High-level engine interface (recommended)
    >>> # User can pass any HF arguments (quantization, device_map, etc.)
    >>> engine = RLLMEngine(
    ...     model_name="meta-llama/Llama-2-7b-chat-hf",
    ...     device_map="auto",
    ...     load_in_4bit=True,
    ...     max_recursion_depth=3
    ... )
    >>> output = engine.generate("Solve this problem...", max_attempt=3)
    >>>
    >>> # Or use factory functions for more control
    >>> model = create_recursive_model(
    ...     "meta-llama/Llama-2-7b-chat-hf",
    ...     quantization="4bit",
    ...     max_depth=3,
    ...     max_call=20
    ... )
    >>> result = model.generate("What is 5*5 + 3*3?", max_attempt=3)
"""