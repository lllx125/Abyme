"""
Recursive engine for processing elaborate/response cycles using RecursiveModel pattern.
"""
from typing import List, Dict, Optional, Any
from .model import RecursiveModel, Model, HuggingFaceModel, DeepSeekModel, GPTModel, ErrorGuardModel
from .magic import magic_prompt, magic_formatter, magic_guard_prompt

# Lazy imports - only load torch/transformers when needed for HuggingFace models
# This allows lightweight usage with DeepSeek/GPT without installing heavy dependencies


def AbymeHuggingFaceModel(model_name: str = "Abyme", **kwargs) -> HuggingFaceModel:
    """
    Factory function to create a HuggingFaceModel with Abyme special tokens.

    This loads a model with custom tokenizer that includes special tokens for
    recursive reasoning: <elaborate>, </elaborate>, <response>, </response>, </run>.

    Args:
        model_name: HuggingFace model identifier or local path (default: "Abyme")
        **kwargs: All arguments supported by HuggingFaceModel:
            - system_prompt: System prompt for the model
            - device: Device to load on ("cuda", "cpu", etc.)
            - torch_dtype: Data type for weights
            - load_in_8bit/load_in_4bit: Quantization options
            - trust_remote_code: Whether to trust remote code
            - generation_config: Dict of generation parameters
            - model_kwargs: Additional model loading arguments
            - use_chat_template: Whether to use chat template
            - chat_template: Custom chat template string

    Returns:
        HuggingFaceModel instance with special tokens injected

    Example:
        >>> model = AbymeHuggingFaceModel(
        ...     model_name="meta-llama/Llama-2-7b-chat-hf",
        ...     load_in_4bit=True,
        ...     system_prompt="You are a helpful assistant."
        ... )
    """
    # Lazy import torch/transformers only when HuggingFace model is used
    import torch
    from transformers import PreTrainedTokenizerBase, AutoModelForCausalLM, BitsAndBytesConfig
    from .tokenization import setup_model_and_tokenizer, get_stopping_token_id, inject_special_tokens

    # Extract model_kwargs and merge with remaining kwargs
    model_kwargs = kwargs.pop('model_kwargs', {})

    # Merge all kwargs for setup_model_and_tokenizer
    setup_kwargs = {**kwargs, **model_kwargs}

    # Remove parameters that aren't for model loading
    setup_kwargs.pop('system_prompt', None)
    setup_kwargs.pop('generation_config', None)
    setup_kwargs.pop('tokenizer_kwargs', None)
    setup_kwargs.pop('use_chat_template', None)
    setup_kwargs.pop('chat_template', None)
    setup_kwargs.pop('device', None)

    # Load model with custom tokenizer (includes special token injection)
    model, tokenizer = setup_model_and_tokenizer(model_name, **setup_kwargs)

    # Create and return HuggingFaceModel instance with pre-loaded model/tokenizer
    return HuggingFaceModel(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        **kwargs
    )

def Abyme_DeepSeek(reasoning: bool = False, max_depth: int = 20, max_call: int = 50, max_parallel_workers: int = 1, print_progress: bool = False) -> RecursiveModel:
    base_model = DeepSeekModel(reasoning=reasoning, system_prompt=magic_prompt)
    guard_model = ErrorGuardModel()
    guard_model = DeepSeekModel(reasoning=reasoning, system_prompt=magic_guard_prompt)
    return RecursiveModel(
        base_model=base_model,
        guard_model=guard_model,
        formatter=magic_formatter,
        max_depth=max_depth,
        max_call=max_call,
        max_parallel_workers=max_parallel_workers,
        print_progress = print_progress
    )


def Abyme_GPT(max_depth: int = 20, max_call: int = 50, max_parallel_workers: int = 1, print_progress: bool = False) -> RecursiveModel:
    base_model = GPTModel(system_prompt=magic_prompt)
    return RecursiveModel(
        base_model=base_model,
        formatter=magic_formatter,
        max_depth=max_depth,
        max_call=max_call,
        max_parallel_workers=max_parallel_workers,
        print_progress = print_progress
    )