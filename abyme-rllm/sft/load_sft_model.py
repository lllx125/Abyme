
from abyme.magic import abyme_system_prompt

def AbymeSFTHuggingFaceModel(model_name: str = "Lixing-Li/Abyme-Llama-3.1-8B-SFT", **kwargs):
    """
    Factory function to create a HuggingFaceModel with Abyme special tokens.

    This loads a fully merged model with custom tokenizer that includes special tokens for
    recursive reasoning: <elaborate>, </elaborate>, <response>, </response>, </run>.

    The default model (Lixing-Li/Abyme-Llama-3.1-8B-SFT) is a fully merged fine-tuned model
    based on Meta-Llama-3.1-8B-Instruct with Abyme-specific training.

    Default configuration is optimized for NVIDIA RTX 4080 (12GB VRAM):
    - BFloat16 compute dtype
    - 4-bit quantization enabled
    - SDPA (Scaled Dot Product Attention - PyTorch 2.0+ native)
    - Auto device mapping

    Args:
        model_name: HuggingFace model identifier
                   (default: "Lixing-Li/Abyme-Llama-3.1-8B-SFT")
        **kwargs: All arguments supported by HuggingFaceModel:
            - system_prompt: System prompt for the model
            - device: Device to load on ("cuda", "cpu", etc.)
            - dtype: Data type for weights (default: torch.bfloat16)
            - load_in_4bit: Use 4-bit quantization (default: True)
            - load_in_8bit: Use 8-bit quantization (default: False)
            - trust_remote_code: Whether to trust remote code (default: True)
            - generation_config: Dict of generation parameters
            - model_kwargs: Additional model loading arguments
            - use_chat_template: Whether to use chat template
            - chat_template: Custom chat template string

    Returns:
        HuggingFaceModel instance with special tokens injected

    Example:
        >>> # Use with defaults (optimized for 12GB GPU)
        >>> model = AbymeSFTHuggingFaceModel()
        >>>
        >>> # Custom configuration
        >>> model = AbymeSFTHuggingFaceModel(
        ...     load_in_8bit=True,  # Use 8-bit instead of 4-bit
        ...     generation_config={"temperature": 0.8}
        ... )
    """
    # Lazy import torch/transformers only when HuggingFace model is used
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from abyme.model import HuggingFaceModel

    # Set defaults optimized for RTX 4080 12GB
    defaults = {
        'load_in_4bit': True,
        'trust_remote_code': True,
        'model_kwargs': {
            'device_map': 'auto',
            'attn_implementation': 'sdpa',  # Scaled Dot Product Attention (PyTorch 2.0+)
            'low_cpu_mem_usage': True,
        }
    }

    # Merge user kwargs with defaults (user kwargs take precedence)
    model_kwargs = defaults['model_kwargs'].copy()
    if 'model_kwargs' in kwargs:
        model_kwargs.update(kwargs.pop('model_kwargs'))

    for key, value in defaults.items():
        if key not in kwargs and key != 'model_kwargs':
            kwargs[key] = value

    kwargs['model_kwargs'] = model_kwargs

    # Extract parameters for later use
    system_prompt = kwargs.pop('system_prompt', None)
    generation_config = kwargs.pop('generation_config', None)
    tokenizer_kwargs = kwargs.pop('tokenizer_kwargs', None)
    use_chat_template = kwargs.pop('use_chat_template', None)
    chat_template = kwargs.pop('chat_template', None)
    device = kwargs.pop('device', None)

    load_in_4bit = kwargs.pop('load_in_4bit', False)
    load_in_8bit = kwargs.pop('load_in_8bit', False)
    trust_remote_code = kwargs.pop('trust_remote_code', True)
    dtype = kwargs.pop('dtype', None)
    torch_dtype = kwargs.pop('torch_dtype', None)  # Legacy support
    if torch_dtype is not None and dtype is None:
        dtype = torch_dtype
    additional_model_kwargs = kwargs.pop('model_kwargs', {})

    # Configure quantization
    quantization_config = None
    if load_in_4bit:
        print("Configuring 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype or torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif load_in_8bit:
        print("Configuring 8-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )

    # Build model loading kwargs
    model_load_kwargs = {
        'trust_remote_code': trust_remote_code,
        **additional_model_kwargs
    }

    if dtype is not None:
        model_load_kwargs['torch_dtype'] = dtype

    if quantization_config is not None:
        model_load_kwargs['quantization_config'] = quantization_config

    # Initialize variables for cleanup
    tokenizer = None
    model = None

    try:
        # Load merged model directly
        print(f"Loading merged model from {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_load_kwargs
        )

        # Load tokenizer
        print(f"Loading tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )

        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Set pad_token to eos_token")

        # Rebuild kwargs for HuggingFaceModel
        hf_kwargs = {}
        if system_prompt is not None:
            hf_kwargs['system_prompt'] = system_prompt
        if generation_config is not None:
            hf_kwargs['generation_config'] = generation_config
        if tokenizer_kwargs is not None:
            hf_kwargs['tokenizer_kwargs'] = tokenizer_kwargs
        if use_chat_template is not None:
            hf_kwargs['use_chat_template'] = use_chat_template
        if chat_template is not None:
            hf_kwargs['chat_template'] = chat_template
        if device is not None:
            hf_kwargs['device'] = device

        # Create HuggingFaceModel instance
        hf_model = HuggingFaceModel(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            system_prompt=abyme_system_prompt,
            **hf_kwargs
        )

        return hf_model

    except Exception as e:
        # Clean up on failure to prevent memory leaks
        print(f"Error loading model: {e}")
        print("Cleaning up memory...")

        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer

        # Clear CUDA cache to free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared")

        raise RuntimeError(f"Failed to load model '{model_name}': {e}") from e

    finally:
        # Clean up intermediate configuration objects
        if 'quantization_config' in locals() and quantization_config is not None:
            del quantization_config