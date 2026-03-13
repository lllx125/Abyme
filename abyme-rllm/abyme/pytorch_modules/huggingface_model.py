"""
HuggingFace model implementation for local models with PyTorch dependencies.

This module contains the HuggingFaceModel class which requires PyTorch and transformers.
It is separated from the main model.py to allow lightweight usage without PyTorch.
"""

from typing import Optional, Dict, Any
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from abc import ABC, abstractmethod


class Model(ABC):
    """Abstract base class for language models."""

    @abstractmethod
    def generate(self, prompt: str, max_attempt: int) -> str:
        """
        Generate a response from the model with retry capability.

        Args:
            prompt: The input prompt string
            max_attempt: Maximum number of retry attempts on failure

        Returns:
            The generated response string

        Raises:
            Exception: If all attempts fail
        """
        pass


class HuggingFaceModel(Model):
    """
    HuggingFace model implementation for local models with full customization.

    Supports loading and running local HuggingFace models with complete control over:
    - Model and tokenizer initialization
    - Generation parameters (temperature, top_p, max_tokens, etc.)
    - Device placement (CPU, CUDA, multi-GPU)
    - Quantization and optimization (8-bit, 4-bit, flash attention)
    - Custom chat templates and system prompts

    Example usage:
    ```python
    # Basic usage with default settings
    model = HuggingFaceModel(
        model_name="meta-llama/Llama-2-7b-chat-hf",
        system_prompt="You are a helpful assistant."
    )

    # Advanced usage with custom configuration
    model = HuggingFaceModel(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        device="cuda:0",
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,
        generation_config={
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "max_new_tokens": 2048,
            "do_sample": True,
            "repetition_penalty": 1.1
        },
        model_kwargs={
            "use_flash_attention_2": True,
            "device_map": "auto"
        }
    )
    ```
    """

    def __init__(
        self,
        model_name: str,
        system_prompt: str = "You are a helpful AI assistant.",
        device: Optional[str] = None,
        torch_dtype: Optional[Any] = None,  # torch.dtype when torch is available
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
        generation_config: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        use_chat_template: bool = True,
        chat_template: Optional[str] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None
    ):
        """
        Initialize a HuggingFace model with full customization.

        Args:
            model_name: HuggingFace model identifier (e.g., "meta-llama/Llama-2-7b-chat-hf")
                       or path to local model directory
            system_prompt: System prompt to prepend to all generations
            device: Device to load model on ("cpu", "cuda", "cuda:0", etc.)
                   If None, automatically selects CUDA if available, otherwise CPU
            torch_dtype: Data type for model weights (torch.float16, torch.bfloat16, torch.float32)
                        If None, uses model's default dtype
            load_in_8bit: Whether to load model in 8-bit precision (requires bitsandbytes)
            load_in_4bit: Whether to load model in 4-bit precision (requires bitsandbytes)
            trust_remote_code: Whether to trust remote code in model config (needed for some models)
            generation_config: Dictionary of generation parameters:
                - temperature (float): Sampling temperature (default: 0.7)
                - top_p (float): Nucleus sampling probability (default: 0.9)
                - top_k (int): Top-k sampling (default: 50)
                - max_new_tokens (int): Maximum tokens to generate (default: 2048)
                - do_sample (bool): Whether to use sampling (default: True)
                - repetition_penalty (float): Penalty for repeating tokens (default: 1.0)
                - num_beams (int): Number of beams for beam search (default: 1)
            model_kwargs: Additional keyword arguments passed to AutoModelForCausalLM.from_pretrained():
                - device_map (str/dict): Device mapping strategy ("auto", "balanced", etc.)
                - use_flash_attention_2 (bool): Use Flash Attention 2 if available
                - attn_implementation (str): Attention implementation ("flash_attention_2", "sdpa", "eager")
                - low_cpu_mem_usage (bool): Reduce CPU memory usage during loading
                - max_memory (dict): Maximum memory per device
            tokenizer_kwargs: Additional keyword arguments passed to AutoTokenizer.from_pretrained():
                - padding_side (str): "left" or "right"
                - truncation_side (str): "left" or "right"
                - model_max_length (int): Maximum sequence length
            use_chat_template: Whether to use the model's chat template for formatting
            chat_template: Custom chat template string (Jinja2 format). If None, uses model's default
            model: Optional pre-loaded model. If provided with tokenizer, skips model loading
            tokenizer: Optional pre-loaded tokenizer. If provided with model, skips tokenizer loading
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError(
                "HuggingFace transformers library is required. "
                "Install with: pip install transformers torch"
            )

        self.model_name = model_name
        self.system_prompt = system_prompt
        self.use_chat_template = use_chat_template
        self.custom_chat_template = chat_template

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Use pre-loaded model and tokenizer if provided, otherwise load them
        if model is not None and tokenizer is not None:
            print(f"Using pre-loaded model and tokenizer for {model_name}")
            self.model = model
            self.tokenizer = tokenizer

            # Apply custom chat template if provided
            if self.custom_chat_template:
                self.tokenizer.chat_template = self.custom_chat_template

            # Set model to evaluation mode
            self.model.eval()
        else:
            # Prepare model loading arguments
            model_load_kwargs: Dict[str, Any] = {
                "trust_remote_code": trust_remote_code,
            }

            # Add dtype if specified
            if torch_dtype is not None:
                model_load_kwargs["torch_dtype"] = torch_dtype

            # Add quantization settings
            if load_in_8bit:
                model_load_kwargs["load_in_8bit"] = True
            elif load_in_4bit:
                model_load_kwargs["load_in_4bit"] = True

            # Add custom model kwargs
            if model_kwargs:
                model_load_kwargs.update(model_kwargs)

            # If no device_map specified and not using quantization, set device
            if "device_map" not in model_load_kwargs and not load_in_8bit and not load_in_4bit:
                model_load_kwargs["device_map"] = self.device

            # Load tokenizer
            tokenizer_load_kwargs: Dict[str, Any] = {
                "trust_remote_code": trust_remote_code,
            }
            if tokenizer_kwargs:
                tokenizer_load_kwargs.update(tokenizer_kwargs)

            print(f"Loading tokenizer from {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                **tokenizer_load_kwargs
            )

            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Apply custom chat template if provided
            if self.custom_chat_template:
                self.tokenizer.chat_template = self.custom_chat_template

            # Load model
            print(f"Loading model from {model_name}...")
            print(f"Model loading configuration: {model_load_kwargs}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_load_kwargs
            )

            # Set model to evaluation mode
            self.model.eval()

        # Store generation configuration with defaults
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "max_new_tokens": 2048,
            "do_sample": True,
            "repetition_penalty": 1.0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # Update with user-provided generation config
        if generation_config:
            self.generation_config.update(generation_config)

        print(f"Model loaded successfully on {self.device}")
        print(f"Generation config: {self.generation_config}")

    def generate(self, prompt: str, max_attempt: int = 1) -> str:
        """
        Generate a response using the local HuggingFace model with retry mechanism.

        Args:
            prompt: The user prompt
            max_attempt: Maximum number of retry attempts on failure (default: 1)

        Returns:
            The generated response text

        Raises:
            Exception: If all attempts fail
        """
        import torch

        last_error = None

        for attempt in range(max_attempt):
            try:
                # Format the prompt with chat template if enabled
                if self.use_chat_template and hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    # Fallback to simple concatenation
                    formatted_prompt = f"{self.system_prompt}\n\nUser: {prompt}\n\nAssistant:"

                # Tokenize input
                inputs = self.tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )

                # Move inputs to model's device
                input_ids = inputs["input_ids"].to(self.model.device)
                attention_mask = inputs["attention_mask"].to(self.model.device)

                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **self.generation_config
                    )

                # Decode only the newly generated tokens
                generated_tokens = outputs[0][input_ids.shape[1]:]
                response = self.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )

                return response.strip()

            except Exception as e:
                last_error = e
                if attempt < max_attempt - 1:
                    print(f"Generation attempt {attempt + 1} failed: {e}. Retrying...")
                continue

        raise Exception(f"All {max_attempt} attempts failed. Last error: {last_error}")

    def update_system_prompt(self, new_prompt: str):
        """
        Update the system prompt for future generations.

        Args:
            new_prompt: The new system prompt to use
        """
        self.system_prompt = new_prompt

    def update_generation_config(self, **kwargs):
        """
        Update generation configuration parameters.

        Args:
            **kwargs: Generation parameters to update (temperature, top_p, max_new_tokens, etc.)

        Example:
            model.update_generation_config(temperature=0.8, max_new_tokens=1024)
        """
        self.generation_config.update(kwargs)
        print(f"Updated generation config: {self.generation_config}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dtype": str(self.model.dtype),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "num_trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "generation_config": self.generation_config,
            "system_prompt": self.system_prompt,
        }
