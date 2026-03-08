"""
Tokenizer setup with smart token initialization for recursive LLM.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def inject_special_tokens(model, tokenizer):
    """
    Safely adds special tokens to the model/tokenizer if they are missing.
    """
    # Define special tokens for recursive reasoning
    special_tokens = [
        "<elaborate>",   # Start of a subtask to be recursively processed
        "</elaborate>",  # End of subtask definition
        "<response>",    # Start of response from recursive call
        "</response>",   # End of response
        "</run>",        # Signal to stop and process elaborations
    ]

    # Check if they already exist in the vocabulary
    existing_tokens = set(tokenizer.get_vocab().keys())
    tokens_to_add = [t for t in special_tokens if t not in existing_tokens]

    if not tokens_to_add:
        return model, tokenizer

    tokenizer.add_special_tokens({
        'additional_special_tokens': tokens_to_add
    })

    model.resize_token_embeddings(len(tokenizer))

    # Smart initialization: Copy weights from semantic proxies
    _initialize_special_tokens(model, tokenizer, tokens_to_add)

    return model, tokenizer


def setup_model_and_tokenizer(model_name, **kwargs):
    """
    Load model and tokenizer with flexible HuggingFace arguments.
    """
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=kwargs.get('trust_remote_code', True)
    )

    # Extract quantization-related kwargs
    load_in_4bit = kwargs.pop('load_in_4bit', False)
    load_in_8bit = kwargs.pop('load_in_8bit', False)

    # Configure quantization if requested
    if load_in_4bit:
        print("Configuring 4-bit quantization...")
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif load_in_8bit:
        print("Configuring 8-bit quantization...")
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_8bit=True
        )

    print(f"Loading model from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    # Automatically inject special tokens if missing
    model, tokenizer = inject_special_tokens(model, tokenizer)

    # Set EOS/Pad tokens if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")

    print("Model and tokenizer setup complete!")
    return model, tokenizer


def _initialize_special_tokens(model, tokenizer, tokens_to_initialize):
    """
    Initialize new token embeddings by copying from semantically similar tokens.
    """
    # Get the embedding layer (Returns nn.Module, but we treat it dynamically)
    embeddings = model.get_input_embeddings()

    # Define semantic proxies
    # Note: </run> uses "branch"/"recurse" semantics (continue elsewhere)
    # rather than "stop" semantics (which conflicts with EOS)
    token_proxies = {
        "<elaborate>": ["plan", "step", "task"],
        "</elaborate>": ["end", "done", "complete"],
        "<response>": ["answer", "result", "output"],
        "</response>": ["end", "done", "complete"],
        "</run>": ["branch", "delegate", "recurse", "checkpoint"],
    }

    if tokens_to_initialize is None:
        tokens_to_initialize = list(token_proxies.keys())

    for special_token in tokens_to_initialize:
        if special_token not in token_proxies:
            continue

        # Check vocab using standard dict method to avoid any 'callable' confusion
        vocab = tokenizer.get_vocab()
        if special_token not in vocab:
            continue

        special_token_id = vocab[special_token]
        proxy_words = token_proxies[special_token]

        proxy_id = None
        for proxy_word in proxy_words:
            proxy_tokens = tokenizer.encode(proxy_word, add_special_tokens=False)
            if len(proxy_tokens) > 0:
                proxy_id = proxy_tokens[0]
                print(f"  {special_token} <- '{proxy_word}' (ID: {proxy_id})")
                break

        if proxy_id is not None:
            # Copy the embedding weights without tracking gradients
            with torch.no_grad():
                # By removing types, the linter won't complain about .weight or __setitem__
                embeddings.weight[special_token_id] = embeddings.weight[proxy_id].clone()
        else:
            print(f"  Warning: No proxy found for {special_token}, using random initialization")


def get_stopping_token_id(tokenizer):
    """
    Get the token ID for the </run> stopping token.
    """
    return tokenizer.convert_tokens_to_ids("</run>")