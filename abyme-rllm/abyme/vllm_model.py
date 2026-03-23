"""
vllm_models.py

Contains inference model wrappers for the Agent 4.0 Architecture.
Includes standard API models and a high-throughput local vLLM model
optimized for A100 GPU continuous batching via a Thread-to-Async bridge.
"""

import time
import uuid
import threading
import asyncio
import os
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from abyme.model import Model
from abyme.magic import abyme_system_prompt


os.environ["VLLM_TORCH_COMPILE_LEVEL"] = "0"
os.environ["VLLM_USE_V1"] = "0"


# ==========================================
# 1. API MODEL (For OpenAI, DeepSeek, etc.)
# ==========================================
class APIModel(Model):
    """
    A thread-safe wrapper for API-based models (OpenAI, DeepSeek API, vLLM Server).
    Because the `openai` client relies on HTTP requests, it natively releases 
    the Python GIL, making it perfectly safe for ThreadPoolExecutor concurrency.
    """
    
    def __init__(self, 
                 model_name: str, 
                 api_key: str, 
                 base_url: Optional[str] = None, 
                 system_prompt: str = abyme_system_prompt,
                 temperature: float = 0.7,
                 max_tokens: int = 2048):
        """
        Initialize the API model.
        Requires the `openai` python package installed (`pip install openai`).
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("The 'openai' package is required for APIModel. Run `pip install openai`.")
            
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize the client (can point to OpenAI or any OpenAI-compatible endpoint)
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, max_attempt: int = 1) -> str:
        last_error = None
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        for attempt in range(max_attempt):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                # Extract and return the text
                return response.choices[0].message.content
                
            except Exception as e:
                last_error = e
                if attempt < max_attempt - 1:
                    time.sleep(1.0 * (attempt + 1))  # Exponential backoff on rate limits
                continue
                
        raise Exception(f"APIModel failed after {max_attempt} attempts. Last error: {last_error}")

# ==========================================
# 2. LOCAL vLLM MODEL (A100 Optimized)
# ==========================================
class LocalVLLMModel(Model):
    def __init__(self, 
                 model_path: str, 
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.9,
                 max_model_len: int = 8192,
                 temperature: float = 0.7,
                 max_tokens: int = 2048,
                 system_prompt: str = abyme_system_prompt):
        
        try:
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("Run `pip install vllm transformers`.")

        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 1. Load the tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # ---------------------------------------------------------
        # FIX: DYNAMICALLY FIND VALID STOP TOKENS FOR ANY MODEL
        # ---------------------------------------------------------
        self.stop_token_ids = []
        if self.tokenizer.eos_token_id is not None:
            self.stop_token_ids.append(self.tokenizer.eos_token_id)
            
        # Common chat termination tokens across different architectures (Llama, Qwen, ChatML, Mistral)
        potential_stop_strings = ["<|eot_id|>", "<|im_end|>", "<|end_of_turn|>", "<|endoftext|>"]
        
        for token_str in potential_stop_strings:
            token_id = self.tokenizer.convert_tokens_to_ids(token_str)
            # If the tokenizer recognizes the token, add it to our stop list
            if token_id is not None and token_id != self.tokenizer.unk_token_id:
                self.stop_token_ids.append(token_id)
                
        # Remove any duplicates
        self.stop_token_ids = list(set(self.stop_token_ids))
        print(f"Registered stop token IDs for this model: {self.stop_token_ids}")
        # ---------------------------------------------------------

        # 2. A100 40GB OPTIMIZED DEFAULTS
        engine_args = AsyncEngineArgs(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="bfloat16",          
            max_num_seqs=256,          
            enforce_eager=True,           
            enable_chunked_prefill=False,       
            trust_remote_code=True
        )
        
        print(f"Loading {model_path} (bfloat16) into vLLM engine on A100...")
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        self.loop = asyncio.new_event_loop()
        self.bridge_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.bridge_thread.start()

    def _run_async_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def generate(self, prompt: str, max_attempt: int = 1) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        last_error = None
        for _ in range(max_attempt):
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._async_generate(formatted_prompt), 
                    self.loop
                )
                return future.result()
            except Exception as e:
                last_error = e
                continue
                
        raise Exception(f"LocalVLLMModel failed after {max_attempt} attempts. Last error: {last_error}")

    async def _async_generate(self, formatted_prompt: str) -> str:
        from vllm import SamplingParams
        
        request_id = str(uuid.uuid4())
        
        # Inject the dynamically built stop token list
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop_token_ids=self.stop_token_ids
        )
        
        engine_inputs = {"prompt": formatted_prompt}
        
        results_generator = self.engine.generate(
            engine_inputs, 
            sampling_params, 
            request_id
        )
        
        final_output = ""
        async for request_output in results_generator:
            final_output = request_output.outputs[0].text
            
        return final_output.strip()
    
    def shutdown(self):
        """Gracefully shuts down the background vLLM loop and engine."""
        print("Initiating graceful vLLM shutdown...")

        # 1. Shut down the engine while the loop is still running so vLLM can
        #    cancel its internal async tasks via call_soon_threadsafe.
        if hasattr(self, 'engine'):
            self.engine.shutdown()

        # 2. Let the loop process the task cancellations before we stop it.
        if hasattr(self, 'loop') and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(asyncio.sleep(0.1), self.loop).result(timeout=5.0)

        # 3. Stop the background asyncio loop.
        if hasattr(self, 'loop') and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)

        # 4. Wait for the bridge thread to finish.
        if hasattr(self, 'bridge_thread') and self.bridge_thread.is_alive():
            self.bridge_thread.join(timeout=5.0)

        # 5. Delete the engine now that all async tasks are cleanly cancelled.
        if hasattr(self, 'engine'):
            del self.engine

        print("vLLM shutdown complete.")