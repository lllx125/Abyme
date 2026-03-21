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
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from abyme.model import Model

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
                 system_prompt: str = "You are a helpful mathematical reasoning agent.",
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
    """
    A high-throughput local model utilizing vLLM's AsyncLLMEngine.
    
    Architecture Note:
    Abyme uses synchronous worker threads. vLLM requires an asynchronous event loop.
    This class acts as a Thread-to-Async Bridge. It spins up a background daemon thread
    running an asyncio event loop. When synchronous workers call `generate()`, it safely
    submits the request to the background loop and blocks until vLLM finishes generating.
    This allows 50+ synchronous threads to saturate the A100 GPU efficiently.
    """
    
    def __init__(self, 
                 model_path: str, 
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.90,
                 max_model_len: int = 8192,
                 temperature: float = 0.7,
                 max_tokens: int = 2048,
                 system_prompt: str = "You are a helpful mathematical reasoning agent."):
        """
        Initialize the local vLLM engine.
        Requires the `vllm` python package installed.
        """
        
        try:
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine
            from vllm import SamplingParams
        except ImportError:
            raise ImportError("Run `pip install vllm`.")

        self.system_prompt = system_prompt
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # A100 40GB OPTIMIZED DEFAULTS
        engine_args = AsyncEngineArgs(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="bfloat16",          # Force 16-bit
            max_num_seqs=256,          # Maximize batch size for 50+ concurrent threads
            enforce_eager=False,       # Use CUDA graphs for faster inference
            disable_log_requests=True,
            trust_remote_code=True
        )
        
        print(f"Loading {model_path} (bfloat16) into vLLM engine on A100...")
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        self.loop = asyncio.new_event_loop()
        self.bridge_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.bridge_thread.start()

    def _run_async_loop(self):
        """
        Target function for the background daemon thread.
        Sets up the asyncio event loop and runs it indefinitely.
        """
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def generate(self, prompt: str, max_attempt: int = 1) -> str:
        """
        Synchronous interface called by RecursiveModel worker threads.
        Thread-safe: Safely submits the generation task to the background async loop.
        """
        # Note: vLLM's standard generate doesn't take chat formats natively in the base generate method 
        # unless using the chat templates wrapper. We will manually format the prompt for the base engine.
        formatted_prompt = f"{self.system_prompt}\n\n{prompt}"
        
        last_error = None
        for _ in range(max_attempt):
            try:
                # Submit the coroutine to the background loop
                future = asyncio.run_coroutine_threadsafe(
                    self._async_generate(formatted_prompt), 
                    self.loop
                )
                
                # Block the calling worker thread until the future resolves
                return future.result()
                
            except Exception as e:
                last_error = e
                continue
                
        raise Exception(f"LocalVLLMModel failed after {max_attempt} attempts. Last error: {last_error}")

    async def _async_generate(self, prompt: str) -> str:
        """
        Asynchronous generation method running inside the vLLM background loop.
        """
        # Generate a unique request ID for vLLM tracking
        request_id = str(uuid.uuid4())
        
        # Submit to the continuous batching engine
        results_generator = self.engine.generate(
            prompt, 
            self.sampling_params, 
            request_id
        )
        
        final_output = ""
        # Asynchronously iterate over the stream of generated tokens
        async for request_output in results_generator:
            # We only care about the final accumulated string
            final_output = request_output.outputs[0].text
            
        return final_output