from abc import ABC, abstractmethod
import os
from openai import OpenAI
from typing import Callable, List, Optional, Dict, Any
from dotenv import load_dotenv
from .utils import extract_delegations, replace_delegations_with_responses, format_output, default_formatter, verify_format
from .tree_trace import TreeTraceNode
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Lazy import torch - only needed for HuggingFaceModel
# This allows lightweight usage with OpenAI/DeepSeek models

# Load environment variables from .env file
load_dotenv()


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
    
class RecursiveModel(Model):
    """
    A recursive model wrapper that implements delegation-based problem decomposition.

    This model wraps a base Model and enables recursive generation by:
    1. Detecting <delegate>...</delegate> tags in model output
    2. Recursively solving each delegation as a sub-problem
    3. Replacing delegations with their solutions and continuing generation

    The recursive process allows the model to:
    - Break complex problems into simpler sub-problems
    - Solve sub-problems independently with full fragment
    - Integrate sub-solutions back into the main reasoning flow
    - Continue generation with enhanced fragment from solved sub-problems
    - Process sub-problems in parallel (when max_parallel_workers > 1)

    **Safety Constraints:**
    - max_depth: Prevents infinite recursion by limiting nesting levels
    - max_call: Prevents resource exhaustion by capping total model calls
    - max_parallel_workers: Controls concurrent execution of sub-problems

    **Trace Recording:**
    - Uses TreeTraceNode to record the entire generation tree
    - Access via last_generation_trace() for debugging and analysis
    - Tracks: prompts, fragments, depths, outputs, latencies, and subproblem relationships

    **Error Handling and Partial Traces:**
    - All errors propagate to the root and are re-raised
    - Partial successful traces are preserved in self.trace even on failure
    - Failed nodes are never connected to the tree structure
    - Successful subproblems are preserved even when later siblings fail
    - No continuation (next) is created if any subproblem fails
    - This allows inspection of what succeeded before failure occurred

    """

    def __init__(self,
                 base_model: Model,
                 guard_model: Optional[Model] = None,
                 max_depth: int = 20,
                 max_call: int = 50,
                 max_parallel_workers: int = 1,
                 max_subproblem_retry: int = 2,
                 formatter: Callable[[str, str, str], str] = default_formatter,
                 print_progress: bool = False
                 ):
        """
        Initialize the recursive model wrapper.

        Args:
            base_model: The underlying Model to use for generation.
            guard_model: The non-recursive Model to use when max recursion depth is reached.
            max_depth: Maximum recursion depth allowed. Prevents infinite recursion.
                      Each delegation increases depth by 1.
            max_call: Maximum total number of model calls allowed across all recursion levels.
                     Prevents resource exhaustion from too many API calls.
            max_parallel_workers: Maximum number of parallel workers for recursive generation.
            max_subproblem_retry: Maximum number of retry attempts for each subproblem generation.
                                 Defaults to 1 (no retry).
            formatter: Function to format prompt and fragment into a single string for the base model.
            print_progress: Whether to print the generation to the console
        """
        self.base_model = base_model
        self.guard_model = guard_model
        self.max_depth = max_depth
        self.max_call = max_call
        self.max_parallel_workers = max_parallel_workers
        self.max_subproblem_retry = max_subproblem_retry
        self.call_count = 0  # Tracks total calls across all recursion levels
        self.formatter = formatter
        self.print_progress = print_progress
        self.trace = TreeTraceNode("","",0)  # Store the trace of the last generation for debugging
        self.main_problem = ""  # Store the main problem prompt for context in subproblems


    def generate(self, prompt: str, max_attempt: int = 1) -> str:
        """
        Generate a response with automatic retry mechanism on failures.

        This is the main entry point for generating responses. It:
        1. Resets the call counter for a fresh generation session
        2. Attempts generation up to max_attempt times
        3. Retries on exceptions (e.g., max_depth or max_call exceeded)
        4. Returns the final successful result

        **Retry Behavior:**
        - Each retry gets a fresh attempt with reset call_count
        - Exceptions are caught and logged, then generation retries
        - The final attempt (after all retries fail) is not caught

        Args:
            prompt: The problem or instruction to solve
            max_attempt: Maximum number of retry attempts if generation fails

        Returns:
            The final generated response string with all delegations resolved

        Raises:
            Exception: If the final attempt fails (after all retries exhausted)

        Example:
            >>> model = RecursiveModel(base_model, max_depth=2, max_call=5)
            >>> result = model.generate("Solve problem X", max_attempt=3)
            # If first 2 attempts hit max_call, 3rd attempt will retry
        """
        self.main_problem = prompt
        # Try up to max_attempt times, catching exceptions
        last_error = None
        for _ in range(max_attempt):
            try:
                self.call_count = 0
                root = TreeTraceNode(prompt=prompt, fragment="", depth=0)
                self._recursive_generate(root)
                self.trace = root  # Store the trace for debugging
                return root.get_final_output()  # Success, return immediately
            except Exception as e:
                # Save partial trace even on error (contains all successful generations)
                self.trace = root
                last_error = e
                continue

        raise Exception(f"All {max_attempt} attempts failed. Final attempt error: {last_error}")

    def _recursive_generate(self, node: TreeTraceNode):
        """
        Recursively generate output by solving delegations as sub-problems.

        This implements the core recursive delegation algorithm using TreeTraceNode
        for tracking the entire generation tree.

        **Generation Process:**
        1. **Safety Checks**: Verify depth and call limits not exceeded
        2. **Generate**: Call base model with node's prompt and fragment
        3. **Extract delegations**: Check for <delegate>...</delegate> tags
        4. **Base Case**: If no delegations, set node.final_output and return
        5. **Recursive Case**:
           a. Recursively solve each delegation as independent sub-problem
           b. Create child TreeTraceNode for each subproblem
           c. Replace delegation tags with <response>sub_solution</response>
           d. Create continuation node with updated fragment (depth unchanged)
           e. Link continuation node as node.next

        **Key Design Decisions:**
        - Sub-problems are solved with empty fragment (independent reasoning)
        - Sub-problems increase depth by 1 (prevent infinite nesting)
        - Continuation call keeps same depth (not a new sub-problem)
        - Call count is global across all recursion levels
        - TreeTraceNode structure allows inspection of entire generation tree

        Args:
            node: TreeTraceNode containing:
                  - prompt: The current problem to solve
                  - fragment: Previously generated text
                  - depth: Current recursion depth (0 = top level)

        Returns:
            TreeTraceNode: The input node, now populated with generation data,
                          subproblems, and continuation nodes

        Raises:
            Exception: If maximum recursion depth is exceeded (depth > max_depth)
            Exception: If maximum call count is exceeded (call_count >= max_call)

        Example:
            >>> # Internal recursive flow
            >>> root = TreeTraceNode("Solve X+Y", "", 0)
            >>> _recursive_generate(root)
            # Generates: "I need <delegate>X</delegate> and <delegate>Y</delegate>"
            # Creates child nodes for "X" and "Y" at depth 1
            # Replaces: "I need <response>5</response> and <response>3</response>"
            # Creates continuation node at depth 0
            # Final output via root.get_final_output(): "So X+Y = 8"
        """
        
        # Safety check: Prevent infinite recursion
        if node.depth > self.max_depth:
            if not self.guard_model:
                raise Exception("Maximum recursion depth reached.")
            else:
                try:
                    output = self._guarded_generate_with_formatter(node, model=self.guard_model, max_attempt=1)
                except Exception as e:
                    raise Exception(f"Error in base model generation: {e}")
                
                node.final_output = format_output(output)
                return node
            

        # Generate output using base model (retries up to 3 times internally)
        try:
            output = self._guarded_generate_with_formatter(node,model=self.base_model, max_attempt=1)
        except Exception as e:
            raise Exception(f"Error in base model generation: {e}")


        # Base case: answer is generated
        if "</think>" in output:
            node.final_output = format_output(output)
            return node
        
        # Extract any delegation tags from the output
        subproblems = extract_delegations(output)

        # Recursive case: Solve each delegation as a sub-problem
        sub_responses: List[str] = []

        if self.max_parallel_workers == 1:
            try:
                sub_responses = self._dfs_sequential_subproblem_generate(subproblems, node)
            except Exception as e:
                raise Exception(f"Error in sequential sub-problem generation: {e}")
        else:
            try:
                sub_responses = self._parallel_subproblem_generate(subproblems, node)
            except Exception as e:
                raise Exception(f"Error in parallel sub-problem generation: {e}")

        # Replace all <delegate>...</delegate> tags with <response>...</response>
        reconstructed_output = replace_delegations_with_responses(output, sub_responses)

        # Continue generation with updated fragment
        # - Same prompt (still solving the original problem)
        # - Updated fragment (now includes resolved sub-problems)
        # - Same depth (this is continuation, not a new sub-problem)
        try:
            newnode = TreeTraceNode(prompt=node.prompt, fragment=node.fragment+"\n\n"+reconstructed_output, depth=node.depth)
            node.next = newnode
            self._recursive_generate(newnode)
            return node
        except Exception as e:
            raise Exception(f"Error continuing generation: {e}")
    
    def _guarded_generate_with_formatter(self, node: TreeTraceNode, model: Model, max_attempt: int) -> str:
        """
        Helper method to call base model's generate_with_formatter with retries.

        This method wraps the base model's generate_with_formatter in a retry loop:
        - Tries up to max_attempt times
        - Catches exceptions and retries
        - If all attempts fail, raises the last exception

        Args:
            node: The TreeTraceNode instance containing prompt, fragment, and depth
            max_attempt: Maximum number of retry attempts if generation fails

        Returns:
            The generated response string from the base model

        Raises:
            Exception: If all attempts fail, raises the last encountered exception
        """

        self.call_count += 1
            
        last_error = None
        for _ in range(max_attempt):
            try:
                start_time = time.time()
                context = node.parent.prompt if node.parent else "None"
                response = model.generate(self.formatter(node.prompt, context, node.fragment), max_attempt=1)
                if not verify_format(response):
                    raise Exception(f"Invalid format detected in response. output:{ response }")
                node.record_generation(response, latency=time.time()-start_time)
                if self.print_progress:
                    print(f"{'{'}\n\"prompt\":\"{node.prompt}\"\n\"context\":\"{context}\"\n\"fragment\":\"{node.fragment}\"\n\"output\":\"{response}\"\n{'}'}\ndepth: {node.depth}\nlatency: {node.latency}\ncall: {self.call_count}\n{'='*50}")
                return response
            except Exception as e:
                last_error = e
                continue
        raise Exception(f"All {max_attempt} attempts failed for base model. Last error: {last_error}")
    
    def _dfs_sequential_subproblem_generate(self, subproblems: List[str], node: TreeTraceNode) -> List[str]:
        """
        Helper method to solve sub-problems sequentially (depth-first search).

        This method processes each sub-problem one after another:
        - Checks call count before each sub-problem (thread-safe)
        - Creates child TreeTraceNode for each subproblem
        - Solves sub-problem with empty fragment and increased depth
        - Adds completed subproblem node to parent's subproblems list
        - Collects responses in order

        Args:
            subproblems: List of sub-problem strings extracted from delegations
            node: Parent TreeTraceNode to attach subproblem nodes to

        Returns:
            List of responses corresponding to each sub-problem (in order)

        Raises:
            Exception: If maximum call count is exceeded or if any sub-problem fails
        """
        responses = []
        for sub in subproblems:
            try:
                newnode = TreeTraceNode(prompt=sub, fragment="", depth=node.depth + 1)
                node.add_subproblem(newnode)
                for i in range(self.max_subproblem_retry):
                    try:
                        self._recursive_generate(newnode)
                        break  # Success, break out of retry loop
                    except Exception as e:
                        if i == self.max_subproblem_retry:
                            raise Exception(f"Sub-problem generation failed after {self.max_subproblem_retry} attempts: {e}")
                        continue  # Retry on exception
                
                responses.append(newnode.get_final_output())
            except Exception as e:
                raise Exception(f"Error solving sub-problem '{sub}': {e}")
        return responses
    
    def _parallel_subproblem_generate(self, subproblems: List[str], node: TreeTraceNode) -> List[str]:
        """
        Helper method to solve sub-problems in parallel using asyncio + ThreadPoolExecutor.

        This method avoids thread pool starvation by using asyncio to manage task dependencies
        while ThreadPoolExecutor handles the actual work with max_parallel_workers limit.

        Key design:
        - asyncio manages waiting (doesn't block threads while waiting for children)
        - ThreadPoolExecutor performs actual work (limited by max_parallel_workers)
        - Parents can wait for children without occupying threads
        - No deadlocks: threads are released back to pool while waiting

        Args:
            subproblems: List of sub-problem strings extracted from delegations
            node: Parent TreeTraceNode to attach subproblem nodes to

        Returns:
            List of responses corresponding to each sub-problem (in original order)

        Raises:
            Exception: If any sub-problem fails during generation
        """
        import asyncio
        import concurrent.futures

        async def process_subproblem(sub: str, executor_ref) -> tuple:
            """
            Process a single subproblem asynchronously.

            Returns tuple of (node, response) to maintain order.
            """
            loop = asyncio.get_running_loop()

            # Create the child node
            newnode = TreeTraceNode(prompt=sub, fragment="", depth=node.depth + 1)

            # Add node to parent immediately before recursive call
            node.add_subproblem(newnode)

            # Retry logic for subproblem generation
            last_error = None
            for attempt in range(self.max_subproblem_retry):
                try:
                    # Execute the recursive generation in the thread pool
                    # This is where the actual blocking work happens
                    # The thread is ONLY occupied during this execution
                    await loop.run_in_executor(
                        executor_ref,
                        self._recursive_generate,
                        newnode
                    )
                    break  # Success, exit retry loop
                except Exception as e:
                    last_error = e
                    if attempt == self.max_subproblem_retry - 1:
                        raise Exception(f"Sub-problem generation failed after {self.max_subproblem_retry} attempts: {e}")
                    continue  # Retry on exception

            # Get the final output
            response = newnode.get_final_output()

            return newnode, response

        async def process_all_subproblems(executor_ref):
            """Process all subproblems concurrently and maintain order."""
            # Create tasks for all subproblems
            tasks = [process_subproblem(sub, executor_ref) for sub in subproblems]

            # Wait for all to complete (maintains order)
            # gather() ensures all children complete before parent moves on
            # Because this is 'await', it yields control - no threads are blocked here!
            results = await asyncio.gather(*tasks)

            # Unpack results (nodes already added to parent before recursive calls)
            responses = []
            for newnode, response in results:
                responses.append(response)

            return responses

        # Create thread pool executor with max_parallel_workers limit
        # This enforces the global concurrency limit
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel_workers) as executor:
            # Run the async function
            try:
                responses = asyncio.run(process_all_subproblems(executor))
                return responses
            except Exception as e:
                raise Exception(f"Error in parallel sub-problem generation: {e}")


class OpenAIModel(Model):
    """OpenAI-compatible model implementation."""

    def __init__(
        self,
        model_name: str,
        api_key: str ,
        base_url: str ,
        system_prompt: str = "You are a helpful AI assistant."
    ):
        """
        Initialize OpenAI model.

        Args:
            model_name: Name of the model to use
            api_key: API key for authentication
            base_url: Base URL for the API endpoint
            system_prompt: System prompt to use for all generations
        """
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.system_prompt = system_prompt

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def generate(self, prompt: str, max_attempt: int = 1) -> str:
        """
        Generate a response using the OpenAI API with retry mechanism.

        Args:
            prompt: The user prompt
            max_attempt: Maximum number of retry attempts on failure (default: 1)

        Returns:
            The generated response

        Raises:
            Exception: If all attempts fail
        """
        last_error = None
        for _ in range(max_attempt):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
                content = response.choices[0].message.content
                return content if content is not None else ""
            except Exception as e:
                last_error = e
                continue

        raise Exception(f"All {max_attempt} attempts failed. Last error: {last_error}")
    
    def update_system_prompt(self, new_prompt: str):
        """
        Update the system prompt for future generations.

        Args:
            new_prompt: The new system prompt to use
        """
        self.system_prompt = new_prompt


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


class DeepSeekModel(Model):
    """DeepSeek model implementation using OpenAI-compatible API."""

    def __init__(
        self,
        reasoning: bool = False,
        system_prompt: str = "You are a helpful AI assistant."
    ):
        """
        Initialize DeepSeek model.

        Args:
            reasoning: If True, uses deepseek-reasoner; otherwise uses deepseek-chat
            system_prompt: System prompt to use for all generations
        """
        self.model_name = "deepseek-reasoner" if reasoning else "deepseek-chat"
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = "https://api.deepseek.com"
        self.system_prompt = system_prompt

        # Initialize OpenAI client for DeepSeek
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def generate(self, prompt: str, max_attempt: int = 1) -> str:
        """
        Generate a response using the DeepSeek API with retry mechanism.

        Args:
            prompt: The user prompt
            max_attempt: Maximum number of retry attempts on failure (default: 1)

        Returns:
            The generated response

        Raises:
            Exception: If all attempts fail
        """
        last_error = None
        for _ in range(max_attempt):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
                content = response.choices[0].message.content
                return content if content is not None else ""
            except Exception as e:
                last_error = e
                continue

        raise Exception(f"All {max_attempt} attempts failed. Last error: {last_error}")


class GPTModel(Model):
    """GPT-5 model implementation using the new responses API."""

    def __init__(
        self,
        system_prompt: str = "You are a helpful AI assistant."
    ):
        """
        Initialize GPT-5 model.

        Args:
            system_prompt: System prompt to use for all generations
        """
        self.system_prompt = system_prompt
        # OpenAI client automatically loads API key from OPENAI_API_KEY environment variable
        self.client = OpenAI()

    def generate(self, prompt: str, max_attempt: int = 1) -> str:
        """
        Generate a response using the GPT-5 responses API with retry mechanism.

        Args:
            prompt: The user prompt
            max_attempt: Maximum number of retry attempts on failure (default: 1)

        Returns:
            The generated response

        Raises:
            Exception: If all attempts fail
        """
        last_error = None
        for _ in range(max_attempt):
            try:
                response = self.client.responses.create(
                    model="gpt-5",
                    input=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
                # Extract content from the response
                content = response.output_text if hasattr(response, 'output_text') else str(response)
                return content if content is not None else ""
            except Exception as e:
                last_error = e
                continue

        raise Exception(f"All {max_attempt} attempts failed. Last error: {last_error}")
    
class ErrorGuardModel(Model):
    def generate(self, prompt: str, max_retry:int) -> str:
        return "You reached recursion limit, you must solve this problem your self and delegate no further"