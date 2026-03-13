"""
Recursive engine for processing elaborate/response cycles using RecursiveModel pattern.
"""
from typing import List,  Optional, Callable, TYPE_CHECKING
from .model import Model, DeepSeekModel, GPTModel, ErrorGuardModel
from .magic import magic_prompt, magic_formatter, magic_guard_prompt
from .utils import *
from .tree_trace import TreeTraceNode
import time

# Lazy imports - only load torch/transformers when needed for HuggingFace models
# This allows lightweight usage with DeepSeek/GPT without installing heavy dependencies
if TYPE_CHECKING:
    from .pytorch_modules import HuggingFaceModel


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
                 max_chain_length: int = 5,
                 max_parallel_workers: int = 1,
                 max_subproblem_retry: int = 2,
                 formatter: Callable[[str, str, str, str], str] = default_formatter,
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
            max_chain_length: Maximum length of the continuation chain (next nodes) at each depth level.
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
        self.max_chain_length = max_chain_length
        self.max_parallel_workers = max_parallel_workers
        self.max_subproblem_retry = max_subproblem_retry
        self.call_count = 0  # Tracks total calls across all recursion levels
        self.formatter = formatter
        self.print_progress = print_progress
        self.trace = TreeTraceNode("","",0,0)  # Store the trace of the last generation for debugging
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
                self.trace = TreeTraceNode(prompt=prompt, fragment="", depth=0, index=0)
                self._recursive_generate(self.trace)
                return self.trace.get_final_output()  # Success, return immediately
            except Exception as e:
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
        
        # Base Case:
        if node.depth > self.max_depth:
            if not self.guard_model:
                node.status = "FAILED"
                raise Exception("Maximum recursion depth reached.")
            else:
                try:
                    output = self._guarded_generate_with_formatter(node, model=self.guard_model, max_attempt=1)
                except Exception as e:
                    node.status = "FAILED"
                    raise Exception(f"Error in base model generation: {e}")

                node.final_output = format_output(output)
                node.status = "COMPLETED"
                if node.parent:
                    self._continue_generation(node.parent)
                return node
        
        if self.call_count >= self.max_call:
            node.status = "FAILED"
            raise Exception("Maximum call count exceeded.")
        
        if node.index > self.max_chain_length:
            node.status = "FAILED"
            raise Exception(f"Maximum chain length exceeded at depth {node.depth}.")

        # Generate output using base model (retries up to 3 times internally)
        try:
            output = self._guarded_generate_with_formatter(node,model=self.base_model, max_attempt=1)
        except Exception as e:
            node.status = "FAILED"
            raise Exception(f"Error in base model generation: {e}")


        # Base case: answer is generated
        if "</{THINK}>" in output:
            node.final_output = format_output(output)
            node.status = "COMPLETED"
            if node.parent:
                self._continue_generation(node.parent)
            return
        
        # AND OR delegation case: extract delegations and solve as subproblems
        if "<{AND}>" in output:
            type = AND
            node.node_type = "AND"
            subproblems = extract_delegations(output, tag=AND)
        elif "<{OR}>" in output: 
            type = OR
            node.node_type = "OR"
            subproblems = extract_delegations(output, tag=OR) 
        
        node.add_subproblems(subproblems)

        # TODO: trigger worker to start working

    
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
            The generated response string from the base model strictly formatted

        Raises:
            Exception: If all attempts fail, raises the last encountered exception
        """

        self.call_count += 1
            
        last_error = None
        for _ in range(max_attempt):
            try:
                start_time = time.time()
                context = node.parent.prompt if node.parent else "None"
                node.status = "GENERATING"
                response = model.generate(self.formatter(node.prompt, self.main_problem, context, node.fragment), max_attempt=1)
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

    def _continue_generation(self, node: TreeTraceNode):
        """
        Helper method to continue generation after sub-problems are solved.

        This method is called when a sub-problem is completed and we need to
        continue generating the main problem with the updated fragment that
        includes the resolved sub-problem.

        Args:
            node: The TreeTraceNode instance representing the main problem that needs to continue generation

        Returns:
            None (the node's next will be updated with the continuation)

        Raises:
            Exception: If generation fails during continuation
        """
        # Check whether the node is completed
        responses = []
        type = AND if node.node_type == "AND" else OR
        for sub in node.subproblems:
            sub_last = sub.get_last()
            # for AND node, we need to wait for all subproblems to complete
            if sub_last.status != "COMPLETED" and type == AND:
                return
            # for OR node, if this guarenteed because this function is triggered only when a subproblem completes
            
            if sub_last.status == "COMPLETED":
                responses.append(sub_last.final_output)
            else:
                # append the partial response from this child
                responses.append("INCOMPLETE\n\nCurrent Progress:"+sub_last.fragment + sub_last.output)
            
        # Replace all tags with <response>...</response>
        reconstructed_output = replace_delegations_with_responses(node.output, responses, type)

        # Continue generation with updated fragment
        # - Same prompt (still solving the original problem)
        # - Updated fragment (now includes resolved sub-problems)
        # - Same depth (this is continuation, not a new sub-problem)
        try:
            newnode = TreeTraceNode(prompt=node.prompt, fragment=node.fragment+"\n\n"+reconstructed_output, depth=node.depth, index=node.index+1)
            node.next = newnode
            self._recursive_generate(newnode)
            return node
        except Exception as e:
            raise Exception(f"Error continuing generation: {e}")


def AbymeHuggingFaceModel(model_name: str = "Abyme", **kwargs):
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
    from .pytorch_modules import HuggingFaceModel

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

def Abyme_API_Models(model: str, max_depth: int = 20, max_call: int = 50, max_parallel_workers: int = 1, print_progress: bool = False) -> RecursiveModel:
    
    if model == "deepseek":
        base_model = DeepSeekModel(system_prompt=magic_prompt)
    elif model == "gpt":
        base_model = GPTModel(system_prompt=magic_prompt)
    elif model == "deepseek-r":
        base_model = DeepSeekModel(reasoning= True, system_prompt=magic_prompt)
    else:
        raise ValueError(f"Unsupported model type: {model}")
    
    guard_model = ErrorGuardModel()
    return RecursiveModel(
        base_model=base_model,
        guard_model=guard_model,
        formatter=magic_formatter,
        max_depth=max_depth,
        max_call=max_call,
        max_parallel_workers=max_parallel_workers,
        print_progress = print_progress
    )