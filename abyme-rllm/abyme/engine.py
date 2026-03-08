"""
Recursive engine for processing elaborate/response cycles.
"""
import re
import torch
from transformers import  PreTrainedTokenizerBase, StoppingCriteria, StoppingCriteriaList
from .tokenization import setup_model_and_tokenizer, get_stopping_token_id
from .extraction import extract_elaborations, format_output


class RLLMEngine:
    """
    The recursive controller that manages the elaborate/response cycle.

    This engine:
    1. Generates text until it encounters </run>
    2. Extracts all <elaborate> blocks
    3. Recursively processes each elaboration
    4. Injects <response> blocks back into context
    5. Resumes generation

    Example:
        >>> from abyme import RLLMEngine
        >>>
        >>> # User can pass any HF arguments (quantization, device_map, etc.)
        >>> engine = RLLMEngine(
        ...     model_name="Abyme/Abyme-V1",
        ...     device_map="auto",
        ...     load_in_4bit=True,
        ...     trust_remote_code=True
        ... )
        >>>
        >>> # The engine handles the recursive loop hidden from the user
        >>> output = engine.generate("Solve this math problem...")
    """

    def __init__(
        self,
        model_name: str,
        max_recursion_depth: int = 3,
        **kwargs
    ):
        """
        Initialize the recursive engine.

        Args:
            model_name: HuggingFace model ID (e.g., "Abyme/Abyme-V1")
            max_recursion_depth: Maximum depth of recursion to prevent infinite loops
            **kwargs: Arguments passed to AutoModelForCausalLM.from_pretrained
                     (e.g., device_map="auto", load_in_4bit=True, torch_dtype=torch.float16)

        Example:
            >>> engine = RLLMEngine(
            ...     "Abyme/Abyme-V1",
            ...     device_map="auto",
            ...     load_in_4bit=True,
            ...     trust_remote_code=True,
            ... )
        """
        print(f"Initializing RLLMEngine with model: {model_name}")

        # Default to safe settings if not provided
        if "device_map" not in kwargs and torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        elif "device_map" not in kwargs:
            kwargs["device_map"] = "cpu"

        # Load model and tokenizer with user-provided kwargs
        self.model, self.tokenizer = setup_model_and_tokenizer(
            model_name,
            **kwargs
        )

        self.stop_token_id: int = get_stopping_token_id(self.tokenizer)
        self.max_recursion_depth: int = max_recursion_depth

    
    def generate(
        self,
        prompt: str,
        current_depth: int = 0,
        full_response: bool = True,
        print_responses: bool = False
    ) -> str:
        """
        Recursively generate text with elaborate/response cycles using BFS logic.

        The engine implements the following recursive loop:

        1. **Initial Generation (Step 1)**:
           - Prepend <think> tag to the prompt to trigger thinking mode
           - Generate text until </run> token is encountered

        2. **Loop Condition Check (Step 2)**:
           - Continue looping while output ends with </run>
           - Exit if no </run> found (generation stopped naturally)

        3. **Elaboration Extraction (Step 3)**:
           - Extract ALL elaboration contents from the output
           - Uses extract_elaborations() to parse content within <elaborate>...</elaborate> tags
           - Returns list of elaboration contents (without the tags themselves)

        4. **Parallel Execution (Step 4)**:
           - Process each elaboration content recursively by calling generate()
           - Each recursive call increments depth and runs with full_response=False
           - Collect all results in child_results list

        5. **Injection and Cleanup (Step 5)**:
           - Replace each <elaborate>...</elaborate> tag with its corresponding response in <response>...</response> tags using regex substitution
           - Uses regex substitution to find and replace tags in order
           - Delete the final </run> tag from the reconstructed text
           - This prepares the context for resuming generation

        6. **Resume Generation (Step 6)**:
           - Continue generating from the reconstructed context
           - Loop back to step 2

        7. **Format Output**:
           - Apply format_output() to clean the final result
           - If full_response=True, return complete text
           - If full_response=False, return only content after final </think> tag

        Args:
            prompt: The input prompt to process
            current_depth: Current recursion depth (internal use, default: 0)
            full_response: If True, returns the full generated text including <think> tags.
                         If False, returns only content after the final </think> tag (default: True)
            print_responses: If True, print the responses whenever </run> is reached or
                           generation ends naturally (default: False)

        Returns:
            Final generated text, formatted according to full_response parameter

        Example:
            >>> output = engine.generate(
            ...     "Solve this math problem: 5 * 5 + 3 * 3",
            ...     full_response=False,
            ...     print_responses=True
            ... )
        """

        # Base case: max depth reached
        if current_depth > self.max_recursion_depth:
            return f"Max recursion depth ({self.max_recursion_depth}) reached."
        
        #prompt += "<think>" # Trigger model to think

        # Step 1: Generate until </run>
        if current_depth == 0:
            output = prompt  
        else:
            output = self._generate_until_stop(prompt)
        
        if print_responses:
            print(f"\n{'='*50}")
            print(output)
            print(f"{'='*50}\n")

        # Step 2: If the output doesn't contain </run>, it means generation stopped for another reason (e.g., max tokens) - return as is
        while output.endswith("</run>"):

            # Step 3: Extract ALL elaborations that need processing
            elaborations = extract_elaborations(output)

            if not elaborations:
                break  # No elaborations found, exit loop

            # Step 4 (Parallel Execution): Process all elaborations
            child_results = []

            for elaboration_content in elaborations:
                result = self.generate(
                    elaboration_content,
                    current_depth + 1,
                    False,
                    print_responses
                )
                child_results.append(result)


            # Step 5: Delete elaboration tags and inject responses, then delete final </run> tag
            reconstructed = output

            # Replace each elaboration tag with its corresponding response using regex
            def replace_elaboration(match):
                # Get the index based on how many replacements we've done
                idx = replace_elaboration.counter
                replace_elaboration.counter += 1
                return f'<response>{child_results[idx]}</response>'

            replace_elaboration.counter = 0
            reconstructed = re.sub(r'<elaborate>.*?</elaborate>', replace_elaboration, reconstructed, flags=re.DOTALL)

            # Delete the final </run> tag
            if reconstructed.endswith('</run>'):
                reconstructed = reconstructed[:-6]  # Remove last 6 characters ('</run>')

            # Step 6: Resume generation
            output = self._generate_until_stop(reconstructed)

        return format_output(output, full_response, self.tokenizer.eos_token)
    
    def _generate_until_stop(self, prompt: str, max_new_tokens: int = 512) -> str:
        """
        Generate text until </run> token is encountered.

        Args:
            prompt: The input prompt
            max_new_tokens: Maximum number of new tokens to generate

        Returns:
            Generated text
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)

        # Track original input length to only check newly generated tokens
        input_length = input_ids.shape[1]

        # Create stopping criteria
        stopping_criteria = StoppingCriteriaList([
            RunTokenStoppingCriteria(self.stop_token_id, self.tokenizer, input_length)
        ])

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stopping_criteria,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (exclude input to remove BOS)
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)

        # Prepend the original prompt to get the full output
        full_text = prompt + generated_text
        return full_text


class RunTokenStoppingCriteria(StoppingCriteria):
    """
    Custom stopping criteria that stops generation when </run> token is encountered.
    Only checks newly generated tokens, not the input prompt.
    """

    def __init__(self, stop_token_id: int, tokenizer: PreTrainedTokenizerBase, input_length: int):
        self.stop_token_id = stop_token_id
        self.tokenizer = tokenizer
        self.input_length = input_length

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs
    ) -> bool:
        # Only check newly generated tokens (after input_length)
        current_length = input_ids.shape[1]

        # If we haven't generated any new tokens yet, don't stop
        if current_length <= self.input_length:
            return False

        # Check if the last generated token is the stop token
        if input_ids[0, -1] == self.stop_token_id:
            return True

        # Also check if </run> appears in newly generated text
        # (in case it's tokenized as multiple tokens)
        # Only look at the last 10 newly generated tokens
        generated_tokens = input_ids[0, self.input_length:]
        check_tokens = generated_tokens[-10:] if len(generated_tokens) > 10 else generated_tokens
        decoded = self.tokenizer.decode(check_tokens, skip_special_tokens=False)
        return "</run>" in decoded