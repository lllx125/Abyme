"""
Utility functions for text processing and XML tag extraction.
"""
import re
from typing import List


def extract_elaborations(text: str) -> List[str]:
    """
    Extract elaborate tags for each run

    Args:
        text: The text to search in

    Returns:
        List of elaboration contents (content within tags only)

    """
    # Use regex to find all occurrences of <elaborate>...</elaborate>
    pattern = r'<elaborate>(.*?)</elaborate>'
    contents = []

    for match in re.finditer(pattern, text, flags=re.DOTALL):
        contents.append(match.group(1))  # Return content only (without tags)

    return contents


def format_output(text: str, full_response: bool = False, eos_token: str = "<｜end▁of▁sentence｜>") -> str:
    """
    Format the output text based on response type.

    Args:
        text: The text to format
        full_response: If True, return the full text. If False, return everything after the final </think> tag.
        eos_token: The EOS token to remove from the end if present (optional)

    Returns:
        Formatted text without EOS token

    """
    result = text

    if not full_response:
        # Find the last occurrence of </think>
        last_think_end = result.rfind('</think>')
        if last_think_end != -1:
            # Return everything after the final </think> tag
            result = result[last_think_end + len('</think>'):]

    # Remove EOS token if present at the end
    if eos_token and result.endswith(eos_token):
        result = result[:-len(eos_token)]

    return result


def replace_elaborations_with_responses(text: str, responses: List[str]) -> str:
    """
    Replace all <elaborate>...</elaborate> tags with corresponding responses from the list.

    This function is extracted from engine.py to be reusable in both the inference engine
    and the SFT data generation pipeline.

    Args:
        text: The text containing <elaborate>...</elaborate> tags
        responses: List of response strings to replace elaborations with (in order)

    Returns:
        Text with all elaborations replaced by <response>...</response> tags

    Raises:
        ValueError: If the number of elaborations doesn't match the number of responses
    """
    # First, count the number of elaboration tags
    elaborations = extract_elaborations(text)

    if len(elaborations) != len(responses):
        raise ValueError(
            f"Number of elaborations ({len(elaborations)}) doesn't match "
            f"number of responses ({len(responses)})"
        )

    # Replace each elaboration tag with its corresponding response
    def replace_elaboration(match):
        idx = replace_elaboration.counter
        replace_elaboration.counter += 1
        return f'<response>{responses[idx]}\n</response>'

    replace_elaboration.counter = 0
    result = re.sub(r'<elaborate>.*?</elaborate>', replace_elaboration, text, flags=re.DOTALL)

    return result

def default_context_formatter(prompt: str, context: str) -> str:
    """
    Default context formatter that combines prompt and context.

    Args:
        prompt: The input prompt string
        context: Additional context string

    Returns:
        Combined string of prompt and context
    """
    return f"Prompt: {prompt}\nContext: {context}"

def verify_format(trace: str) -> bool:
    """
    Verify that the generated output has valid XML-like tag structure.

    This function is used to validate model-generated output when it is produced,
    ensuring it follows the proper format for the Abyme recursive reasoning system.

    **Validation Rules:**

    1. **No Incomplete or Broken Tags**:
       - All tags must be complete (e.g., no "<elabo" without "rate>")
       - Detects partial or malformed tag structures

    2. **No Nested <elaborate> Tags**:
       - <elaborate> tags cannot be nested within themselves
       - Nesting depth must never exceed 1

    3. **All <elaborate> Tags Come in Pairs**:
       - Every <elaborate> opening tag must have a matching </elaborate> closing tag
       - Opening and closing counts must match exactly
       - Closing tags cannot appear before their corresponding opening tags

    4. **No <response> Tags Allowed**:
       - Neither <response> nor </response> tags are permitted
       - The output should only contain elaborations, not responses

    5. **No <think> Opening Tag**:
       - Only the closing tag </think> is allowed
       - No <think> opening tag should exist in the output

    6. **Mutual Exclusivity Requirement**:
       - The output must contain either <elaborate> tags OR a </think> tag, but NOT both
       - At least one of these must be present

    7. **</think> Appears At Most Once**:
       - The closing tag </think> can appear zero or one time
       - Multiple </think> tags are not allowed

    Args:
        trace: The generated output text to validate

    Returns:
        bool: True if all validation rules pass, False if any rule is violated

    Examples:
        Valid outputs:
        >>> verify_format("Solution<elaborate>sub-problem</elaborate>")
        True
        >>> verify_format("Final answer: 42</think>")
        True

        Invalid outputs:
        >>> verify_format("Test<elaborate>nested<elaborate>bad</elaborate></elaborate>")
        False
        >>> verify_format("Both<elaborate>test</elaborate>and</think>")  # Both elaborate and think
        False
        >>> verify_format("Answer</think></think>")  # Multiple </think>
        False
    """
    # Rule 4: No <response> or </response> tags allowed
    if '<response>' in trace or '</response>' in trace:
        return False

    # Rule 5: No <think> opening tag allowed
    if '<think>' in trace:
        return False

    # Rule 1: Check for broken/incomplete tags
    broken_patterns = [
        r'<elabo[^r]',  # Incomplete <elaborate>
        r'</elabo[^r]',  # Incomplete </elaborate>
        r'</thin[^k]',  # Incomplete </think>
    ]
    for pattern in broken_patterns:
        if re.search(pattern, trace):
            return False

    # Rule 3: Check <elaborate> tags are properly paired (balanced and not nested)
    elaborate_open_count = trace.count('<elaborate>')
    elaborate_close_count = trace.count('</elaborate>')

    if elaborate_open_count != elaborate_close_count:
        return False

    # Rule 2: Check for nesting by tracking depth
    depth = 0
    elaborate_pattern = re.compile(r'<elaborate>|</elaborate>')
    for match in elaborate_pattern.finditer(trace):
        if match.group() == '<elaborate>':
            depth += 1
            if depth > 1:  # Nested opening tag
                return False
        else:
            depth -= 1
            if depth < 0:  # Closing before opening
                return False

    # Rule 7: </think> appears at most once
    think_close_count = trace.count('</think>')
    if think_close_count > 1:
        return False

    # Rule 6: Must contain either <elaborate> or </think>, but not both, and at least one
    has_elaborate = elaborate_open_count > 0
    has_think_close = think_close_count > 0

    # Must have at least one
    if not has_elaborate and not has_think_close:
        return False

    # Cannot have both
    if has_elaborate and has_think_close:
        return False

    return True
