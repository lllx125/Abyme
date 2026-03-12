"""
Utility functions for text processing and XML tag extraction.
"""
import re
from typing import List


def extract_delegations(text: str) -> List[str]:
    """
    Extract delegate tags for each run

    Args:
        text: The text to search in

    Returns:
        List of delegation contents (content within tags only)

    """
    # Use regex to find all occurrences of <delegate>...</delegate>
    pattern = r'<delegate>(.*?)</delegate>'
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


def replace_delegations_with_responses(text: str, responses: List[str]) -> str:
    """
    Replace all <delegate>...</delegate> tags with corresponding responses from the list.

    This function is extracted from engine.py to be reusable in both the inference engine
    and the SFT data generation pipeline.

    Args:
        text: The text containing <delegate>...</delegate> tags
        responses: List of response strings to replace delegations with (in order)

    Returns:
        Text with all delegations replaced by <response>...</response> tags

    Raises:
        ValueError: If the number of delegations doesn't match the number of responses
    """
    # First, count the number of delegation tags
    delegations = extract_delegations(text)

    if len(delegations) != len(responses):
        raise ValueError(
            f"Number of delegations ({len(delegations)}) doesn't match "
            f"number of responses ({len(responses)})"
        )

    # Replace each delegation tag with its corresponding response
    def replace_delegation(match):
        idx = replace_delegation.counter
        replace_delegation.counter += 1
        return f'<response>{responses[idx]}\n</response>'

    replace_delegation.counter = 0
    result = re.sub(r'<delegate>.*?</delegate>', replace_delegation, text, flags=re.DOTALL)

    return result

def default_formatter(prompt: str, context: str, fragment: str) -> str:
    """
    Default formatter that combines prompt, context, and fragment.

    Args:
        prompt: The input prompt string
        context: Additional context string
        fragment: previous answer fragment

    Returns:
        Combined string of prompt, context, and fragment
    """
    return f"Prompt: {prompt}\nContext: {context}\nFragment: {fragment}"

def verify_format(trace: str) -> bool:
    """
    Verify that the generated output has valid XML-like tag structure.

    This function is used to validate model-generated output when it is produced,
    ensuring it follows the proper format for the Abyme recursive reasoning system.

    **Validation Rules:**

    1. **No Incomplete or Broken Tags**:
       - All tags must be complete (e.g., no "<elabo" without "rate>")
       - Detects partial or malformed tag structures

    2. **No Nested <delegate> Tags**:
       - <delegate> tags cannot be nested within themselves
       - Nesting depth must never exceed 1

    3. **All <delegate> Tags Come in Pairs**:
       - Every <delegate> opening tag must have a matching </delegate> closing tag
       - Opening and closing counts must match exactly
       - Closing tags cannot appear before their corresponding opening tags

    4. **No <think> Opening Tag**:
       - Only the closing tag </think> is allowed
       - No <think> opening tag should exist in the output

    5. **</think> Appears At Most Once**:
       - The closing tag </think> can appear zero or one time
       - Multiple </think> tags are not allowed

    Args:
        trace: The generated output text to validate

    Returns:
        bool: True if all validation rules pass, False if any rule is violated

    Examples:
        Valid outputs:
        >>> verify_format("Solution<delegate>sub-problem</delegate>")
        True
        >>> verify_format("Solution<response>answer</response>")
        True
        >>> verify_format("Final answer: 42</think>")
        True

        Invalid outputs:
        >>> verify_format("Test<delegate>nested<delegate>bad</delegate></delegate>")
        False
        >>> verify_format("Both<delegate>test</delegate>and</think>")  # Both delegate and think
        False
        >>> verify_format("Both<response>test</response>and</think>")  # Both response and think
        False
        >>> verify_format("Answer</think></think>")  # Multiple </think>
        False
    """

    # Rule 4: No <think> opening tag allowed
    if '<think>' in trace:
        return False

    # Rule 1: Check for broken/incomplete tags
    broken_patterns = [
        r'<elabo[^r]',  # Incomplete <delegate>
        r'</elabo[^r]',  # Incomplete </delegate>
        r'</thin[^k]',  # Incomplete </think>
        r'<respon[^s]',  # Incomplete <response>
        r'</respon[^s]',  # Incomplete </response>
    ]
    for pattern in broken_patterns:
        if re.search(pattern, trace):
            return False

    # Rule 3: Check <delegate> tags are properly paired (balanced and not nested)
    delegate_open_count = trace.count('<delegate>')
    delegate_close_count = trace.count('</delegate>')

    if delegate_open_count != delegate_close_count:
        return False

    # Rule 3: Check <response> tags are properly paired
    response_open_count = trace.count('<response>')
    response_close_count = trace.count('</response>')

    if response_open_count != response_close_count:
        return False

    # Rule 2: Check for nesting by tracking depth
    depth = 0
    delegate_pattern = re.compile(r'<delegate>|</delegate>')
    for match in delegate_pattern.finditer(trace):
        if match.group() == '<delegate>':
            depth += 1
            if depth > 1:  # Nested opening tag
                return False
        else:
            depth -= 1
            if depth < 0:  # Closing before opening
                return False

    # Rule 2: Check <response> tags for nesting
    depth = 0
    response_pattern = re.compile(r'<response>|</response>')
    for match in response_pattern.finditer(trace):
        if match.group() == '<response>':
            depth += 1
            if depth > 1:  # Nested opening tag
                return False
        else:
            depth -= 1
            if depth < 0:  # Closing before opening
                return False

    # Rule 5: </think> appears at most once
    think_close_count = trace.count('</think>')
    if think_close_count > 1:
        return False

    return True
