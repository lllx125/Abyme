"""
Utility functions for text processing and XML tag extraction.
"""
import re
from typing import List

AND = "do"
OR = "try"
RESPONSE = "response"
THINK = "think"
LEAF = "leaf" # not a tag, denoting node state only

VALID_TAGS = {AND, OR, RESPONSE, THINK}


def extract_delegations(text: str, tag:str = AND) -> List[str]:
    """
    Extract delegate tags for each run

    Args:
        text: The text to search in
        tag: The XML tag to extract

    Returns:
        List of delegation contents (content within tags only)

    """
    # Use regex to find all occurrences of <tag>...</tag>
    pattern = rf'<{tag}>(.*?)</{tag}>'
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
        last_think_end = result.rfind(f"</{THINK}>")
        if last_think_end != -1:
            # Return everything after the final </think> tag
            result = result[last_think_end + len(f"</{THINK}>"):]

    # Remove EOS token if present at the end
    if eos_token and result.endswith(eos_token):
        result = result[:-len(eos_token)]

    return result


def replace_delegations_with_responses(text: str, responses: List[str], tag:str = AND) -> str:
    """
    Replace all <tag>...</tag> tags with corresponding responses from the list.

    This function is extracted from engine.py to be reusable in both the inference engine
    and the SFT data generation pipeline.

    Args:
        text: The text containing <tag>...</tag> tags
        responses: List of response strings to replace delegations with (in order)

    Returns:
        Text with all delegations replaced by <response>...</response> tags

    Raises:
        ValueError: If the number of delegations doesn't match the number of responses
    """
    # First, count the number of delegation tags
    delegations = extract_delegations(text, tag)

    if len(delegations) != len(responses):
        raise ValueError(
            f"Number of delegations ({len(delegations)}) doesn't match "
            f"number of responses ({len(responses)})"
        )

    # Replace each delegation tag with its corresponding response
    def replace_delegation(match):
        idx = replace_delegation.counter
        replace_delegation.counter += 1
        return f'<{RESPONSE}>{responses[idx]}\n</{RESPONSE}>'

    replace_delegation.counter = 0
    result = re.sub(rf'<{tag}>.*?</{tag}>', replace_delegation, text, flags=re.DOTALL)

    return result

def default_formatter(prompt: str, main_problem: str = "None", boss_problem: str = "None",fragment: str = "None") -> str:
    """
    Default formatter that combines prompt, main_problem, boss_problem, and fragment.

    Args:
        prompt: The input prompt string
        main_problem: The main problem string
        boss_problem: The boss problem string
        fragment: previous answer fragment

    Returns:
        Combined string of prompt, main_problem, boss_problem, and fragment
    """
    return f"Prompt: {prompt}\nMain Problem: {main_problem}\nBoss Problem: {boss_problem}\nFragment: {fragment}"

def _check_tag_pairing_and_nesting(trace: str, tag: str) -> bool:
    """
    Helper function to verify that tags are properly paired and not nested.

    Args:
        trace: The text to check
        tag: The tag name (e.g., 'do', 'try', 'response')

    Returns:
        bool: True if tags are properly paired and not nested, False otherwise
    """
    open_tag = f'<{tag}>'
    close_tag = f'</{tag}>'

    # Check tags are properly paired
    open_count = trace.count(open_tag)
    close_count = trace.count(close_tag)

    if open_count != close_count:
        return False

    # Check for nesting by tracking depth
    depth = 0
    tag_pattern = re.compile(rf'<{tag}>|</{tag}>')
    for match in tag_pattern.finditer(trace):
        if match.group() == open_tag:
            depth += 1
            if depth > 1:  # Nested opening tag
                return False
        else:
            depth -= 1
            if depth < 0:  # Closing before opening
                return False

    return True


def verify_format(trace: str) -> bool:
    """
    Verify that the generated output has valid XML-like tag structure.

    This function is used to validate model-generated output when it is produced,
    ensuring it follows the proper format for the Abyme recursive reasoning system.

    **Validation Rules:**

    1. **No Incomplete or Broken Tags**:
       - All tags must be complete (only valid tags allowed)
       - Detects partial or malformed tag structures

    2. **No Opening <THINK> Tag**:
       - Only the closing tag </think> is allowed
       - No <think> opening tag should exist in the output

    3. **RESPONSE, AND and OR Tags are Paired and Not Nested**:
       - Every opening tag must have a matching closing tag
       - Tags cannot be nested within themselves
       - Opening and closing counts must match exactly
       - Closing tags cannot appear before their corresponding opening tags

    4. **THINK Close Tag Appears At Most Once**:
       - The closing tag </think> must appear at most once

    5. **No Valid Tags After THINK Close Tag**:
       - No valid tags (AND, OR, RESPONSE, THINK) should appear after </think>

    6. **Only One of AND, OR, or THINK Can Appear**:
       - At most one of AND, OR, or THINK tags can be present in the trace
       - They are mutually exclusive

    Args:
        trace: The generated output text to validate

    Returns:
        bool: True if all validation rules pass, False if any rule is violated

    Examples:
        Valid outputs:
        >>> verify_format("Solution<do>sub-problem</do>")
        True
        >>> verify_format("Solution<response>answer</response>")
        True
        >>> verify_format("Final answer: 42</think>")
        True

        Invalid outputs:
        >>> verify_format("Test<do>nested<do>bad</do></do>")
        False
        >>> verify_format("Both<do>test</do>and</think>")  # Both AND and THINK
        False
        >>> verify_format("Both<do>test</do>and<try>other</try>")  # Both AND and OR
        False
        >>> verify_format("Answer</think>extra content")  # Content after </think>
        False
        >>> verify_format("Answer</think></think>")  # Multiple </think>
        False
    """
    if not trace:
        return True

    # Rule 2: No <think> opening tag allowed
    if f'<{THINK}>' in trace:
        return False

    # Rule 1: Check for incomplete tags - only valid tags are allowed

    # Rule 3: Check RESPONSE, AND, and OR tags are properly paired and not nested
    for tag in [RESPONSE, AND, OR]:
        if not _check_tag_pairing_and_nesting(trace, tag):
            return False

    # Rule 4: </think> must appear at most once
    think_close_count = trace.count(f'</{THINK}>')
    if think_close_count > 1:
        return False

    # Rule 5: No valid tags after </think> close tag
    think_close_pos = trace.rfind(f'</{THINK}>')
    if think_close_pos != -1:
        content_after_think = trace[think_close_pos + len(f'</{THINK}>'):].strip()
        # Check if there are any valid tags in the content after </think>
        for tag in VALID_TAGS:
            if f'<{tag}>' in content_after_think or f'</{tag}>' in content_after_think:
                return False

    # Rule 6: Only one of AND, OR, or THINK can appear at a time
    has_think = think_close_count > 0
    has_and = f'<{AND}>' in trace or f'</{AND}>' in trace
    has_or = f'<{OR}>' in trace or f'</{OR}>' in trace

    # Count how many of them are present
    present_count = sum([has_think, has_and, has_or])
    if present_count > 1:
        return False

    return True
