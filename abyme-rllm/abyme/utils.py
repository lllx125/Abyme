"""
Utility functions for text processing and XML tag extraction.
"""
import re
from typing import List, Optional

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


def get_format_error(text: str, allow_extra_content: bool = False) -> Optional[str]:
    """
    Returns None if the text is valid, or an error string describing why it failed.
    Follows the same logic as verify_output_format_strict.
    """
    if not text:
        return "empty text"

    text_stripped = text.strip()

    if '<think>' in text_stripped:
        return "has <think> open tag"

    if '<response>' in text_stripped or '</response>' in text_stripped:
        return "has <response> tag"

    if '</think>' in text_stripped:
        if text_stripped.count('</think>') != 1:
            return "answer: multiple </think>"
        think_pos = text_stripped.find('</think>')
        if not text_stripped[think_pos + len('</think>'):].strip():
            return "answer: empty after </think>"
        if '## DO' in text_stripped or '## TRY' in text_stripped or '<do>' in text_stripped or '<try>' in text_stripped:
            return "answer: mixed DO/TRY"
        return None

    if allow_extra_content:
        do_pattern = r'##\s*DO\s+(\d+)\s*\n\s*>\s*[^\n]+\n.*?<do>.*?</do>'
    else:
        do_pattern = r'##\s*DO\s+(\d+)\s*\n\s*>\s*[^\n]+\n\s*<do>.*?</do>'
    do_matches = list(re.finditer(do_pattern, text_stripped, re.DOTALL | re.MULTILINE))

    if do_matches:
        numbers = [int(m.group(1)) for m in do_matches]
        for i in range(1, len(numbers)):
            if numbers[i] != numbers[i-1] + 1:
                return "do: non-consecutive numbers"
        if '<try>' in text_stripped or '</try>' in text_stripped or '</think>' in text_stripped or '## TRY' in text_stripped:
            return "do: mixed TRY/ANSWER"
        if not _check_tag_pairing_and_nesting(text_stripped, 'do'):
            return "do: unpaired/nested tags"
        do_open_count = text_stripped.count('<do>')
        do_close_count = text_stripped.count('</do>')
        if do_open_count != do_close_count or do_open_count != len(do_matches):
            return "do: tag count mismatch"
        do_header_count = len(re.findall(r'##\s*DO\s+\d+', text_stripped))
        if do_header_count != len(do_matches):
            return "do: header count mismatch"
        for match in do_matches:
            block_text = text_stripped[match.start():match.end()]
            if not re.search(r'\n\s*>\s*[^\n]+\n', block_text):
                return "do: missing > description line"
        return None

    if allow_extra_content:
        try_pattern = r'##\s*TRY\s+(\d+)\s*\n\s*>\s*[^\n]+\n.*?<try>.*?</try>'
    else:
        try_pattern = r'##\s*TRY\s+(\d+)\s*\n\s*>\s*[^\n]+\n\s*<try>.*?</try>'
    try_matches = list(re.finditer(try_pattern, text_stripped, re.DOTALL | re.MULTILINE))

    if try_matches:
        if len(try_matches) < 2:
            return "try: only 1 TRY block (need >=2)"
        numbers = [int(m.group(1)) for m in try_matches]
        for i in range(1, len(numbers)):
            if numbers[i] != numbers[i-1] + 1:
                return "try: non-consecutive numbers"
        if '<do>' in text_stripped or '</do>' in text_stripped or '</think>' in text_stripped or '## DO' in text_stripped:
            return "try: mixed DO/ANSWER"
        if not _check_tag_pairing_and_nesting(text_stripped, 'try'):
            return "try: unpaired/nested tags"
        try_open_count = text_stripped.count('<try>')
        try_close_count = text_stripped.count('</try>')
        if try_open_count != try_close_count or try_open_count != len(try_matches):
            return "try: tag count mismatch"
        try_header_count = len(re.findall(r'##\s*TRY\s+\d+', text_stripped))
        if try_header_count != len(try_matches):
            return "try: header count mismatch"
        return None

    has_do = '## DO' in text_stripped or '<do>' in text_stripped or '</do>' in text_stripped
    has_try = '## TRY' in text_stripped or '<try>' in text_stripped or '</try>' in text_stripped
    if has_do:
        return "do: partial/malformed structure"
    if has_try:
        return "try: partial/malformed structure"
    return "no valid format (plain text)"


def verify_output_format_strict(text: str, print_reason: bool = False, allow_extra_content: bool = False) -> bool:
    """
    Verify that the output follows one of three strict format cases: DO, TRY, or ANSWER.

    DO case:
        [reasoning, can be none]
        ## DO 1
        > [Brief description]
        [optional content if allow_extra_content=True]
        <do>
        [Sterile, self-contained prompt]
        </do>
        ## DO 2
        > [Brief description]
        [optional content if allow_extra_content=True]
        <do>
        [Sterile, self-contained prompt]
        </do>
        ... (and more DOs)

    TRY case:
        [reasoning, can be none]
        ## TRY 1
        > [Brief description]
        [optional content if allow_extra_content=True]
        <try>
        [Sterile, self-contained prompt]
        </try>
        ## TRY 2
        > [Brief description]
        [optional content if allow_extra_content=True]
        <try>
        [Sterile, self-contained prompt]
        </try>
        ...(and more TRYs, number of TRYs must be >=2)

    ANSWER case:
        [reasoning]
        </think>
        [Concise final answer]

    Strict Requirements:
    - Cases are mutually exclusive (cannot mix DO with TRY or ANSWER)
    - Numbers must be consecutive (each next one is exactly 1 bigger than previous)
    - Tags cannot be nested, broken, or unpaired
    - <think> opening tag cannot exist (only </think> allowed in ANSWER case)
    - Each ## DO/TRY must be paired with exactly one > and one <do>/<try> block
    - Only reasoning can be empty
    - no <response> tags allowed in any case (reasoning must be outside of any tags)

    Args:
        text: The text to validate
        print_reason: If True, prints the reason when validation fails
        allow_extra_content: If True, allows extra content between '>' line and <do>/<try> tag

    Returns:
        bool: True if the text follows one of the three formats exactly, False otherwise
    """
    error = get_format_error(text, allow_extra_content=allow_extra_content)
    if error is None:
        return True
    if print_reason:
        print(f"Validation failed: {error}")
    return False

