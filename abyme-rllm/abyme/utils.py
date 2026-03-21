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
    if not text:
        if print_reason:
            print("Validation failed: Text is empty")
        return False

    text_stripped = text.strip()

    # <think> opening tag cannot exist anywhere
    if '<think>' in text_stripped:
        if print_reason:
            print("Validation failed: <think> opening tag is not allowed (only </think> allowed in ANSWER case)")
        return False

    # No <response> tags allowed in any case
    if '<response>' in text_stripped or '</response>' in text_stripped:
        if print_reason:
            print("Validation failed: <response> tags are not allowed in any case")
        return False

    # Check for ANSWER case (contains </think>)
    if '</think>' in text_stripped:
        # </think> must appear exactly once
        if text_stripped.count('</think>') != 1:
            if print_reason:
                print(f"Validation failed: ANSWER case must have exactly one </think> tag (found {text_stripped.count('</think>')})")
            return False

        # Find position of </think>
        think_pos = text_stripped.find('</think>')

        # Content after </think> must exist and be non-empty
        final_answer = text_stripped[think_pos + len('</think>'):].strip()
        if not final_answer:
            if print_reason:
                print("Validation failed: ANSWER case must have non-empty content after </think>")
            return False

        # ANSWER case cannot contain DO/TRY patterns
        if '## DO' in text_stripped or '## TRY' in text_stripped:
            if print_reason:
                print("Validation failed: ANSWER case cannot contain ## DO or ## TRY headers")
            return False

        # ANSWER case cannot contain <do>/<try> tags
        if '<do>' in text_stripped or '</do>' in text_stripped:
            if print_reason:
                print("Validation failed: ANSWER case cannot contain <do> tags")
            return False
        if '<try>' in text_stripped or '</try>' in text_stripped:
            if print_reason:
                print("Validation failed: ANSWER case cannot contain <try> tags")
            return False

        return True

    # Check for DO case
    # Pattern: ## DO <number>\n> <description>\n[optional content]\n<do>...</do>
    if allow_extra_content:
        # Allow any content between > line and <do> tag
        do_pattern = r'##\s*DO\s+(\d+)\s*\n\s*>\s*[^\n]+\n.*?<do>.*?</do>'
    else:
        # Strict: only whitespace between > line and <do> tag
        do_pattern = r'##\s*DO\s+(\d+)\s*\n\s*>\s*[^\n]+\n\s*<do>.*?</do>'
    do_matches = list(re.finditer(do_pattern, text_stripped, re.DOTALL | re.MULTILINE))

    if do_matches:
        # Extract numbers and verify they're consecutive
        numbers = [int(m.group(1)) for m in do_matches]
        for i in range(1, len(numbers)):
            if numbers[i] != numbers[i-1] + 1:
                if print_reason:
                    print(f"Validation failed: DO numbers must be consecutive (found {numbers[i-1]} followed by {numbers[i]})")
                return False

        # Verify no TRY blocks or </think> tags (cases are mutually exclusive)
        if '<try>' in text_stripped or '</try>' in text_stripped or '</think>' in text_stripped:
            if print_reason:
                print("Validation failed: DO case cannot contain <try> tags or </think> (cases are mutually exclusive)")
            return False

        # Verify no ## TRY headers
        if '## TRY' in text_stripped:
            if print_reason:
                print("Validation failed: DO case cannot contain ## TRY headers (cases are mutually exclusive)")
            return False

        # Verify all <do> tags are properly paired and not nested
        if not _check_tag_pairing_and_nesting(text_stripped, 'do'):
            if print_reason:
                print("Validation failed: <do> tags are not properly paired or are nested")
            return False

        # Verify count: all <do> tags must be accounted for in pattern matches
        do_open_count = text_stripped.count('<do>')
        do_close_count = text_stripped.count('</do>')
        if do_open_count != do_close_count or do_open_count != len(do_matches):
            if print_reason:
                print(f"Validation failed: Found {do_open_count} <do> and {do_close_count} </do> tags, but only {len(do_matches)} complete DO blocks")
            return False

        # Verify all ## DO headers have corresponding complete matches
        do_header_count = len(re.findall(r'##\s*DO\s+\d+', text_stripped))
        if do_header_count != len(do_matches):
            if print_reason:
                print(f"Validation failed: Found {do_header_count} ## DO headers but only {len(do_matches)} complete DO blocks")
            return False

        # Verify each ## DO has exactly one > following it
        for match in do_matches:
            # Extract text from ## DO to end of <do></do> block
            block_text = text_stripped[match.start():match.end()]
            # Check the > on its own line exists
            if not re.search(r'\n\s*>\s*[^\n]+\n', block_text):
                if print_reason:
                    print("Validation failed: Each ## DO must have exactly one > line with description")
                return False

        return True

    # Check for TRY case
    # Pattern: ## TRY <number>\n> <description>\n[optional content]\n<try>...</try>
    if allow_extra_content:
        # Allow any content between > line and <try> tag
        try_pattern = r'##\s*TRY\s+(\d+)\s*\n\s*>\s*[^\n]+\n.*?<try>.*?</try>'
    else:
        # Strict: only whitespace between > line and <try> tag
        try_pattern = r'##\s*TRY\s+(\d+)\s*\n\s*>\s*[^\n]+\n\s*<try>.*?</try>'
    try_matches = list(re.finditer(try_pattern, text_stripped, re.DOTALL | re.MULTILINE))

    if try_matches:
        # Must have at least 2 TRYs
        if len(try_matches) < 2:
            if print_reason:
                print(f"Validation failed: TRY case must have at least 2 TRY blocks (found {len(try_matches)})")
            return False

        # Extract numbers and verify they're consecutive
        numbers = [int(m.group(1)) for m in try_matches]
        for i in range(1, len(numbers)):
            if numbers[i] != numbers[i-1] + 1:
                if print_reason:
                    print(f"Validation failed: TRY numbers must be consecutive (found {numbers[i-1]} followed by {numbers[i]})")
                return False

        # Verify no DO blocks or </think> tags (cases are mutually exclusive)
        if '<do>' in text_stripped or '</do>' in text_stripped or '</think>' in text_stripped:
            if print_reason:
                print("Validation failed: TRY case cannot contain <do> tags or </think> (cases are mutually exclusive)")
            return False

        # Verify no ## DO headers
        if '## DO' in text_stripped:
            if print_reason:
                print("Validation failed: TRY case cannot contain ## DO headers (cases are mutually exclusive)")
            return False

        # Verify all <try> tags are properly paired and not nested
        if not _check_tag_pairing_and_nesting(text_stripped, 'try'):
            if print_reason:
                print("Validation failed: <try> tags are not properly paired or are nested")
            return False

        # Verify count: all <try> tags must be accounted for in pattern matches
        try_open_count = text_stripped.count('<try>')
        try_close_count = text_stripped.count('</try>')
        if try_open_count != try_close_count or try_open_count != len(try_matches):
            if print_reason:
                print(f"Validation failed: Found {try_open_count} <try> and {try_close_count} </try> tags, but only {len(try_matches)} complete TRY blocks")
            return False

        # Verify all ## TRY headers have corresponding complete matches
        try_header_count = len(re.findall(r'##\s*TRY\s+\d+', text_stripped))
        if try_header_count != len(try_matches):
            if print_reason:
                print(f"Validation failed: Found {try_header_count} ## TRY headers but only {len(try_matches)} complete TRY blocks")
            return False

        return True

    # If none of the formats match, provide detailed diagnostics
    if print_reason:
        # Check if there are partial DO structures
        has_do_headers = '## DO' in text_stripped
        has_do_tags = '<do>' in text_stripped or '</do>' in text_stripped
        has_try_headers = '## TRY' in text_stripped
        has_try_tags = '<try>' in text_stripped or '</try>' in text_stripped

        if has_do_headers and has_do_tags:
            do_header_count = len(re.findall(r'##\s*DO\s+\d+', text_stripped))
            do_open_count = text_stripped.count('<do>')
            do_close_count = text_stripped.count('</do>')
            print(f"Validation failed: Found {do_header_count} ## DO headers and {do_open_count}/{do_close_count} <do></do> tags, but format is incorrect.")
            if not allow_extra_content:
                print("  Hint: Try setting allow_extra_content=True if there's content between '>' and '<do>' tag")
            print(f"  Expected format: ## DO N\\n> description\\n<do>content</do>")
        elif has_try_headers and has_try_tags:
            try_header_count = len(re.findall(r'##\s*TRY\s+\d+', text_stripped))
            try_open_count = text_stripped.count('<try>')
            try_close_count = text_stripped.count('</try>')
            print(f"Validation failed: Found {try_header_count} ## TRY headers and {try_open_count}/{try_close_count} <try></try> tags, but format is incorrect.")
            if not allow_extra_content:
                print("  Hint: Try setting allow_extra_content=True if there's content between '>' and '<try>' tag")
            print(f"  Expected format: ## TRY N\\n> description\\n<try>content</try>")
        elif has_do_headers or has_do_tags:
            print("Validation failed: Partial DO structure detected but incomplete or malformed.")
            print(f"  Has ## DO headers: {has_do_headers}")
            print(f"  Has <do> tags: {has_do_tags}")
        elif has_try_headers or has_try_tags:
            print("Validation failed: Partial TRY structure detected but incomplete or malformed.")
            print(f"  Has ## TRY headers: {has_try_headers}")
            print(f"  Has <try> tags: {has_try_tags}")
        else:
            print("Validation failed: Text does not match any of the three required formats (DO, TRY, or ANSWER)")
            print("  Expected formats:")
            print("  - ANSWER: [reasoning]</think>\\n[answer]")
            print("  - DO: [reasoning]## DO 1\\n> desc\\n<do>content</do>...")
            print("  - TRY: [reasoning]## TRY 1\\n> desc\\n<try>content</try>... (minimum 2 TRYs)")
    return False

