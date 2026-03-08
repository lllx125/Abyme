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


def format_output(text: str, full_response: bool, eos_token: str = None) -> str:
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