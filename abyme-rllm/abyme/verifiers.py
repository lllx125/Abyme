"""
Robust math verifiers for extracting and validating answers from LLM output.
Handles LaTeX formatting, messy outputs, and AIME-style integer answers.
"""

import re
from typing import Optional


def clean_latex(text: str) -> str:
    """
    Remove generic LaTeX commands and clean formatting.

    Args:
        text: Input string potentially containing LaTeX commands

    Returns:
        Cleaned string with LaTeX commands removed
    """
    if not text:
        return ""

    # Remove \left and \right commands
    text = re.sub(r'\\left', '', text)
    text = re.sub(r'\\right', '', text)

    # Remove dollar signs
    text = text.replace('$', '')

    # Convert \frac{a}{b} to a/b
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', text)

    # Remove common LaTeX commands but keep their content
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\mathbf\{([^}]+)\}', r'\1', text)

    # Remove remaining backslashes for simple commands
    text = re.sub(r'\\([a-zA-Z]+)', r'\1', text)

    # Clean up extra whitespace
    text = ' '.join(text.split())

    return text.strip()


def extract_answer(text: str) -> Optional[str]:
    """
    Extract the answer from LLM output using priority-based regex matching.

    Priority 1: Look for \\boxed{...} pattern
    Priority 2: Extract the last distinct number (integer or float) from text

    Args:
        text: LLM output text

    Returns:
        Extracted answer string or None if no answer found
    """
    if not text:
        return None

    # Priority 1: Search for \boxed{...} pattern with balanced braces
    # Find all occurrences of \boxed{
    boxed_starts = [m.start() for m in re.finditer(r'\\boxed\{', text)]

    if boxed_starts:
        # Extract content from the last \boxed{ with balanced braces
        last_start = boxed_starts[-1]
        start_pos = last_start + len(r'\boxed{')

        # Find the matching closing brace
        brace_count = 1
        pos = start_pos

        while pos < len(text) and brace_count > 0:
            if text[pos] == '{':
                brace_count += 1
            elif text[pos] == '}':
                brace_count -= 1
            pos += 1

        if brace_count == 0:
            # Successfully found matching brace
            answer = text[start_pos:pos-1]
            return clean_latex(answer).strip()

    # Priority 2: Search for the last distinct number
    # Handle integers, floats, and numbers with commas
    number_pattern = r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?'
    number_matches = re.findall(number_pattern, text)

    if number_matches:
        # Get the last number and remove commas
        last_number = number_matches[-1].replace(',', '')
        return last_number.strip()

    return None


def verify_math(model_output: str, ground_truth: str) -> bool:
    """
    Verify if the model output matches the ground truth answer.

    Handles AIME-style integer answers (0-999) with special normalization.

    Args:
        model_output: The LLM's generated response
        ground_truth: The correct answer

    Returns:
        True if answers match, False otherwise
    """
    # Extract answer from model output
    extracted = extract_answer(model_output)

    if extracted is None:
        return False

    # Normalize both answers: strip whitespace, lowercase
    extracted = extracted.strip().lower()
    ground_truth = ground_truth.strip().lower()

    # Direct string match
    if extracted == ground_truth:
        return True

    # AIME special check: If ground_truth is an integer (0-999),
    # ensure the extracted answer matches it exactly as an integer
    try:
        gt_int = int(ground_truth)
        ex_int = int(float(extracted))  # Handle "7.0" -> 7

        # AIME answers are typically 0-999
        if 0 <= gt_int <= 999:
            return gt_int == ex_int

        # For other integers, also check equality
        return gt_int == ex_int

    except (ValueError, TypeError):
        # Not integers, try float comparison
        pass

    # Try float comparison for decimal answers
    try:
        gt_float = float(ground_truth)
        ex_float = float(extracted)

        # Use relative tolerance for floating point comparison
        return abs(gt_float - ex_float) < 1e-6

    except (ValueError, TypeError):
        # Neither integer nor float match
        pass

    # Try cleaning both with LaTeX cleanup
    cleaned_extracted = clean_latex(extracted)
    cleaned_ground_truth = clean_latex(ground_truth)

    if cleaned_extracted == cleaned_ground_truth:
        return True

    # Final attempt: normalize fractions (1/2 vs 0.5)
    try:
        # Check if ground truth is a fraction
        if '/' in cleaned_ground_truth:
            parts = cleaned_ground_truth.split('/')
            if len(parts) == 2:
                gt_frac_value = float(parts[0]) / float(parts[1])
                ex_value = float(cleaned_extracted)
                return abs(gt_frac_value - ex_value) < 1e-6

        # Check if extracted is a fraction
        if '/' in cleaned_extracted:
            parts = cleaned_extracted.split('/')
            if len(parts) == 2:
                ex_frac_value = float(parts[0]) / float(parts[1])
                gt_value = float(cleaned_ground_truth)
                return abs(ex_frac_value - gt_value) < 1e-6

    except (ValueError, ZeroDivisionError):
        pass

    return False
