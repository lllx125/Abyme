#!/usr/bin/env python3
"""
Test suite for math verifiers.

Tests regex logic for extracting answers from various LaTeX formats,
including edge cases and AIME-style integer answers.

Usage:
    python tests/test_verifiers.py
    pytest tests/test_verifiers.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from abyme.verifiers import clean_latex, extract_answer, verify_math


def test_case_1_standard_box():
    """Test Case 1: Standard boxed answer."""
    model_output = "The answer is \\boxed{42}"
    ground_truth = "42"
    result = verify_math(model_output, ground_truth)
    assert result, f"Failed: Standard boxed answer. Expected True, got {result}"
    print("✓ Test Case 1 PASSED: Standard boxed answer")


def test_case_2_latex_noise():
    """Test Case 2: LaTeX noise with fraction."""
    # Test with integer
    model_output = "The value is \\boxed{100}"
    ground_truth = "100"
    result = verify_math(model_output, ground_truth)
    assert result, f"Failed: LaTeX noise (integer). Expected True, got {result}"
    print("✓ Test Case 2a PASSED: LaTeX noise with integer")

    # Test with fraction
    model_output = "The answer is \\boxed{\\frac{1}{2}}"
    ground_truth = "1/2"
    result = verify_math(model_output, ground_truth)
    assert result, f"Failed: LaTeX noise (fraction). Expected True, got {result}"
    print("✓ Test Case 2b PASSED: LaTeX noise with fraction")


def test_case_3_messy_output():
    """Test Case 3: Messy output with multiple text."""
    model_output = "I think x = 5. Therefore \\boxed{5}"
    ground_truth = "5"
    result = verify_math(model_output, ground_truth)
    assert result, f"Failed: Messy output. Expected True, got {result}"
    print("✓ Test Case 3 PASSED: Messy output with reasoning")


def test_case_4_aime_integer():
    """Test Case 4: AIME-style integer with leading zeros."""
    model_output = "The answer is 007"
    ground_truth = "7"
    result = verify_math(model_output, ground_truth)
    assert result, f"Failed: AIME integer. Expected True, got {result}"
    print("✓ Test Case 4 PASSED: AIME integer with leading zeros")


def test_case_5_failure():
    """Test Case 5: Incorrect answer should fail."""
    model_output = "The answer is 5"
    ground_truth = "50"
    result = verify_math(model_output, ground_truth)
    assert not result, f"Failed: Should return False for wrong answer. Expected False, got {result}"
    print("✓ Test Case 5 PASSED: Correctly identifies wrong answer")


def test_clean_latex():
    """Test LaTeX cleaning function."""
    # Test removing \left and \right
    assert "( 1 + 2 )" == clean_latex(r"\left( 1 + 2 \right)")
    print("✓ PASSED: Clean \\left and \\right")

    # Test fraction conversion
    assert "1/2" == clean_latex(r"\frac{1}{2}")
    print("✓ PASSED: Convert fraction to 1/2")

    # Test dollar sign removal
    assert "x = 5" == clean_latex("$x = 5$")
    print("✓ PASSED: Remove dollar signs")


def test_extract_answer():
    """Test answer extraction function."""
    # Test boxed extraction
    assert "42" == extract_answer("The answer is \\boxed{42}")
    print("✓ PASSED: Extract from \\boxed{}")

    # Test multiple boxed (should get last)
    assert "100" == extract_answer("First \\boxed{50}, then \\boxed{100}")
    print("✓ PASSED: Extract last \\boxed{}")

    # Test number extraction (no boxed)
    assert "123" == extract_answer("The value is 123 here")
    print("✓ PASSED: Extract number without \\boxed{}")

    # Test comma in number
    assert "1200" == extract_answer("The population is 1,200 people")
    print("✓ PASSED: Handle comma in number")

    # Test no answer
    assert extract_answer("No answer here") is None  # Should return None when no number present
    print("✓ PASSED: Handle text with no answer")


def test_advanced_cases():
    """Test advanced edge cases."""
    # Nested braces in boxed
    model_output = "The answer is \\boxed{\\frac{3}{4}}"
    ground_truth = "3/4"
    result = verify_math(model_output, ground_truth)
    assert result, "Failed: Nested braces in boxed"
    print("✓ PASSED: Nested braces in boxed")

    # Float comparison
    model_output = "\\boxed{0.5}"
    ground_truth = "1/2"
    result = verify_math(model_output, ground_truth)
    assert result, "Failed: Float vs fraction comparison"
    print("✓ PASSED: Float vs fraction comparison")

    # Negative numbers
    model_output = "\\boxed{-5}"
    ground_truth = "-5"
    result = verify_math(model_output, ground_truth)
    assert result, "Failed: Negative number"
    print("✓ PASSED: Negative number")

    # Large AIME-style answer
    model_output = "The answer is \\boxed{999}"
    ground_truth = "999"
    result = verify_math(model_output, ground_truth)
    assert result, "Failed: Large AIME answer"
    print("✓ PASSED: Large AIME answer (999)")


def run_all_tests():
    """Run all test cases."""
    print("\n" + "="*60)
    print("Running Verifier Test Suite")
    print("="*60 + "\n")

    try:
        print("Testing clean_latex()...")
        test_clean_latex()
        print()

        print("Testing extract_answer()...")
        test_extract_answer()
        print()

        print("Testing verify_math() - Basic Cases...")
        test_case_1_standard_box()
        test_case_2_latex_noise()
        test_case_3_messy_output()
        test_case_4_aime_integer()
        test_case_5_failure()
        print()

        print("Testing verify_math() - Advanced Cases...")
        test_advanced_cases()
        print()

        print("="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60 + "\n")
        return True

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
