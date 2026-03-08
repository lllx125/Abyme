#!/usr/bin/env python3
"""
Quick unit tests for utility functions without requiring model loading.
This tests the BFS fixes independently of the full engine.
"""

import sys
import os

# Add parent directory to path to import abyme
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from abyme.utils import (
    extract_xml_tags,
    remove_think_tags,
    extract_elaborations,
    extract_paired_elaborations
)


def test_remove_think_tags():
    """Test that remove_think_tags works correctly."""
    print("=" * 60)
    print("TEST: remove_think_tags")
    print("=" * 60)

    # Test 1: Basic think tag removal
    test_text1 = "<think>internal reasoning</think>The answer is 42"
    cleaned1 = remove_think_tags(test_text1)
    print(f"Input:  '{test_text1}'")
    print(f"Output: '{cleaned1}'")
    assert cleaned1 == "The answer is 42", f"Expected 'The answer is 42', got '{cleaned1}'"
    print("✓ Basic think tag removal works\n")

    # Test 2: Multiline think tags
    test_text2 = """<think>
This is
multiline
reasoning
</think>Final answer"""
    cleaned2 = remove_think_tags(test_text2)
    print(f"Input:  (multiline with think tags)")
    print(f"Output: '{cleaned2}'")
    assert "<think>" not in cleaned2, "Think tags should be removed"
    assert "Final answer" in cleaned2, "Content should be preserved"
    print("✓ Multiline think tag removal works\n")

    # Test 3: BOS/EOS token removal
    test_text3 = "Answer<|end_of_sentence|>"
    cleaned3 = remove_think_tags(test_text3)
    print(f"Input:  '{test_text3}'")
    print(f"Output: '{cleaned3}'")
    assert "<|end_of_sentence|>" not in cleaned3, "EOS tokens should be removed"
    assert cleaned3 == "Answer", f"Expected 'Answer', got '{cleaned3}'"
    print("✓ BOS/EOS token removal works\n")

    # Test 4: Multiple special tokens
    test_text4 = "<s><think>reasoning</think>Result</s><|end_of_text|>"
    cleaned4 = remove_think_tags(test_text4)
    print(f"Input:  '{test_text4}'")
    print(f"Output: '{cleaned4}'")
    assert "<s>" not in cleaned4, "<s> should be removed"
    assert "</s>" not in cleaned4, "</s> should be removed"
    assert "<think>" not in cleaned4, "<think> should be removed"
    assert "Result" in cleaned4, "Content should be preserved"
    print("✓ Multiple special token removal works\n")

    print("=" * 60)
    print("✓ ALL remove_think_tags TESTS PASSED")
    print("=" * 60)
    print()


def test_extract_elaborations():
    """Test that extract_elaborations works correctly."""
    print("=" * 60)
    print("TEST: extract_elaborations")
    print("=" * 60)

    test_text = """<think>
Let me solve this step by step:
<elaborate>Calculate 5 * 5</elaborate>
<elaborate>Calculate 3 * 3</elaborate>
<elaborate>Add the results</elaborate>
</think>"""

    elaborations = extract_elaborations(test_text)
    print(f"Found {len(elaborations)} elaborations:")
    for i, elab in enumerate(elaborations, 1):
        print(f"  {i}. {elab.strip()}")

    assert len(elaborations) == 3, f"Expected 3, got {len(elaborations)}"
    assert "5 * 5" in elaborations[0], f"Expected '5 * 5' in first elaboration"
    assert "3 * 3" in elaborations[1], f"Expected '3 * 3' in second elaboration"
    assert "Add" in elaborations[2], f"Expected 'Add' in third elaboration"

    print("\n=" * 60)
    print("✓ ALL extract_elaborations TESTS PASSED")
    print("=" * 60)
    print()


def test_extract_paired_elaborations():
    """Test that extract_paired_elaborations works correctly."""
    print("=" * 60)
    print("TEST: extract_paired_elaborations")
    print("=" * 60)

    # Test 1: With responses
    test_text1 = """
<elaborate>Task A</elaborate>
<response>Result A</response>
<elaborate>Task B</elaborate>
<response>Result B</response>
"""
    pairs1 = extract_paired_elaborations(test_text1)
    print(f"Test 1: {len(pairs1)} pairs with responses")
    for i, (elab, resp) in enumerate(pairs1, 1):
        print(f"  {i}. Elab: '{elab.strip()}', Resp: '{resp.strip() if resp else None}'")

    assert len(pairs1) == 2, f"Expected 2 pairs, got {len(pairs1)}"
    assert pairs1[0][0].strip() == "Task A", "First elaboration should be 'Task A'"
    assert pairs1[0][1].strip() == "Result A", "First response should be 'Result A'"
    print("✓ Paired elaborations with responses work\n")

    # Test 2: Without responses (unprocessed)
    test_text2 = """
<elaborate>Task A</elaborate>
<elaborate>Task B</elaborate>
"""
    pairs2 = extract_paired_elaborations(test_text2)
    print(f"Test 2: {len(pairs2)} pairs without responses")
    for i, (elab, resp) in enumerate(pairs2, 1):
        print(f"  {i}. Elab: '{elab.strip()}', Resp: {resp}")

    assert len(pairs2) == 2, f"Expected 2 pairs, got {len(pairs2)}"
    assert pairs2[0][1] is None, "First response should be None"
    assert pairs2[1][1] is None, "Second response should be None"
    print("✓ Unpaired elaborations work\n")

    print("=" * 60)
    print("✓ ALL extract_paired_elaborations TESTS PASSED")
    print("=" * 60)
    print()


def main():
    """Run all unit tests."""
    print("\n")
    print("=" * 60)
    print("    ABYME UTILITY FUNCTIONS UNIT TESTS")
    print("=" * 60)
    print("\n")

    try:
        test_remove_think_tags()
        test_extract_elaborations()
        test_extract_paired_elaborations()

        print("\n")
        print("=" * 60)
        print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
        print("=" * 60)
        print("\nThe BFS fixes are working correctly!")
        print("\nNext steps:")
        print("  1. Install dependencies (pip install -e . in a venv)")
        print("  2. Run full test with model (scripts/debug_bfs.py)")
        print("  3. Verify BFS logic with actual model generation")
        print("=" * 60)

    except AssertionError as e:
        print("\n")
        print("=" * 60)
        print("✗✗✗ TEST FAILED ✗✗✗")
        print("=" * 60)
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print("\n")
        print("=" * 60)
        print("✗✗✗ UNEXPECTED ERROR ✗✗✗")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
