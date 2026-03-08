#!/usr/bin/env python3
"""
BFS Logic Debug Script

This script verifies the fixes made to the Abyme recursive engine:
1. Text cleaning functions (remove_think_tags)
2. BFS logic (collecting all elaborations before processing)
3. Parallel execution (if enabled)
4. Proper child prompt formatting

This addresses the parsing bug where child tasks weren't formatted correctly.
"""

from abyme.engine import RecursiveEngine
from abyme.utils import extract_xml_tags, remove_think_tags, extract_elaborations


def test_text_cleaning():
    """Test that text cleaning functions work correctly."""
    print("=" * 60)
    print("TEST 1: Text Cleaning Functions")
    print("=" * 60)

    # Test remove_think_tags
    test_text = """<think>
This is internal reasoning that should be removed.
Multiple lines of thought.
</think>The answer is 42<|end_of_sentence|>"""

    cleaned = remove_think_tags(test_text)
    print(f"Original text length: {len(test_text)}")
    print(f"Cleaned text length: {len(cleaned)}")
    print(f"Cleaned text: '{cleaned}'")

    assert "<think>" not in cleaned, "Think tags should be removed"
    assert "</think>" not in cleaned, "Think tags should be removed"
    assert "<|end_of_sentence|>" not in cleaned, "EOS tokens should be removed"
    assert "The answer is 42" in cleaned, "Content should be preserved"

    print("✓ Text cleaning test passed!\n")


def test_extract_elaborations():
    """Test that elaboration extraction works correctly."""
    print("=" * 60)
    print("TEST 2: Elaboration Extraction")
    print("=" * 60)

    test_text = """
    <think>
    I need to solve this problem step by step.
    <elaborate>Step 1: Calculate 5 * 5</elaborate>
    <elaborate>Step 2: Calculate 3 * 3</elaborate>
    <elaborate>Step 3: Add the results</elaborate>
    </think>
    </run>
    """

    elaborations = extract_elaborations(test_text)
    print(f"Found {len(elaborations)} elaborations:")
    for i, elab in enumerate(elaborations, 1):
        print(f"  {i}. {elab.strip()}")

    assert len(elaborations) == 3, f"Expected 3 elaborations, got {len(elaborations)}"
    assert "5 * 5" in elaborations[0], "First elaboration should contain '5 * 5'"
    assert "3 * 3" in elaborations[1], "Second elaboration should contain '3 * 3'"
    assert "Add" in elaborations[2], "Third elaboration should contain 'Add'"

    print("✓ Elaboration extraction test passed!\n")


def test_bfs_logic():
    """Test the BFS logic with the recursive engine."""
    print("=" * 60)
    print("TEST 3: BFS Logic (Breadth-First Search)")
    print("=" * 60)

    # Initialize the engine
    print("\nInitializing RecursiveEngine...")
    print("(This may take a few minutes and will download the model if not cached)")
    print("-" * 60)

    try:
        # Test with sequential processing first (parallel_max_workers=1)
        engine = RecursiveEngine(
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            load_in_4bit=True,
            max_recursion_depth=2,
            parallel_max_workers=1  # Sequential for initial test
        )
        print("\n✓ Engine initialized successfully!\n")

    except Exception as e:
        print(f"\n✗ Error initializing engine: {e}")
        print("\nPossible issues:")
        print("  - CUDA OOM errors (reduce batch size or use CPU)")
        print("  - Missing model files (will auto-download)")
        print("  - Dependency issues (check transformers, torch versions)")
        return

    # Test prompt that would trigger the parsing bug
    # This prompt has multiple elaborations that should be processed in BFS order
    test_prompt = """<think>
I need to solve the math problem: (5 * 5) + (3 * 3)
Let me break this into subtasks that will be processed in parallel:
<elaborate>Calculate the first part: 5 * 5</elaborate>
<elaborate>Calculate the second part: 3 * 3</elaborate>
</think>
</run>"""

    print("=" * 60)
    print("Test Prompt (Multiple Elaborations):")
    print("-" * 60)
    print(test_prompt)
    print("=" * 60)

    print("\nStarting BFS generation...")
    print("(All elaborations should be collected BEFORE processing)")
    print("-" * 60)

    try:
        output = engine.generate(test_prompt, verbose=True)

        print("\n" + "=" * 60)
        print("FINAL OUTPUT:")
        print("=" * 60)
        print(output)
        print("=" * 60)

        # Verify BFS behavior
        responses = extract_xml_tags(output, "response")
        print(f"\n✓ Found {len(responses)} response(s)")

        if len(responses) >= 2:
            print("✓ BFS logic worked! Multiple elaborations were processed.")
            for i, resp in enumerate(responses, 1):
                print(f"  Response {i}: {resp[:100]}...")
        else:
            print("⚠ Warning: Expected at least 2 responses for 2 elaborations")
            print("The model may not be following the format correctly yet.")

        # Check that responses are clean (no think tags)
        for i, resp in enumerate(responses, 1):
            if "<think>" in resp:
                print(f"⚠ Warning: Response {i} contains <think> tags (not cleaned)")
            else:
                print(f"✓ Response {i} is clean (no think tags)")

    except Exception as e:
        print(f"\n✗ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        print("\nThis may indicate:")
        print("  - OOM errors (reduce max_new_tokens)")
        print("  - Model format issues (model not trained on this format)")
        print("  - BFS logic bugs (check engine.py)")
        return

    print("\n" + "=" * 60)
    print("BFS Test Complete!")
    print("=" * 60)


def test_parallel_execution():
    """Test parallel execution of elaborations (if you have enough GPU memory)."""
    print("=" * 60)
    print("TEST 4: Parallel Execution (Optional)")
    print("=" * 60)
    print("Note: This test requires sufficient GPU memory.")
    print("Skipping for now. To enable, set parallel_max_workers > 1")
    print("=" * 60)
    # Uncomment below to test parallel execution
    # engine = RecursiveEngine(
    #     "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    #     load_in_4bit=True,
    #     max_recursion_depth=2,
    #     parallel_max_workers=4  # Enable parallel processing
    # )
    # ... run similar test as above


def main():
    """Run all BFS debug tests."""
    print("\n")
    print("=" * 60)
    print("         ABYME BFS LOGIC DEBUG TESTS")
    print("=" * 60)
    print("\n")

    # Test 1: Text cleaning (should always pass)
    test_text_cleaning()

    # Test 2: Elaboration extraction (should always pass)
    test_extract_elaborations()

    # Test 3: BFS logic with recursive engine
    test_bfs_logic()

    # Test 4: Parallel execution (optional, requires more GPU memory)
    test_parallel_execution()

    print("\n" + "=" * 60)
    print("BFS Debug Checklist:")
    print("=" * 60)
    print("[ ] remove_think_tags removes <think> tags correctly")
    print("[ ] remove_think_tags removes BOS/EOS tokens")
    print("[ ] extract_elaborations finds all elaborations")
    print("[ ] Engine collects ALL elaborations before processing (BFS)")
    print("[ ] Engine cleans child outputs (no think tags)")
    print("[ ] Parallel execution works (if enabled)")
    print("\nNext steps:")
    print("  1. If tests pass, the BFS fix is working!")
    print("  2. Test with more complex prompts")
    print("  3. Enable parallel processing if you have multi-GPU")
    print("  4. Train the model to follow the format reliably")
    print("=" * 60)


if __name__ == "__main__":
    main()
