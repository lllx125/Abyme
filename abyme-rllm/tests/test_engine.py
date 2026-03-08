#!/usr/bin/env python3
"""
Day 1 Verification Script

This script tests the core functionality of the Abyme recursive engine:
1. Model and tokenizer loading with special tokens
2. Stopping at </run> tokens
3. XML tag extraction
4. Recursive elaboration processing

Since the model isn't trained yet, we use a manually injected prompt
to force the recursive behavior.
"""

from abyme.engine import RLLMEngine


def test_recursive_engine():
    try:
        engine = RLLMEngine(
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            load_in_4bit=True,
            max_recursion_depth=1
        )
        print("\n Engine initialized successfully!\n")

    except Exception as e:
        print(f"\n Error initializing engine: {e}")
        return

    # Test prompt with few-shot examples to teach the model the tag format
    # The examples demonstrate:
    # 1. How to use <think> for reasoning
    # 2. How to use <elaborate> for subtasks
    # 3. How to end with </run>
    test_prompt = """
Now you:
Question: Calculate (5 * 5) + (3 * 3)
<think>
I need to calculate (5 * 5) + (3 * 3).
<elaborate> Calculate 5 * 5 <think></elaborate>
<elaborate> Calculate 3 * 3 <think></elaborate>
</run>"""

    print("=" * 60)
    print("Test Prompt:")
    print("-" * 60)
    print(test_prompt)
    print("=" * 60)

    try:
        output = engine.generate(test_prompt, print_responses=True)

        print("\n" + "=" * 60)
        print("FINAL OUTPUT:")
        print("=" * 60)
        print(output)
        print("=" * 60)

    except Exception as e:
        print(f"\n Error during generation: {e}")
        return


if __name__ == "__main__":
    test_recursive_engine()

'''
============================================================
Test Prompt:
------------------------------------------------------------

Now you:
Question: Calculate (5 * 5) + (3 * 3)
<think>
I need to calculate (5 * 5) + (3 * 3).
<elaborate> Calculate 5 * 5 <think></elaborate>
<elaborate> Calculate 3 * 3 <think></elaborate>
</run>
============================================================

==================================================

Now you:
Question: Calculate (5 * 5) + (3 * 3)
<think>
I need to calculate (5 * 5) + (3 * 3).
<elaborate> Calculate 5 * 5 <think></elaborate>
<elaborate> Calculate 3 * 3 <think></elaborate>
</run>
==================================================


==================================================
 Calculate 5 * 5 <think>
First, I recognize that I need to calculate the product of 5 and 5.

Next, I'll multiply the two numbers together, which gives me 25.

Therefore, the final result of 5 multiplied by 5 is 25.
</think>

**Solution:**

To calculate \(5 \times 5\), follow these simple steps:

1. **Multiply the numbers:**
   
   \[
   5 \times 5 = 5 + 5 + 5 + 5 + 5
   \]
   
2. **Add the numbers together:**
   
   \[
   5 + 5 = 10 \\
   10 + 5 = 15 \\
   15 + 5 = 20 \\
   20 + 5 = 25
   \]
   
3. **Final Answer:**
   
   \[
   5 \times 5 = \boxed{25}
   \]<｜end▁of▁sentence｜>
==================================================


==================================================
 Calculate 3 * 3 <think>
First, I need to understand the problem. It seems like there are two multiplication operations: 3 multiplied by 3 and then the result multiplied by 3 again.

I'll start by calculating the first multiplication. Multiplying 3 by 3 gives me 9.

Next, I'll take that result and multiply it by 3. So, 9 multiplied by 3 equals 27.

Therefore, the final answer is 27.
</think>

Sure! Let's solve the problem step by step.

**Problem:** Calculate \( 3 \times 3 \).

**Solution:**

1. **First Multiplication:**
   \[
   3 \times 3 = 9
   \]

2. **Second Multiplication (if applicable):**
   If there are multiple multiplication operations, perform them sequentially. For example, if the problem is \( 3 \times 3 \times 3 \):
   \[
   3 \times 3 = 9
   \]
   \[
   9 \times 3 = 27
   \]

**Final Answer:**
\[
\boxed{27}
\]<｜end▁of▁sentence｜>
==================================================


============================================================
FINAL OUTPUT:
============================================================

Now you:
Question: Calculate (5 * 5) + (3 * 3)
<think>
I need to calculate (5 * 5) + (3 * 3).
<response>

**Solution:**

To calculate \(5 \times 5\), follow these simple steps:

1. **Multiply the numbers:**
   
   \[
   5 \times 5 = 5 + 5 + 5 + 5 + 5
   \]
   
2. **Add the numbers together:**
   
   \[
   5 + 5 = 10 \\
   10 + 5 = 15 \\
   15 + 5 = 20 \\
   20 + 5 = 25
   \]
   
3. **Final Answer:**
   
   \[
   5 \times 5 = \boxed{25}
   \]</response>
<response>

Sure! Let's solve the problem step by step.

**Problem:** Calculate \( 3 \times 3 \).

**Solution:**

1. **First Multiplication:**
   \[
   3 \times 3 = 9
   \]

2. **Second Multiplication (if applicable):**
   If there are multiple multiplication operations, perform them sequentially. For example, if the problem is \( 3 \times 3 \times 3 \):
   \[
   3 \times 3 = 9
   \]
   \[
   9 \times 3 = 27
   \]

**Final Answer:**
\[
\boxed{27}
\]</response>
**Final Answer**
\[
\boxed{27}
\]
'''
