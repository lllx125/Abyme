"""
Test script for AbymeSFTHuggingFaceModel
"""
import sys

from sft.load_sft_model import AbymeSFTHuggingFaceModel
from abyme.vllm_model import LocalVLLMModel
from abyme.recursive_engine import RecursiveEngine

prompt = "What is the third derivative of e^{x^2}"
prompt = "There are exactly three positive real numbers $ k $ such that the function\n$ f(x) = \\frac{(x - 18)(x - 72)(x - 98)(x - k)}{x} $\ndefined over the positive real numbers achieves its minimum value at exactly two positive real numbers $ x $. Find the sum of these three values of $ k $.\n\nPlease provide your final answer in the format \\boxed{answer}, where answer is an integer between 0 and 999."


def main():
    model_name = "Lixing-Li/Abyme-Trained-Iteration-5"
    model = LocalVLLMModel(model_path=model_name)
    print("Model loaded successfully!")
    engine = RecursiveEngine(
        base_model=model,
        print_progress=True,
        max_call=2000,
        max_depth=12,
        max_workers=50
    )
    output = engine.generate(prompt)
    print(output)


if __name__ == "__main__":
    main()
