"""
Test script for AbymeSFTHuggingFaceModel

This test loads the fine-tuned Abyme-Llama-3.1-8B model and runs a simple generation test.
"""
import sys

from sft.load_sft_model import AbymeSFTHuggingFaceModel
from abyme.core import RecursiveModel

prompt = "What is the third derivative of e^{x^2}"
#prompt = "There are exactly three positive real numbers $ k $ such that the function\n$ f(x) = \\frac{(x - 18)(x - 72)(x - 98)(x - k)}{x} $\ndefined over the positive real numbers achieves its minimum value at exactly two positive real numbers $ x $. Find the sum of these three values of $ k $.\n\nPlease provide your final answer in the format \\boxed{answer}, where answer is an integer between 0 and 999."
prompt = "What is the least positive integer multiple of 30 that can be written with only the digits 0 and 2?\n\nVerify that the answer is 2220"

def main():
    model = AbymeSFTHuggingFaceModel()
    print("Model loaded successfully!")
    #output = model.generate(prompt)
    abyme = RecursiveModel(base_model=model, print_progress=True)
    output = abyme.generate(prompt)
    print(output)


if __name__ == "__main__":
    main()
