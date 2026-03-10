from abyme.core import Abyme_DeepSeek
from abyme.tree_trace import *
import time

prompt = "Find the derivative of f(x) = (x^2 + 1)^(x^3) using logarithmic differentiation, then evaluate f'(1). don't simplify ln()"

def main():
  model = Abyme_DeepSeek(reasoning = False, print_progress=True, max_parallel_workers=5)
  print("model loaded")
  start_time = time.time()
  model.generate(prompt)
  latency = time.time() - start_time
  print("generation complete")
  print("="*50)
  print("total call: ", total_calls(model.trace))
  print("max depth: ", max_depth(model.trace))
  print("max subproblems: ", max_subproblem(model.trace))
  print("parallel latency: ", parallel_latency(model.trace))
  print("actual latency:",latency)

if __name__ == "__main__":
  main()
  
  
