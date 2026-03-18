import sys
import os

# 1. Get the absolute path to the directory containing 'main.py'
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the path to the parent directory (project_root)
parent_dir = os.path.dirname(current_dir)

# 3. Add the parent directory to sys.path
sys.path.append(parent_dir)

from abyme.core import *
from abyme.tree_trace import *
import time

prompt = r"Each vertex of a regular octagon is independently colored either red or blue with equal probability. The probability that the octagon can then be rotated so that all of the blue vertices end up at positions where there were originally red vertices is $\\tfrac{m}{n}$, where $m$ and $n$ are relatively prime positive integers. What is $m+n$?\n\nPlease provide your final answer in the format \\boxed{answer}, where answer is an integer between 0 and 999. You have no access to coding so you may not use any brute force methods"
#prompt = r"Let $ABCDE$ be a convex pentagon with $AB=14, BC=7, CD=24, DE=13, EA=26,$ and $\\angle B=\\angle E=60^\\circ$. For each point $X$ in the plane, define $f(X)=AX+BX+CX+DX+EX$. The least possible value of $f(X)$ can be expressed as $m+n\\sqrt{p}$, where $m$ and $n$ are positive integers and $p$ is not divisible by the square of any prime. Find $m+n+p$.\n\nPlease provide your final answer in the format \\boxed{answer}, where answer is an integer between 0 and 999.  You have no access to coding so you may not use any brute force methods"
#149
#prompt = "Find the derivative of f(x) = (x^2 + 1)^(x^3) using logarithmic differentiation, then evaluate f'(1). leave answer with ln()"


def main():
    # Changed Abyme_DeepSeek to Abyme_API_Models to match your function definition
    model = Abyme_API_Models(
        model="deepseek", 
        print_progress=True, 
        max_parallel_workers=10, 
        max_depth=7, 
        max_chain_length=5,
        max_call=3000
    )
    
    print("Model initialized and worker pool ready.")
    
    start_time = time.time()
    
    # Run the recursive multi-threaded generation
    final_result = model.generate(prompt)
    
    latency = time.time() - start_time
    
    print("\n" + "="*50)
    print("GENERATION COMPLETE")
    print("="*50)
    
    print(f"Final Answer: {final_result}")
    
    # Metrics extracted via the TreeTraceNode utility functions
    total_calls_count = total_calls(model.trace)
    
    print(f"Total LLM Calls:       {total_calls_count}")
    print(f"Max Tree Depth:        {max_depth(model.trace)}")
    print(f"Max Subproblems:       {max_subproblems(model.trace)}")
    print(f"Max Output Characters: {max_output_character(model.trace)}")
    
    # Latency comparisons to prove the parallel worker pool is functioning
    theoretical_parallel = parallel_latency(model.trace)
    print(f"Theoretical Par. Time: {theoretical_parallel:.2f}s")
    print(f"Actual Wall Latency:   {latency:.2f}s")
    print(f"Average Call Latency:  {latency / total_calls_count:.2f}s")
    
    print(f"Nodes Per Level:       {nodes_per_level(model.trace)}")

if __name__ == "__main__":
    main()
  
  
  
#   def solve_mti_1500():
#     # Initial state vector V_0
#     V = [7, 12, 19, 24, 33, 42, 51, 60, 67, 81]
    
#     # Starting constant C_1
#     C = 3
    
#     # Iterate through 1,500 steps
#     for n in range(1, 1501):
#         V_next = [0] * 10
#         for i in range(10):
#             # Apply the update rule: 
#             # v_{n, i} = (v_{n-1, i} + v_{n-1, (i+1) % 10} * C_n) % 100
#             V_next[i] = (V[i] + V[(i + 1) % 10] * C) % 100
        
#         # Update current vector for the next iteration
#         V = V_next
        
#         # Calculate C for the next step: C_{n+1} = (C_n + 7) % 101
#         C = (C + 7) % 101
        
#     return V

# if __name__ == "__main__":
#     result = solve_mti_1500()
#     print(f"Final State Vector V_1500: {result}")
