from abyme.core import *
from abyme.tree_trace import *
import time

prompt = r"""

\section*{Problem: The Binary Mutation Chain (BMC-900)}

Let $\mathbf{S}_n = [a_n, b_n, c_n, d_n, e_n]$ be a state vector in the space $\mathbb{Z}_{256}^5$. 
The initial state at $n=0$ is:
\[ \mathbf{S}_0 = [13, 42, 89, 121, 204] \]

The vector evolves for exactly \textbf{900 steps}. For each step $n \in \{1, 2, \dots, 900\}$, the transformation is defined by:
\begin{enumerate}
    \item $a_n = (a_{n-1} \oplus b_{n-1} + C_n) \pmod{256}$
    \item $b_n = (b_{n-1} \oplus c_{n-1} + 17) \pmod{256}$
    \item $c_n = (c_{n-1} \oplus d_{n-1} + a_n) \pmod{256}$
    \item $d_n = (d_{n-1} \oplus e_{n-1} + 123) \pmod{256}$
    \item $e_n = (e_{n-1} \oplus a_n + n) \pmod{256}$
\end{enumerate}
Where $\oplus$ denotes the bitwise XOR operation and $C_n$ is a rotating constant:
\[ C_n = \begin{cases} 55 & \text{if } n \text{ is odd} \\ 110 & \text{if } n \text{ is even} \end{cases} \]

\subsection*{Requirement}
You must provide the full state vector $\mathbf{S}_{900} = [a_{900}, b_{900}, c_{900}, d_{900}, e_{900}]$. 

YOU HAVE NO ACCESS TO CODE, YOU MUST DO THIS STEP BY STEP MANUALLY
\]

"""
#prompt = "Let $ABCDE$ be a convex pentagon with $AB=14, BC=7, CD=24, DE=13, EA=26,$ and $\\angle B=\\angle E=60^\\circ$. For each point $X$ in the plane, define $f(X)=AX+BX+CX+DX+EX$. The least possible value of $f(X)$ can be expressed as $m+n\\sqrt{p}$, where $m$ and $n$ are positive integers and $p$ is not divisible by the square of any prime. Find $m+n+p$.\n\nPlease provide your final answer in the format \\boxed{answer}, where answer is an integer between 0 and 999."
#149
#prompt = 'There are $ n $ values of $ x $ in the interval $ 0 < x < 2\\pi $ where $ f(x) = \\sin(7\\pi \\cdot \\sin(5x)) = 0 $. For $ t $ of these $ n $ values of $ x $, the graph of $ y = f(x) $ is tangent to the $ x $-axis. Find $ n + t $.\n\nPlease provide your final answer in the format \\boxed{answer}, where answer is an integer between 0 and 999.'
#371
prompt = 'Each vertex of a regular octagon is independently colored either red or blue with equal probability. The probability that the octagon can then be rotated so that all of the blue vertices end up at positions where there were originally red vertices is $\\tfrac{m}{n}$, where $m$ and $n$ are relatively prime positive integers. What is $m+n$?\n\nPlease provide your final answer in the format \\boxed{answer}, where answer is an integer between 0 and 999.'
#prompt = "Find the derivative of f(x) = (x^2 + 1)^(x^3) using logarithmic differentiation, then evaluate f'(1). leave answer with ln()"
prompt = "Analyze whether GOOGL worth investing"

def main():
  model = Abyme_DeepSeek(print_progress=True, max_parallel_workers=10, max_depth=5, max_call=3000)
  print("model loaded")
  start_time = time.time()
  model.generate(prompt)
  latency = time.time() - start_time
  print("generation complete")
  print("="*50)
  print("final answer: ",model.trace.get_final_output())
  print("total call: ", total_calls(model.trace))
  print("max depth: ", max_depth(model.trace))
  print("max subproblems: ", max_subproblem(model.trace))
  print("max output characters: ", max_output_character(model.trace))
  print("parallel latency: ", parallel_latency(model.trace))
  print("actual latency:",latency)
  print("average latency: ", latency/total_calls(model.trace))
  print("nodes per level: ", nodes_per_level(model.trace))

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
