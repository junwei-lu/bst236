# Summary



1. **Depth-first Search**: 
      - Use for **exploring all paths**, connectivity checks, or any scenario where deep exploration is required.


2. **Breadth-first Search**: 
      - Use for **shortest paths**, level-wise exploration, and when you need to explore all possibilities nearest to the starting point first.



3. **Dynamic Programming**:
      - Use for problems with **overlapping subproblems** and **optimal substructure**, where you can break a problem into smaller, solvable subproblems and reuse results.


## **Which Algorithm to Use?**
| **Problem Type**                               | **Use BFS**    | **Use DFS**    | **Use DP**                     |
|------------------------------------------------|----------------|----------------|--------------------------------|
| **Shortest Path**                              | ✅             | ❌             | ❌                             |
| **Exploring All Paths**                        | ❌             | ✅             | ❌                             |
| **Finding Any Path**                           | ❌             | ✅             | ❌                             |
| **Generating All Solutions**                   | ❌             | ✅             | ❌                             |
| **Overlapping Subproblems (e.g., Fibonacci)**  | ❌             | ❌             | ✅                             |
| **Optimal Substructure (e.g., Knapsack)**      | ❌             | ❌             | ✅                             |

