# Divide and Conquer

Let us continue with the climbing stairs problem.

!!! question "Climbing stairs"

    Given a staircase with $n$steps, where you can climb $1$or $2$steps at a time, how many different ways are there to reach the top?


We have learned to solve the problem using [deep-first search with memoization trick](climb_stairs.md#method-2-memoized-search) and [dynamic programming](climb_stairs.md#method-3-dynamic-programming). The time complexity of the two methods are $O(n)$. Can we further reduce the time complexity?



## Method 4: Closed Form Solution

Let $dp[i]$be the number of ways to reach the $i^{th}$step. The recurrence relation for the climbing stairs problem is:

$$dp[i] = dp[i-1] + dp[i-2] $$

This can be expressed in matrix form as:

$$
\begin{bmatrix}
dp[i] \\
dp[i-1]
\end{bmatrix}
=
\begin{bmatrix}
1 & 1 \\
1 & 0
\end{bmatrix}
\begin{bmatrix}
dp[i-1] \\
dp[i-2]
\end{bmatrix}
$$

Let $A$ be the transformation matrix:

$$
A = 
\begin{bmatrix}
1 & 1 \\
1 & 0
\end{bmatrix}
$$

Then, the state vector at step $i$ can be expressed as:

$$
\begin{bmatrix}
dp[i] \\
dp[i-1]
\end{bmatrix}
=
A
\begin{bmatrix}
dp[i-1] \\
dp[i-2]
\end{bmatrix}
$$

By applying this transformation iteratively, we can express the state vector at step $n $as:

$$
\begin{bmatrix}
dp[n] \\
dp[n-1]
\end{bmatrix}
=
A^{n-1}
\begin{bmatrix}
dp[1] \\
dp[0]
\end{bmatrix}
$$

Given the initial conditions $dp[1] = 1$ and $dp[0] = 1$, the initial state vector is $[1,1]^\top$:


To further find a closed form for $dp[n]$, we perform eigen decomposition on matrix $A$. Solve the characteristic equation $\det(A - \lambda I) = 0$.

$$
   \det
   \begin{bmatrix}
   1-\lambda & 1 \\
   1 & -\lambda
   \end{bmatrix}
   = (1-\lambda)(-\lambda) - 1 = \lambda^2 - \lambda - 1 = 0
$$

   The roots of this equation are the eigenvalues:

$$
   \lambda_1 = \frac{1 + \sqrt{5}}{2}, \quad \lambda_2 = \frac{1 - \sqrt{5}}{2}
$$

Whence we have the eigenvalues of $A$, we can express $A$ as $PDP^{-1}$, where $D$ is the diagonal matrix of eigenvalues and $P$ is the matrix of eigenvectors. Actually, we do not really need to compute $P$ but we just write both $P$ and $D$ here:
$$
   P = \begin{bmatrix} \lambda_1 & \lambda_2 \\ 1 & 1 \end{bmatrix}, \quad D = \begin{bmatrix} \lambda_1 & 0 \\ 0 & \lambda_2 \end{bmatrix}
$$
It is easy to compute see that $A^{n-1} = PD^{n-1}P^{-1}$ and 
$$
\begin{bmatrix}
dp[n] \\
dp[n-1]
\end{bmatrix}
=
P D^{n-1} P^{-1}
\begin{bmatrix}
dp[1] \\
dp[0]
\end{bmatrix}
$$

Even if we do not compute $P$ and $P^{-1}$, we can see that the closed form for $dp[n]$ is given by:

$$
   dp[n] = c_1 \lambda_1^n + c_2 \lambda_2^n
$$

where $c_1$and $c_2$ are constants determined by the initial conditions. Solving for these constants using the initial conditions $dp[1] = 1$ and $dp[0] = 1$, we find:

$$
c_1 = \frac{1}{\sqrt{5}}\left(\frac{1 + \sqrt{5}}{2}\right), \quad c_2 = -\frac{1}{\sqrt{5}}\left(\frac{1 - \sqrt{5}}{2}\right)
$$

Thus, the closed form of $dp[n]$ is:

$$
dp[n] = \frac{1}{\sqrt{5}} \left( \left(\frac{1 + \sqrt{5}}{2}\right)^{n+1} - \left(\frac{1 - \sqrt{5}}{2}\right)^{n+1} \right)
$$

Here is the code to compute the closed form:

```python
def climbing_stairs_closed_form(n: int) -> int:
    """Climbing stairs: Closed form solution"""
    sqrt_5 = math.sqrt(5)
    return int((((1 + sqrt_5) / 2) ** (n + 1) - ((1 - sqrt_5) / 2) ** (n + 1)) / sqrt_5)
```


## Method 5: Divide and Conquer

While the closed form solution provides a direct computation of the number of ways to climb the stairs, it involves floating-point arithmetic, which can lead to precision issues for very large values of $n$. Additionally, it requires the computation of powers of irrational numbers, which can be computationally expensive. Last but not least, the "human computation complexity" of computing the matrix eigenvalues and deriving the closed form might be far beyond the worst algorithm.

To address the limitations of the closed form method, we want to get one step back to the matrix multiplication:
$$
\begin{bmatrix}
dp[n] \\
dp[n-1]
\end{bmatrix}
=
A^{n-1}
\begin{bmatrix}
dp[1] \\
dp[0]
\end{bmatrix}
$$
As $A$ is an integer matrix, this will not involve any floating-point arithmetic. However, the computation complexity of computing the $n$th power of a $d$-by-$d$ matrix $A$ is $O(d^3n)$. 

!!! note "A trick of matrix multiplication"
    Notice that the complexity of the original climbing stairs problem can be reduced to $O(d^2n)$ by computing $A(A(\cdots(A(Av)))\cdots)$ instead of $(A\cdots A)v$.

Is there a way to further reduce the time complexity? We want to introduce a new algorithm idea: **Divide and Conquer**.


To compute $A^{n}$ efficiently, we can apply the divide-and-conquer strategy and it is called **binary exponentiation** when applied to power computation. Here are the key steps of binary exponentiation:

- If the exponent is zero: Return the identity matrix, as any matrix raised to the power of zero is the identity matrix.
- If the exponent is even: Compute $A^{n/2}$ recursively, then square the result to obtain $A^n$.
- If the exponent is odd: Compute $A^{(n-1)/2}$ recursively, square the result, and then multiply by $A$ once more to obtain $A^n$.

Here is the code to compute the power of a matrix using binary exponentiation:


```python
import numpy as np

def binpow(A: np.ndarray, n: int) -> np.ndarray:
    """Calculate the power of matrix A to the n using binary exponentiation."""
    if n == 0:
        # Return the identity matrix of the same size as A
        return np.eye(A.shape[0], dtype=int)
    res = binpow(A, n // 2)
    if n % 2 == 0:
        return res @ res
    else:
        return res @ res @ A

# Applied to the climbing stairs problem
def climbing_stairs_binpow(n: int) -> int:
    A = np.array([[1, 1], [1, 0]], dtype=int)
    return binpow(A, n)[0][0]
```

Recall the [time complexity anlysis](../chapter_computational_complexity/time_complexity.md#logarithmic-time) for logrithmic complexity, logarithmic order reflects situations where "the size is halved each round." In the case of binary exponentiation, the size of the problem is halved each round, so the time complexity is $O(d^3\log n)$.

## Divide-and-Conquer Search Approach

The divide-and-conquer technique can be effectively applied to search problems, such as binary search:

- **Problem Division**: Binary search divides the main problem (searching within an array) into smaller subproblems (searching within half of the array) by comparing the middle element with the target.
- **Independence of Subproblems**: Each subproblem in binary search is handled independently, without interference from other subproblems.
- **No Need for Merging Solutions**: Since binary search aims to locate a specific element, there's no requirement to merge solutions from subproblems. Solving a subproblem directly solves the main problem.

This strategy enhances search efficiency because, unlike brute-force methods that eliminate one possibility per iteration, divide-and-conquer can discard half of the possibilities.

Consider the binary search problem:
!!! question

    Given a sorted array `nums` of length $n$, where all elements are distinct, find the element `target`.

From a divide-and-conquer perspective, the subproblem for the search interval $[i, j]$ is denoted as $f(i, j)$.

Starting with the original problem $f(0, n-1)$, binary search proceeds as follows:

1. Determine the midpoint $m$ of the search interval $[i, j]$ and use it to eliminate half of the interval.
2. Recursively solve the subproblem with the reduced interval, which could be $f(i, m-1)$ or $f(m+1, j)$.
3. Repeat steps `1.` and `2.` until the `target` is found or the interval is empty, resulting in a return.

The diagram below illustrates the divide-and-conquer process of binary search for the element $6$ in an array.

![The divide-and-conquer process of binary search](divide_n_conquer.assets/binary_search_recur.png)

In the implementation, a recursive function `dfs()` is declared to solve the problem $f(i, j)$:

```python
def dfs(nums: list[int], target: int, i: int, j: int) -> int:
    """Binary search: problem f(i, j)"""
    # If the interval is empty, indicating no target element, return -1
    if i > j:
        return -1
    # Calculate midpoint index m
    m = (i + j) // 2
    if nums[m] < target:
        # Recursive subproblem f(m+1, j)
        return dfs(nums, target, m + 1, j)
    elif nums[m] > target:
        # Recursive subproblem f(i, m-1)
        return dfs(nums, target, i, m - 1)
    else:
        # Found the target element, thus return its index
        return m

def binary_search(nums: list[int], target: int) -> int:
    """Binary search"""
    n = len(nums)
    # Solve problem f(0, n-1)
    return dfs(nums, target, 0, n - 1)
```




## Summary 

The Climbing Stairs Problem explores elegant algorithmic ideas for optimization and problem-solving. Here is a concise overview of the methods analyzed:

1. **Depth-First Search (DFS)**  
      - **Approach**: Recursively solve the problem by breaking it into subproblems.
      - **Time Complexity**: $O(2^n)$ due to overlapping subproblems.
      - **Drawback**: Computational inefficiency from redundant calculations.

2. **Memoized Search**  
      - **Approach**: Uses an array to store previously computed results, avoiding recomputation of overlapping subproblems.
      - **Time Complexity**: $O(n)$, with significant improvement over DFS.
      - **Space Complexity**: $O(n)$ for the memoization array.

3. **Dynamic Programming (DP)**  
      - **Approach**: Computes in a bottom-up manner.
      - **Time Complexity**: $O(n)$.
      - **Space Optimization**: By using rolling variables, space complexity is reduced from $O(n)$ to $O(1)$.

4. **Closed Form Solution**  
      - **Approach**: Derives a mathematical formula for $dp[n]$ using eigenvalues of the transformation matrix.
      - **Time Complexity**: $O(n)$ for power computation.
      - **Drawback**: Susceptible to floating-point precision issues for large $n$.

5. **Divide and Conquer (Binary Exponentiation)**  
      - **Approach**: Efficiently computes matrix powers using recursive halving. 
      - **Time Complexity**: $O(\log n)$.