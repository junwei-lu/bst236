# Dynamic Programming

<u>Dynamic programming</u> is a crucial algorithmic approach that breaks down a problem into smaller subproblems, storing their solutions to prevent redundant calculations, thus greatly enhancing time efficiency.

We have encountered dynamic programming in the [climb stairs problem](climb_stairs.md#method-3-dynamic-programming). In this section, we will delve into the general strategy of dynamic programming by addressing these two questions:
1. How can we identify if a problem is suitable for dynamic programming?
2. What are the comprehensive steps to solve a dynamic programming problem?

## Identifying Dynamic Programming

**Problems apt for dynamic programming typically align with the "decision tree model"**, which can be depicted using a tree structure where each node signifies an action, and each path represents a sequence of actions.

In essence, if a problem involves explicit action concepts and the solution is derived through a series of actions, it fits the action tree model and can generally be tackled using dynamic programming.

Additionally, there are some "bonus points" for identifying dynamic programming problems.

- The problem involves descriptions of maximization (minimization) or finding the most (least) optimal solution.
- The problem's states can be represented using a list, multi-dimensional matrix, or tree, and a state has a recursive relationship with its neighboring states.
- Given a certain state, its future development is only related to the current state and unrelated to all past states experienced. This is also known as the "Markovian" property.

For instance, in the stair climbing problem, given state $i$, it will progress to states $i+1$ and $i+2$, corresponding to jumping 1 step and 2 steps respectively. When making these choices, we do not need to consider the states before state $i$, as they do not influence the future of state $i$.

Conversely, there are also some "penalty points".

- The goal of the problem is to find all possible solutions, not just the optimal solution. Such problems are usually solved by [breadth-first search](bfs.md).
- The problem description has clear characteristics of permutations and combinations, requiring the return of specific multiple solutions.

If a problem fits the decision tree model and has evident "bonus points", we can assume it is a dynamic programming problem and verify it during the solution process.

## Steps to Solve Dynamic Programming

The process of solving dynamic programming problems varies with the nature and complexity of the problem but generally follows these steps: 
- describe actions, 
- define states, 
- establish a $dp$ table, 
- derive state transition equations, 
- determine boundary conditions.

To vividly illustrate the problem-solving steps, we use a classic problem, "Minimum Path Sum", as an example.

!!! question

    Given an $n \times m$ two-dimensional grid `grid`, each cell in the grid contains a non-negative integer representing the cost of that cell. The robot starts from the top-left cell and can only move down or right at each step until it reaches the bottom-right cell. Return the minimum path sum from the top-left to the bottom-right.

The figure below shows an example, where the given grid's minimum path sum is $13$.

![Minimum Path Sum Example Data](dp_solution_pipeline.assets/min_path_sum_example.png)

**First step: Consider each round of actions, define the state, and thereby obtain the $dp$ table**

Each round of actions in this problem is to move one step down or right from the current cell. Suppose the row and column indices of the current cell are $[i, j]$, then after moving down or right, the indices become $[i+1, j]$ or $[i, j+1]$. Therefore, the state should include two variables: the row index and the column index, denoted as $[i, j]$.

The state $[i, j]$ corresponds to the subproblem: the minimum path sum from the starting point $[0, 0]$ to $[i, j]$, denoted as $dp[i, j]$.

Thus, we obtain the two-dimensional $dp$ matrix shown in the figure below, whose size is the same as the input grid $grid$.

![State definition and DP table](dp_solution_pipeline.assets/min_path_sum_solution_state_definition.png)

**Second step: Identify the optimal substructure, then derive the state transition equation**

For the state $[i, j]$, it can only be derived from the cell above $[i-1, j]$ or the cell to the left $[i, j-1]$. Therefore, the optimal substructure is: the minimum path sum to reach $[i, j]$ is determined by the smaller of the minimum path sums of $[i, j-1]$ and $[i-1, j]$.

Based on the above analysis, the state transition equation shown in the figure below can be derived:

$$
dp[i, j] = \min(dp[i-1, j], dp[i, j-1]) + grid[i, j]
$$

![Optimal substructure and state transition equation](dp_solution_pipeline.assets/min_path_sum_solution_state_transition.png)

!!! note

    Based on the defined $dp$ table, think about the relationship between the original problem and the subproblems, and find out how to construct the optimal solution to the original problem from the optimal solutions to the subproblems, i.e., the optimal substructure.

    Once we have identified the optimal substructure, we can use it to build the state transition equation.

**Third step: Determine boundary conditions and state transition order**

In this problem, the states in the first row can only come from the states to their left, and the states in the first column can only come from the states above them, so the first row $i = 0$ and the first column $j = 0$ are the boundary conditions.

As shown in the figure below, since each cell is derived from the cell to its left and the cell above it, we use loops to traverse the matrix, the outer loop iterating over the rows and the inner loop iterating over the columns.

![Boundary conditions and state transition order](dp_solution_pipeline.assets/min_path_sum_solution_initial_state.png)

!!! note

    Boundary conditions are used in dynamic programming to initialize the $dp$ table, and in search to prune.
    
    The core of the state transition order is to ensure that when calculating the solution to the current problem, all the smaller subproblems it depends on have already been correctly calculated.

Based on the above analysis, we can directly write the dynamic programming code. 

```python
def min_path_sum_dp(grid: list[list[int]]) -> int:
    """Minimum path sum: Dynamic programming"""
    n, m = len(grid), len(grid[0])
    # Initialize dp table
    dp = [[0] * m for _ in range(n)]
    dp[0][0] = grid[0][0]
    # State transition: first row
    for j in range(1, m):
        dp[0][j] = dp[0][j - 1] + grid[0][j]
    # State transition: first column
    for i in range(1, n):
        dp[i][0] = dp[i - 1][0] + grid[i][0]
    # State transition: the rest of the rows and columns
    for i in range(1, n):
        for j in range(1, m):
            dp[i][j] = min(dp[i][j - 1], dp[i - 1][j]) + grid[i][j]
    return dp[n - 1][m - 1]
```

The figure below shows the state transition process of the minimum path sum, traversing the entire grid, **thus the time complexity is $O(nm)$**.

The array `dp` is of size $n \times m$, **therefore the space complexity is $O(nm)$**.

=== "<1>"
    ![Dynamic programming process of minimum path sum](dp_solution_pipeline.assets/min_path_sum_dp_step1.png)

=== "<2>"
    ![min_path_sum_dp_step2](dp_solution_pipeline.assets/min_path_sum_dp_step2.png)

=== "<3>"
    ![min_path_sum_dp_step3](dp_solution_pipeline.assets/min_path_sum_dp_step3.png)

=== "<4>"
    ![min_path_sum_dp_step4](dp_solution_pipeline.assets/min_path_sum_dp_step4.png)

=== "<5>"
    ![min_path_sum_dp_step5](dp_solution_pipeline.assets/min_path_sum_dp_step5.png)

=== "<6>"
    ![min_path_sum_dp_step6](dp_solution_pipeline.assets/min_path_sum_dp_step6.png)

=== "<7>"
    ![min_path_sum_dp_step7](dp_solution_pipeline.assets/min_path_sum_dp_step7.png)

=== "<8>"
    ![min_path_sum_dp_step8](dp_solution_pipeline.assets/min_path_sum_dp_step8.png)

=== "<9>"
    ![min_path_sum_dp_step9](dp_solution_pipeline.assets/min_path_sum_dp_step9.png)

=== "<10>"
    ![min_path_sum_dp_step10](dp_solution_pipeline.assets/min_path_sum_dp_step10.png)

=== "<11>"
    ![min_path_sum_dp_step11](dp_solution_pipeline.assets/min_path_sum_dp_step11.png)

=== "<12>"
    ![min_path_sum_dp_step12](dp_solution_pipeline.assets/min_path_sum_dp_step12.png)

### Space Optimization

Since each cell is only related to the cell to its left and above, we can use a single-row array to implement the $dp$ table.

Please note, since the array `dp` can only represent the state of one row, we cannot initialize the first column state in advance, but update it as we traverse each row:

```python
def min_path_sum_dp_comp(grid: list[list[int]]) -> int:
    """Minimum path sum: Space-optimized dynamic programming"""
    n, m = len(grid), len(grid[0])
    # Initialize dp table
    dp = [0] * m
    # State transition: first row
    dp[0] = grid[0][0]
    for j in range(1, m):
        dp[j] = dp[j - 1] + grid[0][j]
    # State transition: the rest of the rows
    for i in range(1, n):
        # State transition: first column
        dp[0] = dp[0] + grid[i][0]
        # State transition: the rest of the columns
        for j in range(1, m):
            dp[j] = min(dp[j - 1], dp[j]) + grid[i][j]
    return dp[m - 1]
```
