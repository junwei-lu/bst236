# To Do

- [X] Add a checklist for preparing the course
- [x] Github classroom setup.
- [x] All installation instructions.
- [ ] Cluster setup.
- [ ] Find interesting, available datasets for homework.

# Course plan

## Workflow

    1. Principles of good coding
    2. Git and GitHub
    3. Makefile
    4. Virtual environment

## Data Structures

    1. Python: List (also as a stack), dict, deque, 
       * Big O, Leetcode: Two Sum
    2. Numpy: Arrays
    3. Pandas: DataFrames
    4. Avoid for loops
    5. Input/Output: 1brc 

## Algorithms 

    1. Complexity
    2. Example: climbing the stairs
       * Recursion (dfs)
       * Memoization/pruning (wayToSteps[i] = wayToStep.append is worse than =[0]*(n+1))
       * Dynamic Programming (dp[i] vs a,b,c)
       * Super power (divide and conquer $O(\log n)$ complexity)
       * HW: Frog jump
    3. Example: 0-1 Knapsack
       * Introduce the high level ideas of backtracking, deep first search
       * Write a template code with comments
       * dfs / bfs (introduce (deque) for python )
       * pruning / memoization
       * Dynamic Programming (+ memory optimization)
       * Fractional knapsack: Greedy
       * HW: DP: ATCG Longest Common Subsequence

## Numerical Linear Algebra

### BasicsOperations
1. Matrix eigenvector, singular vectors
2. What is fast operations (no sqrt, always matrix multiplications)
3. Operations: matrix multiplication

### Linear equation

- LU, LL decomposition; 
- Computation complexity
- Sensitivity analysis (why never inverse)
- (D+L+U)x = b -> (D+L)x = - Ux + b -> x = (D+L)\Ux + (D+L)\b (may skip as conjugate grad is better)
- Leave conjugate gradient descent to optimization

### Eigenvectors

- Power method (sensitivity; conditions; bad examples)
- QR decomposition is a multiple column normalization (complexity, sensitivity)
- eigen gap matters
- Full vector, QR->RQ (mention there exists better algorithms)
- HW: Why X = AX, QR gives the right results;

### Advanced:
- matrix-free (Check chatgpt, Nicol)
- DnC
- Randomized
- Homework: Covariance of normal (temperature ？）, Hession of logistic,cupy matrix multiplication; 

## Optimization

### Concepts

- Convexity
- Convergence: linear convergence, sublinear convergence
- pytorch autograd

### First Order Methods

- Gradient descent
- Accelerated gradient descent (Momentum)
- Proximal gradient descent

### Second Order Methods

- Newton's method
- Quasi-Newton methods: BFGS


### Stochastic Optimization

- SGD
- Adam 
- Pytorch optimizer


### Miscellaneous

- Nonconvex optimization
- Combinatorial optimization
