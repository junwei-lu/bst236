# Linear Equation Solvers

We want to solve the linear equation $Ax = b$ for $x$ given $A \in \mathbb{R}^{n \times n}$ and $b \in \mathbb{R}^n$. As we stated in the previous section, we will not discuss the details of the algorithms but focus on the practical issues when solving linear equations.

## Basic Facts of Linear Equation Solvers

**Time Complexity**. The time complexity of solving a linear equation is generally $O(n^3)$.

**Sensitivity**. The sensitivity of solving a linear equation depends on the condition number of the matrix $A$.

**Definition**(Condition Number of Matrix). The condition number of a matrix $A$ is defined as:
$$
\kappa(A) = \|A\| \|A^{-1}\|.
$$

**Theorem**(Sensitivity of Linear Equation Solver). Suppose $A \in \mathbb{R}^{n \times n}$ is non-singular, $b \in \mathbb{R}^n$ is non-zero, and $\delta A \in \mathbb{R}^{n \times n}$ satisfies $\|A^{-1}\| \|\delta A\| < 1$. If $x$ and $x + \delta x$ are solutions to the linear systems:
$$
Ax = b \quad \text{and} \quad (A + \delta A)(x + \delta x) = b + \delta b,
$$
then the following holds:
$$
\frac{\|\delta x\|}{\|x\|} \leq \frac{\kappa(A)}{1 - \kappa(A) \frac{\|\delta A\|}{\|A\|}} \left( \frac{\|\delta A\|}{\|A\|} + \frac{\|\delta b\|}{\|b\|} \right).
$$

## Algorithms to Solve Linear Equations

We will only give a high-level overview of the algorithms to solve linear equations. The linear solvers are so well-developed now that for the most time, it is efficient enough to use the built-in functions in the programming languages.

The most important philosophy in designing a numerical linear algebra algorithm is to reduce the general problem into several simpler problems. And the most common way to do this is to use different kinds of **matrix decompositions**. For linear equations, it is easy to solve a linear equation $Lx = b$ if $L$ is a lower triangular matrix or $Ux = b$ if $U$ is an upper triangular matrix. 
Therefore, we aims to decompose the matrix $A$ into a product of triangular matrices.

**LU Decomposition $A = LU$**. Any invertible matrix $A$ can be decomposed into a product of a lower triangular matrix $L$ and an upper triangular matrix $U$. The LU decomposition can be computed by [Gaussian elimination with partial pivoting](https://en.wikipedia.org/wiki/Gaussian_elimination). We then solve the linear equation $Ax = LUx = b$ by solving $Ly = b$ and $Ux = y$. In Python, the standard `numpy.linalg.solve` or `scipy.linalg.solve` function solves the linear equation by this method.

**Cholesky Decomposition $A = LL^T$**. Any symmetric positive definite matrix $A$ can be decomposed into a product of a lower triangular matrix $L$ and its transpose $L^T$. We then solve the linear equation $Ax = b$ by solving $Ly = b$ and $L^Tx = y$.

The LU decomposition needs approximately $2n^3/3$ operations, while the Cholesky decomposition needs approximately $n^3/3$ operations. Therefore, if $A$ is positive definite, we should use the Cholesky decomposition. Also note that the time complexity of solving a triangular linear equation is $O(n^2)$. So for linear equations, matrix decompositions are bottlenecks of the performance.

However, you do not need to implement the Cholesky decomposition by yourself. The `linalg.solve` function 
has a built-in `assume_a` parameter:
- `assume_a = 'gen'` - General matrix (default)
- `assume_a = 'sym'` - Symmetric matrix
- `assume_a = 'pos'` - Symmetric positive definite

The following code compares the running time of the standard `linalg.solve` function and the Cholesky decomposition method.

```python
import numpy as np
from scipy import linalg
import time

# Create a symmetric positive definite matrix
n = 5000
np.random.seed(0)
X = np.random.rand(n, n)
A = np.dot(X, X.T) # Also a sample covariance matrix
b = np.random.rand(n)

# Time np.linalg.solve()
start_time = time.time()
x1 = linalg.solve(A, b)
linalg_time = time.time() - start_time

# Use positive definite argument
start_time = time.time()
x3 = linalg.solve(A, b, assume_a='pos')
solve_pos_time = time.time() - start_time

# Use symmetric argument
start_time = time.time()
x4 = linalg.solve(A, b, assume_a='sym')
solve_sym_time = time.time() - start_time

# Manual Cholesky decomposition
start_time = time.time()
L = linalg.cholesky(A)
y = linalg.solve_triangular(L, b, lower=True)  # Forward substitution
x2 = linalg.solve_triangular(L.T, y, lower=False)  # Backward substitution
cholesky_time = time.time() - start_time

# Print results
print(f'General solve Time: {linalg_time:.6f} seconds') # 0.669566 seconds
print(f'Symmetric solve Time: {solve_sym_time:.6f} seconds') # 1.184631 seconds
print(f'Symmetric positive definite solve Time: {solve_pos_time:.6f} seconds') # 0.468433 seconds
print(f'Manual Cholesky Decomposition Time: {cholesky_time:.6f} seconds') # 0.369518 seconds
```

The manual Cholesky decomposition is slightly faster than the `linalg.solve` function with the `assume_a='pos'` argument. For most time, we do not need to implement the Cholesky decomposition by ourselves. You can refer to the [scipy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve.html) for more details.

## Tips for Solving Linear Equations

**Tip 1**. Never use `inv(A) @ B`. Always use `solve(A, B)`. Most of the time `inv(A)` is not numerically stable.

**Tip 2**. Even if you need to solve the linear equation $Ax = b$ for multiple $b$ vectors, e.g., in iterative optimization algorithms, you should not save `inv(A)`. Instead, you should save the LU decomposition of $A$ (or Cholesky decomposition if $A$ is positive definite) and solve the triangular linear equations in the iterations. The time complexity of solving a triangular linear equation is $O(n^2)$, same as matrix multiplication.

**Tip 3**. When $A$ is low rank plus a diagonal matrix, we can use the Woodbury matrix identity to speed up the computation.

**Woodbury Matrix Identity**. 
$$
(D+UV)^{-1}=D^{-1}-D^{-1}U(I+VD^{-1}U)^{-1}VD^{-1}
$$

If $A = D + UV \in \mathbb{R}^{n \times n}$ with $D$ is a diagonal matrix and $U, V \in \mathbb{R}^{n \times k}$, then directly computing the inverse of $A$ is $O(n^3)$. However, using the Woodbury matrix identity, we only need to solve the linear equation of the matrix $I + VD^{-1}U$ which is only $k$ by $k$. The overall time complexity using the Woodbury matrix identity reduces to $O(k^3+k^2n)$.

For example, when solving a ridge regression
$$
\hat{x} = \arg\min_x \|Ax - b\|^2 + \lambda \|x\|^2 = (A^T A + \lambda I)^{-1} A^T b,
$$
we can use the Woodbury matrix identity to speed up the computation when the dimension is much larger than the sample size.
