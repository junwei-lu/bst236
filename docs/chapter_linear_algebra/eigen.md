# Eigenvalues and Eigenvectors

We now discuss the numerical algorithms for computing eigenvalues and eigenvectors. We will mainly focus on the symmetric matrices with real eigenvalues and eigenvectors. For the general case, most of the tasks in data science need to conduct the singular value decomposition (SVD). 

Recall the symmetric matrix $A$ has the eigenvalue decomposition $A = U\Lambda U^T$, where $U = (u_1, u_2, \cdots, u_n)$ is an orthogonal matrix with the columns $u_1, u_2, \cdots, u_n$ being the eigenvectors of $A$ and $\Lambda = \text{diag}(\lambda_1, \lambda_2, \cdots, \lambda_n)$ is a diagonal matrix with diagonal entries $\lambda_1\ge \lambda_2\ge \cdots \ge \lambda_n$ being the eigenvalues of $A$.

## Power Method

Here we consider the positive definite matrix $A$ with the eigenvalues $\lambda_1 > \lambda_2 \ge \cdots \ge \lambda_n >0$.

The power method applies the power
$$
A^k = U \Lambda^k U^T = \sum_{j=1}^n \lambda_j^k u_j u_j^T.
$$

Let us take a vector $x_0 \in \mathbb{R}^n$ which can be expressed as:

$$
x_0 = \alpha_1 u_1 + \alpha_2 u_2 + \cdots + \alpha_n u_n,
$$

where $\alpha_i \in \mathbb{R}$. Thus, we have:

$$
A^k x_0 = \sum_{j=1}^n \alpha_j A^k u_j
$$

$$
= \lambda_1^k \left[ \alpha_1 u_1 + \sum_{j=2}^n \alpha_j \left( \frac{\lambda_j}{\lambda_1} \right)^k u_j \right].
$$

Therefore, it follows that:

$$
\lim_{k \to \infty} \frac{A^k x_0}{\lambda_1^k} = \alpha_1 u_1.
$$

This indicates that, when $\alpha_1 \neq 0$ and $k$ is sufficiently large, the vector:

$$
x_k = \frac{A^k x_0}{\lambda_1^k}
$$

is a very good approximation of an eigenvector of $A$.

This motivates the **power method** to find the largest eigenvalue and its corresponding eigenvector.
$$
\begin{aligned}
x_{k+1} &= A x_k \\
x_{k+1} &= \frac{x_{k+1}}{||x_{k+1}||}
\end{aligned}
$$
Notice that we normalize the vector $x_{k+1}$ in each iteration as the eigenvector is only determined by the direction.

```python
import numpy as np
A = np.array([[4, 1],
              [2, 3]])
tol = 1e-6
max_iter = 1000
# Initialize a random vector
n = A.shape[0]
x = np.random.rand(n)

for _ in range(max_iter):
    # Multiply A with the vector
    x = A @ x
    x_norm = np.linalg.norm(x)    
    # Normalize the vector
    x = x / x_norm
    # Check for convergence
    if np.linalg.norm(x_new - x) < tol:
        break
```

**Convergence and Sensitivity**: We can see from the above analysis that the power method convergence speed is determined by $\lambda_2/\lambda_1$. If $\lambda_2$ is close to $\lambda_1$, the convergence is slow. This also tells us that the sensitivity of eigenvalues is determined by **eigengap** $\Delta_k = |\lambda_k - \lambda_{k+1}|$. If $\Delta_k$ is small, it is more difficult to distinguish $u_k$ from $u_{k+1}$.

**Other eigenvectors**: The power method above only finds the largest eigenvalue and its corresponding eigenvector. As $A^{-1}$ has the largest eigenvalues $\lambda_n^{-1}$ with the eigenvector $u_n$, we can use the power method to $A^{-1}$ to get the smallest eigenvalue and its corresponding eigenvector. This is called the **inverse power method**.
$$
\begin{aligned}
x_{k+1} &= A^{-1} x_k \\
x_{k+1} &= \frac{x_{k+1}}{||x_{k+1}||}
\end{aligned}
$$
We can see that the inverse power method needs to solve a linear system. You can improve the efficiency by apply the [LU decomposition](linear_equation.md#tips-for-solving-linear-equations) first. Similar to the above analysis, the convergence speed of the inverse power method is determined by $\lambda_n/\lambda_{n-1}$.

We can further apply the inverse power method to $A - \mu I$ with some shift $\mu$ to accelerate the convergence. This is called the **shifted inverse power method**. Without loss of generality, if we want to compute $\lambda_1$, and we assume $0<|\lambda_1 - \mu| < |\lambda_2 - \mu| \le \cdots \le |\lambda_n - \mu|$. The convergence speed is determined by $|\lambda_k - \mu|/|\lambda_{k+1} - \mu|$, the shift $\mu$ should be chosen as close to $\lambda_k$ as possible to accelerate the convergence. You may worry that when $\mu$ is too close to $\lambda_k$, linear equation for $A-\mu I$ is ill-conditioned. Actually, it can be shown that the illness of $A-\mu I$ will not affect the convergence of the inverse power method.


## QR Method

Though the power method is simple and easy to implement, it depends on the eigenvalue distribution. Besides, we can only find one eigenvalue and its corresponding eigenvector using the power method. We now discuss how to get the full eigen-decomposition using the QR method.

**QR Decomposition**: For any matrix $A$, there exists a QR decomposition $A = QR$, where $Q$ is an orthogonal matrix and $R$ is an upper triangular matrix. QR decomposition can be computed by the [Householder transformation](https://en.wikipedia.org/wiki/QR_decomposition) with the complexity $O(n^3)$.

QR decomposition can be applied to any matrix $A$, but the $Q, R$ could be complex if $A$ is not symmetric. So we mainly focus on the symmetric matrices in the following discussion. Also, for rectangular matrix $A \in \mathbb{R}^{m \times n}$ with $m > n$, we can also apply the QR decomposition as 
$$
A = Q \begin{pmatrix} R \\ 0 \end{pmatrix}
$$
where $R$ is an upper triangular matrix.

**QR Iteration**: We can use the QR iteration to find the eigenvalues and eigenvectors: Given $A_0 = A$, 
$$
\begin{aligned}
A_{k} &= Q_k R_k \\
A_{k+1} &= R_k Q_k
\end{aligned}
$$

**Theorem**(Convergence of QR Iteration): If $A$ has distinct eigenvalues $|\lambda_1| > |\lambda_2| > \cdots > |\lambda_n|>0$, then $A_k$ will converge to an upper triangular matrix with the eigenvalues on the diagonal. $\tilde Q_k = Q_1 Q_2 \cdots Q_k$ will converge to the eigenvectors. 

![QR Iteration](./eigen.assets/QR.gif)

In fact, we can see from the QR iteration that
$$
A^k = Q_1 Q_2 \cdots Q_k R_k R_{k-1} \cdots R_1.
$$
Therefore, the QR algorithm can be seen as a more sophisticated variation of the basic "power" eigenvalue algorithm. 

Here's a simple implementation of the QR iteration method:

```python
import numpy as np

# Create a symmetric test matrix
n = 4
A = np.random.rand(n, n)
A = (A + A.T) / 2  

# Initialize
A_k = A.copy()
V = np.eye(n)  # To accumulate eigenvectors
tol = 1e-8
max_iter = 1000

# QR iteration
for _ in range(max_iter):
    # QR decomposition
    Q, R = np.linalg.qr(A_k)
    
    # Update A_k
    A_k = R @ Q
    
    # Accumulate eigenvectors
    V = V @ Q
    
    # Check convergence of off-diagonal elements
    if np.sum(np.abs(A_k - np.diag(np.diag(A_k)))) < tol:
        break

# Eigenvalues are on the diagonal of final A_k
eigenvals = np.diag(A_k)
```

The QR iteration method above is just the basic version. It can be further accelerated by applying the shift technique like the power method. In fact, the state-of-the-art [Francis algorithm](https://en.wikipedia.org/wiki/QR_algorithm) applies the famous implicit double-shift with no QR decompositions being explicitly performed. 



