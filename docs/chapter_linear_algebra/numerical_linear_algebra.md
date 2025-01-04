# Introduction to Numerical Linear Algebra

There are two major tasks in numerical linear algebra:

1. Solving linear systems
2. Computing eigenvalues and eigenvectors

As this course is an introductory course of general computation, we will not cover the details of algorithms solving these two problems. Instead, we will focus on the practical implementations. We suggest the readers to refer to the textbooks on numerical linear algebra for more details.

## Complexity of Matrix Operations

We discussed the two important standards for a good algorithm in the [Complexity Analysis](../chapter_computational_complexity/index.md):

- **Time complexity**: $A$ is $\ell$ by $m$ matrix, $B$ is $m$ by $n$ matrix, the time complexity of $AB$ is $O(\ell m n)$.
- **Space complexity**: Storage of $A$ and $B$ requires $O(\ell m + m n)$ space.

Actually, the time complexity of multiplying two $n$ by $n$ matrices can be faster than $O(n^3)$. Let $\omega$ be the exponent of the time complexity $O(n^\omega)$ of multiplying two $n$ by $n$ matrices. The naive range is $2 \le \omega \le 3$. It is still an open problem to find the minimum $\omega$. Till 2024, the best $\omega$ is 2.371339. We refer the readers to the [Wikipedia page](https://en.wikipedia.org/wiki/Computational_complexity_of_matrix_multiplication) for more details.



However, the real computation time is not only determined by the time complexity. In practice, even two algorithms with the same time complexity can have different running time. 
To understand why this situation occurs, we first need to understand the storage structure of modern computers. Modern computers use a multi-level storage system: registers, caches, main memory (RAM), disks, and tapes. When a computer directly exchanges information with registers, calculations occur by transferring the required data sequentially from higher-level to lower-level storage, starting from registers. Data at the end of the calculation process is then sequentially stored back into lower-level storage. As the storage level decreases (e.g., from registers to disks or tapes), the access speed to the data slows significantly. Generally, there are quantitative differences between these levels. Registers are extremely fast and efficient, while disks and tapes are relatively slow. Therefore, the capacity of disks and tapes is large, while the capacities of caches and registers are relatively small.
Based on this hierarchical storage structure of computers, when designing software, we should aim to minimize data transfers between external storage, registers, and main memory. 

![Storage Structure of Modern Computers](./numerical_linear_algebra.assets/storage_pyramid.png)

To summary, in modern computers, <u>computation is much faster than data communication</u>. Usually the bottleneck of your code is determined by the volume of data communication.
Suppose a particular computational task requires a total of $f$ operations and $m$ data retrievals; we define:

$$q = \frac{f}{m}$$

to represent the average number of computations performed per data retrieval. Our goal is to maximize $q$ to improve the efficiency of the computation.

The following table shows the ratio $q$ for some common matrix operations.

| Typical Operation | $f$ (Number of Computations) | $m$ (Number of Data Retrievals) | $q = f/m$ (Ratio) |
|------------------|------------------------------|--------------------------------|-------------------|
| $y \gets y + \alpha x$ | $2n$ | $3n + 1$ | $\frac{2}{3}$ |
| $y \gets y + Ax$ | $2n^2$ | $n^2 + 3n$ | $2$ |
| $C \gets C + AB$ | $2n^3$ | $4n^2$ | $\frac{n}{2}$ |



From the **table**, it is evident that the efficiency of matrix-matrix operations is the highest, with an average of computations per data retrieval. Therefore, in the design of matrix-related algorithms, we tend to maximize the use of matrix-matrix operations <u>when the computation complexity is same</u>. 

We want to emphasize again that the above analysis only applies to when the computation complexities are same. When you want to compute $ABx$ for the matrices $A, B$ and vector $x$, the complexity of $(AB)x$ is $O(n^3)$ while the complexity of $A(Bx)$ is $O(n^2)$. Therefore, we should compute $Bx$ first in this case.

## Sensitivity and Stability


Besides the time complexity and space complexity, we also need to consider the following two standards for numerical algorithms:

- **Sensitivity**: How much the numerical problem is affected by the small perturbations in the input data
- **Stability**: When running the algorithm, how do the roundoff (and other) errors affect the accuracy of the computed solution.

Notice that the sensitivity is fully determined by the problem itself and independent of the algorithm to solve it, while the stability is determined by the algorithm.

### Sensitivity Analysis

**Definition**(Sensitivity). We want to compute $f(x)$ for a given $x$. If $x$ is perturbed by $\delta x$, the output is $f(x + \delta x)$. Whyen $|\delta x|/|x|$ is small, we want to find a constant $c(x)$ as small as possible such that
$$
|f(x + \delta x) - f(x)| \le c(x) \frac{|\delta x|}{|x|}.
$$
We call $c(x)$ the **condition number** of the problem. We say the problem is **well-conditioned** if $c(x)$ is bounded, and **ill-conditioned** if $c(x)$ is large. 

We can see that when $f(x)$ is differentiable
$$
c(x) \approx \frac{|f'(x)||x|}{|f(x)|}.
$$
However, for complex problems, it is not easy to compute the condition number. We will discuss the condition number of linear algebra problems in the following section.


### Backward Error Analysis

We all know that computers using the floating point arithmetic. 
For example, suppose the computer can only store 2 digits after the decimal point. When we compute $\pi + e$, the computer actually computes float$(\pi + e) = 3.14 + 2.72 = 5.86$ with a roundoff error $\epsilon \approx 0.00012$. Therefore, it is important to analyze how the roundoff errors affect the accuracy of algorithms. Even if a numerical problem is well-conditioned, a bad algorithm can still lead to a large error.

**Definition**(Backward Error Analysis). Suppose $\hat{f}$ is the algorithm to compute $f$. Suppose there exists a $\delta x$ such that 
$$
\hat{f}(x) = f(x + \delta x), \quad |\delta x| \le \epsilon |x|,
$$ 
where $\epsilon$ relates to the machine precision and the problem. We say the algorithm is **numerically stable** if $\epsilon$ is small.

Combining the sensitivity analysis and the backward error analysis, we can have the overall error bound of the result:
$$
\frac{|\hat{f}(x) - f(x)|}{|f(x)|} = \frac{|f(x + \delta x) - f(x)|}{|f(x)|} \le c(x) \frac{|\delta x|}{|x|} \le c(x) \epsilon.
$$
We can get a reliable result when we apply a numerically stable algorithm to a well-conditioned problem.

In this course, we will not discuss the details of backward error analysis as most of the algorithms we will use are numerically stable. Instead, we will just give a simple tip for possible numerical instabilities in practice.

**Tip**: Be careful with operations (especially multiplication and division) involving vastly different magnitudes.

Example 1. Suppose we want to compute: $(10^8 \times 1.23456789)/10^8$

This can be done in two different orders:

Method 1: Multiply then divide
1. Compute $10^8 \times 1.23456789 = 123456789.0$ 
2. Divide by $10^8$: $123456789.0/10^8 = 1.23456789$
This may have overflow error in the first step.

Method 2: Divide then multiply
1. Compute $1.23456789/10^8= 0.00000001$
2. Multiply by $10^8$: $0.00000001 \times 10^8 = 1.00000000$
The roundoff error makes the result not accurate.

Method 3: Divide numbers with same order of magnitude
1. Compute $10^8/10^8= 1$
2. Multiply by $1.23456789$: $1 \times 1.23456789 = 1.23456789$

Example 2. Suppose we want to compute $e^{x} - e^{-x}$ for $x$ around 0. It is better to compute $e^x - e^{-x} = 2x + \frac{2x^3}{3!} + \frac{2x^5}{5!} + \cdots$

