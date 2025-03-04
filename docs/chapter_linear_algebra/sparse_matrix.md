# Sparse Matrix

Sparse matrices are matrices where most elements are zero. They are common in many applications like:

- Social networks (adjacency matrices)
- Genome-wide association studies (SNP-by-gene matrices)
- Language models (word-by-word matrices)
- Numerical PDEs

## Basic Operations

The `scipy.sparse` module provides several sparse matrix formats. Refer to the [scipy documentation](https://docs.scipy.org/doc/scipy/reference/sparse.html) for more details. The most commonly used are:

- CSR (Compressed Sparse Row): efficient for row operations and matrix-vector products (thus this is most used)
- CSC (Compressed Sparse Column): efficient for column operations
- COO (Coordinate): good for constructing matrices
- LIL (List of Lists): good for incremental matrix construction

Here's how to create and manipulate sparse matrices:

```python
import numpy as np
from scipy import sparse

# Create a sparse matrix from dense
dense_matrix = np.array([[1, 0, 0], [0, 2, 3], [4, 0, 0]])
sparse_csr = sparse.csr_matrix(dense_matrix)

# Basic properties
print(sparse_csr.data)      # Non-zero elements
print(sparse_csr.indices)   # Column indices
print(sparse_csr.indptr)    # Row pointer
print(sparse_csr.shape)     # Matrix dimensions
print(sparse_csr.nnz)       # Number of non-zero elements

# Basic operations
A = sparse_csr
B = sparse.csr_matrix([[1, 1, 0], [0, 1, 0], [0, 0, 1]])

# Addition and multiplication
C = A + B
D = A @ B                   # Matrix multiplication
E = A.multiply(B)          # Element-wise multiplication

# Slicing
row_slice = A[1:3, :]
element = A[0, 1]

# Convert back to dense if needed
dense_A = A.toarray()
```

## Solving Linear Equations

For sparse linear systems $Ax = b$, scipy provides specialized solvers in the `scipy.sparse.linalg` module that are more efficient than dense matrix methods. Refer to the [scipy documentation](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html) for more details.


1. **For medium-sized systems (n < 10,000):**

	```python
	from scipy.sparse import linalg

	# Create a sparse positive definite matrix
	n = 1000
	A = sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), format='csr')
	b = np.ones(n)

	# Direct solver 
	x = linalg.spsolve(A, b)
	```
    
    The direct solver is the more accurate and predictable performance but it has higher memory usage and slower for very large systems.

2. **For large symmetric positive definite matrices:**
   ```python
   # Try conjugate gradient first with a preconditioner
   M = sparse.diags([A.diagonal()])  # Diagonal preconditioner
   x, info = linalg.cg(A, b, M=M, rtol=1e-10)
   
   # If CG fails or is slow, try direct solver
   if info != 0:
       x = linalg.spsolve(A, b)
   ```

   The conjugate gradient is an iterative method and the fastest and most memory efficient method for large symmetric positive definite matrices. Preconditioning is important for the conjugate gradient to converge. However, compared to the direct solver, the iterative methods are less accurate and may have convergence issues when the matrix is ill-conditioned.


3. **For very large non-symmetric matrices:**
   ```python
   # Always use preconditioner for large systems
   x, info = linalg.gmres(A, b, rtol=1e-10)
   # If memory is an issue, switch to BiCG
   if memory_error:
       x, info = linalg.bicg(A, b, rtol=1e-10)
   ```

   The GMRES is the most robust iterative method for large non-symmetric matrices. However, it requires more memory than the conjugate gradient. If the memory is an issue, we can switch to BiCG.

Please check the `info` return value for convergence of the iterative methods.


## Eigenvalues and Eigenvectors

For large sparse matrices, we usually only need a few eigenvalues/eigenvectors. The `scipy.sparse.linalg` module provides methods to compute them efficiently:

```python
from scipy.sparse.linalg import eigs, eigsh, svds
# Create a large sparse symmetric matrix
n = 1000
A = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(n, n), format='csr')

# Compute k largest magnitude eigenvalues
k = 5
eigenvalues, eigenvectors = eigs(A, k=k, which='LM')  # 'LM' selects largest magnitude eigenvalues
# For symmetric matrices
eigenvalues, eigenvectors = eigsh(A, k=k, which='LA')  # 'LA' selects largest algebraic eigenvalues
# Top k singular values and vectors
U, S, VT = svds(A, k=k)

# Different options for 'which':
# 'LM' : largest magnitude
# 'SM' : smallest magnitude
# 'LA' : largest algebraic (only for symmetric)
# 'SA' : smallest algebraic (only for symmetric)
```


## Performance Tips

1. **Choose the right format**:
   
      - CSR: best for row slicing, matrix-vector products
      - CSC: best for column slicing
      - COO: best for constructing matrices
      - LIL: best for incremental construction

2. **Avoid operations that create dense intermediates**:
   
   For example, the trick of saving the LU decomposition of the matrix to avoid recomputing it is no longer efficient for sparse matrices.

3. **Use specialized sparse operations**:
   ```python
   # Bad: converts to dense
   diagonal = np.diag(A.toarray())
   
   # Good: stays sparse
   diagonal = A.diagonal()
   ```

4. **Preconditioning for iterative solvers**:
   
      - Use appropriate preconditioners for faster convergence
      - Common choices: diagonal, incomplete LU, incomplete Cholesky

5. **Memory efficiency**:
      - Monitor memory usage with large matrices
      - Consider using `float32` instead of `float64` if precision allows

Remember that sparse matrices are most beneficial when the number of non-zero elements is much smaller than the total size of the matrix (typically < 10% non-zero).
