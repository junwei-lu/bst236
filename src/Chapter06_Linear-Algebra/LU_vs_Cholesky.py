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

# Time Cholesky decomposition
start_time = time.time()
L = linalg.cholesky(A)
y = linalg.solve_triangular(L, b, lower=True)  # Forward substitution
x2 = linalg.solve_triangular(L.T, y, lower=False)  # Backward substitution
cholesky_time = time.time() - start_time

# Use symmetric argument
start_time = time.time()
x3 = linalg.solve(A, b, assume_a='pos')
solve_pos_time = time.time() - start_time

# Use symmetric argument
start_time = time.time()
x4 = linalg.solve(A, b, assume_a='sym')
solve_sym_time = time.time() - start_time

# Print results
print(f'General solve Time: {linalg_time:.6f} seconds') # 0.669566 seconds
print(f'Symmetric solve Time: {solve_sym_time:.6f} seconds') # 1.184631 seconds
print(f'Symmetric positive definite solve Time: {solve_pos_time:.6f} seconds') # 0.468433 seconds
print(f'Manually Cholesky Decomposition Time: {cholesky_time:.6f} seconds') # 0.369518 seconds
#print(f'LDL Decomposition Time: {ldl_time:.6f} seconds') # 0.369518 seconds