import numpy as np
from numpy.linalg import norm
from scipy import linalg, optimize
import time
import matplotlib.pyplot as plt

# Set dimensions: m x n times n x p
n = 1000

# Set random seed for reproducibility
np.random.seed(42)

# Generate random matrices A and B
A = np.random.randn(n, n)
B = np.random.randn(n, n)
# Try different sketch sizes and measure time/error
sketch_sizes = [1, 10, 20, 50, 100, 200, 400, 800,1000]
times = []
errors = []

# Compute true product once
C_true = A @ B

for s in sketch_sizes:
    # Time the sketched multiplication
    start = time.time()
    
    # Create random projection matrix
    S = np.random.randn(n, s) / np.sqrt(s)
    
    # Compute sketched matrices and product
    Y = A @ S
    Z = S.T @ B
    C_approx = Y @ Z
    
    times.append(time.time() - start)
    
    # Calculate relative error
    error = norm(C_true - C_approx, 'fro') / norm(C_true, 'fro')
    errors.append(error)

# Create figure with two y-axes
fig, ax1 = plt.subplots()
sketch_ratio = np.array(sketch_sizes) / n

# Plot running time
ax1.set_xlabel('Sketch size / Matrix size')
ax1.set_ylabel('Running time (s)', color='tab:blue')
ax1.plot(sketch_ratio, times, color='tab:blue', marker='o')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Plot error on second y-axis
ax2 = ax1.twinx()
ax2.set_ylabel('Relative error', color='tab:red')
ax2.plot(sketch_ratio, errors, color='tab:red', marker='s')
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title('Sketch Size vs Time and Error')
plt.show()


# # Randomized Linear Algebra

# def sketch_linear_system(A, b, sketch_size):
#     m, n = A.shape
#     # Create a sketching matrix S (e.g., Gaussian)
#     S = np.random.randn(sketch_size, m) / np.sqrt(sketch_size)  # sketch_size << m
#     # Compute the sketched system
#     A_sketch = S @ A
#     b_sketch = S @ b
#     # Solve the smaller least squares problem
#     x_approx, _, _, _ = np.linalg.lstsq(A_sketch, b_sketch)
#     return x_approx


# A = A @ A.T + np.eye(n)
# b = np.random.randn(n)

# # Time np.linalg.solve()
# start_time = time.time()
# x1 = linalg.solve(A, b)
# linalg_time = time.time() - start_time


# for s in sketch_sizes:
#     # Time the sketched multiplication
#     start = time.time()
    
#     # Create random projection matrix
#     x_approx = sketch_linear_system(A, b, s)
    
#     times.append(time.time() - start)
    
#     # Calculate relative error
#     error = norm(x1 - x_approx, 2) / norm(x1, 2)
#     errors.append(error)

# # Create figure with two y-axes
# fig, ax1 = plt.subplots()
# sketch_ratio = np.array(sketch_sizes) / n

# # Plot running time
# ax1.set_xlabel('Sketch size / Matrix size')
# ax1.set_ylabel('Running time (s)', color='tab:blue')
# ax1.plot(sketch_ratio, times, color='tab:blue', marker='o')
# ax1.tick_params(axis='y', labelcolor='tab:blue')

# # Plot error on second y-axis
# ax2 = ax1.twinx()
# ax2.set_ylabel('Relative error', color='tab:red')
# ax2.plot(sketch_ratio, errors, color='tab:red', marker='s')
# ax2.tick_params(axis='y', labelcolor='tab:red')

# plt.title('Linear System: Sketch Size vs Time and Error')
# plt.show()