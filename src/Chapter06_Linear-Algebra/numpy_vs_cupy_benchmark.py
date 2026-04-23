import numpy as np
import cupy as cp
import time


# Set sizes for benchmarking
sizes = [1000, 4000, 8000]

for size in sizes:
    print(f"\n=== Testing with {size}x{size} matrices ===")
    
    # Create test data
    np_A = np.random.rand(size, size).astype(np.float32)
    np_B = np.random.rand(size, size).astype(np.float32)
    np_b = np.random.rand(size).astype(np.float32)
    
    # Make matrix invertible for solve
    np_A_solve = np_A + np.eye(size) * size
    
    cp_A = cp.asarray(np_A)
    cp_B = cp.asarray(np_B)
    cp_b = cp.asarray(np_b)
    cp_A_solve = cp.asarray(np_A_solve)
    
    # 1. Matrix Multiplication
    print("\nMatrix Multiplication:")
    
    # NumPy
    start = time.time()
    np_result = np.matmul(np_A, np_B)
    np_time = time.time() - start
    print(f"NumPy time: {np_time:.4f} seconds")
    
    # CuPy
    start = time.time()
    cp_result = cp.matmul(cp_A, cp_B)
    # Synchronize GPU to get accurate timing
    cp.cuda.stream.get_current_stream().synchronize()
    cp_time = time.time() - start
    print(f"CuPy time:  {cp_time:.4f} seconds")
    print(f"Speedup:    {np_time/cp_time:.2f}x")
    
    # 2. Linear Equation Solve
    print("\nLinear Equation Solving:")
    
    # NumPy
    start = time.time()
    np_solve = np.linalg.solve(np_A_solve, np_b)
    np_time = time.time() - start
    print(f"NumPy time: {np_time:.4f} seconds")
    
    # CuPy
    start = time.time()
    cp_solve = cp.linalg.solve(cp_A_solve, cp_b)
    cp.cuda.stream.get_current_stream().synchronize()
    cp_time = time.time() - start
    print(f"CuPy time:  {cp_time:.4f} seconds")
    print(f"Speedup:    {np_time/cp_time:.2f}x")
    
    # 3. SVD
    print("\nSingular Value Decomposition:")
    
    # NumPy
    start = time.time()
    np_U, np_S, np_Vh = np.linalg.svd(np_A, full_matrices=False)
    np_time = time.time() - start
    print(f"NumPy time: {np_time:.4f} seconds")
    
    # CuPy
    start = time.time()
    cp_U, cp_S, cp_Vh = cp.linalg.svd(cp_A, full_matrices=False)
    cp.cuda.stream.get_current_stream().synchronize()
    cp_time = time.time() - start
    print(f"CuPy time:  {cp_time:.4f} seconds")
    print(f"Speedup:    {np_time/cp_time:.2f}x") 