import numpy as np
import scipy.sparse as sparse
from scipy.sparse import linalg
import time

def create_sparse_matrix(size, density=0.01):
    """Create a sparse matrix with given size and density."""
    # Create a random sparse matrix
    matrix = sparse.random(size, size, density=density, format='csr')
    # Make it diagonally dominant to ensure it's invertible
    matrix = matrix + sparse.eye(size, format='csr') * size
    return matrix

def naive_solve(A, b):
    """Solve Ax = b using numpy's dense solver."""
    start_time = time.time()
    x = np.linalg.solve(A.toarray(), b)
    end_time = time.time()
    return x, end_time - start_time

def sparse_direct_solve(A, b):
    """Solve Ax = b using scipy's sparse direct solver."""
    start_time = time.time()
    x = linalg.spsolve(A, b)
    end_time = time.time()
    return x, end_time - start_time

def gmres_solve(A, b, tol=1e-10):
    """Solve Ax = b using GMRES iterative solver."""
    start_time = time.time()
    # ILU preconditioner
    # P = linalg.spilu(A)
    # M = linalg.LinearOperator(matvec=P.solve, shape=A.shape, dtype=A.dtype)
    # x, info = linalg.gmres(A, b, M=M, rtol=tol)
    x, info = linalg.gmres(A, b, rtol=tol)
    end_time = time.time()
    return x, end_time - start_time, info

def bicg_solve(A, b, tol=1e-10):
    """Solve Ax = b using BiCG iterative solver."""
    start_time = time.time()
    x, info = linalg.bicg(A, b, rtol=tol)
    end_time = time.time()
    return x, end_time - start_time, info

def relative_error(x_true, x_approx):
    """Compute the relative error between the true and approximate solutions."""
    return np.linalg.norm(x_true - x_approx) / np.linalg.norm(x_true)

def compare_solvers(sizes):
    """Compare different solvers for different matrix sizes."""
    results = {
        'size': [],
        'naive': [],
        'spsolve': [],
        'gmres': [],
        'bicg': [],
        'rel_err_spsolve': [],
        'rel_err_gmres': [],
        'rel_err_bicg': []
    }
    
    for size in sizes:
        print(f"\nTesting with matrix size {size}x{size}")
        
        # Create sparse matrix and right-hand side
        A = create_sparse_matrix(size)
        b = np.ones(size)
        
        results['size'].append(size)
        
        # Test naive solver (only for smaller matrices)
        if size <= 5000:  # Limit for dense solver to avoid memory issues
            try:
                x_naive, time_naive = naive_solve(A, b)
                results['naive'].append(time_naive)
                print(f"Naive solve time: {time_naive:.4f} seconds")
            except MemoryError:
                results['naive'].append(None)
                print("Naive solve: Memory error")
                continue  # Skip other solvers if naive fails
        else:
            x_naive, time_naive = naive_solve(A, b)
            results['naive'].append(time_naive)
            print(f"Naive solve time: {time_naive:.4f} seconds")
        
        # Test sparse direct solver
        try:
            x_spsolve, time_spsolve = sparse_direct_solve(A, b)
            results['spsolve'].append(time_spsolve)
            rel_err = relative_error(x_naive, x_spsolve)
            results['rel_err_spsolve'].append(rel_err)
            print(f"Sparse direct solve time: {time_spsolve:.4f} seconds, relative error: {rel_err:.2e}")
        except Exception as e:
            results['spsolve'].append(None)
            results['rel_err_spsolve'].append(None)
            print(f"Sparse direct solve: {str(e)}")
        
        # Test GMRES
        try:
            x_gmres, time_gmres, info_gmres = gmres_solve(A, b)
            results['gmres'].append(time_gmres)
            rel_err = relative_error(x_naive, x_gmres)
            results['rel_err_gmres'].append(rel_err)
            print(f"GMRES time: {time_gmres:.4f} seconds (info: {info_gmres}), relative error: {rel_err:.2e}")
        except Exception as e:
            results['gmres'].append(None)
            results['rel_err_gmres'].append(None)
            print(f"GMRES: {str(e)}")
        
        # Test BiCG
        try:
            x_bicg, time_bicg, info_bicg = bicg_solve(A, b)
            results['bicg'].append(time_bicg)
            rel_err = relative_error(x_naive, x_bicg)
            results['rel_err_bicg'].append(rel_err)
            print(f"BiCG time: {time_bicg:.4f} seconds (info: {info_bicg}), relative error: {rel_err:.2e}")
        except Exception as e:
            results['bicg'].append(None)
            results['rel_err_bicg'].append(None)
            print(f"BiCG: {str(e)}")
    
    return results

def main():
    # Define three different matrix sizes to test
    # Small, medium, and large
    sizes = [1000, 5000, 10000]
    
    # Compare solvers
    results = compare_solvers(sizes)
    
    # Print summary
    print("\nSummary:")
    print("=" * 80)
    print(f"{'Size':<10} {'Naive':<15} {'SpSolve':<15} {'GMRES':<15} {'BiCG':<15}")
    print(f"{'(Time)':<10} {'(Time)':<15} {'(Time/Error)':<15} {'(Time/Error)':<15} {'(Time/Error)':<15}")
    print("-" * 80)
    
    for i, size in enumerate(results['size']):
        naive = f"{results['naive'][i]:.4f}" if results['naive'][i] is not None else "N/A"
        
        if results['spsolve'][i] is not None and results['rel_err_spsolve'][i] is not None:
            spsolve = f"{results['spsolve'][i]:.4f}/{results['rel_err_spsolve'][i]:.2e}"
        else:
            spsolve = "N/A"
            
        if results['gmres'][i] is not None and results['rel_err_gmres'][i] is not None:
            gmres = f"{results['gmres'][i]:.4f}/{results['rel_err_gmres'][i]:.2e}"
        else:
            gmres = "N/A"
            
        if results['bicg'][i] is not None and results['rel_err_bicg'][i] is not None:
            bicg = f"{results['bicg'][i]:.4f}/{results['rel_err_bicg'][i]:.2e}"
        else:
            bicg = "N/A"
        
        print(f"{size:<10} {naive:<15} {spsolve:<15} {gmres:<15} {bicg:<15}")

if __name__ == "__main__":
    main()