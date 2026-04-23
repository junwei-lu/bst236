import numpy as np
from scipy.sparse.linalg import eigs, eigsh, svds
import time

# Create a symmetric matrix
n = 1000
np.random.seed(0)
A = np.random.rand(n, n)
A = A + A.T  # Make symmetric

k = 10  # Number of eigenvalues for partial computations

# Time full eigendecomposition (eig)
start_time = time.time()
eigenvalues, eigenvectors = np.linalg.eig(A)
eig_time = time.time() - start_time

# Time eigenvalues only (eigvals)
start_time = time.time()
eigenvalues = np.linalg.eigvals(A)
eigvals_time = time.time() - start_time

# Time symmetric eigendecomposition (eigh)
start_time = time.time()
eigenvalues, eigenvectors = np.linalg.eigh(A)
eigh_time = time.time() - start_time

# Time symmetric eigenvalues only (eigvalsh)
start_time = time.time()
eigenvalues = np.linalg.eigvalsh(A)
eigvalsh_time = time.time() - start_time

# Time full SVD
start_time = time.time()
U, S, VT = np.linalg.svd(A, full_matrices=False)
svd_time = time.time() - start_time

# Time singular values only
start_time = time.time()
singular_values = np.linalg.svdvals(A)
svdvals_time = time.time() - start_time

# Time partial eigendecomposition (k largest eigenvalues)
start_time = time.time()
eigenvalues, eigenvectors = eigs(A, k=k, which='LM')
partial_eigs_time = time.time() - start_time

# Time partial symmetric eigendecomposition
start_time = time.time()
eigenvalues, eigenvectors = eigsh(A, k=k, which='LA')
partial_eigsh_time = time.time() - start_time

# Time partial SVD
start_time = time.time()
U, S, VT = svds(A, k=k)
partial_svd_time = time.time() - start_time

# Print results in a organized way
print("\nFull decomposition times:")
print(f"{'Full eigendecomposition (eig):':<40} {eig_time:.6f} seconds")
print(f"{'Eigenvalues only (eigvals):':<40} {eigvals_time:.6f} seconds")
print(f"{'Symmetric eigendecomposition (eigh):':<40} {eigh_time:.6f} seconds")
print(f"{'Symmetric eigenvalues only (eigvalsh):':<40} {eigvalsh_time:.6f} seconds")
print(f"{'Full SVD:':<40} {svd_time:.6f} seconds")
print(f"{'Singular values only (svdvals):':<40} {svdvals_time:.6f} seconds")

print("\nPartial decomposition times (k={k} components):")
print(f"{'Partial eigendecomposition (eigs):':<40} {partial_eigs_time:.6f} seconds")
print(f"{'Partial symmetric eigen (eigsh):':<40} {partial_eigsh_time:.6f} seconds")
print(f"{'Partial SVD (svds):':<40} {partial_svd_time:.6f} seconds")