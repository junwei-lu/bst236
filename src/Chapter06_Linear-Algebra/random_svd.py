import numpy as np

def randomized_svd(A, rank, s):
    m, n = A.shape

    # Generate a random Gaussian matrix
    S = np.random.randn(n, s)

    # Form the sample matrix Z, which is m x k
    Y = A @ S

    # Orthonormalize Y using QR decomposition
    Q, _ = np.linalg.qr(Y)

    # Obtain the low-rank approximation of A
    B = Q.T @ A

    # Perform SVD on the low-rank approximation
    U_tilde, Sigma, Vt = np.linalg.svd(B, full_matrices=False)

    # Obtain the final singular vectors
    U = Q @ U_tilde

    return U, Sigma, Vt


