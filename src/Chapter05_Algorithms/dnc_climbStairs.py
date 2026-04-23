import numpy as np

"""
Binary exponentiation algorithm for matrix power:

1. Base case:
   - If exponent is 0, return identity matrix
   - This is because any matrix raised to power 0 equals identity matrix

2. Even exponent case:
   - If n is even, compute A^(n/2) recursively 
   - Square the result to get A^n
   - This works because A^n = (A^(n/2))^2

3. Odd exponent case:
   - If n is odd, compute A^(n/2) recursively
   - Square the result to get A^(n-1)
   - Multiply by A one more time to get A^n
   - This works because A^n = (A^(n/2))^2 * A
"""

def binpow(A: np.ndarray, n: int) -> np.ndarray:
    """Calculate the power of matrix A to the n using binary exponentiation."""
    if n == 0:
        # Return the identity matrix of the same size as A
        return np.eye(A.shape[0], dtype=int)
    res = binpow(A, n // 2)
    if n % 2 == 0:
        return res @ res
    else:
        return res @ res @ A

# Applied to the climbing stairs problem
def climbing_stairs_binpow(n: int) -> int:
    A = np.array([[1, 1], [1, 0]], dtype=int)
    return binpow(A, n)[0][0]