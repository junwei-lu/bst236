import numpy as np

class Solutions:
    """Binary exponentiation algorithm for matrix power:
        
        1. Base case (n=0): Return identity matrix
        2. Even exponent: A^n = (A^(n/2))^2 
        3. Odd exponent: A^n = (A^(n/2))^2 * A
        
        This divide-and-conquer approach reduces time complexity from O(n) to O(log n).
    """
    def binpow(self, A: np.ndarray, n: int) -> np.ndarray:
        # Base case (n=0)
        if n == 0:
            return np.eye(2)

        # Divide step
        res = self.binpow(A, n//2)

        # Conquer step
        # Even exponent: A^n = (A^(n/2))^2 
        if n%2 == 0:
            return res @ res
        else:
            return res @ res @ A
        # Odd exponent: A^n = (A^(n/2))^2 * A

    def climbStairs(self, n: int) -> int:
        """Climbing stairs: Binary exponentiation algorithm"""
        A = np.array([[1, 1], [1, 0]], dtype=int)
        return self.binpow(A, n)[0][0]