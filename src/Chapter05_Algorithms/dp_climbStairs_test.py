class Solution:
    def climbStairs(self, n: int) -> int:
        """Climbing stairs: Dynamic programming"""
        if n == 1 or n ==2:
            return n
        # Initialize dp table, used to store subproblem solutions
        # dp = [0] * (n+1)
        # Initial state: preset the smallest subproblem solution
        a, b = 1, 2
        # State transition: gradually solve larger subproblems from smaller ones
        for _ in range(3,n+1):
            b, a = a, a+b
        return b
    
    # Save space
    
