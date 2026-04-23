def climbing_stairs_dp(n: int) -> int:
    """Climbing stairs: Dynamic programming"""
    if n == 1 or n == 2:
        return n
    # Initialize dp table, used to store subproblem solutions
    dp = [0] * (n + 1)
    # Initial state: preset the smallest subproblem solution
    dp[1], dp[2] = 1, 2
    # State transition: gradually solve larger subproblems from smaller ones
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]