def min_path_sum_dp(grid: list[list[int]]) -> int:
    """Minimum path sum: Dynamic programming"""
    n, m = len(grid), len(grid[0])
    # Initialize dp table
    dp = [[0] * m for _ in range(n)]
    dp[0][0] = grid[0][0]
    # State transition: first row
    for j in range(1, m):
        dp[0][j] = dp[0][j - 1] + grid[0][j]
    # State transition: first column
    for i in range(1, n):
        dp[i][0] = dp[i - 1][0] + grid[i][0]
    # State transition: the rest of the rows and columns
    for i in range(1, n):
        for j in range(1, m):
            dp[i][j] = min(dp[i][j - 1], dp[i - 1][j]) + grid[i][j]
    return dp[n - 1][m - 1]