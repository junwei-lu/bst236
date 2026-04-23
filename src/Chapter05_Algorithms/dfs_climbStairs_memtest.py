"""
When a value is calculated for the first time, 
we record it in mem[i] for later use.
When that value needs to be calculated again, 
we can directly retrieve the result from mem[i],
thus avoiding redundant calculations of that subproblem.
"""

class Solution:
    def dfs(self, i: int, mem: list[int]) -> int:
        """Search"""
        # Known dp[1] and dp[2], return them
        if i == 1 or i == 2:
            mem[i] = i
            return i
        # dp[i] = dp[i-1] + dp[i-2]
        if mem[i] != -1:
            return mem[i]
        count = self.dfs(i - 1,mem) + self.dfs(i - 2,mem)
        mem[i] = count
        return count

    def climbStairs(self, n: int) -> int:
        """Climbing stairs: Search"""
        mem = [-1] * (n+1)
        return self.dfs(n,mem)
