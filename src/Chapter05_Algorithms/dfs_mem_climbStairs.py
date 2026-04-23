"""
When a value is calculated for the first time, 
we record it in mem[i] for later use.
When that value needs to be calculated again, 
we can directly retrieve the result from mem[i],
thus avoiding redundant calculations of that subproblem.
"""

class Solution:
    def dfs(self, i: int, mem: list[int]) -> int:
        """Memoized search"""
        # Known dp[1] and dp[2], return them
        if i == 1 or i == 2:
            return i
        # If there is a record for dp[i], return it
        if mem[i] != -1:
            return mem[i]
        # dp[i] = dp[i-1] + dp[i-2]
        count = self.dfs(i - 1, mem) + self.dfs(i - 2, mem)
        # Record dp[i]
        mem[i] = count
        return count

    def climb_stairs_dfs_mem(self, n: int) -> int:
        """Climbing stairs: Memoized search"""
        # mem[i] records the total number of solutions for climbing to the ith step, -1 means no record
        mem = [-1] * (n + 1)
        return self.dfs(n, mem)
