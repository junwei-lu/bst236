class Solutions:
    def climbStairs(self, n: int) -> int:
        """Climbing stairs: Closed form solution"""
        sqrt_5 = math.sqrt(5)
        return int((((1 + sqrt_5) / 2) ** (n + 1) - ((1 - sqrt_5) / 2) ** (n + 1)) / sqrt_5) 