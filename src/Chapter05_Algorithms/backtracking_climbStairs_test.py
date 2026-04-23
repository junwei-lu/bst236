class Solution:
    # problem = n 
    # choices = [1,2]
    # state = list of choices e.g [1,1,2,1,..]
    def backtrack_climbStairs(self, problem, state, choices=[1, 2], result=[]):
        """
        A generic DFS template for backtracking problems.
        """
        # Check if the current state is a complete solution
        if sum(state) == problem:
            result.append(state[:])  # Append a copy of the solution
            return

        # Iterate through all possible choices
        for choice in choices:  # choices = [1, 2]
            # Pruning: Check if the choice is valid
            if sum(state) + choice <= problem:
                state.append(choice)  # Trying
                self.backtrack_climbStairs(problem, state, choices, result)  # Explore further with this choice
                state.pop()  # Retreating: Undo the choice

    def climbStairs(self, n: int) -> int:
        problem = n
        choices = [1, 2]
        state = []
        result = []
        self.backtrack_climbStairs(problem, state, choices, result)
        return len(result)



    