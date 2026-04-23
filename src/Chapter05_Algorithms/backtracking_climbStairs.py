def is_solution(state: list[int], n: int) -> bool:
    """A state is a solution if the sum of moves equals n (i.e. reached the top)."""
    return sum(state) == n

def generate_choices(state: list[int], n: int) -> list[int]:
    """For climbing stairs, the choices are always 1 or 2 steps."""
    return [1, 2]

def is_valid(choice: int, state: list[int], n: int) -> bool:
    """A choice is valid if adding it to the current state does not exceed n."""
    return sum(state) + choice <= n

def backtrack(n: int, state: list[int], choices: list[int], result: list[list[int]]) -> None:
    """
    A generic DFS template for backtracking problems applied to the climbing stairs problem.
    - state: list of moves (e.g., [1, 2, 1] means first move 1 step, then 2 steps, then 1 step).
    - n: the total number of steps in the staircase.
    """
    if is_solution(state, n):
        result.append(state[:])  # Append a copy of the complete solution
        return

    for choice in generate_choices(state, n):
        if is_valid(choice, state, n):  # Pruning: do not exceed n steps
            state.append(choice)        # Trying
            backtrack(n, state, choices, result)  # Explore further
            state.pop()                 # Retreating: Undo the choice

def climbing_stairs_paths(n: int) -> int:
    """
    Returns the number of distinct ways to climb a staircase with n steps,
    where you can climb either 1 or 2 steps at a time.
    """
    result = []
    backtrack(n, [], [1, 2], result)
    return len(result)

# Example usage:
if __name__ == "__main__":
    n = 4
    print(f"Number of ways to climb {n} steps: {climbing_stairs_paths(n)}")
