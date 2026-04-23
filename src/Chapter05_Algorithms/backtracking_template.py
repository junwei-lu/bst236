def backtrack(problem, state, choices, result):
    """
    A generic DFS template for backtracking problems.
    """
    # Check if the current state is a complete solution
    if is_solution(state, problem):
        result.append(state[:])  # Append a copy of the solution
        return

    # Iterate through all possible choices
    for choice in generate_choices(state, problem):
        if is_valid(choice, state, problem):  # Pruning:Check if the choice is valid
            state.append(choice)  # Trying
            backtrack(problem, state, choices, result)  # Explore further with this choice
            state.pop()  # Retreating: Undo the choice