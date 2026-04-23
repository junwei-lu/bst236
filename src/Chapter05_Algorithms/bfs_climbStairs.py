from collections import deque

# result = # of moves
def bfs_climbStairs(n: int, choices: list[int]):
    # Initialize queue
    result = 0
    queue = deque([(0, result)]) 
    # Initialize visited set
    visited = []
    while queue:
        # Get the current state
        current_state, current_result = queue.popleft() 

        # Skip if current state is already visited
        if current_state in visited:
            continue
        visited.append(current_state)

        # Check if current state is the goal
        if current_state == n:
            return current_result

        # Explore next possible states
        next_states = current_state + choices
        for next_state in [i for i in next_states if i<=n]:
            # Compute new result based on current result and next state
            visited.append(next_state)
            next_result = current_result+1
            queue.append((next_state, next_result))

    return None # No goal state found