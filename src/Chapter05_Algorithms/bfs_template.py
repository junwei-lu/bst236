from collections import deque

def bfs_template(problem, start_state):
    # Initialize queue
    result = []
    queue = deque([(start_state, result)]) 
    # Initialize visited set
    visited = []
    while queue:
        # Get the current state
        current_state, current_result = queue.popleft() 

        # Check if current state is the goal
        if is_solution(current_state):
            return current_result

        # Explore next possible states
        for next_state in generate_choices(current_state,problem):
            if is_valid(next_state,problem):
                if next_state in visited:
                    continue # Memoization
                visited.append(next_state)
                # Compute new result based on current result and next state
                next_result = compute_result(current_result, next_state)
                queue.append((next_state, next_result))
    return None # No goal state found