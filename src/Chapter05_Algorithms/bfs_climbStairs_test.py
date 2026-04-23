from collections import deque
# choices  = [1,2]
# problem = n
# start_state = 0
# state = which stairs we are
# result = # of steps 
def bfs_climbStairs(n, choices):
    # Initialize queue
    result = 0
    start_state = 0
    queue = deque([(start_state, result)]) 
    # Initialize visited set
    visited = []
    while queue:
        # Get the current state
        current_state, current_result = queue.popleft() 

        # Check if current state is the goal
        if current_state == n:
            return current_result

        # Explore next possible states
        for next_state in choices + current_state:
            if next_state <= n:
                if next_state in visited:
                    continue # Memoization
                visited.append(next_state)
                # Compute new result based on current result and next state
                next_result = current_result + 1
                queue.append((next_state, next_result))
    return None # No goal state found