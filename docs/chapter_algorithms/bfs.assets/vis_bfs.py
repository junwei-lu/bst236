import matplotlib.pyplot as plt
from collections import deque

def min_moves_to_climb(n, steps):
    # BFS Initialization
    queue = deque([(0, 0, [0])])  # (current position, number of moves, path)
    visited = set()  # To avoid revisiting the same step
    visited.add(0)
    
    while queue:
        position, moves, path = queue.popleft()
        
        # If we reach the top, return the number of moves and path
        if position == n:
            return moves, path
        
        # Explore all possible moves
        for step in steps:
            next_position = position + step
            if next_position <= n and next_position not in visited:
                visited.add(next_position)
                queue.append((next_position, moves + 1, path + [next_position]))
    
    return -1, []  # If reaching the top is not possible

def visualize_stairs(path):
    # Number of stairs
    n = path[-1]
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, n // 2))
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, n + 1)
    ax.axis('off')  # Hide axes
    
    # Draw stairs
    for i in range(n + 1):
        ax.hlines(i, -0.5, 0.5, colors='black', linewidth=1)
        ax.text(-0.7, i, f'Step {i}', verticalalignment='center', fontsize=9)
    
    # Draw movements
    for i in range(len(path) - 1):
        start = path[i]
        end = path[i + 1]
        # Draw an arrow from start to end
        ax.annotate('', xy=(0, end), xytext=(0, start),
                    arrowprops=dict(arrowstyle='->, head_width=0.4', color='blue', lw=2))
        # Label the move
        ax.text(0.1, (start + end) / 2, f'+{end - start}', verticalalignment='center', fontsize=9, color='blue')
    
    # Show the plot
    plt.title('Stair Climbing Visualization')
    plt.show()

# Example usage
n = 107
steps_list = [2, 3, 5]
moves, path = min_moves_to_climb(n, steps_list)
print(f"Minimum moves: {moves}")
print(f"Path taken: {path}")

# Visualize the path
if moves != -1:
    visualize_stairs(path)
else:
    print("It's not possible to reach the top with the given steps.")
