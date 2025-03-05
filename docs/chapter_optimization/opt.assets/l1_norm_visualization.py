import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import os

# Set the style
plt.style.use('ggplot')

def create_l1_ball_animation():
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Create the L1 ball (diamond shape in 2D)
    l1_ball_vertices = np.array([
        [1, 0],   # Right
        [0, 1],   # Top
        [-1, 0],  # Left
        [0, -1]   # Bottom
    ])
    l1_ball = Polygon(l1_ball_vertices, closed=True, fill=False, 
                    edgecolor='black', linewidth=2, label='L1 Ball (||y||_1 = 1)')
    ax.add_patch(l1_ball)
    
    # Choose a fixed vector x (as mentioned, not at a vertex of the L1 ball)
    # Let's choose a vector that has unequal components so the maximum is clear
    x = np.array([0.4, 0.7])  # The component with larger absolute value is x[1]
    x_norm = np.max(np.abs(x))  # For scaling visualization
    
    # Normalize x for better visualization
    x_normalized = x / np.linalg.norm(x)
    
    # Draw the fixed vector x (scaled for visibility)
    x_arrow = ax.arrow(0, 0, x_normalized[0], x_normalized[1], 
                      head_width=0.05, head_length=0.1, fc='red', ec='red',
                      label='Vector x')
    
    # Text for the fixed vector x
    ax.text(x_normalized[0] + 0.05, x_normalized[1] + 0.05, 'x', 
            fontsize=12, color='red')
    
    # Initialize the vector y (we'll animate this)
    # Start with y at the (1,0) position
    initial_y = np.array([1, 0])
    y_arrow = ax.arrow(0, 0, initial_y[0], initial_y[1], head_width=0.05, head_length=0.1, 
                     fc='blue', ec='blue', label='Vector y')
    
    # Text for vector y
    y_text = ax.text(initial_y[0] + 0.05, initial_y[1] + 0.05, 'y', fontsize=12, color='blue')
    
    # Initialize projection line (inner product visualization)
    projection_line = ax.plot([], [], 'g--', linewidth=1.5)[0]
    
    # Inner product value text
    inner_product_text = ax.text(0.1, -1.3, '', fontsize=12)
    
    # Maximum point marker
    max_point_marker = ax.plot([], [], 'ko', markersize=8)[0]
    
    # Title and labels
    ax.set_title(r'Variational Form of L1-Norm: $\max_{\|y\|_1 \leq 1} \langle y, x \rangle$', fontsize=14)
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    
    # Create custom legend elements
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Vector x'),
        Line2D([0], [0], color='blue', lw=2, label='Vector y (||y||_1 = 1)'),
        Line2D([0], [0], color='green', linestyle='--', lw=2, label='Inner Product Projection'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, 
               label='Maximum Inner Product Point')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Initialize maximum inner product values
    max_inner_product = -np.inf
    max_y = None
    
    # Animation update function
    def update(frame):
        nonlocal max_inner_product, max_y, y_arrow
        
        # Move y along the vertices of the L1 ball
        if frame < 100:
            # First quadrant: from (1,0) to (0,1)
            t = frame / 99
            y = np.array([(1-t), t])
        elif frame < 200:
            # Second quadrant: from (0,1) to (-1,0)
            t = (frame - 100) / 99
            y = np.array([-(t), (1-t)])
        elif frame < 300:
            # Third quadrant: from (-1,0) to (0,-1)
            t = (frame - 200) / 99
            y = np.array([-(1-t), -(t)])
        else:
            # Fourth quadrant: from (0,-1) to (1,0)
            t = (frame - 300) / 99
            y = np.array([(t), -(1-t)])
        
        # Clear previous arrow
        if y_arrow in ax.patches:
            y_arrow.remove()
            
        # Draw new y arrow
        y_arrow = ax.arrow(0, 0, y[0], y[1], head_width=0.05, head_length=0.1, 
                         fc='blue', ec='blue')
        
        # Update y text position
        y_text.set_position((y[0] + 0.05, y[1] + 0.05))
        
        # Calculate inner product
        inner_product = np.dot(x, y)
        
        # Keep track of maximum inner product
        if inner_product > max_inner_product:
            max_inner_product = inner_product
            max_y = y.copy()
        
        # Update projection line
        # Project y onto x
        projection_scalar = inner_product / np.linalg.norm(x)**2
        projection_vector = projection_scalar * x
        
        projection_line.set_data([0, projection_vector[0], y[0]], 
                                [0, projection_vector[1], y[1]])
        
        # Update inner product text
        inner_product_text.set_text(f'Inner Product: {inner_product:.4f}\nMax Inner Product: {max_inner_product:.4f}')
        
        # Mark the current maximum point
        if np.allclose(y, max_y, rtol=1e-5, atol=1e-5):
            max_point_marker.set_data([y[0]], [y[1]])
        else:
            max_point_marker.set_data([], [])
        
        # Add special highlight when we're at a vertex
        if (np.allclose(y, [1, 0], rtol=1e-5, atol=1e-5) or 
            np.allclose(y, [0, 1], rtol=1e-5, atol=1e-5) or 
            np.allclose(y, [-1, 0], rtol=1e-5, atol=1e-5) or 
            np.allclose(y, [0, -1], rtol=1e-5, atol=1e-5)):
            y_arrow.set(color='cyan')
        
        # Additional annotation for the optimal point: j* = argmax |x_j|
        if frame == 125:  # Around where the maximum should be achieved
            j_star = np.argmax(np.abs(x))
            sign_x_j_star = np.sign(x[j_star])
            optimal_y = np.zeros(2)
            optimal_y[j_star] = sign_x_j_star
            ax.plot(optimal_y[0], optimal_y[1], 'ro', markersize=10)
            ax.text(optimal_y[0] - 0.3, optimal_y[1] - 0.2, 
                   rf'$y^* = (0, \text{{sign}}(x_{{j^*}}))$', fontsize=12, color='red')

        return [y_arrow, projection_line, inner_product_text, max_point_marker, y_text]
    
    # Create animation
    frames = 400  # Complete rotation around the L1 ball
    animation = FuncAnimation(fig, update, frames=frames, interval=30, blit=True)
    
    # Save animation as GIF
    animation.save('l1_norm_visualization.gif', writer='pillow', fps=30, dpi=100)
    print("Animation saved as 'l1_norm_visualization.gif'")
    
    # Display final frame
    plt.close()
    
    # Additional plot showing the final state with the optimal point
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # L1 ball
    ax.add_patch(Polygon(l1_ball_vertices, closed=True, fill=False, 
                      edgecolor='black', linewidth=2))
    
    # Vector x
    ax.arrow(0, 0, x_normalized[0], x_normalized[1], 
            head_width=0.05, head_length=0.1, fc='red', ec='red')
    ax.text(x_normalized[0] + 0.05, x_normalized[1] + 0.05, 'x', fontsize=12, color='red')
    
    # Optimal y
    j_star = np.argmax(np.abs(x))
    sign_x_j_star = np.sign(x[j_star])
    optimal_y = np.zeros(2)
    optimal_y[j_star] = sign_x_j_star
    
    ax.arrow(0, 0, optimal_y[0], optimal_y[1], 
            head_width=0.05, head_length=0.1, fc='blue', ec='blue')
    ax.text(optimal_y[0] + 0.05, optimal_y[1] + 0.05, 'y*', fontsize=12, color='blue')
    
    # Explanation
    ax.set_title(r'Optimal Solution: $y^* = (0, \text{sign}(x_{j^*}))$ where $j^* = \arg\max_{j} |x_j|$', fontsize=14)
    ax.text(-1.4, -1.3, 
           f'Vector x = ({x[0]}, {x[1]})\n' + 
           f'j* = {j_star+1} (component with max |x_j| = {np.abs(x[j_star])})\n' +
           f'sign(x_{{j*}}) = {sign_x_j_star}\n' +
           f'Optimal y* = {tuple(optimal_y)}\n' +
           f'Maximum inner product = {np.dot(x, optimal_y):.4f}',
           fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.savefig('l1_norm_final_state.png')
    print("Final state saved as 'l1_norm_final_state.png'")
    
if __name__ == "__main__":
    create_l1_ball_animation() 