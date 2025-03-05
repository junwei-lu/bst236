import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import os

# Set the style
plt.style.use('ggplot')

def create_l2_ball_animation():
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Create the L2 ball (circle in 2D)
    l2_ball = Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=2, label='L2 Ball (||y||_2 = 1)')
    ax.add_patch(l2_ball)
    
    # Choose a fixed vector x
    x = np.array([0.7, 0.4])  # Some arbitrary vector
    
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
    # Start with y at (1,0) position
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
    ax.set_title(r'Variational Form of L2-Norm: $\max_{\|y\|_2 \leq 1} \langle y, x \rangle$', fontsize=14)
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    
    # Create custom legend elements
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Vector x'),
        Line2D([0], [0], color='blue', lw=2, label='Vector y (||y||_2 = 1)'),
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
        
        # Move y around the unit circle
        angle = 2 * np.pi * frame / 360  # 360 frames for a full circle
        y = np.array([np.cos(angle), np.sin(angle)])
        
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
        
        # Add special highlight when y is parallel to x (maximizer)
        if np.allclose(y, x_normalized, rtol=1e-2, atol=1e-2):
            y_arrow.set(color='cyan')
            
            # Add annotation for optimal y
            if frame == 200:  # Some arbitrary frame when y is close to optimal
                ax.plot(x_normalized[0], x_normalized[1], 'ro', markersize=10)
                ax.text(x_normalized[0] - 0.4, x_normalized[1] - 0.1, 
                       r'$y^* = \frac{x}{||x||_2}$', fontsize=12, color='red')

        return [y_arrow, projection_line, inner_product_text, max_point_marker, y_text]
    
    # Create animation
    frames = 360  # Complete rotation around the L2 ball
    animation = FuncAnimation(fig, update, frames=frames, interval=30, blit=True)
    
    # Save animation as GIF
    animation.save('l2_norm_visualization.gif', writer='pillow', fps=30, dpi=100)
    print("Animation saved as 'l2_norm_visualization.gif'")
    
    # Display final frame
    plt.close()
    
    # Additional plot showing the final state with the optimal point
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # L2 ball
    ax.add_patch(Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=2))
    
    # Vector x
    ax.arrow(0, 0, x_normalized[0], x_normalized[1], 
            head_width=0.05, head_length=0.1, fc='red', ec='red')
    ax.text(x_normalized[0] + 0.05, x_normalized[1] + 0.05, 'x', fontsize=12, color='red')
    
    # Optimal y (parallel to x)
    optimal_y = x_normalized.copy()
    
    ax.arrow(0, 0, optimal_y[0], optimal_y[1], 
            head_width=0.05, head_length=0.1, fc='blue', ec='blue')
    ax.text(optimal_y[0] + 0.05, optimal_y[1] + 0.05, 'y*', fontsize=12, color='blue')
    
    # Explanation
    ax.set_title(r'Optimal Solution: $y^* = \frac{x}{||x||_2}$ (Unit vector in direction of x)', fontsize=14)
    ax.text(-1.4, -1.3, 
           f'Vector x = ({x[0]}, {x[1]})\n' + 
           f'||x||_2 = {np.linalg.norm(x):.4f}\n' +
           f'Optimal y* = x/||x||_2 = ({x_normalized[0]:.4f}, {x_normalized[1]:.4f})\n' +
           f'Maximum inner product = ||x||_2 = {np.linalg.norm(x):.4f}',
           fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    # Add a visual explanation of Cauchy-Schwarz inequality
    ax.text(-1.4, 1.2, 
           r'By Cauchy-Schwarz inequality:' + '\n' +
           r'$\langle y, x \rangle \leq ||y||_2 \cdot ||x||_2$' + '\n' +
           r'Equality holds when $y$ is parallel to $x$',
           fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.savefig('l2_norm_final_state.png')
    print("Final state saved as 'l2_norm_final_state.png'")
    
if __name__ == "__main__":
    create_l2_ball_animation() 