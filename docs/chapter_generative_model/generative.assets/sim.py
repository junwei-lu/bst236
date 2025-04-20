import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def grad_log_mixture(x):
    """
    Compute the gradient of f(x)=log(p(x)) for the mixture density
    p(x) = 0.5 * N(x | -3,1) + 0.5 * N(x | 3,1).
    
    Since the Gaussian density (ignoring constant factors) is:
         N(x | μ, 1) ∝ exp(-0.5*(x-μ)^2),
    we can compute the gradient as:
         f'(x) = - [0.5*(x+3)*exp(-0.5*(x+3)**2) + 0.5*(x-3)*exp(-0.5*(x-3)**2)] /
                   [0.5*exp(-0.5*(x+3)**2) + 0.5*exp(-0.5*(x-3)**2)]
    """
    w1, w2 = 0.5, 0.5
    a = np.exp(-0.5 * (x + 3)**2)
    b = np.exp(-0.5 * (x - 3)**2)
    # Add a small constant to prevent division by zero
    denominator = w1*a + w2*b
    # Ensure no division by zero with a small epsilon
    denominator = np.maximum(denominator, 1e-10)
    return -(w1*(x + 3)*a + w2*(x - 3)*b) / denominator

# Set random seed for reproducibility
np.random.seed(42)

# Simulation parameters
num_samples = 10000   # number of trajectories/samples
T = 200               # total number of time steps
eps = 0.1             # time-step size

# Define grid for density estimation (x-axis)
x_grid = np.linspace(-8, 8, 400)

# Initialize samples from N(0, 1)
X = np.random.randn(num_samples)*0.001

# Record density evolution: each row corresponds to the density estimate at a time step.
density_matrix = []

# Estimate initial density via KDE
kde = gaussian_kde(X)
density_matrix.append(kde(x_grid))

# Simulate the Langevin dynamics over T steps
for t in range(1, T):
    # Update samples: noise is scaled by sqrt(2 * epsilon)
    noise = np.sqrt(2 * eps) * np.random.randn(num_samples)
    X = X - eps * grad_log_mixture(X) + noise
    
    # Check for and handle NaN or infinite values
    X = np.nan_to_num(X, nan=0.0, posinf=8.0, neginf=-8.0)
    
    kde = gaussian_kde(X)
    density_matrix.append(kde(x_grid))

# Convert list to a numpy array of shape (T, len(x_grid))
density_matrix = np.array(density_matrix)

# Plot the heatmap: time is on the vertical axis and x on the horizontal axis.
plt.figure(figsize=(8, 6))
plt.imshow(density_matrix, aspect='auto', 
           extent=[x_grid[0], x_grid[-1], T, 0],
           cmap='viridis')
plt.colorbar(label='Density')
plt.xlabel('x')
plt.ylabel('Time step')
plt.title('Evolution of Density via Langevin Dynamics\n(Target: Mixture of Gaussians)')
plt.show()