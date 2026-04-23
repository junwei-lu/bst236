import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import imageio
import time
import matplotlib.pyplot as plt

# Monkey patch for compatibility with newer NumPy versions
# Some versions of Gym check for np.bool8 which may not exist in newer NumPy
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

# --------------------------
# 1. Define the Policy Network
# --------------------------
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=32):
        """
        A simple MLP that maps state observations to a probability distribution
        over actions using softmax.
        """
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        pi = Categorical(probs=self.fc2(x))
        action_probs = torch.softmax(self.fc2(x), dim=-1)
        return action_probs

# --------------------------
# 2. Monte Carlo Sampling Function
# --------------------------
def run_episode(env, policy, gamma=0.99, render=False):
    """
    Run a single episode using the current policy.
    
    Parameters:
      env: The Gym environment.
      policy: The policy network.
      gamma: Discount factor (passed for future compatibility).
      render (bool): If True, render the environment using the default render mode.
      
    Returns:
      episode_rewards (list): Rewards collected during the episode.
      episode_log_probs (list): Log probabilities for the actions taken.
    """
    state = env.reset() #s0
    if isinstance(state, tuple):  # Handle new gym API where reset returns (state, info)
        state = state[0]
    
    episode_rewards = []
    episode_log_probs = []
    done = False

    while not done:
        if render:
            env.render()
            time.sleep(0.02)
        
        # Convert state to tensor properly
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        action_probs = policy(state_tensor)
        m = Categorical(probs=action_probs)
        action = m.sample()
        log_prob = m.log_prob(action)

        try:
            # Try new Gym API (returning 5 values)
            next_state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
        except ValueError:
            # Fall back to older Gym API (returning 4 values)
            next_state, reward, done, info = env.step(action.item())
            
        episode_rewards.append(reward)
        episode_log_probs.append(log_prob)
        state = next_state

    return episode_rewards, episode_log_probs

# --------------------------
# 3. Combined Loss Function
# --------------------------
def compute_combined_loss(env, policy, batch_size=10, gamma=0.99):
    """
    Sample a batch of episodes, compute their discounted returns, and combine
    them with the log probabilities to compute the policy gradient loss.
    
    This function combines:
      - Sampling a batch of trajectories,
      - Computing the discounted returns,
      - Aggregating the loss over the batch.
    
    Parameters:
      env: The Gym environment.
      policy: The policy network.
      batch_size (int): Number of episodes in the batch.
      gamma (float): Discount factor.
      
    Returns:
      loss (torch.Tensor): The computed policy gradient loss averaged over the batch.
    """
    total_loss = 0.0
    
    for _ in range(batch_size):
        episode_rewards, episode_log_probs = run_episode(env, policy, gamma, render=False)
        
        # Compute the discounted return R for the episode.
        R = 0.0
        for t, r in enumerate(episode_rewards):
            R += (gamma ** t) * r
        
        # Aggregate the loss over the episode.
        for log_prob in episode_log_probs:
            total_loss += -log_prob * R  # Negative sign converts gradient descent to ascent.
    
    return total_loss / batch_size

# --------------------------
# 4. Function to Record and Save an Episode as GIF
# --------------------------
def run_episode_save_gif(env, policy, gamma=0.99, filename="episode.gif", sleep_time=0.02):
    """
    Run an episode with frame recording and then save the frames as a GIF.
    
    Parameters:
      env: The Gym environment.
      policy: The policy network.
      gamma: Discount factor (not used directly here).
      filename (str): The filename for the saved GIF.
      sleep_time (float): Pause between frames (optional, for a smoother GIF).
    """
    frames = []
    state = env.reset()
    if isinstance(state, tuple):  # Handle new gym API where reset returns (state, info)
        state = state[0]
        
    done = False
    
    while not done:
        # Get the rendered frame from the environment
        frame = env.render()
        
        # Skip frame if it's None
        if frame is not None:
            frames.append(frame)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        action_probs = policy(state_tensor)
        m = Categorical(action_probs)
        action = m.sample()
        
        try:
            # Try new Gym API (returning 5 values)
            next_state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
        except ValueError:
            # Fall back to older Gym API (returning 4 values)
            next_state, reward, done, info = env.step(action.item())
            
        state = next_state
        time.sleep(sleep_time)
    
    # Only save GIF if we captured valid frames
    if frames:
        try:
            imageio.mimsave(filename, frames, fps=30)
            print(f"Saved replay GIF to: {filename}")
        except Exception as e:
            print(f"Error saving GIF: {e}")
    else:
        print(f"No frames captured, GIF not saved.")

# --------------------------
# 5. Training Loop with GIF Replay
# --------------------------
def train_lunar_lander(num_epochs=1000, batch_size=32, learning_rate=1e-3, gamma=0.99, save_gif=True, gif_interval=10):
    """
    Train the Lunar Lander agent using a combined loss function. Periodically
    replay one episode and save it as a GIF for visual inspection.
    
    Parameters:
      num_epochs (int): Number of training epochs.
      batch_size (int): Number of episodes to sample per epoch.
      learning_rate (float): Learning rate for the optimizer.
      gamma (float): Discount factor.
      save_gif (bool): If True, save a replay of the trajectory as a GIF.
      gif_interval (int): Save a GIF every this many epochs.
    """
    # Create the environment with explicit render_mode for RGB array output
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = Policy(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    
    # Track loss history for plotting
    loss_history = []
    
    for epoch in range(num_epochs):
        loss = compute_combined_loss(env, policy, batch_size, gamma)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Store loss value
        loss_history.append(loss.item())
        
        print(f"Epoch: {epoch + 1}, Loss: {loss.item():.4f}")
        
        # Save a replay of the trajectory as a GIF every gif_interval epochs
        if save_gif and (epoch + 1) % gif_interval == 0:
            gif_filename = f"epoch_{epoch + 1}.gif"
            try:
                run_episode_save_gif(env, policy, gamma, filename=gif_filename, sleep_time=0.02)
            except Exception as e:
                print(f"Failed to save GIF: {e}")
        
        # Plot loss history every 100 epochs
        if (epoch + 1) % 100 == 0:
            plot_loss_history(loss_history, epoch + 1)
    
    # Final loss plot
    if num_epochs % 100 != 0:
        plot_loss_history(loss_history, num_epochs)
        
    env.close()
    
    return loss_history

def plot_loss_history(loss_history, current_epoch):
    """
    Plot the loss history up to the current epoch.
    
    Parameters:
        loss_history (list): List of loss values.
        current_epoch (int): Current epoch number.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.title(f'Training Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Save the plot
    plt.savefig(f'loss_history_epoch_{current_epoch}.png')
    plt.close()
    print(f"Loss history plot saved as loss_history_epoch_{current_epoch}.png")

if __name__ == '__main__':
    train_lunar_lander(save_gif=True, gif_interval=100)