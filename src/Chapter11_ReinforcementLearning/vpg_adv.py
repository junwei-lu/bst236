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
# 1. Define the Policy and Value Networks
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
        action_probs = torch.softmax(self.fc2(x), dim=-1)
        return action_probs

class Value(nn.Module):
    def __init__(self, state_dim, hidden_size=32):
        """
        A simple MLP that estimates the value function V(s)
        """
        super(Value, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        value = self.fc2(x)
        return value

# --------------------------
# 2. Monte Carlo Sampling Function
# --------------------------
def run_episode(env, policy, value_net=None, gamma=0.99, render=False):
    """
    Run a single episode using the current policy.
    
    Parameters:
      env: The Gym environment.
      policy: The policy network.
      value_net: The value network for state value estimation.
      gamma: Discount factor.
      render (bool): If True, render the environment using the default render mode.
      
    Returns:
      episode_rewards (list): Rewards collected during the episode.
      episode_log_probs (list): Log probabilities for the actions taken.
      episode_states (list): States visited during the episode.
      episode_values (list): Value estimations for each state (if value_net provided).
    """
    state = env.reset()
    if isinstance(state, tuple):  # Handle new gym API where reset returns (state, info)
        state = state[0]
    
    episode_rewards = []
    episode_log_probs = []
    episode_states = []
    episode_values = []
    done = False

    while not done:
        if render:
            env.render()
            time.sleep(0.02)
        
        # Convert state to tensor properly
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        episode_states.append(state_tensor)
        
        # Get action probabilities and sample action
        action_probs = policy(state_tensor)
        m = Categorical(action_probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        
        # Get value estimate if value network is provided
        if value_net is not None:
            value = value_net(state_tensor)
            episode_values.append(value)

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

    return episode_rewards, episode_log_probs, episode_states, episode_values

# --------------------------
# 3. Combined Loss Function with GAE
# --------------------------
def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Parameters:
      rewards (list): List of rewards for each time step.
      values (list): List of value estimates for each state.
      gamma (float): Discount factor.
      lam (float): GAE lambda parameter for bias-variance trade-off.
      
    Returns:
      advantages (list): GAE advantages for each time step.
      returns (list): Discounted returns for each time step (for value function training).
    """
    # Convert lists to tensors
    rewards = torch.tensor(rewards)
    
    # Extract values from their tensors
    values = torch.cat(values).squeeze()
    
    # Add a zero value for the terminal state
    values_next = torch.cat([values[1:], torch.tensor([0.0])])
    
    # Calculate delta: r + gamma*V(s') - V(s)
    deltas = rewards + gamma * values_next - values
    
    # Calculate GAE advantages
    advantages = []
    advantage = 0.0
    
    for delta in reversed(deltas):
        advantage = delta + gamma * lam * advantage
        advantages.insert(0, advantage)
    
    # Convert to tensor
    advantages = torch.tensor(advantages)
    
    # Calculate returns for value function training
    returns = advantages + values
    
    return advantages, returns

def compute_combined_loss(env, policy, value_net, batch_size=10, gamma=0.99, lam=0.95):
    """
    Sample a batch of episodes, compute their advantages using GAE, and combine
    them with the log probabilities to compute the policy gradient loss.
    
    Parameters:
      env: The Gym environment.
      policy: The policy network.
      value_net: The value network.
      batch_size (int): Number of episodes in the batch.
      gamma (float): Discount factor.
      lam (float): GAE lambda parameter.
      
    Returns:
      policy_loss (torch.Tensor): The computed policy gradient loss.
      value_loss (torch.Tensor): The value function loss.
    """
    policy_loss = 0.0
    value_loss = 0.0
    
    for _ in range(batch_size):
        episode_rewards, episode_log_probs, episode_states, episode_values = run_episode(
            env, policy, value_net, gamma, render=False
        )
        
        # Compute advantages and returns using GAE
        advantages, returns = compute_advantages(episode_rewards, episode_values, gamma, lam)
        
        # Compute policy loss using advantages
        for i, log_prob in enumerate(episode_log_probs):
            policy_loss += -log_prob * advantages[i].detach()  # Detach to avoid backprop through advantages
        
        # Compute value loss (MSE between predicted values and returns)
        values_tensor = torch.cat(episode_values).squeeze()
        returns_tensor = returns.detach()  # Detach to avoid backprop through returns
        value_loss += torch.mean((values_tensor - returns_tensor) ** 2)
    
    # Average over batch
    policy_loss = policy_loss / batch_size
    value_loss = value_loss / batch_size
    
    return policy_loss, value_loss

# --------------------------
# 4. Function to Record and Save an Episode as GIF
# --------------------------
def run_episode_save_gif(env, policy, value_net, gamma=0.99, filename="episode.gif", sleep_time=0.02):
    """
    Run an episode with frame recording and then save the frames as a GIF.
    
    Parameters:
      env: The Gym environment.
      policy: The policy network.
      value_net: The value network.
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
def train_lunar_lander(num_epochs=1000, batch_size=32, learning_rate=1e-3, gamma=0.99, lam=0.95, save_gif=True, gif_interval=10):
    """
    Train the Lunar Lander agent using GAE and a value function. Periodically
    replay one episode and save it as a GIF for visual inspection.
    
    Parameters:
      num_epochs (int): Number of training epochs.
      batch_size (int): Number of episodes to sample per epoch.
      learning_rate (float): Learning rate for the optimizer.
      gamma (float): Discount factor.
      lam (float): GAE lambda parameter.
      save_gif (bool): If True, save a replay of the trajectory as a GIF.
      gif_interval (int): Save a GIF every this many epochs.
    """
    # Create the environment with explicit render_mode for RGB array output
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize policy and value networks
    policy = Policy(state_dim, action_dim)
    value_net = Value(state_dim)
    
    # Set up optimizers for both networks
    policy_optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    value_optimizer = optim.Adam(value_net.parameters(), lr=learning_rate)
    
    # Track loss history for plotting
    policy_loss_history = []
    value_loss_history = []
    
    for epoch in range(num_epochs):
        # Compute policy and value losses
        policy_loss, value_loss = compute_combined_loss(env, policy, value_net, batch_size, gamma, lam)
        
        # Update policy
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()
        
        # Update value function
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()
        
        # Store loss values
        policy_loss_history.append(policy_loss.item())
        value_loss_history.append(value_loss.item())
        
        print(f"Epoch: {epoch + 1}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}")
        
        # Save a replay of the trajectory as a GIF every gif_interval epochs
        if save_gif and (epoch + 1) % gif_interval == 0:
            gif_filename = f"gae/epoch_{epoch + 1}.gif"
            try:
                run_episode_save_gif(env, policy, value_net, gamma, filename=gif_filename, sleep_time=0.02)
            except Exception as e:
                print(f"Failed to save GIF: {e}")
        
        # Plot loss history every 100 epochs
        if (epoch + 1) % 100 == 0:
            plot_loss_history(policy_loss_history, value_loss_history, epoch + 1)
    
    # Final loss plot
    if num_epochs % 100 != 0:
        plot_loss_history(policy_loss_history, value_loss_history, num_epochs)
        
    env.close()
    
    return policy_loss_history, value_loss_history

def plot_loss_history(policy_loss_history, value_loss_history, current_epoch):
    """
    Plot the policy and value loss histories up to the current epoch.
    
    Parameters:
        policy_loss_history (list): List of policy loss values.
        value_loss_history (list): List of value loss values.
        current_epoch (int): Current epoch number.
    """
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(policy_loss_history) + 1), policy_loss_history)
    plt.title('Policy Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Policy Loss')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(range(1, len(value_loss_history) + 1), value_loss_history)
    plt.title('Value Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Value Loss')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'gae/loss_history_epoch_{current_epoch}.png')
    plt.close()
    print(f"Loss history plot saved as loss_history_epoch_{current_epoch}.png")

if __name__ == '__main__':
    train_lunar_lander(save_gif=True, gif_interval=100)