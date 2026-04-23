from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

# Create the environment
env_id = "LunarLander-v3"
n_envs = 16
env = make_vec_env(env_id, n_envs=n_envs)

# Create the evaluation envs
eval_envs = make_vec_env(env_id, n_envs=5)

# Adjust evaluation interval depending on the number of envs
eval_freq = int(1e5)
eval_freq = max(eval_freq // n_envs, 1)

# Create evaluation callback to save best model
# and monitor agent performance
eval_callback = EvalCallback(
    eval_envs,
    best_model_save_path="./logs/",
    eval_freq=eval_freq,
    n_eval_episodes=10,
)

# Instantiate the agent
# Hyperparameters from https://github.com/DLR-RM/rl-baselines3-zoo
model = PPO(
    "MlpPolicy",
    env,
    n_steps=1024,
    batch_size=64,
    gae_lambda=0.98,
    gamma=0.999,
    n_epochs=4,
    ent_coef=0.01,
    verbose=1,
)

# Train the agent (you can kill it before using ctrl+c)
try:
    model.learn(total_timesteps=int(5e6), callback=eval_callback)
except KeyboardInterrupt:
    pass

# Load best model
model = PPO.load("logs/best_model.zip")

# Create a single environment for rendering
eval_env = make_vec_env(env_id, n_envs=1)
import imageio
import numpy as np
from PIL import Image
import os

# Create directory for gifs if it doesn't exist
os.makedirs("ppo", exist_ok=True)

# Render episodes with the best model and save as gif
for episode in range(3):  # Save 3 episodes as gifs
    frames = []
    obs = eval_env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        # Render the frame
        frame = eval_env.render(mode='rgb_array')
        frames.append(frame)
        
        # Get action and step environment
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        episode_reward += reward
        
        if done:
            print(f"Episode {episode+1} reward: {episode_reward}")
            break
    
    # Save frames as gif
    print(f"Saving episode {episode+1} as gif...")
    imageio.mimsave(f"gifs/episode_{episode+1}.gif", 
                   [np.array(frame) for frame in frames],
                   fps=30)

print("All episodes saved as gifs in the 'gifs' directory")

# Close the environment
eval_env.close()