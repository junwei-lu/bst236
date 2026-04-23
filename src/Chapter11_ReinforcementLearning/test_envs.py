
#%%
import gym

# Test LunarLander
env = gym.make("LunarLander-v2", render_mode="human")
state, _ = env.reset() # Start at s0

#%%
for _ in range(100):
    action = env.action_space.sample()
    state, reward, terminated, truncated, _ = env.step(action)
    env.render()
    time.sleep(0.02)
    if terminated or truncated:
        state, _ = env.reset()
env.close()

