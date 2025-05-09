# Proximal Policy Optimization

Proximal Policy Optimization (PPO) addresses a fundamental challenge in reinforcement learning: how can we maximize policy improvement using our current data while preventing catastrophic performance degradation from overly aggressive updates?

Unlike vanilla policy gradient methods that maintain proximity between policies in parameter space, PPO recognizes that small parameter changes can sometimes lead to dramatic performance differences. This sensitivity makes large step sizes risky in traditional policy gradient approaches, limiting their sample efficiency. PPO introduces mechanisms to prevent harmful policy updates, allowing for larger improvement steps while maintaining stability, which typically results in faster and more reliable performance gains. In summary, we have the following two main ideas:

- Policy gradient estimates have high variance
- Small changes in policy’s parameters can lead to large changes in policy
- Our goal is not change $\theta$ but to change $\pi_{\theta}$ gradually.

![proximal policy optimization](./rl.assets/proximal.png)

Like the [proximal gradient descent](../chapter_optimization/proximal_gradient_descent.md), PPO aims to find a loss function that is an approximation of loss function

$$
L(\theta) = \mathbb{E}_{s, a \sim \pi_{\theta}} \left[  A^{\pi_{\theta}}(s,a)\right].
$$

We can approximate the above loss function around the current policy $\pi_{\theta_k}$ by:

$$
L(\theta) \approx \mathbb{E}_{s \sim \pi_{\theta_k}, a \sim \pi_{\theta}} \left[  A^{\pi_{\theta_k}}(s,a)\right] = \mathbb{E}_{s \sim \pi_{\theta_k}, a \sim \pi_{\theta_k}} \left[  \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s,a)\right].
$$

Then in each epoch, we can update the policy $\pi_{\theta_k}$ by: 

$$
\theta_{k+1} =  \arg \max_{\theta} \mathbb{E}_{s, a \sim \pi_{\theta_k}} \left[ L(s,a,\theta_k,\theta) \right],
$$

where $L(s,a,\theta_k,\theta) = 
\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}  A^{\pi_{\theta_k}}(s,a)$ is an approximation of $L(\theta)$ at the current policy $\pi_{\theta_k}$.



## Proximal Policy Optimization Loss


Proximal Policy Optimization (PPO) further modify the proximal loss above by adding a clipping term to prevent the policy from changing too much:

$$
L(s,a,\theta_k,\theta) = \min\left(
\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}  A^{\pi_{\theta_k}}(s,a),
g(\epsilon, A^{\pi_{\theta_k}}(s,a))
\right),
$$

where

$$
g(\epsilon, A) = \left\{
    \begin{array}{ll}
    (1 + \epsilon) A & A \geq 0 \\
    (1 - \epsilon) A & A < 0.
    \end{array}
    \right.
$$

![PPO Objective](./rl.assets/clip.png)

Let $r$ be the density ratio $\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}$. The above figure shows PPO loss.

To figure out what intuition to take away from this, let's look at a single state-action pair $(s,a)$, and think of cases.

- **Advantage is positive**: Suppose the advantage for that state-action pair is positive, in which case its contribution to the objective reduces to

$$L(s,a,\theta_k,\theta) = \min\left(
\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}, (1 + \epsilon)
\right)  A^{\pi_{\theta_k}}(s,a).$$

Because the advantage is positive, the objective will increase if the action becomes more likely—that is, if $\pi_{\theta}(a|s)$ increases. But the min in this term puts a limit to how much the objective can increase. Once $\pi_{\theta}(a|s) > (1+\epsilon) \pi_{\theta_k}(a|s)$, the min kicks in and this term hits a ceiling of $(1+\epsilon) A^{\pi_{\theta_k}}(s,a)$. Thus: the new policy does not benefit by going far away from the old policy.

- **Advantage is negative**: Suppose the advantage for that state-action pair is negative, in which case its contribution to the objective reduces to

$$L(s,a,\theta_k,\theta) = \max\left(
\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}, (1 - \epsilon)
\right)  A^{\pi_{\theta_k}}(s,a).$$

Because the advantage is negative, the objective will increase if the action becomes less likely—that is, if $\pi_{\theta}(a|s)$ decreases. But the max in this term puts a limit to how much the objective can increase. Once $\pi_{\theta}(a|s) < (1-\epsilon) \pi_{\theta_k}(a|s)$, the max kicks in and this term hits a ceiling of $(1-\epsilon) A^{\pi_{\theta_k}}(s,a)$. Thus, again: the new policy does not benefit by going far away from the old policy.

We summarize the PPO loss as follows:

!!! abstract "Proximal Policy Optimization"
    - Initialize policy parameters $\theta$ randomly
    - For k = 1, 2, ... do:
        - Generate trajectories $\mathcal{D}_k = \{\tau_i\}$ using policy $\pi_{\theta_k}$
        - Calculate reward-to-go $\hat{R}_t$ for each trajectory
        - Compute advantage $A_t$ using [chosen method](./policy_grad.md#estimating-the-advantage-function)
        - **Actor**: Update policy: $\theta_{k+1} = \arg \min_{\theta} \sum_{\tau \in \mathcal{D}_k}  \sum_{t=0}^{T} \left[ L(s_t,a_t,\theta_k,\theta) \right]$ by some [stochastic gradient descent method](../chapter_optimization/sgd.md)
        - **Critic**: Update value function: $\phi_k = \arg \max_{\phi} \sum_{\tau \in \mathcal{D}_k} \sum_{t=0}^{T} \left( V_{\phi}(s_t) - \hat{R}_t \right)^2$ (Or use the [TD error](mdp.md#value-function-estimation) to update the value function)

### Exploration vs Exploitation

In reinforcement learning, balancing exploration (trying new actions to discover potentially better strategies) and exploitation (using known good actions to maximize immediate rewards) is crucial for optimal learning. Without sufficient exploration, agents may get stuck in suboptimal policies, while too much exploration can waste resources on unproductive actions. 

For PPO, we can add an entropy regularization term to the loss function to encourage exploration.

$$
\theta_{k+1} =  \arg \max_{\theta} \mathbb{E}_{s, a \sim \pi_{\theta_k}} \left[ L(s,a,\theta_k,\theta) \right] + \beta \mathbb{E}_{s \sim \pi_{\theta}} \left[ H(\pi_{\theta}(s)) \right],
$$

where $H(\pi_{\theta}(s)) = -\mathbb{E}_{a \sim \pi_{\theta}(s)} [\log \pi_{\theta}(a|s)]$ is the entropy of the policy and $\beta$ is a hyperparameter.


A policy with larger entropy corresponds to a more uniform or less certain distribution over actions, encouraging the agent to explore a wider range of possible actions instead of repeatedly selecting the same few actions.




### Stable Baselines3 Implementation

[Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) is a popular library for reinforcement learning. It modularizes many reinforcement learning algorithms, including PPO.

```python
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
```

