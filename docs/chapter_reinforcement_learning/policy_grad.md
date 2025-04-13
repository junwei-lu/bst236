# Policy Gradient

In reinforcement learning, our goal is to find a policy $\pi_{\theta}$ that maximizes the expected total reward over time. We can express this mathematically as:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[R(\tau)\right], \text{ where } R(\tau) = \sum_{t=0}^{\infty} \gamma^t r_t
$$

From our study of [optimization](../chapter_optimization/gradient_descent.md), we know we can maximize this objective using gradient ascent:

$$
\theta_{k+1} = \theta_k + \alpha \nabla J(\theta_k)
$$

The key challenge is computing $\nabla J(\theta_t)$, known as the **policy gradient**. Here's how we derive it:

$$
\begin{align*}
\nabla_{\theta} J(\pi_{\theta}) &= \nabla_{\theta} \mathbb{E}_{\tau \sim \pi_{\theta}} \left[R(\tau)\right] & \\
&= \nabla_{\theta} \int_{\tau} P(\tau|\theta) R(\tau) & \text{Expand expectation} \\
&= \int_{\tau} \nabla_{\theta} P(\tau|\theta) R(\tau) & \text{Bring gradient under integral} \\
&= \int_{\tau} P(\tau|\theta) \nabla_{\theta} \log P(\tau|\theta) R(\tau) & \text{Log-derivative trick} \\
&= \mathbb{E}_{\tau \sim \pi_{\theta}} \left[\nabla_{\theta} \log P(\tau|\theta) R(\tau)\right] & \text{Return to expectation form} \\
\therefore \nabla_{\theta} J(\pi_{\theta}) &= \mathbb{E}_{\tau \sim \pi_{\theta}} \left[\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau)\right] & \text{Final expression}
\end{align*}
$$

This shows we can estimate the policy gradient through sampling. Given a dataset $\mathcal{D} = \{\tau_1, \tau_2, \cdots, \tau_n\}$ of trajectories from policy $\pi_{\theta}$, where each trajectory $\tau_i = (s_0^i, a_0^i, s_1^i, a_1^i, \cdots, s_T^i, a_T^i)$ (we'll drop the superscript $i$ for simplicity), we can estimate the gradient as:

$$
\hat{g}(\theta) = \frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau),
$$

This leads to the **vanilla policy gradient** update:

$$
\theta_{k+1} = \theta_k + \alpha \hat{g}(\theta_k)
$$

## Vanilla Policy Gradient Algorithm

Here's the step-by-step process for implementing the vanilla policy gradient:

!!! abstract "Vanilla Policy Gradient Algorithm"
    - Start with randomly initialized policy parameters $\theta$
    - Repeat until convergence:
        - Generate trajectories $\mathcal{D} = \{\tau_i\}$ using current policy $\pi_{\theta_k}$
        - Calculate returns $R(\tau)$ for each trajectory
        - Estimate policy gradient: $\hat{g}(\theta_k) = \frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta_k}(a_t |s_t) R(\tau)$
        - Update policy: $\theta_{k+1} = \theta_k + \alpha \hat{g}(\theta_k)$

    - Return final policy parameters $\theta$

This algorithm is **on-policy**, meaning it requires interaction with the environment using the current policy. While this approach works, it can be inefficient in practice. There are alternative **off-policy** methods that we won't cover in this course.

## Reducing Variance in Policy Gradients

The vanilla policy gradient method often suffers from high variance in its estimates, leading to unstable training. Let's explore ways to reduce this variance while keeping our estimates unbiased.

Looking at our policy gradient expression:

$$
\nabla_{\theta} J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau)\right].
$$

This formulation has a problem: it assigns credit to all actions based on the total reward $R(\tau)$, even for actions that occurred before the reward was received. This doesn't make sense - actions should only be reinforced based on their future consequences.

We can improve this by using the "reward-to-go" formulation:

$$
\nabla_{\theta} J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \sum_{t'=t}^T R(s_{t'}, a_{t'})\right].
$$

Here, $\hat{R}_t = \sum_{t'=t}^T R(s_{t'}, a_{t'})$ represents the future rewards from time $t$ onward.

In general, the policy gradient can be written as:

$$
\nabla_{\theta} J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \Phi_t\right],
$$

where $\Phi_t$ can be:
- Total reward: $\Phi_t = R(\tau)$
- Reward-to-go: $\Phi_t = \hat{R}_t$
- Baseline-adjusted: $\Phi_t = \hat{R}_t - b(s_t)$
- Action-value: $\Phi_t = Q^{\pi_{\theta}}(s_t, a_t)$
- Advantage: $\Phi_t = A^{\pi_{\theta}}(s_t, a_t) = Q^{\pi_{\theta}}(s_t, a_t) - V^{\pi_{\theta}}(s_t)$

### Estimating the Advantage Function

The most common approach uses the **advantage function** $A_t = A^{\pi}(s_t, a_t) = Q^{\pi}(s_t, a_t) - V^{\pi}(s_t)$. We can estimate it in several ways:

1. Monte Carlo Sampling:
   $$
   A_t = \hat{R}_t - V(s_t)
   $$

2. Temporal Difference (TD(0)):
   $$
   A_t = r_t + \gamma V(s_{t+1}) - V(s_t)
   $$

3. k-step Advantage:
   $$
   A^{(k)}_t = \sum_{l=0}^{k-1} \gamma^l r_{t+l} + \gamma^k V(s_{t+k}) - V(s_t)
   $$

4. Generalized Advantage Estimation (GAE):
   $$
   A_t = \sum_{l=0}^{\infty} \left(\gamma \lambda\right)^l \delta_{t+l}
   $$
   where $\delta_{t+l} = r_{t+l} + \gamma V(s_{t+l+1}) - V(s_{t+l})$ is the TD error.

Here's the complete algorithm with advantage estimation:

!!! abstract "Policy Gradient with Advantage Estimation"
    - Initialize policy parameters $\theta$ randomly
    - Repeat until convergence:
        - Generate trajectories $\mathcal{D}_k = \{\tau_i\}$ using policy $\pi_{\theta_k}$
        - Calculate reward-to-go $\hat{R}_t$ for each trajectory
        - Compute advantage $A_t$ using chosen method
        - Estimate gradient: $\hat{g}(\theta_k) = \frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta_k}(a_t |s_t) A_t$
        - Update policy: $\theta_{k+1} = \theta_k + \alpha \hat{g}(\theta_k)$
        - Update value function: $\phi_k = \arg \min_{\phi} \sum_{\tau \in \mathcal{D}_k} \sum_{t=0}^{T} \left( V_{\phi}(s_t) - \hat{R}_t \right)^2$

### Implementing Policy Gradient in PyTorch

Let's implement a simple policy network using PyTorch. We'll use a neural network with one hidden layer for our policy $\pi_{\theta}(a|s)$:

```python
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)
```

We can convert the policy gradient into a loss function:

$$
\mathcal{L}(\theta) = -\frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^{T}  \log \pi_{\theta}(a_t |s_t) R(\tau),
$$

Here's how we implement the key components:

```python
from torch.distributions.categorical import Categorical

policy_net = PolicyNetwork(state_dim, hidden_dim, action_dim)

def get_policy(state):
    logits = policy_net(state)
    return Categorical(logits=logits) 

def get_action(obs):
    return get_policy(obs).sample().item()

def get_log_prob(obs, action):
    return get_policy(obs).log_prob(action)

def compute_loss(obs, act, weights):
    logp = get_policy(obs).log_prob(act)
    return -(logp * weights).mean()
```

The `Categorical` class in PyTorch handles discrete action distributions, with `log_prob` computing the log probability of actions under the current policy.


































