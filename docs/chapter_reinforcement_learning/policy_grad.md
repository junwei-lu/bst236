# Policy Gradient

Recall the objective of RL is to find a policy $\pi_{\theta}$ that maximizes the expected cumulative reward

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[R(\tau)\right], \text{ where } R(\tau) = \sum_{t=0}^{\infty} \gamma^t r_t
$$

We have learned from [optimization](../chapter_optimization/gradient_descent.md) that we could maximize the objective by gradient ascent.

$$
\theta_{k+1} = \theta_k + \alpha \nabla J(\theta_k)
$$

So the key is to compute $\nabla J(\theta_t)$ which is called the **policy gradient**. We have the following derivation:

$$
\begin{align*}
\nabla_{\theta} J(\pi_{\theta}) &= \nabla_{\theta} \mathbb{E}_{\tau \sim \pi_{\theta}} \left[R(\tau)\right] & \\
&= \nabla_{\theta} \int_{\tau} P(\tau|\theta) R(\tau) & \text{Expand expectation} \\
&= \int_{\tau} \nabla_{\theta} P(\tau|\theta) R(\tau) & \text{Bring gradient under integral} \\
&= \int_{\tau} P(\tau|\theta) \nabla_{\theta} \log P(\tau|\theta) R(\tau) & \text{Log-derivative trick} \\
&= \mathbb{E}_{\tau \sim \pi_{\theta}} \left[\nabla_{\theta} \log P(\tau|\theta) R(\tau)\right] & \text{Return to expectation form} \\
\therefore \nabla_{\theta} J(\pi_{\theta}) &= \mathbb{E}_{\tau \sim \pi_{\theta}} \left[\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau)\right] & \text{Expression for grad-log-prob}
\end{align*}
$$

Therefore, we can represent the policy gradient as the expectation. This means that we can apply the [stochastic gradient ascent](../chapter_optimization/sgd.md) to estimate the policy gradient by sampling trajectories from the policy.

Given a dataset $\mathcal{D} = \{\tau_1, \tau_2, \cdots, \tau_n\}$ of trajectories sampled from the policy $\pi_{\theta}$, each trajectory $\tau_i = (s_0^i, a_0^i, s_1^i, a_1^i, \cdots, s_T^i, a_T^i)$ and we will omit the superscript $i$ for simplicity. The gradient of the policy can be estimated by

$$
\hat{g}(\theta) = \frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau),
$$

We call this method **vanilla policy gradient**:

$$
\theta_{k+1} = \theta_k + \alpha \hat{g}(\theta_k)
$$

## Vanilla Policy Gradient 

We now can write the pseudo-code for the vanilla policy gradient algorithm:

- Initialize policy parameters $\theta$ randomly
- For $k = 0, 1, 2, \ldots$ do

    - Generate a set of trajectories $\mathcal{D} = \{\tau_i\}$ by executing the current policy $\pi_{\theta_k}$
    - Compute the returns $R(\tau)$ for each trajectory
    - Estimate the policy gradient: $\hat{g}(\theta_k) = \frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta_k}(a_t |s_t) R(\tau)$
    - Update the policy parameters: $\theta_{k+1} = \theta_k + \alpha \hat{g}(\theta_k)$.

- Return $\theta$

You can see that the vanilla policy gradient above requires to generate a series of trajectories to estimate the policy gradient. This implies that we need to be able to interact with the environment using the given policy. 
This is called **on-policy** algorithm. In practice, simulating from the environment is not always feasible, there is another family of reinforcement learning algorithms called **off-policy** algorithm. We will not cover them in this course.


### PyTorch for Vanilla Policy Gradient

Let's implement the vanilla policy gradient algorithm using PyTorch. First, we'll define a simple policy network $\pi_{\theta}(a|s)$ with one hidden layer neural network. Here the action space is discrete with `action_dim` actions.

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

We can then convert the policy gradient to a loss function:

$$
\mathcal{L}(\theta) = -\frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^{T}  \log \pi_{\theta}(a_t |s_t) R(\tau),
$$

Therefore, we can define the loss function as follows:

```python
from torch.distributions.categorical import Categorical

policy_net = PolicyNetwork(state_dim, hidden_dim, action_dim)
# make function to compute action distribution
def get_policy(state):
    logits = policy_net(state)
    return Categorical(logits=logits) 

# make action selection function (outputs int actions, sampled from policy)
def get_action(obs):
    return get_policy(obs).sample().item()

# make function to compute log probability of an action under current policy
def get_log_prob(obs, action):
    return get_policy(obs).log_prob(action)

# make loss function whose gradient, for the right data, is policy gradient
def compute_loss(obs, act, weights):
    logp = get_policy(obs).log_prob(act)
    return -(logp * weights).mean()
```

`Categorical` is a distribution class in PyTorch that represents a categorical distribution. It has a method `log_prob` that computes the log probability of an action under the distribution.
































