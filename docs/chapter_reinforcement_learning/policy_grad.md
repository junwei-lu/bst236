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

![humanoid](rl.assets/humanoid.gif)

You can see that the vanilla policy gradient above requires to generate a series of trajectories to estimate the policy gradient. This implies that we need to be able to interact with the environment using the given policy. 
This is called **on-policy** algorithm. In practice, simulating from the environment is not always feasible, there is another family of reinforcement learning algorithms called **off-policy** algorithm. We will not cover them in this course.

## Variance Reduction

Examine our most recent expression for the policy gradient:

$$
\nabla_{\theta} J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau)\right].
$$

Taking a step with this gradient pushes up the log-probabilities of each action in proportion to R(\tau), the sum of all rewards ever obtained. But this doesn’t make much sense.

Agents should really only reinforce actions on the basis of their consequences. Rewards obtained before taking an action have no bearing on how good that action was: only rewards that come after.

It turns out that this intuition shows up in the math, and we can show that the policy gradient can also be expressed by

$$
\nabla_{\theta} J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \sum_{t'=t}^T R(s_{t'}, a_{t'})\right].
$$

We’ll call this form the “reward-to-go policy gradient,” because the sum of rewards after a point in a trajectory,

$$
\hat{R}_t \doteq \sum_{t'=t}^T R(s_{t'}, a_{t'}),
$$

is called the **reward-to-go** from that point, and this policy gradient expression depends on the reward-to-go from state-action pairs.


In fact, the policy gradient can be written as the general form

$$
\nabla_{\theta} J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \Phi_t\right],
$$

where $\Phi_t$ is some function of the trajectory which could be:

- $\Phi_t = R(\tau)$,

- $\Phi_t = \sum_{t'=t}^T R(s_{t'}, a_{t'})$,

- $\Phi_t = \sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1}) - b(s_t)$ for any function $b(s_t)$.

- On-Policy Action-Value Function: $\Phi_t = Q^{\pi_{\theta}}(s_t, a_t)$

- The Advantage Function: $\Phi_t = A^{\pi_{\theta}}(s_t, a_t) = Q^{\pi_{\theta}}(s_t, a_t) - V^{\pi_{\theta}}(s_t)$

In practice, $V^{\pi}(s_t)$ cannot be computed exactly, so it has to be approximated. This is usually done with a neural network, $V_{\phi}(s_t)$, which is updated concurrently with the policy (so that the value network always approximates the value function of the most recent policy).

The simplest method for learning $V_{\phi}$, used in most implementations of policy optimization algorithms, is to minimize a mean-squared-error objective:

$$
\phi_k = \arg \min_{\phi} \mathbb{E}_{s_t, \hat{R}_t \sim \pi_k} \left( V_{\phi}(s_t) - \hat{R}_t \right)^2,
$$

where $\pi_k$ is the policy at epoch $k$. This is done with one or more steps of gradient descent, starting from the previous value parameters $\phi_{k-1}$.










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


































