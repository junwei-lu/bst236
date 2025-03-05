# Stochastic Gradient Descent

Stochastic Gradient Descent (SGD) is a popular optimization algorithm used to minimize an objective function, particularly in machine learning and deep learning. Unlike traditional gradient descent, which uses the entire dataset to compute gradients, SGD updates the model parameters using only a single data point or a small batch of data, making it more efficient for large datasets.

## Stochastic Gradient Descent Algorithm

The basic SGD algorithm iteratively updates the model parameters as follows:

$$
x_{t+1} = x_t - \eta_t \nabla f(x_t; \xi_t)
$$

where $\eta_t$ is the learning rate, and $\nabla f(x_t; \xi_t)$ is the gradient of the loss function with respect to a randomly selected data point $\xi_t$.

## Mini-Batch Gradient Descent

Mini-batch gradient descent is a variant of SGD that uses a small batch of data points to compute the gradient, balancing the efficiency of SGD with the stability of full-batch gradient descent.

**Mini-Batch Gradient Descent Algorithm**: For a mini-batch $\mathcal{B}_t$, update the parameters as follows:

$$
x_{t+1} = x_t - \eta_t \frac{1}{|\mathcal{B}_t|} \sum_{\xi \in \mathcal{B}_t} \nabla f(x_t; \xi)
$$

Mini-batch gradient descent reduces the variance of parameter updates, leading to more stable convergence.

## Adaptive Gradient Descent

Adaptive gradient descent algorithms adjust the learning rate for each parameter individually, allowing for more efficient optimization.

### Adagrad

Adagrad adapts the learning rate based on the historical gradient information:

$$
x_{t+1} = x_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \nabla f(x_t)
$$

where $G_t$ is the sum of the squares of past gradients, and $\epsilon$ is a small constant to prevent division by zero.

### RMSprop

RMSprop modifies Adagrad by introducing a decay factor to control the accumulation of past gradients:

$$
G_t = \gamma G_{t-1} + (1 - \gamma) \nabla f(x_t)^2
$$

$$
x_{t+1} = x_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \nabla f(x_t)
$$

where $\gamma$ is the decay rate.

## Adam

Adam (Adaptive Moment Estimation) combines the ideas of momentum and RMSprop, maintaining an exponentially decaying average of past gradients and squared gradients.

**Adam Algorithm**:

Compute the biased first moment estimate: $m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla f(x_t)$

Compute the biased second raw moment estimate: $v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla f(x_t))^2$

Correct the bias in the first and second moments: $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$

Update the parameters: $x_{t+1} = x_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$

Adam is widely used due to its robustness and efficiency in training deep neural networks.
