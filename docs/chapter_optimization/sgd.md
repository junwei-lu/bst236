# Stochastic Gradient Descent

Stochastic Gradient Descent (SGD) is a popular optimization algorithm in machine learning and deep learning. 

The SGD problem is to solve:

$$
\min_{x\in X} F(x) = \frac{1}{n} \sum_{i=1}^n f_i(x)
$$

where $n$ is the sample size and it is typically too large to compute the full gradient $\nabla F(x)$ in one go.

Recalling the idea of [randomized linear algebra](../chapter_linear_algebra/stochastic_matrix.md), we can replace the full gradient with its unbiased estimator: $\nabla f_{\xi}(x)$ where $\xi$ is uniformly sampled from $\{1, 2, \cdots, n\}$. 

This motivates the basic SGD algorithm:

$$
x_{t+1} = x_t - \eta_t g_t
$$

where in the following of the note, $g_t$ is used to denote any unbiased estimator of $\nabla F(x_t)$, i.e.,

$$
\mathbb{E}[g_t] = \nabla F(x_t).
$$

## Mini-Batch Gradient Descent

Mini-batch gradient descent is a variant of SGD that uses a small batch of data points to compute the gradient, balancing the efficiency of SGD with the stability of full-batch gradient descent.

**Mini-Batch Gradient Descent Algorithm**: For a mini-batch $\mathcal{B}_t \subset \{1, 2, \cdots, n\}$, update the parameters as follows:

$$
x_{t+1} = x_t - \eta_t \frac{1}{|\mathcal{B}_t|} \sum_{i\in\mathcal{B}_t} \nabla f_i(x_t)
$$

Mini-batch gradient descent reduces the variance of parameter updates, leading to more stable convergence.

## PyTorch DataLoader and Optimizer

Mini-batch gradient descent can be implemented in PyTorch by the `DataLoader` class. The `DataLoader` is a PyTorch class that provides a way to load data from a dataset into a batch. Then the stochastic gradient descent can be implemented by the `torch.optim.SGD` function.

```python
import torch
from torch.utils.data import DataLoader

# Define the dataset
dataset = ... # PyTorch requires a dataset to be a subclass of torch.utils.data.Dataset

# Define the DataLoader
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
```

The mini-batch gradient descent can be implemented by the following training code.

```python
# Define the model class. Here we use a simple linear regression model as an example.
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        This is used to define and initialize the parameters of the model.
        All parameters in model should be nn.Parameter.
        Here we define a linear regression model with weight and bias.
        We will discuss other types of model class in the next chapter of deep learning.
        """
        super().__init__() # Call the constructor of the parent class
        # Initialize weight and bias as learnable parameters
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim))
        self.bias = nn.Parameter(torch.randn(output_dim))

    def forward(self, x):
        """
        This is used to define the mathematics of the model.
        """
        return x @ self.weight.T + self.bias

# Below is the training code:
# Define the loss function, e.g., MSELoss
loss_fn = nn.MSELoss()
# Define the optimizer 
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # PyTorch uses SGD as the mini-batch gradient descent optimizer

# Define the training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        outputs = model(batch) # Compute the output of the model
        loss = loss_fn(outputs, batch) # Compute the loss 
        # Backward pass
        optimizer.zero_grad() # Zero the gradients of the model parameters
        loss.backward() # Compute the gradient of the loss with respect to the model parameters
        optimizer.step() # Update the model parameters: x_{t+1} = x_t - lr * g_t
```

Here the mini-batch gradient descent can be implemented by PyTorch function `torch.optim.SGD` with the following parameters:

- `params=model.parameters()` passes all the trainable parameters from your model to the optimizer
- `lr=0.01` sets the learning rate (step size) to 0.01
- `momentum=0.9` You can even revise the mini-batch gradient descent by setting adding momentum to the optimizer with the momentum parameter.


We can write our own mini-batch gradient descent optimizer by the following code:

```python
lr = 0.01 # Set the learning rate
# 1. Manually zero the gradients
for param in model.parameters():
    if param.grad is not None:
        param.grad.zero_()
# 2. Compute gradients
loss.backward()
# 3. Update parameters using explicit SGD rule
with torch.no_grad():  # Prevent tracking history for parameter updates
    for param in model.parameters():
        if param.grad is not None:
            # The core SGD update: θ = θ - lr * gt
            param.data = param.data - lr * param.grad
```




## Adaptive Gradient Descent

Adaptive gradient descent algorithms adjust the learning rate for each parameter individually, allowing for more efficient optimization. If the partial derivative $\partial f(x) / \partial x_j$ is smaller, the $j$-th entry $x_{t,j}$ is less likely to be updated and we tend to take a larger step. On the other hand, if the partial derivative $\partial f(x) / \partial x_j$ is larger, the $j$-th entry $x_{t,j}$ was updated more significantly and we tend to take a smaller step. The idea of adaptive gradient descent is to measure the standard deviation of the partial derivatives and adjust the learning rate accordingly.

### Adagrad

Adagrad adapts the learning rate based on the standard deviation of the historical gradient information. Let $x_{t,j}$ be the $j$-th component of $x_t$, then Adagrad updates each entry of $x_t$ as follows:

$$
\begin{align*}
G_{t,j} &= G_{t-1,j} + g_{t,j}^2\\
x_{t+1,j} &= x_{t,j} - \frac{\eta}{\sqrt{G_{t,j} + \epsilon}} g_{t,j}
\end{align*}
$$

for all $j=1,2,\cdots,d$, where $G_{t,j}$ is the sum of the squares of past gradients, and $\epsilon$ is a small constant to prevent division by zero. For the notation simplicity, we will write the above Adagrad update as:

$$
\begin{align*}
G_{t} &= G_{t-1} + g_{t}^2\\
x_{t+1} &= x_{t} - \frac{\eta}{\sqrt{G_{t} + \epsilon}} g_{t}
\end{align*}
$$

where $g_{t}^2$ is the entry-wise square of $g_t$.

Adagrad can be implemented by the following code:

```python
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
```

### RMSprop

RMSprop modifies Adagrad by introducing a decay factor to control the accumulation of past gradients:

$$
G_{t} = \gamma G_{t-1} + (1 - \gamma) g_{t}^2
$$

$$
x_{t+1} = x_{t} - \frac{\eta}{\sqrt{G_{t} + \epsilon}} g_{t}
$$

where $\gamma$ is the decay rate.

RMSprop can be implemented by the following code:

```python
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
```

## Adam

Adam (Adaptive Moment Estimation) combines the ideas of momentum and RMSprop, maintaining an exponentially decaying average of past gradients and squared gradients.

**Adam Algorithm**:


$$
\begin{align*}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t\\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t}\\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^t}\\
x_{t+1} &= x_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
\end{align*}
$$

Adam is widely used due to its robustness and efficiency in training deep neural networks. It can be implemented by the following code:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))
```


