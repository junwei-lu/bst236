# Introduction to PyTorch

[PyTorch](https://pytorch.org/) is a popular deep learning library that provides tensor computation with GPU acceleration and automatic differentiation capabilities.

## Basic Operations

**Create and manipulate**:

```python
import torch
import numpy as np
import pandas as pd

# Creating a basic PyTorch tensor
tensor = torch.tensor([1, 2, 3, 4, 5])

# Tensor Operations
squared_tensor = tensor ** 2     # Element-wise squaring
sum_value = tensor.sum()         # Calculate sum

# Creating tensors with specific properties
zeros = torch.zeros(3, 4)                 # Tensor of zeros
ones = torch.ones(2, 3)                   # Tensor of ones
random_tensor = torch.rand(2, 3)          # Random values between 0 and 1
range_tensor = torch.arange(0, 10, 2)     # Range with step size
```

**Reshaping and manipulating**:

```python
# Reshaping
original = torch.arange(6)
reshaped = original.reshape(2, 3)
flattened = reshaped.flatten()
reshaped.shape      # torch.Size([2, 3])
flattened.shape     # torch.Size([6])

# Concatenation
tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])
concat_result = torch.cat([tensor1, tensor2])
stacked_result = torch.stack([tensor1, tensor2])  # New dimension
```

**Indexing and masking**:

```python
# Indexing
values = torch.tensor([1, 2, 3, 4, 5, 6])
subset = values[2:5]

# Boolean masking
mask = values > 3
filtered = values[mask]  # Returns tensor([4, 5, 6])

# Finding indices matching condition
indices = torch.where(values % 2 == 0)
```

**Converting Between NumPy, Pandas, and PyTorch**:

```python
# NumPy array to PyTorch tensor
np_array = np.array([1, 2, 3])
tensor_from_np = torch.from_numpy(np_array)

# PyTorch tensor to NumPy array
tensor = torch.tensor([4, 5, 6])
np_from_tensor = tensor.numpy()

# Pandas DataFrame to PyTorch tensor
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
tensor_from_df = torch.tensor(df.values)

# PyTorch tensor to Pandas DataFrame
tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
df_from_tensor = pd.DataFrame(tensor_2d.numpy())
```

## Linear Algebra

PyTorch provides a comprehensive linear algebra library in `torch.linalg`. You can find the official documentation [here](https://pytorch.org/docs/stable/linalg.html). The functions basically have the same names as the ones in NumPy and SciPy.

Below we list some of the most commonly used functions for matrix operations, solving linear equations, and eigenvalue problems.

```python
# Matrix multiplication
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])
C = A @ B  # or torch.matmul(A, B)

# Solving linear equations
b = torch.tensor([5, 6])
x = torch.linalg.solve(A, b)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = torch.linalg.eig(A)

# Use torch.linalg.eigh for symmetric/Hermitian matrices

# Singular value decomposition
U, S, Vh = torch.linalg.svd(A, full_matrices=False)
```

Notice that `torch.linalg.solve` does not provide the option to use a specific solver. It will use the LU decomposition by default. If your matrix is symmetric positive definite, you can use `torch.linalg.cholesky` to solve the linear equations.

```python
A = torch.tensor([[1, 2], [2, 5]])
L = torch.linalg.cholesky(A)
y = torch.linalg.triangular_solve(L, b, upper=False)
x = torch.linalg.triangular_solve(L.T, y, upper=True)
```

## Automatic Differentiation

PyTorch's automatic differentiation system (autograd) enables gradient-based optimization for training neural networks.

```python
# Basic autograd example
x = torch.tensor([2.0, 3.0], requires_grad=True)  # Enable gradient tracking
y = torch.sum(x * x)  # y = x^2

# Compute gradient of z with respect to x
y.backward()

# Access gradients
x.grad == 2 * x  # Should be 2*x: tensor([4., 6.])
```

You need to zero the gradients by `x.grad.zero_()` before computing the gradient at a new point or for a new function. Otherwise, PyTorch will accumulate the gradients.

```python 
# Zeroing gradients before computing new ones
x.grad.zero_()
y = torch.sum(x**3)
y.backward()
x.grad == 3*x**2
```

**Partial Derivatives**:

```python
# Computing partial derivatives
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
z = torch.sum(x * y)
z.backward()
print(x.grad)  # Should be tensor([4., 5., 6.])
```

**Gradients with control flow**:

```python
def f(x):
    y = x * 2
    while y.norm() < 1000:
        y = y * 2
    return y

x = torch.tensor([0.5], requires_grad=True)
y = f(x)
y.backward()
print(x.grad)  # Gradient depends on control flow path taken
```

### Gradient Update

When implementing parameter updates like gradient descent in PyTorch, it's crucial to use `torch.no_grad()` to prevent autograd from tracking operations. Here is an example of what happens if we update the parameter without `torch.no_grad()`.

```python
# BAD: Update without torch.no_grad()
x = torch.tensor([2.0], requires_grad=True)
y = x**2
y.backward()
x = x - 0.1 * x.grad  # Creates a new tensor, loses gradient connection
print(x.grad)  # None - gradient information is lost!

# GOOD: Update with torch.no_grad()
x = torch.tensor([2.0], requires_grad=True)
y = x**2
y.backward()
with torch.no_grad():
    x -= 0.1 * x.grad  # Updates in-place without building computational graph
```

Notice that we need to disable the gradient tracking for parameter updates by `with torch.no_grad()`. Otherwise, the parameter will be part of the computation graph and the gradient will be disconnected from the original gradient.

!!! warning "Pitfall of Parameter Updates in PyTorch"
    Always use `torch.no_grad()` when manually updating parameters in optimization algorithms in PyTorch. And use `x.grad.zero_()` to zero out the gradients before computing the new ones.


Also, you should not write `x -= 0.1 * x.grad` as `x = x - 0.1 * x.grad` because it will

- Creates a brand new tensor and assigns it to variable `x`, which is inefficient for memory usage.
- The new tensor `x` loses the connection to the computational graph
- The right-hand side is an expression involving `x.grad` which has `requires_grad=True`. PyTorch will start tracking gradients for the parameter update itself

In general, you should update the parameters by **in-place operations**. For simple gradient update, you can use `x -= 0.1 * x.grad`. For general parameter updates, you can first compute the value by a new variable `x_new = update_rule(x,x.grad)` and then use `x.copy_(x_new)` or `x[:] = x_new` to copy the value back to `x`. When you use `x = g(x,x.grad)`, you're creating a completely new tensor and assigning it to the variable `x`. This breaks the computational graph connection to the original tensor.


We still use the gradient descent update as an example below. You can refer to the [Frank-Wolfe Algorithm](./gradient_descent.md#example-constrained-lasso) or [Proximal Gradient Descent](./proximal_gradient_descent.md) for more realistic examples.

```python
x = torch.tensor([2.0], requires_grad=True)
y = x**2
y.backward()
x_new = x - 0.1 * x.grad # Update x using x_new to avoid recreating x
x.copy_(x_new) # or x[:] = x_new
```

!!! warning "Pitfall of In-place Operations in PyTorch"
    If the algorithm has updating rule like $x_{t+1} = g(x_t,\nabla f(x_t))$ for some function $g$, avoid using `x = g(x,x.grad)` in the updating step. You should use in-place operations like `x.copy_(g(x,x.grad))` or `x[:] = g(x,x.grad)` instead. 


We will introduce more about how to use PyTorch to implement the optimization algorithms with setting up the dataloader, model, and optimizer in the [future lecture](sgd.md#pytorch-optimizer-pipeline).


