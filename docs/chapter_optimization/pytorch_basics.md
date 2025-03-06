# PyTorch Tensor

[PyTorch](https://pytorch.org/) is a popular deep learning library that provides tensor computation with GPU acceleration and automatic differentiation capabilities.

## Basic Operations



**Create and manipulate**:

```python
import torch

# Creating a basic PyTorch tensor
tensor = torch.tensor([1, 2, 3, 4, 5])

# Tensor Operations
squared_tensor = tensor ** 2     # Element-wise squaring
mean_value = tensor.mean()       # Calculate mean
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
import numpy as np
import pandas as pd

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

## Automatic Differentiation

PyTorch's automatic differentiation system (autograd) enables gradient-based optimization for training neural networks.

```python
# Basic autograd example
x = torch.tensor([2.0, 3.0], requires_grad=True)  # Enable gradient tracking
y = x * x  # y = x^2
z = y.sum()  # z = sum(y)

# Compute gradient of z with respect to x
z.backward()

# Access gradients
print(x.grad)  # Should be 2*x: tensor([4., 6.])
```

**Gradient for non-scalar outputs**:

```python
# For non-scalar outputs, specify gradient argument
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = x * x
y.backward(torch.ones_like(y))  # Equivalent to sum() then backward()
print(x.grad)  # Should be 2*x
```

**Detaching computation**:

```python
# Stop gradient tracking
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * x
z = y.detach()  # Detach from computation graph
z.backward(torch.ones_like(z))  # Has no effect on x.grad
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
