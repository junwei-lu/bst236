# PyTorch Pitfalls

This section serves as a collection of common pitfalls and best practices when using PyTorch. You may not encounter all of them, but it's still useful to know what to look out for.


## Loss function

The cross entropy loss function `nn.CrossEntropyLoss()`  in PyTorch expects raw logits, not softmaxed outputs.

That means for linear classifier $O = Wx + b$, we predict $\hat{y} = \text{softmax}(O)$, but we should pass the raw logits $O$ to the loss function, not the softmaxed outputs $\hat{y}$.

#### Common Pitfalls

- You passed softmaxed outputs to a loss that expects raw logits

```python
# Wrong
predictions = torch.softmax(model(data), dim=1)
loss = nn.CrossEntropyLoss()(predictions, targets)
# Correct
predictions = model(data)
loss = nn.CrossEntropyLoss()(predictions, targets)
```

- Inconsistent handling of class dimension in classification tasks

```python
# Wrong: targets include class dim
  predictions = model(data)  # [batch_size, num_classes]
  targets = F.one_hot(targets, num_classes)  # [batch_size, num_classes]
  loss = nn.CrossEntropyLoss()(predictions, targets)  # Error!
  
# Correct: targets are class indices
  targets = targets  # [batch_size] with class indices
  loss = nn.CrossEntropyLoss()(predictions, targets)
```

## Dropout

Dropout is only applied during training. When evaluating the model on the validation set or test set, you should turn it off.

### Common Pitfalls

- You should not use `F.dropout()` in the `forward` method, otherwise the model will implement dropout in evaluation mode as well. Instead, use `nn.Dropout()` in the model definition.

```python
# Wrong
import torch.nn.functional as F
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.dropout(x, p=0.5)

# Correct
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.dropout(x)
```

### Best Practices

- Use `nn.Dropout()` in the model definition and never use `F.dropout()`.
- Always use `model.train()` before the training and use `model.eval()` in evaluation.


## Gradient Accumulation

When accumulating gradients across multiple batches, remember to zero the gradients before each backward pass to avoid incorrect gradient accumulation.

#### Common Pitfalls

- You forgot to `.zero_grad()`  before `.backward()`

```python
# Wrong
optimizer.zero_grad()
for batch in dataloader:
    loss = model(batch)
    loss.backward()

# Correct
for batch in dataloader:
    loss = model(batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

- Accumulating losses and taking the gradient

```python
# Problematic pattern
for batch in dataloader:
    outputs = model(batch)
    current_loss = nn.MSELoss()(outputs, batch.target)
    loss += current_loss.item()  # In-place operation!

loss.backward()  
```

There are two problems. 

First, Calling `.item()` detaches the value from the computation graph, so when you do: `loss += current_loss.item()`, you are backpropagating on a float—not a proper tensor with gradients.

Second, `loss += current_loss` is an in-place operation, and it can break the computation graph. You should use `loss = loss + current_loss` to keep the computation graph intact.








#### Best Practices

- Zero gradients between backward passes. By default, always use this code snippet:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

- If you need to write your own optimizer or manipulate gradients, e need to disable the gradient tracking for parameter updates by `with torch.no_grad()`.

```python
with torch.no_grad():
    x -= 0.1 * x.grad
```

- The only exception is when you need to update the parameters, e.g. when you are using gradient accumulation. Sometimes, to save the memory, we may not call `optimizer.step()` until accumulating the gradients for several batches:

```python
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss.backward()
    # Don't call optimizer.step() here
    if i % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## In-place operations

In-place operations in PyTorch modify tensors directly rather than creating new ones. While they can save memory, they come with important limitations and potential pitfalls.
Most of the methods end with `_` in PyTorch are in-place operations. Examples of common in-place operations:


- `tensor.add_(value)`
- `tensor.mul_(value)`
- `tensor.copy_(source)`
- `tensor.zero_()`
- `tensor.zero_()`

In PyTorch, 

- `x += 1` is an in-place operation that modifies the tensor directly, equivalent to calling `x.add_(1)`. The underscore suffix in PyTorch methods indicates in-place operations.
- `x = x + 1` creates a new tensor with the result and assigns it to the variable `x`. The original tensor is not modified.

### Common Pitfalls

- Breaking the computational graph

The most serious pitfall is modifying tensors that are part of an active computational graph:

```python
x = torch.tensor([1.0], requires_grad=True)
y = x * 2
x.add_(1)  # In-place modification of x
z = y + 3  # y depends on the original value of x
z.backward()  # RuntimeError: leaf variable has been modified by an inplace operation
```
When you modify `x` in-place, the relationship between `x` and `y` is invalidated because `y` was computed using the original value of `x`.

### Best Practices

- Avoid in-place operations on tensors that require gradients
- Use `x.clone()` or `x.copy_()` to create a new tensor with the same value as `x`, but not part of the computational graph:
- When modifying parameters in-place, do so after optimizer steps and use `with torch.no_grad()`.











## Shape Manipulation

Use `squeeze()` when:
- You specifically want to remove dimensions of size 1
- You need to normalize tensor shape to remove singleton dimensions
- You're matching shapes for operations that don't support broadcasting
- You're working with models that add extra dimensions (like unsqueeze)

```python
# Good use of squeeze() - removing specific singleton dimensions
x = torch.randn(10, 1, 20, 1)
squeezed = x.squeeze()  # Removes all dimensions of size 1: [10, 20]
squeezed_specific = x.squeeze(1)  # Only removes dim 1: [10, 20, 1]
```

Use `flatten()` when:
- You specifically want to flatten consecutive dimensions
- You want more readable and self-documenting code
- You want to preserve batch dimensions (using start_dim)
- You need the specific semantic meaning of flattening

```python
# Good use of flatten() - clear intention
x = torch.randn(32, 3, 224, 224)  # [batch, channels, height, width]
flattened = x.flatten(start_dim=1)  # [32, 3*224*224] - preserve batch dimension
```

Use `reshape()` when:
- You're unsure about contiguity of your tensor
- Your code needs to work with potentially non-contiguous tensors
- You want safer code that won't throw contiguity errors
- You need a general-purpose reshaping solution

```python
# Good use of reshape() - more robust
x = some_function_that_might_return_non_contiguous_tensor()
reshaped = x.reshape(batch_size, num_features)  # Works regardless of contiguity
```

Use `view()` when:
- Your tensor is definitely contiguous
- You need maximum performance (no data copying)
- You're reshaping in a straightforward way without changing element order
- You're in a performance-critical section of code

```python
# Good use of view()
x = torch.randn(32, 3, 224, 224)  # Fresh tensor is contiguous
flattened = x.view(32, -1)  # Efficiently reshape without copying
```

### Decision Flowchart

1. **Are you removing dimensions of size 1?**
    - If yes → Use `squeeze()`

2. **Are you flattening consecutive dimensions?**
    - If yes → Use `flatten()` for readability

3. **Are you unsure about tensor contiguity?**
    - If yes → Use `reshape()` for safety

4. **Is the tensor definitely contiguous and performance critical?**
    - If yes → Use `view()` for maximum performance

#### Best Practices

- Be Explicit About Your Model's Output Shape
```python
def forward(self, x):
    # Be explicit about output shape
    x = self.linear(x)  # Shape: [batch_size, 1]
    return x.squeeze(1)  # Shape: [batch_size]
```

- Use Shape Assertions for Critical Layers
```python
def forward(self, x):
    assert x.dim() == 2, f"Expected input to have 2 dimensions, got {x.dim()}"
    output = self.linear(x)
    output = output.squeeze(1)
    assert output.dim() == 1, f"Expected output to have 1 dimension, got {output.dim()}"
    return output
```





#### Common Pitfalls

Here are some common pitfalls when manipulating tensor shapes in PyTorch:

**Contiguity-Related Pitfalls**

- **Using `view()` on non-contiguous tensors**
  ```python
  x = torch.randn(4, 5).transpose(0, 1)  # Non-contiguous after transpose
  x.view(-1)  # ERROR: view size is not compatible with input tensor's size and stride
  ```
  *Solution:* Use `reshape()` or call `contiguous()` first: `x.contiguous().view(-1)`

- **Forgetting that operations like `transpose()`, `permute()`, and slicing create non-contiguous tensors**
  ```python
  x = torch.randn(10, 20, 30)
  y = x.permute(2, 0, 1)  # Now non-contiguous
  ```
*Solution:* Never use `view()` unless you are sure the tensor is contiguous.

**Dimension-Related Pitfalls**

- **Incorrectly calculating dimensions for reshape**
  ```python
  x = torch.randn(10, 3, 224, 224)
  # Wrong: miscalculated dimensions
  x.reshape(10, 150528)  # Should be 3*224*224 = 150528
  # Correct: use -1 for automatic calculation
  x.reshape(10, -1) # or x.flatten(start_dim=1)
  ```
  *Solution:* Never compute the dimensions manually. Use `-1` for automatic calculation or clearly document the math

**Broadcasting and Operation Pitfalls**

Pytorch supports broadcasting, which means that it will automatically expand tensors to the same shape when performing operations. Though it is convenient some times, it can also lead to unexpected results.

- **Unexpected broadcasting behavior**
  ```python
  a = torch.randn(10, 1)
  b = torch.randn(1, 20)
  c = a * b  # Results in shape [10, 20] through broadcasting
  ```
  *Solution:* Be explicit about intended broadcasting with `expand` or `repeat`


- **Dimension mismatch between predictions and targets**
  ```python
  model = nn.Linear(10, 1)
  data = torch.randn(10)
  target_data = torch.randn(10)
  predictions = model(data)  # Shape: [batch_size, 1]
  targets = target_data      # Shape: [batch_size]
  # Wrong: this will not raise an error, but the loss will be incorrect by broadcasting
  loss = nn.MSELoss()(predictions, targets)  
  # Correct: 
  predictions = model(data).squeeze()
  loss = nn.MSELoss()(predictions, targets)
  ```

  *Solution:* In your `forward` method, make sure to squeeze the output of the model to match the target shape.

