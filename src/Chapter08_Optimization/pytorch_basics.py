#%% Importing libraries
import torch
import numpy as np
import pandas as pd

#%% Creating a basic PyTorch tensor
tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

# Tensor Operations
squared_tensor = tensor ** 2     # Element-wise squaring
sum_value = tensor.sum()         # Calculate sum
mean_value = tensor.mean()	   # Calculate mean

#%% Creating tensors with specific properties
zeros = torch.zeros(3, 4)                 # Tensor of zeros
ones = torch.ones(2, 3)                   # Tensor of ones
random_tensor = torch.rand(2, 3)          # Random values between 0 and 1
range_tensor = torch.arange(0, 10, 2)     # Range with step size

#%% Reshaping
original = torch.arange(6)
reshaped = original.reshape(2, 3)
flattened = reshaped.flatten()
reshaped.shape
flattened.shape

#%% Concatenation
tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])
concat_result = torch.cat([tensor1, tensor2])
stacked_result = torch.stack([tensor1, tensor2])  # New dimension

#%% Indexing
values = torch.tensor([1, 2, 3, 4, 5, 6])
subset = values[2:5]

#%% Boolean masking
mask = values > 3
filtered = values[mask]  # Returns tensor([4, 5, 6])

#%% Finding indices matching condition
indices = torch.where(values % 2 == 0)

#%% NumPy array to PyTorch tensor
np_array = np.array([1, 2, 3])
tensor_from_np = torch.from_numpy(np_array)

#%% PyTorch tensor to NumPy array
tensor = torch.tensor([4, 5, 6])
np_from_tensor = tensor.numpy()

#%% Pandas DataFrame to PyTorch tensor
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
tensor_from_df = torch.tensor(df.values)

#%% PyTorch tensor to Pandas DataFrame
tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
df_from_tensor = pd.DataFrame(tensor_2d.numpy())