# Convolutional Neural Networks


Convolutional Neural Networks (CNNs) are a specialized type of neural network designed for processing structured grid data such as images. Unlike fully connected networks where each neuron is connected to all neurons in the previous layer, CNNs use a mathematical operation called convolution that applies localized filters to the input data.

The key intuition behind CNNs is to exploit spatial locality - nearby pixels in images tend to be related. By using convolutional filters that operate on local regions of the input, CNNs can efficiently identify local patterns like edges, textures, and shapes that are important for tasks like image recognition.

## The Convolution Operation

Given an input image $\mathbf{I}$ and a kernel (filter) $\mathbf{K}$, the 2D convolution operation is defined as:

$$
(\mathbf{I} * \mathbf{K})(i,j) = \sum_{m} \sum_{n} \mathbf{I}(i+m, j+n) \mathbf{K}(m,n)
$$

This operation slides the kernel $\mathbf{K}$ over the input image $\mathbf{I}$, performing element-wise multiplication at each location and summing the results to produce a single output value at each position.

### Padding and Stride

The convolution operation has two important hyperparameters:

- **Padding**: Adding zeros around the input to control the spatial dimensions of the output. If we have a $n \times n$ image and a $k \times k$ kernel:
  - Without padding: Output size is $(n - k + 1) \times (n - k + 1)$
  - With padding $p$: Output size is $(n + 2p - k + 1) \times (n + 2p - k + 1)$

- **Stride**: The step size for moving the kernel. With a stride of $s$, the output size becomes:
  - $\lfloor (n + 2p - k) / s + 1 \rfloor \times \lfloor (n + 2p - k) / s + 1 \rfloor$
  
In PyTorch, we can use the `nn.Conv2d` layer to perform the 2D convolution operation.

```python
conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
```

Below is an example of a 2D convolution operation with `kernel_size=3`, `stride=1`, and `padding=1`.

![CNN](./nn.assets/cnn.gif)

## Pooling Operations

Pooling layers reduce the spatial dimensions of the feature maps, providing:
1. Computational efficiency
2. Some degree of translation invariance
3. Control over overfitting

There are two types of pooling operations:

- Max pooling: takes the maximum value within a local region
- Average pooling: takes the average value within a local region


In PyTorch, we can use the `nn.MaxPool2d` layer to perform the max pooling operation. Below is an example of a max pooling operation with `kernel_size=2`, `stride=2`.

```python
pool = nn.MaxPool2d(kernel_size, stride)
```

![Max Pooling](./nn.assets/maxpool.gif)

Similarly, we can use the `nn.AvgPool2d` layer to perform the average pooling operation.

## CNN Architectures

The typical CNN architecture consists of:
1. **Convolutional layers**: Extract local patterns
2. **Activation functions** (typically ReLU): Add non-linearity
3. **Pooling layers**: Reduce spatial dimensions
4. **Fully connected layers**: Final classification/regression

## PyTorch for CNNs

PyTorch provides a convenient API for building CNNs using the `torch.nn` module. The basic building blocks include:

- `nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)`: 2D convolutional layer
- `nn.MaxPool2d(kernel_size, stride)`: Max pooling layer
- `nn.AvgPool2d(kernel_size, stride)`: Average pooling layer
- `nn.ReLU()`: ReLU activation function

### Simple CNN with Sequential API

```python
import torch
import torch.nn as nn

# Define a simple CNN for MNIST (28x28 grayscale images)
input_channels, output_size = 1, 10
model = nn.Sequential(
    # First convolutional block
    nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 16x14x14
    
    # Second convolutional block
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32x7x7
    
    # Flatten and fully connected layers
    nn.Flatten(),  # Output: 32*7*7 = 1568
    nn.Linear(32*7*7, 128),
    nn.ReLU(),
    nn.Linear(128, output_size)
)
```

### Custom CNN using nn.Module

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, input_channels, output_size):
        super().__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, output_size)
        
    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Second block
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Flatten and fully connected
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# Initialize the model
input_channels, output_size = 1, 10
model = SimpleCNN(input_channels, output_size)

# Print the model
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.shape}")
```

### More Complex CNN Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

class DeepCNN(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_size):
        super().__init__()
        
        # First block: two convolutions followed by pooling
        self.block1 = nn.Sequential(
            ConvBlock(input_channels, hidden_channels),
            ConvBlock(hidden_channels, hidden_channels),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Second block: two convolutions followed by pooling
        self.block2 = nn.Sequential(
            ConvBlock(hidden_channels, hidden_channels*2),
            ConvBlock(hidden_channels*2, hidden_channels*2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(hidden_channels*2, output_size)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Initialize the model
input_channels, hidden_channels, output_size = 3, 64, 10
model = DeepCNN(input_channels, hidden_channels, output_size)
```
