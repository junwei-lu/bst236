#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

def print_network_parameters(net):
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")

#%% Easy way to define a network
net = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
print_network_parameters(net)

#%% You can further add more layers to the network
net.append(nn.ReLU())
net.append(nn.Linear(10, 10))
print_network_parameters(net)

#%% You can even use for loop to add layers
net = nn.Sequential(nn.Linear(5,10))
depth = 3
for _ in range(depth):
    net.append(nn.ReLU())
    net.append(nn.Linear(10, 10))
print_network_parameters(net)

x = torch.randn(1, 5)
print(net(x))

#%% Define a one-hidden-layer network
class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size): # Define and initialize the network parameters
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x): # Define the forward pass of the network
        return self.net(x)

# Initialize the model
input_size, hidden_size, output_size = 784, 256, 10
net = TwoLayerNet(input_size, hidden_size, output_size)
print_network_parameters(net)

#%%
# Print the model
print_network_parameters(net)
# %%

# %% Define a custom activation function

class CustomTwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.alpha = nn.Parameter(torch.ones(1))

    def my_activation(self, x):
        return torch.where(x <= 0, 1 / (1 + self.alpha * torch.exp(-x)), x)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.my_activation(x)
        x = self.linear3(x)
        return x

net = CustomTwoLayerNet(input_size, hidden_size, output_size)
print_network_parameters(net)

# %% Define multiple-hidden-layer network with given depth

# First define a building block of one-hidden-layer neural network
class Block(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

# Then define a multi-layer neural network by repeating the building block
class MultiLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, depth, output_size):
        super().__init__()
        self.input_linear = nn.Linear(input_size, hidden_size)
        self.blocks = nn.Sequential(*[Block(hidden_size, hidden_size) for _ in range(depth-2)]) # repeat the building block depth times
        self.output_linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.input_linear(x)
        x = F.relu(x)
        x = self.blocks(x)
        x = self.output_linear(x)
        return x
    
net = MultiLayerNet(input_size, hidden_size, depth=4, output_size=output_size)
print_network_parameters(net)

# %% Dropout
import torch.nn as nn

class TwoLayerDropoutNet(nn.Module):
    def __init__(self, D_in, H, D_out): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(H, D_out)
        )

    def forward(self, x): 
        return self.net(x)

net = TwoLayerNet(input_size, hidden_size, output_size)
print_network_parameters(net)

# %% Batch Normalization
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)