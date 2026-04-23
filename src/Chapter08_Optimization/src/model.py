# model.py
import torch
import torch.nn as nn

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Simple linear regression model.
        
        Args:
            input_dim (int): Dimension of input features
            output_dim (int): Dimension of output predictions
        """
        super().__init__()  # Call the constructor of the parent class
        
        # Initialize weight and bias as learnable parameters
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim))
        self.bias = nn.Parameter(torch.randn(output_dim))

    def forward(self, x):
        # Perform the linear transformation: y = xW^T + b
        return x @ self.weight.T + self.bias

# A simple hidden layer block with a linear layer and ReLU activation.
class HiddenLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(HiddenLayer, self).__init__()
        # Define a linear layer followed by ReLU activation
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Forward pass through the hidden layer
        x = self.fc(x)
        x = self.relu(x)
        return x

# A simple network using one hidden layer for classification.
class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNet, self).__init__()
        # Initialize the hidden layer
        self.hidden = HiddenLayer(input_dim, hidden_dim)
        # Output layer
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Forward pass through the hidden layer and then the output layer
        x = self.hidden(x)
        x = self.output(x)
        return x
