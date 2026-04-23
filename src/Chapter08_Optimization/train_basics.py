import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# Custom Dataset
class LinearRegressionDataset(Dataset):
    def __init__(self, num_samples=100):
        self.x = torch.randn(num_samples, 1)
        self.y = 3 * self.x + 2 + torch.randn(self.x.size()) * 2

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__() # Call the constructor of the parent class
        # Initialize weight and bias as learnable parameters
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim))
        self.bias = nn.Parameter(torch.randn(output_dim))

    def forward(self, x):
        # Perform the linear transformation
        return x @ self.weight.T + self.bias

# Model for logistic regression
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Initialize weight and bias as learnable parameters
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim))
        self.bias = nn.Parameter(torch.randn(output_dim))
        # Sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Linear transformation followed by sigmoid activation
        linear = x @ self.weight.T + self.bias
        return self.sigmoid(linear)




# DataLoader
dataset = LinearRegressionDataset(num_samples=100)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Model
model = LinearRegressionModel(input_dim=1, output_dim=1)

# Loss and Optimizer
criterion = nn.MSELoss()




# SGD
optimizer = optim.SGD(model.parameters(), lr=0.01)
# Adam
# optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))

# Training Loop
num_epochs = 100
for epoch in range(num_epochs): 
    for inputs, targets in dataloader: # every batch 10; go through all the data
        outputs = model(inputs) # y = f(x) = wx + b
        loss = criterion(outputs, targets) # loss = (y - y_true)^2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # x = x - lr * gt Perform a single optimization step to update parameter.
        # compute gradient norm
        grad_norm = sum(p.grad.norm() for p in model.parameters())

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Gradient Norm: {grad_norm.item():.4f}')
