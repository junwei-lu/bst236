import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    """Base model class that all models should inherit from"""
    def __init__(self):
        super(BaseModel, self).__init__()
        
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        torch.save(self.state_dict(), path)
        
    def load_checkpoint(self, path):
        """Load model from checkpoint"""
        self.load_state_dict(torch.load(path))
        
class ConvNet(BaseModel):
    """Example CNN architecture"""
    def __init__(self, in_channels, num_classes):
        super(ConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the flattened size based on your input and architecture
        # This is an example, adjust as needed
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def get_model(config):
    """
    Factory function to create a model based on config
    
    Args:
        config: Configuration containing model parameters
        
    Returns:
        model: The instantiated model
    """
    if config.model_name == 'convnet':
        model = ConvNet(
            in_channels=config.in_channels,
            num_classes=config.num_classes
        )
    # Add more model options as needed
    else:
        raise ValueError(f"Model {config.model_name} not supported")
        
    return model 