#%%
import torch.nn as nn
from config import config_TinyVGG

class TinyVGG(nn.Module):
    """
    A tiny VGG-like network for CIFAR-10.
    Architecture: Two convolutional blocks followed by a linear classifier.
    """
    def __init__(self, input_channels=config_TinyVGG["input_channels"], 
                 num_classes=config_TinyVGG["num_classes"]):
        super(TinyVGG, self).__init__()
        self.features = nn.Sequential(
            # Block 1: Two conv layers then max pool.
            nn.Conv2d(input_channels, config_TinyVGG["conv1_channels"], 
                     kernel_size=config_TinyVGG["kernel_size"], 
                     padding=config_TinyVGG["padding"]),
            nn.ReLU(),
            nn.Conv2d(config_TinyVGG["conv1_channels"], config_TinyVGG["conv1_channels"], 
                     kernel_size=config_TinyVGG["kernel_size"], 
                     padding=config_TinyVGG["padding"]),
            nn.ReLU(),
            nn.MaxPool2d(2),  # reduces 32x32 -> 16x16

            # Block 2: Two conv layers then max pool.
            nn.Conv2d(config_TinyVGG["conv1_channels"], config_TinyVGG["conv2_channels"], 
                     kernel_size=config_TinyVGG["kernel_size"], 
                     padding=config_TinyVGG["padding"]),
            nn.ReLU(),
            nn.Conv2d(config_TinyVGG["conv2_channels"], config_TinyVGG["conv2_channels"], 
                     kernel_size=config_TinyVGG["kernel_size"], 
                     padding=config_TinyVGG["padding"]),
            nn.ReLU(),
            nn.MaxPool2d(2)   # reduces 16x16 -> 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config_TinyVGG["conv2_channels"] * 8 * 8, num_classes)  # final linear layer for classification
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# For testing the model architecture independently.
if __name__ == "__main__":
    from torchsummary import summary
    model = TinyVGG()
    summary(model, (3, 32, 32))
    #%% Test the model
    x = torch.randn(1, 3, 32, 32)
    print(model(x).shape)
# %%
#%% 
    # Import ResNet18
    import torchvision.models as models
    resnet18 = models.resnet18(pretrained=True)
    print(resnet18)
# %%
