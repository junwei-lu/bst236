
# Fine-tuning


Fine-tuning is a powerful technique in computer vision where a pre-trained model (typically trained on a large dataset like ImageNet) is adapted to a new, often smaller dataset. Instead of training a model from scratch, fine-tuning leverages the knowledge already captured in the pre-trained model's weights. This approach is particularly effective when you have limited training data. The process typically involves freezing the early layers of the network (which capture generic features like edges and textures) while retraining the later layers (which capture more task-specific features). Fine-tuning generally requires less computational resources and training time compared to training from scratch, and often results in better performance, especially for datasets similar to the one used for pre-training.


PyTorch's `torchvision` package  provides multiple pre-trained models. You can find the full list [here](https://pytorch.org/vision/main/models.html). We will use ResNet18 as an example.

```python
import torch
import torchvision.models as models

# Load pre-trained ResNet18
resnet18 = models.resnet18(pretrained=True)

# For inference
resnet18.eval()
input_tensor = torch.randn(1, 3, 224, 224)
output = resnet18(input_tensor)

# Print the model architecture
print(resnet18)

# ResNet(
#   (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#   (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (relu): ReLU(inplace=True)
#   (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#   (layer1): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (layer2): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# ...
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
#   (fc): Linear(in_features=512, out_features=1000, bias=True)
# )
```

Once you print the model architecture, you can see the final fully connected layer is a linear layer with 512 output features. 


## Fine-tuning last layer

To fine-tune the pre-trained ResNet on a new task, with new number of classes, we need to replace the final fully connected layer with a new linear layer with the number of output features equal to the number of classes in the new task. Then we need to unfreeze the last layer using `param.requires_grad = True` and freeze all other layers using `param.requires_grad = False`.
    
```python
# For fine-tuning on a new task
num_classes = 10  # New number of classes
resnet18.fc = nn.Linear(512, num_classes)  # Replace the final fully connected layer
for param in resnet18.parameters():
    param.requires_grad = False  # Freeze all layers

for param in resnet18.fc.parameters():
    param.requires_grad = True  # Unfreeze only the last layer

# Use resnet18 for the training ...
```