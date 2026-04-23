#%% Load CIFAR10 dataset
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

# Convert images to tensors for training
transform = transforms.ToTensor()
# Download and load training dataset
trainset = datasets.CIFAR10(root='./data', train=True,
                          download=True, transform=transform)
print(len(trainset)) # 50000
# Download and load test dataset
testset = datasets.CIFAR10(root='./data', train=False,
                          download=True, transform=transform)
print(len(testset)) # 10000

#%% Visualize CIFAR10 dataset
import matplotlib.pyplot as plt
import numpy as np

def show_image(image_tensor):
    # Convert to numpy and transpose to (H, W, C)
    image_np = image_tensor.numpy().transpose(1, 2, 0)
    # Plot the image
    plt.imshow(image_np)
    plt.axis('off')
    plt.show()

image, _ = trainset[0]
print(image.shape) # torch.Size([3, 32, 32])
show_image(image)

#%% Data Augmentation
def apply_transform(img, aug, num_times=6):
    imgs = [aug(img) for _ in range(num_times)]
    plt.figure(figsize=(num_times * 2, 2))
    for i, im in enumerate(imgs):
        img_np = im.numpy().transpose(1, 2, 0)
        plt.subplot(1, num_times, i + 1)
        plt.imshow(img_np)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

##% Flip
apply_transform(image, torchvision.transforms.RandomHorizontalFlip())
apply_transform(image, torchvision.transforms.RandomRotation(degrees=15))

#%% Color Jitter
apply_transform(image, torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))

#%% Crop
apply_transform(image, torchvision.transforms.RandomResizedCrop(size=(32, 32), scale=(0.5, 1.5), ratio=(0.5, 1.5)))
#%% Compose
from torchvision import transforms
transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(degrees=15),
    torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    torchvision.transforms.RandomResizedCrop(size=(32, 32), scale=(0.5, 1.5), ratio=(0.5, 1.5))
])
apply_transform(image, transform)trainset_aug = datasets.CIFAR10(root='./data', train=True,
                          download=True, transform=transform)
trainloader = DataLoader(trainset_aug, batch_size=128, shuffle=True)

#%% 
