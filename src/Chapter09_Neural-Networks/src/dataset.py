#%%
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class CustomCIFAR(Dataset):
    """
    A custom CIFAR-10 dataset that inherits from torch.utils.data.Dataset.
    It allows for user-defined augmentations and provides visualization capabilities.
    """
    def __init__(self, train=True, subset_size=None, transform=None, target_transform=None):
        self.train = train
        # Set default transforms if none provided.
        if transform is None:
            if train:
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),  # Data augmentation
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
                ])
        else:
            self.transform = transform

        self.target_transform = target_transform

        # Load the CIFAR-10 dataset.
        self.dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=train,
            download=True,
            transform=self.transform,
            target_transform=self.target_transform
        )

        self.class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # Optionally use a subset of the dataset.
        if subset_size is not None:
            self.dataset = Subset(self.dataset, range(min(subset_size, len(self.dataset))))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_loader(self, batch_size=64, shuffle=True):
        """
        Returns a DataLoader for the dataset.
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def get_train_val_loaders(self, batch_size=64, validation_split=0.2, shuffle=True):
        """
        Returns separate train and validation DataLoaders based on the current dataset.
        
        Args:
            batch_size: Batch size for both loaders
            validation_split: Fraction of the dataset to use for validation
            shuffle: Whether to shuffle the data
            
        Returns:
            train_loader, val_loader: DataLoader objects for training and validation
        """
        from torch.utils.data import random_split
        
        # Calculate split sizes
        dataset_size = len(self)
        val_size = int(validation_split * dataset_size)
        train_size = dataset_size - val_size
        
        # Split the dataset
        train_dataset, val_dataset = random_split(
            self,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
        
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=len(val_dataset),
            shuffle=False,  # No need to shuffle validation data
        )
        
        return train_loader, val_loader

    def visualize_samples(self, num_samples=8):
        """
        Visualizes a few samples from the dataset using matplotlib.
        Assumes normalization of mean=0.5 and std=0.5.
        """
        # Collect first num_samples items.
        samples = [self[i] for i in range(num_samples)]
        images, labels = zip(*samples)
        # Unnormalize images for display.
        images = [img * 0.5 + 0.5 for img in images]

        fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))
        for idx, ax in enumerate(axes):
            # Rearrange dimensions for display.
            ax.imshow(np.transpose(images[idx].numpy(), (1, 2, 0)))
            ax.set_title(str(self.class_names[labels[idx]]))
            ax.axis('off')
        plt.show()



#%%
if __name__ == "__main__":
    # Create a small subset for quick testing.
    custom_dataset = CustomCIFAR(train=True, subset_size=16)
    loader = custom_dataset.get_loader(batch_size=4)
    for images, labels in loader:
        # Check the shape of the batch
        print("Batch shape:", images.shape)
        break
    # Visualize the samples
    custom_dataset.visualize_samples(num_samples=8)

    #%%
    class_names = custom_dataset.class_names
    print( getattr(loader.dataset, 'class_names', None))



# %%
