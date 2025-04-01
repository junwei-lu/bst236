import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class CustomDataset(Dataset):
    """
    Custom dataset class for your specific data
    """
    def __init__(self, data_path, transform=None, split='train'):
        """
        Initialize the dataset
        
        Args:
            data_path (str): Path to the data directory
            transform (callable, optional): Optional transform to be applied on a sample
            split (str): 'train', 'val', or 'test' split
        """
        self.data_path = data_path
        self.transform = transform
        self.split = split
        
        # Load data
        self.data, self.labels = self._load_data()
        
    def _load_data(self):
        """Load data from disk"""
        # Implementation depends on your data format
        # Example:
        # data = np.load(os.path.join(self.data_path, f'{self.split}_data.npy'))
        # labels = np.load(os.path.join(self.data_path, f'{self.split}_labels.npy'))
        # return data, labels
        pass
        
    def __len__(self):
        """Return the size of the dataset"""
        return len(self.data)
        
    def __getitem__(self, idx):
        """Return a sample from the dataset"""
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label

def get_dataloader(config):
    """
    Create data loaders for train, validation, and test sets
    
    Args:
        config: Configuration object with data parameters
        
    Returns:
        dict: Dictionary containing train, val, and test dataloaders
    """
    # Define transformations
    # Example:
    # train_transform = transforms.Compose([...])
    # val_transform = transforms.Compose([...])
    
    # Create datasets
    train_dataset = CustomDataset(
        config.data_path, 
        transform=train_transform, 
        split='train'
    )
    
    val_dataset = CustomDataset(
        config.data_path, 
        transform=val_transform, 
        split='val'
    )
    
    test_dataset = CustomDataset(
        config.data_path, 
        transform=val_transform, 
        split='test'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    } 