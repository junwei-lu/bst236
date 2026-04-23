# dataset.py
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class LinearRegressionDataset(Dataset):
    def __init__(self, num_samples=100):
        """
        Create a synthetic dataset for linear regression.
        
        Args:
            num_samples (int): Number of data points to generate.
        """
        # Generate input data with a uniform distribution between -10 and 10
        self.x = torch.linspace(-10, 10, num_samples).view(-1, 1)
        
        # Generate target data: y = 3x + 2 + noise
        # True relationship is y = 3x + 2, but we add some random noise
        self.y = 3 * self.x + 2 + torch.randn(self.x.size()) * 2

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# Custom dataset for loading images and labels.
# For demonstration, we assume images are stored in a directory,
# and the label is encoded as the first character of the filename.
class CarvanaDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = os.listdir(data_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):  # Get an item from the dataset
        # Load image from the specified directory
        img_filename = self.image_files[index]
        img_path = os.path.join(self.data_dir, img_filename)
        image = Image.open(img_path).convert("L")  # Convert image to grayscale

        # For demonstration, assume the label is the first character of the filename.
        # In practice, use a proper labeling strategy.
        label = int(img_filename[0])

        if self.transform:
            image = self.transform(image)
        
        # Flatten the image (for the simple one hidden layer model)
        image = image.view(-1)
        
        return image, label

def get_loaders(num_samples, batch_size):
    """
    Create data loaders for training and validation.
    
    Args:
        num_samples (int): Number of samples for each dataset.
        batch_size (int): Batch size.
        
    Returns:
        tuple: (train_loader, val_loader) - Data loaders for training and validation.
    """
    train_dataset = LinearRegressionDataset(num_samples=num_samples)
    val_dataset = LinearRegressionDataset(num_samples=num_samples // 4)  # Smaller validation set
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader
