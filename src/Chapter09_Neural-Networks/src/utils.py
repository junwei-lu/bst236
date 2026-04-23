import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import torchvision
from sklearn.metrics import confusion_matrix
import seaborn as sns

class FeatureExtractor:
    """
    Class for extracting features from intermediate layers of a model.
    """
    def __init__(self, model, layer_indices):
        """
        Initialize the feature extractor.
        
        Args:
            model: The PyTorch model to extract features from
            layer_indices: List of layer indices to extract features from
        """
        self.model = model
        self.layer_indices = layer_indices
        self.features = [None] * len(layer_indices)
        self.hooks = []
        
        # Register hooks for each layer
        for i, layer_idx in enumerate(layer_indices):
            hook = self._register_hook(i, layer_idx)
            self.hooks.append(hook)
    
    def _register_hook(self, feature_idx, layer_idx):
        """Register a forward hook on the specified layer."""
        def hook_fn(module, input, output):
            self.features[feature_idx] = output.detach()
        
        # Register the hook on the specified layer
        return self.model.features[layer_idx].register_forward_hook(hook_fn)
    
    def extract_features(self, x):
        """
        Extract features by passing input through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            List of feature maps
        """
        self.model.eval()
        with torch.no_grad():
            _ = self.model(x)
        return self.features
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()


def visualize_layer_outputs(model, image, device, save_path=None, fig_size=(15, 5)):
    """
    Visualize the outputs of different layers of the TinyVGG model.
    
    Args:
        model: TinyVGG model
        image: Input image tensor of shape [1, C, H, W]
        device: Device to run the model on
        save_path: Path to save the figure, if None the figure will be displayed
        fig_size: Size of the figure
        
    Returns:
        matplotlib figure
    """
    # Ensure model is in eval mode
    model.eval()
    
    # Move image to device
    image = image.to(device)
    
    # Create feature extractor
    # Layer 4 is after first block (16x16 feature maps)
    # Layer 9 is after second block (8x8 feature maps)
    extractor = FeatureExtractor(model, [4, 9])
    
    # Extract features
    features = extractor.extract_features(image)
    
    # Remove hooks
    extractor.remove_hooks()
    
    # Create figure with 3 subplots (input, block1, block2)
    fig, axes = plt.subplots(1, 3, figsize=fig_size)
    
    # Plot input image
    input_img = image[0].cpu().numpy()
    # Denormalize image (assuming normalization with mean=0.5, std=0.5)
    input_img = np.transpose(input_img * 0.5 + 0.5, (1, 2, 0))
    axes[0].imshow(input_img)
    axes[0].set_title('Input Image (32x32)')
    axes[0].axis('off')
    
    # Plot Block 1 features (16x16)
    block1_features = features[0][0]  # [C, 16, 16]
    # Select the first 16 feature maps (or fewer if there are less)
    num_features = min(16, block1_features.size(0))
    # Create a grid of feature maps
    block1_grid = torchvision.utils.make_grid(
        block1_features[:num_features].unsqueeze(1).cpu(),
        nrow=4,
        normalize=True,
        padding=2
    )
    # Convert from [C, H, W] to [H, W, C] for plotting
    block1_grid = block1_grid.permute(1, 2, 0).numpy()
    axes[1].imshow(block1_grid)
    axes[1].set_title('Block 1 Output (16x16)')
    axes[1].axis('off')
    
    # Plot Block 2 features (8x8)
    block2_features = features[1][0]  # [C, 8, 8]
    # Select the first 16 feature maps (or fewer if there are less)
    num_features = min(16, block2_features.size(0))
    # Create a grid of feature maps
    block2_grid = torchvision.utils.make_grid(
        block2_features[:num_features].unsqueeze(1).cpu(),
        nrow=4,
        normalize=True,
        padding=2
    )
    # Convert from [C, H, W] to [H, W, C] for plotting
    block2_grid = block2_grid.permute(1, 2, 0).numpy()
    axes[2].imshow(block2_grid)
    axes[2].set_title('Block 2 Output (8x8)')
    axes[2].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show figure
    if save_path:
        plt.savefig(save_path)
    
    return fig


def visualize_model_layers(model, dataloader, num_images=1, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), save_dir=None):
    """
    Visualize layers for a random image from the dataloader.
    
    Args:
        model: TinyVGG model
        dataloader: DataLoader containing images
        device: Device to run the model on
        save_dir: Directory to save visualizations, if None they will be displayed
    """
    # Get a random batch
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    
    # Select the first num_images images
    image = images[:num_images]
    label = labels[:num_images].item()
    
    # Get class names (if available)
    class_names = getattr(dataloader.dataset, 'class_names', None)
    if class_names and label < len(class_names):
        class_name = class_names[label]
    else:
        class_name = f"Class {label}"
    
    print(f"Visualizing features for image of {class_name}")
    
    # Generate visualization
    save_path = f"{save_dir}/{class_name}_features.png" if save_dir else None
    fig = visualize_layer_outputs(model, image, device, save_path)
    
    return fig

def compute_confusion_matrix(model, dataloader, device=None, class_names=None):
    """
    Compute and visualize the confusion matrix for a model on a given dataloader.
    
    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader containing validation/test data
        device: Device to run the model on (defaults to CPU if None)
        
    Returns:
        matplotlib figure containing the confusion matrix visualization
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set model to evaluation mode
    model.eval()
    
    if class_names is None:
        class_names = getattr(dataloader.dataset, 'class_names', None)
    
    # Lists to store true labels and predictions
    all_targets = []
    all_predictions = []
    
    # Disable gradient computation for inference
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get predicted class indices
            _, predictions = torch.max(outputs, 1)
            
            # Store targets and predictions
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    
    # Convert lists to numpy arrays
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    
    # Get the number of classes
    if class_names is None:
        num_classes = len(np.unique(np.concatenate([all_targets, all_predictions])))
        class_names = [str(i) for i in range(num_classes)]
    else:
        num_classes = len(class_names)
    
    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_predictions, labels=range(num_classes))
    
    # Normalize confusion matrix (optional)
    with np.errstate(divide='ignore', invalid='ignore'):  # Handle division by zero
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized, nan=0)  # Replace NaN with 0
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot confusion matrix with seaborn heatmap
    ax = sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
    
    # Add labels and title
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Rotate x-axis labels for better readability if there are many classes
    if num_classes > 10:
        plt.xticks(rotation=45, ha='right')
    
    # Create and return the figure
    fig = plt.gcf()
    plt.tight_layout()
    
    return fig

def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

def log_embeddings(model, dataloader, logger, class_names, device=None, n_samples=100):
    """
    Extract features from images and log them as embeddings to TensorBoard.
    
    Args:
        model: The model from which to extract features
        dataloader: DataLoader containing the dataset
        logger: TensorBoard SummaryWriter instance
        class_names: List of class names for metadata
        device: Device to run the model on
        n_samples: Number of samples to visualize
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get random batch
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    
    # Select random subset
    images, labels = select_n_random(images, labels, n=n_samples)
    
    # Get class labels for each image
    class_label_names = [class_names[lab] for lab in labels]
    
    # Set model to evaluation mode
    model.eval()
    
    # Extract features (from the output of the conv layers, before the classifier)
    with torch.no_grad():
        # Move images to device
        images = images.to(device)
        
        # Run images through model features (convolutional layers)
        features_maps = model.features(images)
        
        # Global average pooling to get a 1D feature vector
        features = torch.mean(features_maps, dim=[2, 3])  # Average over spatial dimensions
        
    # Move features back to CPU for logging
    features = features.cpu()
    
    # Log embeddings to TensorBoard
    logger.add_embedding(
        features,
        metadata=class_label_names,
        label_img=images.cpu(),
        global_step=0
    )
    
    # Ensure embeddings are written to disk
    logger.flush()
