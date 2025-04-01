import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
from tqdm import tqdm
import numpy as np

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch
    
    Args:
        model: The neural network model
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cuda/cpu)
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / total, 
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """
    Validate the model
    
    Args:
        model: The neural network model
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on (cuda/cpu)
        
    Returns:
        tuple: (validation loss, validation accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': running_loss / total, 
                'acc': 100. * correct / total
            })
    
    val_loss = running_loss / total
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

def train_model(model, dataloaders, config):
    """
    Full training loop
    
    Args:
        model: The neural network model
        dataloaders: Dictionary containing train and validation dataloaders
        config: Configuration object with training parameters
        
    Returns:
        model: Trained model
    """
    device = torch.device(config.device)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=config.lr_patience, 
        verbose=True
    )
    
    # Tensorboard writer
    writer = SummaryWriter(log_dir=config.log_dir)
    
    # Training loop
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, 
            dataloaders['train'], 
            criterion, 
            optimizer, 
            device
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, 
            dataloaders['val'], 
            criterion, 
            device
        )
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(config.checkpoint_dir, f"best_model.pt")
            model.save_checkpoint(save_path)
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            
        # Periodic checkpoint saving
        if (epoch + 1) % config.save_every == 0:
            save_path = os.path.join(config.checkpoint_dir, f"model_epoch_{epoch+1}.pt")
            model.save_checkpoint(save_path)
            
        # Early stopping
        if early_stopping_counter >= config.early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
            
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
    # Load best model
    best_model_path = os.path.join(config.checkpoint_dir, "best_model.pt")
    model.load_checkpoint(best_model_path)
    
    # Close tensorboard writer
    writer.close()
    
    return model 

class Trainer:
    def __init__(
        self, 
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        criterion,
        hparams,
        log_dir="results/logs"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.hparams = hparams
        
        # Initialize TensorBoard writer
        self.writer = get_tensorboard_writer(log_dir)
        
        # Log hyperparameters
        self.log_hyperparameters()
        
        # Log the model graph
        self.log_model_graph()
        
    def log_hyperparameters(self):
        """Log hyperparameters to TensorBoard"""
        # Convert hparams to dict if it's not already
        hparams_dict = vars(self.hparams) if not isinstance(self.hparams, dict) else self.hparams
        
        # Add model and optimizer info
        hparams_dict['model_type'] = self.model.__class__.__name__
        hparams_dict['optimizer'] = self.optimizer.__class__.__name__
        
        # Log hyperparameters
        self.writer.add_hparams(
            hparams_dict,
            {'dummy_metric': 0}  # Required by TensorBoard but we'll update with real metrics later
        )
        
    def log_model_graph(self):
        """Log model graph to TensorBoard"""
        # Get a sample input from the train loader to visualize the graph
        for data, _ in self.train_loader:
            self.writer.add_graph(self.model, data[:1])
            break
            
    def train(self):
        best_val_loss = float('inf')
        
        for epoch in range(self.hparams['epochs']):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                train_loss += loss.item()
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Log gradients histograms (periodically to save space)
                if batch_idx % 100 == 0:
                    self.log_gradients()
                
                self.optimizer.step()
                
                # Log batch loss
                self.writer.add_scalar('Loss/train_batch', 
                                      loss.item(), 
                                      epoch * len(self.train_loader) + batch_idx)
            
            # Calculate average training loss for the epoch
            avg_train_loss = train_loss / len(self.train_loader)
            
            # Validation phase
            val_loss = self.evaluate()
            
            # Log epoch metrics
            self.writer.add_scalar('Loss/train', avg_train_loss, epoch)
            self.writer.add_scalar('Loss/validation', val_loss, epoch)
            
            # Save checkpoint if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
            
            print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Close the TensorBoard writer when training is done
        self.writer.close()
    
    def evaluate(self):
        """Evaluate the model on validation data"""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                output = self.model(data)
                loss = self.criterion(output, target)
                val_loss += loss.item()
        
        return val_loss / len(self.val_loader)
    
    def log_gradients(self):
        """Log histograms of model parameter gradients"""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'gradients/{name}', 
                                         param.grad, 
                                         global_step=self.global_step)
    
    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint"""
        checkpoint_dir = self.hparams.get('checkpoint_dir', 'models/checkpoints')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        checkpoint_path = os.path.join(
            checkpoint_dir, 
            f"model_epoch_{epoch}_valloss_{val_loss:.4f}.pt"
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }, checkpoint_path)
        
        print(f"Checkpoint saved to {checkpoint_path}") 