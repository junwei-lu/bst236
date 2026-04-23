import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from dataset import CustomCIFAR
from model import TinyVGG
from utils import visualize_model_layers, compute_confusion_matrix, log_embeddings
from config import train_config, config_TinyVGG

import datetime
import argparse
from tqdm import tqdm
import multiprocessing
from torch.utils.tensorboard import SummaryWriter
import wandb

class ParallelTrainer:
    def __init__(self, train_config, model_config, save_freq=None, subset_size=None):
        # Store configs
        self.train_config = train_config
        self.model_config = model_config
        self.subset_size = subset_size
        
        # Initialize timestamp and run name
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{train_config['run_name']}_DataParallel_{self.timestamp}"
        
        # Set device and extract hyperparameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gpus = torch.cuda.device_count()
        print(f"Using {self.num_gpus} GPUs")
        
        self.epochs = train_config["epochs"]
        self.batch_size = train_config["batch_size"] * self.num_gpus  # Multiply batch size by number of GPUs
        self.learning_rate = train_config["learning_rate"]
        
        # Checkpoint saving frequency (None means no checkpoints)
        self.save_freq = save_freq
        
        # Initialize logger and wandb
        self._setup_logging()
        
        # Setup data, model, and optimizer
        self._setup_data()
        self._setup_model()
        
    def _setup_logging(self):
        """Initialize WandB and TensorBoard loggers."""
        # Initialize WandB
        self.wandb_run = wandb.init(
            project=self.train_config["project_name"],
            name=self.run_name,
            config={**self.train_config, **self.model_config, "num_gpus": self.num_gpus}
        )
        
        # Initialize TensorBoard logger
        log_dir = os.path.join('logs/', self.run_name)
        self.logger = SummaryWriter(log_dir=log_dir)
        
    def _setup_data(self):
        """Load and prepare the dataset."""
        if self.subset_size is not None: # Train for smaller subset of data
            data = CustomCIFAR(subset_size=self.subset_size)
        else: # Train for all data
            data = CustomCIFAR()
        self.class_names = data.class_names
        
        # Determine optimal number of workers
        num_workers = min(8, max(4, multiprocessing.cpu_count() // 2))
        
        # Get data loaders with appropriate batch size
        self.train_loader, self.val_loader = data.get_train_val_loaders(
            batch_size=self.batch_size, 
            validation_split=self.train_config["validation_split"]
        )
        
    def _setup_model(self):
        """Initialize the model, loss function, and optimizer with DataParallel."""
        # Create the base model
        model = TinyVGG(
            input_channels=self.model_config["input_channels"],
            num_classes=self.model_config["num_classes"]
        )
        
        # Wrap model with DataParallel
        self.model = DataParallel(model).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Initialize gradient scaler for automatic mixed precision
        self.scaler = GradScaler()
        
        # Try to add model graph to tensorboard
        try:
            dataiter = iter(self.train_loader)
            images, _ = next(dataiter)
            images = images.to(self.device)
            self.logger.add_graph(self.model.module, images)
            print("Successfully added model graph to TensorBoard")
        except Exception as e:
            print(f"Failed to add model graph to TensorBoard: {e}")
            print("Continuing training without model graph visualization")
    
    def train(self):
        """Run the training loop with evaluation."""
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            train_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
            for batch_idx, (inputs, targets) in enumerate(train_pbar):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                
                # Forward pass with automatic mixed precision
                with autocast(device_type='cuda'):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Backward and optimize with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                running_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.0 * correct / total:.2f}%'
                })

            avg_loss = running_loss / len(self.train_loader)
            train_accuracy = 100.0 * correct / total
            print(f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")  
            
            # Log metrics
            self.wandb_run.log({
                "train_loss": avg_loss,
                "train_accuracy": train_accuracy,
                "epoch": epoch + 1
            }) 
            self.logger.add_scalar(tag='Loss/train', scalar_value=avg_loss, global_step=epoch)
            self.logger.add_scalar(tag='Accuracy/train', scalar_value=train_accuracy, global_step=epoch)

            # Evaluate after each epoch
            self.evaluate(epoch)
            
            # Save checkpoint if save_freq is specified and it's time to save
            if self.save_freq is not None and (epoch + 1) % self.save_freq == 0:
                self._save_checkpoint(epoch)
                
    def _save_checkpoint(self, epoch):
        """Save a checkpoint of the model."""
        os.makedirs(self.train_config["checkpoint_dir"], exist_ok=True)
        checkpoint_path = f"{self.train_config['checkpoint_dir']}/{self.run_name}_epoch{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.module.state_dict(),  # Save without DataParallel wrapper
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        # Log checkpoint to wandb
        self.wandb_run.log({f"checkpoint_epoch_{epoch+1}": checkpoint_path})
      
    def evaluate(self, epoch):
        """Evaluate the model and log metrics."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Val]")
            for batch_idx, (inputs, targets) in enumerate(val_pbar):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Forward pass with automatic mixed precision
                with autocast(device_type='cuda'):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    
                running_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.0 * correct / total:.2f}%'
                })

        avg_loss = running_loss / len(self.val_loader)
        val_accuracy = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{self.epochs}], Val Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        
        if epoch == self.epochs - 1:
            self.final_val_loss = avg_loss
            self.final_val_accuracy = val_accuracy
            # Log hyperparameters to TensorBoard
            hparams = {**self.train_config, **self.model_config, "num_gpus": self.num_gpus}
            self.logger.add_hparams(hparams, metric_dict={
                'final_val_loss': self.final_val_loss,
                'final_val_accuracy': self.final_val_accuracy
            })
        
        # Log metrics
        self.wandb_run.log({
            "val_loss": avg_loss,
            "val_accuracy": val_accuracy,
            "epoch": epoch + 1
        }) 
        self.logger.add_scalar(tag='Loss/val', scalar_value=avg_loss, global_step=epoch)
        self.logger.add_scalar(tag='Accuracy/val', scalar_value=val_accuracy, global_step=epoch)
        
        # Visualize feature maps and confusion matrix only on the last epoch or periodically
        if epoch == self.epochs - 1 or epoch % 5 == 0:
            feature_fig = visualize_model_layers(self.model.module, self.val_loader, device=self.device)
            conf_matrix_fig = compute_confusion_matrix(
                self.model.module, 
                self.val_loader, 
                self.device, 
                class_names=self.class_names
            )
            
            # Log figures to TensorBoard
            self.logger.add_figure(tag='Feature Maps/Epoch', figure=feature_fig, global_step=epoch)
            self.logger.add_figure(tag='Confusion Matrix/Epoch', figure=conf_matrix_fig, global_step=epoch)
    
    def cleanup(self):
        """Close loggers and finish the run."""
        self.wandb_run.finish()
        self.logger.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='DataParallel PyTorch Training')
    parser.add_argument('--save-freq', type=int, default=None, 
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument("--subset-size", type=int, default=None,
                        help='Number of samples to train on (default: None)')
    args = parser.parse_args()
    
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    
    if num_gpus < 1:
        print("No GPUs available. Running on CPU.")
    else:
        print(f"Found {num_gpus} GPUs for training")
    
    # Create trainer and run training
    trainer = ParallelTrainer(train_config, config_TinyVGG, save_freq=args.save_freq, subset_size=args.subset_size)
    
    # Run training
    trainer.train()
    
    # Clean up resources
    trainer.cleanup()

if __name__ == "__main__":
    main() 