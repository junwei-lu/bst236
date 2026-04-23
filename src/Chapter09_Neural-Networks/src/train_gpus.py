import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
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

def setup(rank, world_size):
    """
    Setup distributed training environment for the current process
    """
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # Set GPU to use for this process
    torch.cuda.set_device(rank)
    # Enable cuDNN benchmark for fixed input size (faster convolutions)
    torch.backends.cudnn.benchmark = True

def cleanup():
    """
    Clean up the distributed environment
    """
    dist.destroy_process_group()

class DistributedTrainer:
    def __init__(self, rank, world_size, train_config, model_config, save_freq=None, subset_size=None):
        # Store configs and distributed info
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == 0)
        self.train_config = train_config
        self.model_config = model_config
        self.subset_size = subset_size
        
        # Initialize timestamp and run name
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{train_config['run_name']}_GPUs_{self.timestamp}"
        
        # Set device and extract hyperparameters
        self.device = torch.device(f"cuda:{rank}")
        self.epochs = train_config["epochs"]
        self.batch_size = train_config["batch_size"] // world_size  # Adjust batch size per GPU
        self.learning_rate = train_config["learning_rate"] * world_size  # Scale learning rate
        
        # Checkpoint saving frequency (None means no checkpoints)
        self.save_freq = save_freq
        
        # Initialize logger and wandb (only on the main process)
        if self.is_main_process:
            self._setup_logging()
        
        # Setup data, model, and optimizer
        self._setup_data()
        self._setup_model()
        
    def _setup_logging(self):
        """Initialize WandB and TensorBoard loggers (only on the main process)."""
        # Initialize WandB
        self.wandb_run = wandb.init(
            project=self.train_config["project_name"],
            name=self.run_name,
            config={**self.train_config, **self.model_config, "world_size": self.world_size}
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
        
        # Create train/val datasets from the data object
        full_dataset = data.dataset
        
        # Calculate validation split
        dataset_size = len(full_dataset)
        val_size = int(self.train_config["validation_split"] * dataset_size)
        train_size = dataset_size - val_size
        
        # Split the dataset
        from torch.utils.data import random_split
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        # Create distributed sampler for training data
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            seed=42
        )
        
        # Determine optimal number of workers
        num_workers = max(1, (multiprocessing.cpu_count() // self.world_size))
        
        # Create dataloaders with distributed sampler, pinned memory, and workers
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        # For validation, we don't need a distributed sampler
        # Each process will evaluate on the complete val set (for simplicity)
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        
    def _setup_model(self):
        """Initialize the model, loss function, and optimizer with DDP."""
        self.model = TinyVGG(
            input_channels=self.model_config["input_channels"],
            num_classes=self.model_config["num_classes"]
        ).to(self.device)
        
        # Wrap model with DistributedDataParallel
        self.model = DDP(self.model, device_ids=[self.rank])
        
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Initialize gradient scaler for automatic mixed precision
        self.scaler = GradScaler('cuda')
        
        # Try to add model graph to tensorboard (only on main process)
        if self.is_main_process:
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
            # Set epoch for sampler (important for proper shuffling)
            self.train_loader.sampler.set_epoch(epoch)
            
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            train_pbar = None
            if self.is_main_process:  # Only show progress bar on main process
                train_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
                iterator = train_pbar
            else:
                iterator = self.train_loader
                
            for batch_idx, (inputs, targets) in enumerate(iterator):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                
                # Forward pass with automatic mixed precision
                with autocast():
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
                
                # Update progress bar on main process
                if self.is_main_process and train_pbar is not None:
                    train_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100.0 * correct / total:.2f}%'
                    })

            # Synchronize for metrics calculation
            dist.barrier()
            
            # Gather metrics from all processes
            all_running_loss = [0.0] * self.world_size
            all_correct = [0] * self.world_size
            all_total = [0] * self.world_size
            
            dist.all_gather_object(all_running_loss, running_loss)
            dist.all_gather_object(all_correct, correct)
            dist.all_gather_object(all_total, total)
            
            # Calculate global metrics (only on main process)
            if self.is_main_process:
                global_running_loss = sum(all_running_loss)
                global_correct = sum(all_correct)
                global_total = sum(all_total)
                
                avg_loss = global_running_loss / (len(self.train_loader) * self.world_size)
                train_accuracy = 100.0 * global_correct / global_total
                
                print(f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
                
                # Log metrics
                self.wandb_run.log({
                    "train_loss": avg_loss,
                    "train_accuracy": train_accuracy,
                    "epoch": epoch + 1,
                }) 
                self.logger.add_scalar(tag='Loss/train', scalar_value=avg_loss, global_step=epoch)
                self.logger.add_scalar(tag='Accuracy/train', scalar_value=train_accuracy, global_step=epoch)

            # Evaluate after each epoch (all processes participate)
            self.evaluate(epoch)
            
            # Save checkpoint if save_freq is specified and it's time to save
            if self.is_main_process and self.save_freq is not None and (epoch + 1) % self.save_freq == 0:
                self._save_checkpoint(epoch)
                
    def _save_checkpoint(self, epoch):
        """Save a checkpoint of the model (only on main process)."""
        os.makedirs(self.train_config["checkpoint_dir"], exist_ok=True)
        checkpoint_path = f"{self.train_config['checkpoint_dir']}/{self.run_name}_epoch{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.module.state_dict(),  # Save without DDP wrapper
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
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Forward pass with automatic mixed precision
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    
                running_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        # Synchronize for metrics calculation
        dist.barrier()
        
        # Gather metrics from all processes
        all_running_loss = [0.0] * self.world_size
        all_correct = [0] * self.world_size
        all_total = [0] * self.world_size
        
        dist.all_gather_object(all_running_loss, running_loss)
        dist.all_gather_object(all_correct, correct)
        dist.all_gather_object(all_total, total)
        
        # Calculate global metrics (only on main process)
        if self.is_main_process:
            global_running_loss = sum(all_running_loss)
            global_correct = sum(all_correct)
            global_total = sum(all_total)
            
            avg_loss = global_running_loss / (len(self.val_loader) * self.world_size)
            val_accuracy = 100.0 * global_correct / global_total
            
            print(f"Epoch [{epoch+1}/{self.epochs}], Val Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
            
            if epoch == self.epochs - 1:
                self.final_val_loss = avg_loss
                self.final_val_accuracy = val_accuracy
                # Log hyperparameters to TensorBoard
                hparams = {**self.train_config, **self.model_config, "world_size": self.world_size}
                self.logger.add_hparams(hparams, metric_dict={
                    'final_val_loss': self.final_val_loss,
                    'final_val_accuracy': self.final_val_accuracy
                })
            
            # Log metrics
            self.wandb_run.log({
                "val_loss": avg_loss,
                "val_accuracy": val_accuracy,
                "epoch": epoch + 1,
            }) 
            self.logger.add_scalar(tag='Loss/val', scalar_value=avg_loss, global_step=epoch)
            self.logger.add_scalar(tag='Accuracy/val', scalar_value=val_accuracy, global_step=epoch)
            
            # Only generate visualizations on the last epoch or periodically to save compute
            if epoch == self.epochs - 1 or epoch % 5 == 0:
                # Visualize feature maps and confusion matrix (only on main process)
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
        if self.is_main_process:
            self.wandb_run.finish()
            self.logger.close()

def train_worker(rank, world_size, args):
    """
    Training function that runs on each GPU
    """
    # Setup the process group
    setup(rank, world_size)
    
    print(f"Running on GPU {rank}/{world_size-1}")
    
    # Create trainer and run training
    trainer = DistributedTrainer(rank, world_size, train_config, config_TinyVGG, save_freq=args.save_freq, subset_size=args.subset_size)
    
    # Run training
    trainer.train()
    
    # Clean up resources
    trainer.cleanup()
    
    # Clean up the process group
    cleanup()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Multi-GPU PyTorch Training')
    parser.add_argument('--save-freq', type=int, default=None, 
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--num-gpus', type=int, default=None,
                        help='Number of GPUs to use (default: all available)')
    parser.add_argument("--subset-size", type=int, default=None,
                        help='Number of samples to train on (default: None)')
    args = parser.parse_args()
    
    # Get the number of available GPUs
    world_size = torch.cuda.device_count() if args.num_gpus is None else min(args.num_gpus, torch.cuda.device_count())
    
    if world_size < 1:
        print("No GPUs available. Running on CPU is not supported by this script.")
        return
    
    print(f"Using {world_size} GPUs for training")
    
    # Use multiprocessing to spawn one process per GPU
    mp.spawn(
        train_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main() 

# torchrun --nproc_per_node=NUM_GPUS src/train_gpus.py