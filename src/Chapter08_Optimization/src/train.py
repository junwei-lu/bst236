# train.py
import math
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import datetime
import subprocess
import os
import sys
from utils import *
from tqdm import tqdm


import math
import torch
from torch.optim.optimizer import Optimizer

class AGD(Optimizer):
    def __init__(self, params, lr=0.01):
        """
        Implements Nesterov's Accelerated Gradient Descent (AGD).

        Update formulas:
            y_t = x_t + β_t * m_t,       where β_t = t / (t+3)
            m_{t+1} = β_t * m_t + lr * ∇f(y_t)
            x_{t+1} = x_t - m_{t+1}

        This optimizer is designed to be used in the standard PyTorch style:
            optimizer = AGD(model.parameters(), lr=0.01)
            for data in loader:
                optimizer.zero_grad()
                loss = f(model(input))  # loss computed at the lookahead y_t
                loss.backward()
                optimizer.step()

        Args:
            params (iterable): Iterable of parameters to optimize.
            lr (float): Learning rate.
        """
        defaults = dict(lr=lr)
        super(AGD, self).__init__(params, defaults)

    def zero_grad(self):
        """
        Overrides the default zero_grad.

        For each parameter, updates the parameter to the lookahead iterate:
            y_t = x_t + β_t * m_t,  with β_t = t / (t+3)
        Then zeros out its gradient.

        This ensures that the loss (and its gradient) are computed at y_t.
        """
        for group in self.param_groups:
            for p in group['params']:
                # Initialize per-parameter state if not present.
                state = self.state.setdefault(p, {'x': p.data.clone(),          # true iterate x_t
                                                    'm': torch.zeros_like(p.data), # momentum m_t
                                                    'step': 0})                  # iteration counter t
                # Compute β_t = t / (t+3); when t is zero, β_t = 0 so y_t = x_t.
                t = state['step']
                beta = t / (t + 3) if (t + 3) != 0 else 0.0
                # Set parameter to the lookahead iterate: y_t = x_t + β_t * m_t.
                p.data.copy_(state['x'] + beta * state['m'])
                # Zero any existing gradient.
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    def step(self):
        """
        Performs one AGD update step.

        Using the gradient computed at the lookahead iterate y_t (set in zero_grad()),
        update the momentum and the true iterate as follows:

            m_{t+1} = β_t * m_t + lr * ∇f(y_t)
            x_{t+1} = x_t - m_{t+1}

        Finally, update the parameter value to x_{t+1} and reset the gradient.
        """
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                # Skip if no gradient was computed.
                if p.grad is None:
                    continue
                # Retrieve the per-parameter state.
                state = self.state[p]
                # Increment iteration counter (t).
                state['step'] += 1
                t = state['step']
                # Compute β_t = t / (t+3)
                beta = t / (t + 3)
                # p.data currently holds y_t (the lookahead iterate).
                grad = p.grad.data
                # Update momentum: m_{t+1} = β_t * m_t + lr * ∇f(y_t)
                m_new = beta * state['m'] + lr * grad
                # Update the true iterate: x_{t+1} = x_t - m_{t+1}
                x_new = state['x'] - m_new
                # Save the new momentum and true iterate into the state.
                state['m'].copy_(m_new)
                state['x'].copy_(x_new)
                # Update the parameter to the new iterate.
                p.data.copy_(x_new)
                # Reset the gradient.
                p.grad.zero_()




class Trainer:
    """
    A class to handle the training process of a PyTorch model.
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        device,
        num_epochs,
        use_logger=False,
        save_frequency=10,
        hyperparams=None,
        git_commit=False
    ):
        """
        Initialize the Trainer with model and training parameters.
        
        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            loss_fn: Loss function
            optimizer: Optimizer for parameter updates
            device: Device to use for computation
            num_epochs: Number of training epochs
            use_logger: Whether to use logging functionality
            save_frequency: How often to save model checkpoints (epochs)
            hyperparams: Dictionary of hyperparameters to log
            git_commit: Whether to automatically create a git commit after training
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.use_logger = use_logger
        self.save_frequency = save_frequency
        self.hyperparams = hyperparams
        self.git_commit = git_commit
        
        # Extract save directories from hyperparameters if provided
        self.checkpoints_dir = hyperparams.get('CHECKPOINTS_DIR', 'checkpoints') if hyperparams else 'checkpoints'
        self.results_dir = hyperparams.get('RESULTS_DIR', 'results') if hyperparams else 'results'
        self.logs_dir = hyperparams.get('LOGS_DIR', 'logs') if hyperparams else 'logs'
        
        # Logger setup
        self.logger = None
        if self.use_logger:
            self.logger = Logger(log_dir=self.logs_dir)
            self.logger.start_capture()
            # Log timestamp at the beginning
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"{'='*80}{timestamp}")
            # Log hyperparameters if provided
            if self.hyperparams:
                self.logger.log_hyperparameters(self.hyperparams)
        
        # Variables for tracking training progress
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Store start time for commit message
        self.start_time = datetime.datetime.now()
    
    def _train_epoch(self, epoch, pbar=None):
        """Train the model for one epoch."""
        self.model.train()  # Set the model to training mode
        running_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            # Move data to the specified device
            data = data.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            predictions = self.model(data)
            loss = self.loss_fn(predictions, targets)
            running_loss += loss.item()

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update progress bar with current loss if provided
            if pbar:
                pbar.update(1)
                pbar.set_postfix(train_loss=f"{loss.item():.3f}")
        
        avg_loss = running_loss / len(self.train_loader)
        return avg_loss
    
    def _evaluate_model(self, pbar=None):
        """Evaluate the model on validation data."""
        self.model.eval()  # Set the model to evaluation mode
        running_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(self.val_loader):
                # Move data to the specified device
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                predictions = self.model(data)
                loss = self.loss_fn(predictions, targets)
                running_loss += loss.item()
        
        avg_loss = running_loss / len(self.val_loader)
        return avg_loss
    
    def _save_checkpoint(self, epoch):
        """Save model checkpoint."""
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        save_checkpoint(checkpoint, filename=f"checkpoint_{epoch}.pth.tar", directory=self.checkpoints_dir)
        print(f"Saved checkpoint {epoch} with validation loss: {self.val_losses[-1]:.4f}")
    
    def _make_git_commit(self, final_train_loss, final_val_loss):
        """Create a git commit with training results."""
        try:
            # Check if this is a git repository
            subprocess.check_output(["git", "rev-parse", "--is-inside-work-tree"], stderr=subprocess.DEVNULL)
            
            # Format the training duration
            end_time = datetime.datetime.now()
            duration = end_time - self.start_time
            hours, remainder = divmod(duration.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            # Create commit message
            commit_time = end_time.strftime('%Y-%m-%d %H:%M:%S')
            commit_message = (
                f"Training completed at {commit_time}\n\n"
                f"Duration: {duration_str}\n"
                f"Final Training Loss: {final_train_loss:.6f}\n"
                f"Final Validation Loss: {final_val_loss:.6f}\n"
                f"Epochs: {self.num_epochs}\n"
                f"Learning Rate: {self.hyperparams.get('LEARNING_RATE', 'N/A') if self.hyperparams else 'N/A'}"
            )
            
            # Check if there are modified or new files in the output directories
            subprocess.run(["git", "add", self.checkpoints_dir, self.results_dir, self.logs_dir])
            
            # Create the commit
            subprocess.run(["git", "commit", "-m", commit_message])
            
            print("\nCreated git commit with training results.")
            print(f"Commit message:\n{commit_message}")
            
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"\nFailed to create git commit: {str(e)}")
            print("This might not be a git repository or git might not be installed.")
    
    def train(self):
        """Run the full training process."""
        print(f"Training on device: {self.device}")
        
        # Training loop over epochs
        for epoch in range(self.num_epochs):
            # Print epoch header
            print(f"\nEpoch {epoch+1} / {self.num_epochs}")
            
            # Create a progress bar for the entire training process (all batches)
            total_steps = len(self.train_loader)
            with tqdm(total=total_steps, file=sys.stdout, 
                     desc=f"Epoch {epoch+1}/{self.num_epochs}",
                     bar_format='{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]') as pbar:
                
                # Train for one epoch
                train_loss = self._train_epoch(epoch, pbar)
                self.train_losses.append(train_loss)
                
                # Evaluate on validation set
                val_loss = self._evaluate_model()
                self.val_losses.append(val_loss)
                
                # Update progress bar with both losses
                pbar.set_postfix(train_loss=f"{train_loss:.3f}", val_loss=f"{val_loss:.3f}")
            
            # Check if this is the best validation loss and print notification
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                print(f"<<<<<< reach best val loss : {val_loss} >>>>>>")
                
                # Save best model
                checkpoint = {
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict()
                }
                save_checkpoint(checkpoint, filename="best_model.pth.tar", directory=self.checkpoints_dir)
            
            # Save model if needed
            if epoch % self.save_frequency == 0:
                self._save_checkpoint(epoch)
        
        # Log final training metrics
        if self.use_logger and self.logger:
            self.logger.log_final_metrics(self.train_losses[-1], self.val_losses[-1])
        
        # Visualize final model predictions
        visualize_predictions(self.model, device=self.device, save_dir=self.results_dir)
        
        # Plot training and validation loss curves
        plot_loss_curves(self.train_losses, self.val_losses, save_dir=self.results_dir)
        
        print("Training completed!")
        
        # Print log file path if logging was used
        if self.use_logger and self.logger:
            log_path = self.logger.get_log_file_path()
            print(f"Log file saved to: {log_path}")
            # Stop logging
            self.logger.stop_capture()
        
        # Make git commit if enabled
        if self.git_commit:
            self._make_git_commit(self.train_losses[-1], self.val_losses[-1])
    
    def get_log_file_path(self):
        """Get the path to the log file if logging is enabled."""
        if self.use_logger and self.logger:
            return self.logger.get_log_file_path()
        return None
