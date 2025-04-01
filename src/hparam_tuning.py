from torch.utils.tensorboard import SummaryWriter
import os
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product

def run_hparam_experiment(model_class, dataset_class, hparam_ranges):
    """
    Run hyperparameter tuning experiments.
    
    Args:
        model_class: The model class to instantiate
        dataset_class: The dataset class to instantiate
        hparam_ranges: Dictionary of hyperparameter names to lists of values
    """
    # Create base log directory
    log_dir = "results/logs/hparam_tuning"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Generate all combinations of hyperparameters
    keys = list(hparam_ranges.keys())
    values = list(hparam_ranges.values())
    combinations = list(product(*values))
    
    # Dictionary to collect all results for the hparam dashboard
    hparam_results = {}
    
    # Run an experiment for each combination
    for i, combination in enumerate(combinations):
        # Create hyperparameter dictionary for this experiment
        hparams = {keys[j]: combination[j] for j in range(len(keys))}
        print(f"Running experiment {i+1}/{len(combinations)}: {hparams}")
        
        # Create experiment directory
        run_name = f"run_{i}_" + "_".join([f"{k}={v}" for k, v in hparams.items()])
        exp_dir = os.path.join(log_dir, run_name)
        writer = SummaryWriter(exp_dir)
        
        # Initialize dataset, model, criterion, and optimizer
        train_dataset = dataset_class(train=True, **hparams.get('dataset_params', {}))
        val_dataset = dataset_class(train=False, **hparams.get('dataset_params', {}))
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=hparams.get('batch_size', 32),
            shuffle=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=hparams.get('batch_size', 32),
            shuffle=False
        )
        
        model = model_class(**hparams.get('model_params', {}))
        
        criterion = nn.CrossEntropyLoss()  # Adjust based on your task
        
        # Get optimizer based on hparams
        optimizer_name = hparams.get('optimizer', 'Adam')
        lr = hparams.get('learning_rate', 0.001)
        
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=hparams.get('momentum', 0.9))
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr)  # Default
        
        # Train the model
        best_val_loss = float('inf')
        epochs = hparams.get('epochs', 10)
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for data, target in train_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    output = model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            accuracy = correct / total
            
            # Log metrics
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('Loss/validation', avg_val_loss, epoch)
            writer.add_scalar('Accuracy/validation', accuracy, epoch)
            
            # Update best validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
        
        # Store results for hparam dashboard
        hparam_dict = {
            'hparam/' + k: v for k, v in hparams.items()
        }
        
        metric_dict = {
            'metrics/best_val_loss': best_val_loss,
            'metrics/final_accuracy': accuracy
        }
        
        # Log hyperparameters and final metrics
        writer.add_hparams(hparam_dict, metric_dict)
        writer.close()
        
        # Save for the overall hparam dashboard
        hparam_results[run_name] = {
            'hparams': hparams,
            'metrics': {'val_loss': best_val_loss, 'accuracy': accuracy}
        }
    
    return hparam_results 