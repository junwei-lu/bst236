import os
import argparse
import torch
import random
import numpy as np
from datetime import datetime
from config import get_config
from model import get_model
from dataset import get_dataloader
from train import train_model
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_dirs(config):
    """Create necessary directories"""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)

def save_config(config, save_path):
    """Save configuration to a JSON file"""
    config_dict = vars(config)
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=4)

def main(args):
    # Set random seed for reproducibility
    torch.manual_seed(get_config().train_config["seed"])
    
    # Create directories
    os.makedirs(get_config().train_config["checkpoint_dir"], exist_ok=True)
    os.makedirs(get_config().train_config["log_dir"], exist_ok=True)
    
    # Initialize datasets and data loaders
    train_dataset = get_dataloader(get_config())[0]
    val_dataset = get_dataloader(get_config())[1]
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=get_config().train_config["batch_size"],
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=get_config().train_config["batch_size"],
        shuffle=False
    )
    
    # Initialize model
    model = get_model(get_config())
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if get_config().train_config["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=get_config().train_config["learning_rate"])
    elif get_config().train_config["optimizer"] == "SGD":
        optimizer = optim.SGD(
            model.parameters(), 
            lr=get_config().train_config["learning_rate"],
            momentum=0.9
        )
    
    # Initialize trainer
    trainer = train_model(model, train_loader, val_loader, optimizer, criterion, get_config())
    
    # Start training
    trainer.train()
    
    if args.hparam_tuning:
        from hparam_tuning import run_hparam_experiment
        
        # Define hyperparameter ranges to search
        hparam_ranges = {
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [32, 64, 128],
            'optimizer': ['Adam', 'SGD'],
            'epochs': [5, 10]
        }
        
        # Run hyperparameter tuning
        results = run_hparam_experiment(get_model, get_dataloader(get_config())[0].__class__, hparam_ranges)
        print("Hyperparameter tuning completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a neural network with TensorBoard logging')
    parser.add_argument('--hparam_tuning', action='store_true', help='Run hyperparameter tuning')
    args = parser.parse_args()
    
    main(args) 