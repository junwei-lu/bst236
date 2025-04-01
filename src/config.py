import argparse
import os
import json
from types import SimpleNamespace

def get_default_config():
    """
    Get default configuration
    
    Returns:
        SimpleNamespace: Configuration object with default values
    """
    config = SimpleNamespace(
        # Model
        model_name="convnet",
        in_channels=3,
        num_classes=10,
        
        # Data
        data_path="data/processed",
        batch_size=32,
        num_workers=4,
        
        # Training
        learning_rate=0.001,
        num_epochs=100,
        device="cuda",
        lr_patience=5,
        early_stopping_patience=10,
        save_every=5,
        
        # Paths
        checkpoint_dir="models/checkpoints",
        log_dir="results/logs",
        results_dir="results",
    )
    
    return config

def load_config_from_json(json_path):
    """
    Load configuration from a JSON file
    
    Args:
        json_path (str): Path to the JSON config file
        
    Returns:
        SimpleNamespace: Configuration object
    """
    with open(json_path, 'r') as f:
        config_dict = json.load(f)
        
    return SimpleNamespace(**config_dict)

def get_config(args):
    """
    Get configuration from command line arguments and config file
    
    Args:
        args: Command line arguments
        
    Returns:
        SimpleNamespace: Configuration object
    """
    # Start with default config
    config = get_default_config()
    
    # Update from config file if provided
    if hasattr(args, 'config') and args.config:
        file_config = load_config_from_json(args.config)
        for key, value in vars(file_config).items():
            setattr(config, key, value)
    
    # Update from command line arguments
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            setattr(config, key, value)
    
    return config 