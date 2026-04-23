import datetime
import os
import copy
import itertools
import wandb
from train import Trainer
from config import train_config, config_TinyVGG
from torch.utils.tensorboard import SummaryWriter

"""
Below is the code for the TensorBoard grid search.
"""

def define_hparam_space():
    """
    Define the hyperparameter search space.
    This function can be easily extended to include more hyperparameters.
    
    Returns:
        dict: Mapping of hyperparameter names to their search space values
    """
    return {
        # Training hyperparameters
        'learning_rate': [1e-4, 1e-3, 5e-3],
        
        # Model hyperparameters
        # 'conv2_channels': [64, 128]
        
        # Add more hyperparameters as needed
        # 'batch_size': [32, 64, 128],
        # 'conv1_channels': [32, 64, 128],
    }


def run_with_hyperparams(train_cfg, model_cfg, hparams=None):
    """
    Run training with specified hyperparameters and return the final validation loss.
    
    Args:
        train_cfg (dict): Training configuration dictionary
        model_cfg (dict): Model configuration dictionary
        hparams (dict, optional): Specific hyperparameters to override defaults
        
    Returns:
        float: Final validation loss
    """

    # Create deep copies to avoid modifying the original configs
    train_cfg = copy.deepcopy(train_cfg)
    model_cfg = copy.deepcopy(model_cfg)
    
    # Override configs with provided hyperparameters
    if hparams:
        for param_name, param_value in hparams.items():
            # Determine which config dict contains this parameter
            if param_name in train_cfg:
                train_cfg[param_name] = param_value
            elif param_name in model_cfg:
                model_cfg[param_name] = param_value
    
    # Modify run name to include key hyperparameters
    original_run_name = train_cfg.get('run_name', 'run')
    train_cfg['run_name'] = f"{original_run_name}_hparam_tuning"
    
    # Create a trainer instance with our configs
    trainer = Trainer(train_cfg, model_cfg, subset_size=100)
    # Run training
    trainer.train()       
    # Get the final validation loss directly from the trainer
    final_val_loss = getattr(trainer, 'final_val_loss', float('inf'))       
    # Clean up
    trainer.cleanup()
        
    return final_val_loss
    


def grid_search_hyperparams():
    """
    Perform grid search over the hyperparameter space and log results to TensorBoard.
    """
    # Get hyperparameter search space
    hparam_space = define_hparam_space()
    
    # Create all combinations of hyperparameters
    keys = list(hparam_space.keys())
    values = list(hparam_space.values())
    hparam_combinations = list(itertools.product(*values))
    
    # Run training for each hyperparameter combination
    best_loss = float('inf')
    best_hparams = None
    
    print(f"Starting hyperparameter search with {len(hparam_combinations)} combinations")
    
    for i, combination in enumerate(hparam_combinations):
        # Create hyperparameter dictionary for this run
        hparams = {keys[j]: combination[j] for j in range(len(keys))}
        
        print(f"Run {i+1}/{len(hparam_combinations)}: Testing {hparams}")
        
        # Run training with these hyperparameters
        val_loss = run_with_hyperparams(train_config, config_TinyVGG, hparams)

        
        # Track best hyperparameters
        if val_loss < best_loss:
            best_loss = val_loss
            best_hparams = hparams
        
    return best_hparams, best_loss


'''
Below is the code for the WandB sweep.
'''

def define_wandb_sweep_config():
    """
    Define a WandB sweep configuration with hyperparameters to tune.
    
    Returns:
        dict: WandB sweep configuration
    """
    sweep_config = {
        'method': 'random',  # We can use 'random', 'grid', or 'bayes'
        'metric': {
            'name': 'val_loss',  # The metric we want to optimize
            'goal': 'minimize'   # We want to minimize the loss
        },
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform_values',  # Log distribution for learning rate
                'min': 1e-5,
                'max': 1e-2
            },
            'conv2_channels': {
                'values': [64, 128, 256]  # Discrete values for conv2_channels
            }
            # Add more hyperparameters as needed
            # 'batch_size': {
            #     'distribution': 'q_log_uniform_values',
            #     'q': 8,
            #     'min': 16,
            #     'max': 128
            # },
            # 'conv1_channels': {
            #     'values': [32, 64, 128, 256]
            # },
        }
    }
    
    return sweep_config


def train_with_wandb_config():
    """
    Training function that will be called by wandb.agent
    with different hyperparameter configurations.
    """
    # Initialize a new wandb run
    with wandb.init() as run:
        # Copy the base configurations
        train_cfg = copy.deepcopy(train_config)
        model_cfg = copy.deepcopy(config_TinyVGG)
        
        # Get hyperparameters from wandb config
        wandb_config = dict(run.config)
        
        # Apply hyperparameters from wandb config to our configs
        for param_name, param_value in wandb_config.items():
            if param_name in train_cfg:
                train_cfg[param_name] = param_value
            elif param_name in model_cfg:
                model_cfg[param_name] = param_value
        
        # Set a unique run name
        run_id = wandb.run.id
        train_cfg['run_name'] = f"wandb_sweep_{run_id}"
        
        # Create and run trainer
        trainer = Trainer(train_cfg, model_cfg)
        trainer.train()
        
        # Get final validation loss
        final_val_loss = getattr(trainer, 'final_val_loss', float('inf'))
        
        # Log the final validation loss to wandb
        wandb.log({"val_loss": final_val_loss})
        
        # Clean up
        trainer.cleanup()


def run_wandb_sweep(project_name, count=5):
    """
    Run a WandB sweep to find optimal hyperparameters.
    
    Args:
        project_name (str): WandB project name
        entity (str, optional): WandB entity/username
        count (int): Number of runs to perform in the sweep
        
    Returns:
        str: Sweep ID of the created sweep
    """
    # Ensure WandB is logged in
    wandb.login()
    
    # Define sweep configuration
    sweep_config = define_wandb_sweep_config()
    
    # Initialize the sweep
    sweep_id = wandb.sweep(
        sweep_config, 
        project=project_name
    )
    
    print(f"Sweep ID: {sweep_id}")
    
    # Start the sweep agent that will run training with different hyperparameters
    wandb.agent(sweep_id, function=train_with_wandb_config, count=count)
    
    print(f"\nWandB sweep completed.")
    
    return sweep_id


if __name__ == "__main__":
    # Choose which hyperparameter tuning method to use
    use_wandb_sweep = False  # Set to False to use grid search instead
    
    if use_wandb_sweep:
        # Run WandB sweep
        sweep_id = run_wandb_sweep(project_name=train_config["project_name"], count=6)  # Run 6 different hyperparameter combinations
        print(f"WandB sweep completed with ID: {sweep_id}")
    else:
        # Run TensorBoard grid search
        best_hparams, best_loss = grid_search_hyperparams()
        print(f"Grid search completed. Best hyperparameters: {best_hparams}, best loss: {best_loss:.4f}")
    
