# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from model import LinearRegressionModel
from train import Trainer
from dataset import get_loaders
from config import (
    DEVICE, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, 
    NUM_SAMPLES, INPUT_DIM, OUTPUT_DIM,
    CHECKPOINTS_DIR, RESULTS_DIR, LOGS_DIR
)

def main():
    # Create hyperparameters dictionary
    hyperparams = {
        'DEVICE': DEVICE,
        'NUM_EPOCHS': NUM_EPOCHS,
        'BATCH_SIZE': BATCH_SIZE,
        'LEARNING_RATE': LEARNING_RATE,
        'NUM_SAMPLES': NUM_SAMPLES,
        'INPUT_DIM': INPUT_DIM,
        'OUTPUT_DIM': OUTPUT_DIM,
        'CHECKPOINTS_DIR': CHECKPOINTS_DIR,
        'RESULTS_DIR': RESULTS_DIR,
        'LOGS_DIR': LOGS_DIR
    }
    
    # Create data loaders for training and validation
    train_loader, val_loader = get_loaders(
        num_samples=NUM_SAMPLES,
        batch_size=BATCH_SIZE
    )

    # Initialize the model
    model = LinearRegressionModel(INPUT_DIM, OUTPUT_DIM).to(DEVICE)
    
    # Define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    # Create a trainer instance with logging enabled
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        use_logger=True,  # Enable logging
        save_frequency=10,
        hyperparams=hyperparams,
        git_commit=True   # Enable auto git commit after training
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main() 