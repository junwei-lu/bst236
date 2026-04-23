# config.py
import torch
import os

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
NUM_EPOCHS = 100
BATCH_SIZE = 10
LEARNING_RATE = 0.001

# Linear Regression Dataset parameters
NUM_SAMPLES = 100

# Model parameters for linear regression
INPUT_DIM = 1
OUTPUT_DIM = 1

# Save directories
CHECKPOINTS_DIR = os.path.join("output", "checkpoints")
RESULTS_DIR = os.path.join("output", "results")
LOGS_DIR = os.path.join("output", "logs")

# Create directories if they don't exist
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

