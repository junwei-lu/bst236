# Simple TensorBoard logging utility functions
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

def get_tensorboard_writer(log_dir="results/logs"):
    """Create a TensorBoard writer with timestamped directory"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(log_dir, f"run_{timestamp}")
    return SummaryWriter(log_dir) 