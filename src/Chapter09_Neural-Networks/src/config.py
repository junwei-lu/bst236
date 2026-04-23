# Configuration parameters for training
train_config = {
    "project_name": "pytorch-cnn-cv-finaltest",
    "run_name": "cifar10-tinyvgg-run-tuning",
    "checkpoint_dir": "checkpoints",
    "epochs": 10,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "validation_split": 0.2,
    "num_workers": 1
}


# Configuration parameters for TinyVGG model
config_TinyVGG = {
    "input_channels": 3,
    "num_classes": 10,
    "conv1_channels": 64,
    "conv2_channels": 128,
    "kernel_size": 3,
    "padding": 1
}
