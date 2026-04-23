#!/bin/bash
#SBATCH --job-name=tinyVGG_multi_gpu
#SBATCH --output=tinyVGG_multi-%j.out
#SBATCH --error=tinyVGG_multi-%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

# Print node information
echo "Job running on $(hostname)"
echo "Available GPUs:"
nvidia-smi

# Load modules or activate conda environment as needed
# Uncomment and modify the below lines according to your environment
# module load cuda
# source activate your_conda_env

# Change to the directory containing the script
cd $SLURM_SUBMIT_DIR

# Set NCCL environment variables for better performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=^docker0,lo

# Option 1: Run with torchrun (recommended)
torchrun --nproc_per_node=4 src/train_gpus.py --save-freq 100

# Option 2: Run with Python directly (alternative)
# python src/train_gpus.py --num-gpus 4 --save-freq 5

# Print completion message
echo "Training completed at $(date)" 