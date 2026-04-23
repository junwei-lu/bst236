#!/bin/bash
#SBATCH --job-name=tinyVGG_train
#SBATCH --output=tinyVGG-%j.out
#SBATCH --error=tinyVGG-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G

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

# Run the training script
python src/train.py

# Print completion message
echo "Training completed at $(date)" 