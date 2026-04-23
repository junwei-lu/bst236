#!/bin/bash
#SBATCH --job-name=tinyVGG_dataparallel
#SBATCH --output=tinyVGG_dp-%j.out
#SBATCH --error=tinyVGG_dp-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
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

# Set environment variables for better GPU performance
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run the training script with DataParallel
python src/train_dataparallel.py --save-freq 5

# Print completion message
echo "Training completed at $(date)" 