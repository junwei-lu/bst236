#!/bin/bash
#SBATCH --job-name=multiprocess_job    # Name of the job
#SBATCH --output=slurm_%j.log          # Log file 
#SBATCH --error=slurm_%j.err           # Error file 
#SBATCH --time=24:00:00                # Time limit (24 hours)
#SBATCH --ntasks=1                     # Keep this as 1 for Python multiprocessing
#SBATCH --cpus-per-task=8              # Number of CPU cores to allocate
#SBATCH --mem=8G                       # Memory per node 

echo "Job started"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"

# virtual environment
source venv/bin/activate
# Run the Python script
# python3 multiprocessing_template.py
python3 test_multiprocess.py

echo "Job ended"