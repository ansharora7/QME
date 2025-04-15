#!/bin/bash

#SBARCH --nodelist=uri-gpu006
#SBATCH --partition=gpu-preempt  # Partition for preemptible GPUs
#SBATCH --gres=gpu:rtx8000  # Specific GPU type and count
#SBATCH --mem=80GB  # Total requested memory
#SBATCH -t 24:00:00  # Job time limit
#SBATCH -c 4  # Number of CPU cores per task
#SBATCH -o slurm-%j.out  # Output file with job ID in the filename

# Load required modules and activate environment
module load conda/latest
conda activate sail
module load cuda/12.6
/modules/apps/cuda/12.6/samples/bin/x86_64/linux/release/deviceQuery

# Define the Python script to run
SCRIPT_NAME="ag_finetune.py"
export HF_HOME=/scratch3/workspace/amritanshmis_umass_edu-mixing_expers
export HF_TOKEN=''
# Check if the script exists and run it if found
if [ -f "$SCRIPT_NAME" ]; then
    echo "Running script: $SCRIPT_NAME"
    python $SCRIPT_NAME

else
    echo "Script not found: $SCRIPT_NAME"
    exit 1
fi
