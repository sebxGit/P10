#!/bin/bash
#SBATCH --job-name=p10
#SBATCH --output=result_architecture_tuning_classification_%j.out
#SBATCH --error=error_architecture_tuning_classification_%j.err
#SBATCH --time=12:00:00 # 
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=192G

singularity exec --nv energycontainerblackbox.sif python3 architecture_tuning_classification_col.py