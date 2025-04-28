#!/bin/bash
#SBATCH --job-name=p10
#SBATCH --output=result_GB_%j.out
#SBATCH --error=error_GB_%j.err
#SBATCH --time=12:00:00 # 
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=125
#SBATCH --mem=175G

singularity exec --nv energycontainerblackbox.sif python3 baselines_tuning.py --model GradientBoosting