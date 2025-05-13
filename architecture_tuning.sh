#!/bin/bash
#SBATCH --job-name=p10
#SBATCH --output=result_baselines_run_%j.out
#SBATCH --error=error_baselines_run_%j.err
#SBATCH --time=12:00:00 # 
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=20
#SBATCH --mem=175G

srun --pty singularity exec --nv energycontainerblackbox.sif python3 architecture_tuning.py