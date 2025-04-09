#!/bin/bash
#SBATCH --job-name=p9_blackbox_run
#SBATCH --output=result_%j.out
#SBATCH --error=error_%j.err
#SBATCH --time=12:00:00 # 
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=20
#SBATCH --mem=175G

singularity exec --nv energycontainerblackbox.sif python3 baselines_tuning.py --model LSTM