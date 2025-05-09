#!/bin/bash
#SBATCH --job-name=p10
#SBATCH --output=result_LSTM_FIX1_%j.out
#SBATCH --error=error_LSTM_FIX1_%j.err
#SBATCH --time=12:00:00 # 
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=40
#SBATCH --mem=90G

singularity exec --nv energycontainerblackbox.sif python3 baselines_tuning_fix1.py --model LSTM