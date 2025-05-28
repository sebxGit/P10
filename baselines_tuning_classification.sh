#!/bin/bash
#SBATCH --job-name=p10
#SBATCH --output=result_classification_tuning_%j.out
#SBATCH --error=error_classification_tuning_%j.err
#SBATCH --time=12:00:00 # 
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=192G

singularity exec --nv energycontainerblackbox.sif python3 baselines_tuning_classification.py --dataset Colorado --pred_len 24  --model DPAD --individual True