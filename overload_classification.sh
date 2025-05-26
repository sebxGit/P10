#!/bin/bash
#SBATCH --job-name=p10
#SBATCH --output=result_overload_classification_%j.out
#SBATCH --error=error_overload_classification_%j.err
#SBATCH --time=12:00:00 # 
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20
#SBATCH --mem=50G

singularity exec --nv energycontainerblackbox.sif python3 overload_classification.py --dataset SDU --threshold 500 --multiplier 2 --downscaling 13 --models LSTM --individual True 