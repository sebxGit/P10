#!/bin/bash
#SBATCH --job-name=p10
#SBATCH --output=result_xPatch_%j.out
#SBATCH --error=error_xPatch_%j.err
#SBATCH --time=12:00:00 # 
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=20
#SBATCH --mem=192G

singularity exec --nv energycontainerblackbox.sif python3 baselines_tuning.py --dataset SDU --pred_len 24  --model xPatch --mixed True --individual False