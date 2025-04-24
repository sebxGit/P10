#!/bin/bash
#SBATCH --job-name=p10
#SBATCH --output=result_%j.out
#SBATCH --error=error_%j.err
#SBATCH --time=12:00:00 # 
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=20
#SBATCH --mem=175G

singularity exec --nv p10container.sif python3 baselines_tuning_DeepSpeed.py --model Fredformer