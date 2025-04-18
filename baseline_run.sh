#!/bin/bash
#SBATCH --job-name=p10
#SBATCH --output=result_%j.out
#SBATCH --error=error_%j.err
#SBATCH --time=12:00:00 # 
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=20
#SBATCH --mem=175G

singularity exec --nv energycontainerblackbox.sif python3 baselines.py --input_size 22 --seq_len 12 --batch_size 8 --criterion AsymmetricMAEandMSELoss --optimizer Adam  --max_epochs 1000 --n_features 7 --hidden_size 100 --num_layers 1 --dropout 0 --learning_rate 0.001 --num_workers 5 --scaler MinMaxScaler