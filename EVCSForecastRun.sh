#!/bin/bash
#SBATCH --job-name=EVCSForecastRun
#SBATCH --output=result_EVCSForecastRun_%j.out
#SBATCH --error=error_EVCSForecastRun_%j.err
#SBATCH --time=12:00:00 # 
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=20
#SBATCH --mem=192G


MODELS=("LSTM", "GRU", "AdaBoost", "RandomForest", "GradientBoosting", "xPatch", "PatchMixer")

for model in "${MODELS[@]}"; do
    echo "Running model: $model"
    singularity exec --nv energycontainerblackbox.sif \
        python3 model_run.py \
        --model "$model" \
        --load False \
        --individual True \
        --dataset SDU \
        --threshold 250
done
