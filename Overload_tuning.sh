#!/bin/bash
#SBATCH --job-name=model_tuning
#SBATCH --output=result_model_tuning_%j.out
#SBATCH --error=error_model_tuning_%j.err
#SBATCH --time=12:00:00 # 
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=20
#SBATCH --mem=192G


MODELS=("LSTM" "GRU" "AdaBoostRegressor" "RandomForestRegressor" "GradientBoostingRegressor" "xPatch" "PatchMixer" "DPAD")
for model in "${MODELS[@]}"; do
    echo "Running model: $model"
    singularity exec --nv energycontainerblackbox.sif \
        python3 baselines_tuning_classification_SDU.py \
        --model "$model" \
        --individual True \
        --multiplier 2 \
        --downscaling 13 \
        --dataset SDU \
        --threshold 250 || echo "Model $model failed due to missing hyperparameters."
done
