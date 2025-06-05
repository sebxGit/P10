#!/bin/bash
#SBATCH --job-name=EVCSForecastRun
#SBATCH --output=result_overload%j.out
#SBATCH --error=error_overload_%j.err
#SBATCH --time=12:00:00 # 
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=20
#SBATCH --mem=192G


MODELS=("LSTM" "GRU" "AdaBoostRegressor" "RandomForestRegressor" "GradientBoostingRegressor" "xPatch" "PatchMixer")
for model in "${MODELS[@]}"; do
    echo "Running model: $model"
    singularity exec --nv energycontainerblackbox.sif \
        python3 overload_classification_col.py \
        --models "$model" \
        --individual True \
        --multiplier 2 \
        --downscaling 13 \
        --dataset Colorado \
        --threshold 500 || echo "Model $model failed due to missing hyperparameters."
done
