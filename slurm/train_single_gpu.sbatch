#!/bin/bash
#SBATCH --job-name=llama2_lora_single
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# >>> YOUR ENV SETUP (example)
# module load cuda/12.2  # if your cluster uses modules
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate llama2-lora
# <<<

# Hugging Face token must be available if model is gated
# export HUGGING_FACE_HUB_TOKEN=...

python scripts/prepare_dolly.py --subset_ratio 0.4 --seed 42 --outdir data/processed

python scripts/train_lora.py \
  --config configs/training.yaml
