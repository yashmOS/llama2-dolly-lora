#!/bin/bash
#SBATCH --job-name=llama2_lora_fsdp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# >>> YOUR ENV SETUP (example)
# module load cuda/12.2
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate llama2-lora
# <<<

python scripts/prepare_dolly.py --subset_ratio 0.4 --seed 42 --outdir data/processed

# torchrun uses one process per GPU
torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE scripts/train_lora.py \
  --config configs/training.yaml \
  --fsdp
