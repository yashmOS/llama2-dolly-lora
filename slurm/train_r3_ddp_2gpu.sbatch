#!/bin/bash
#SBATCH --job-name=r3_ddp_qlora
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:30:00
#SBATCH --output=runs/r3_ddp_qlora/slurm.out
#SBATCH --error=runs/r3_ddp_qlora/slurm.err

set -euo pipefail
type module >/dev/null 2>&1 && module purge || true

# Activate venv & env
source .venv/bin/activate
set -a; [ -f .env ] && source .env; set +a

# allocator hint for fewer frags
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"
export WANDB_MODE=${WANDB_MODE:-offline}

# echo GPU vis to confirm two slots are visible
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

accelerate launch \
  --config_file configs/accelerate/ddp_2gpu.yaml \
  scripts/train_lora.py \
  --config configs/experiment/r3_ddp_qlora.yaml
