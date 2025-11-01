#!/bin/bash
#SBATCH --job-name=r2_attn_mlp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --time=2-00:00:00
#SBATCH --output=runs/r2_attn_mlp/slurm.out
#SBATCH --error=runs/r2_attn_mlp/slurm.err

source ~/project/llama2-dolly-lora/.venv/bin/activate
if [ -f ~/project/llama2-dolly-lora/.env ]; then
  set -a; source ~/project/llama2-dolly-lora/.env; set +a
fi

export PYTHONUNBUFFERED=1
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_PROJECT=${WANDB_PROJECT:-llama2-dolly}
export WANDB_ENTITY=${WANDB_ENTITY:-$USER}
export TRANSFORMERS_VERBOSITY=info
export HF_LOG_LEVEL=info

accelerate launch --config_file configs/accelerate/single_gpu.yaml \
  scripts/train_lora.py --config configs/experiment/r2_attn_mlp.yaml
