#!/bin/bash
#SBATCH --job-name=r3_fsdp_scaling
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:2
#SBATCH --time=0-12:00:00
#SBATCH --output=runs/r3_fsdp_scaling/slurm.out
#SBATCH --error=runs/r3_fsdp_scaling/slurm.err

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

accelerate launch --config_file configs/accelerate/fsdp_2gpu.yaml \
  scripts/train_lora_fsdp.py --config configs/experiment/r3_fsdp_scaling.yaml
