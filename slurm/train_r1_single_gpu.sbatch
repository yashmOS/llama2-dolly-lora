#!/bin/bash
#SBATCH --job-name=r1_attn_only
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --time=2-00:00:00
#SBATCH --output=runs/r1_attn_only/slurm.out

# 1) Activate venv
source ~/project/llama2-dolly-lora/.venv/bin/activate

# 2) Load secrets in .env 
if [ -f ~/project/llama2-dolly-lora/.env ]; then
  set -a; source ~/project/llama2-dolly-lora/.env; set +a
fi

# 3) Explicitly export project tags
export WANDB_PROJECT=${WANDB_PROJECT:-llama2-dolly}

# 4) Launch
accelerate launch --config_file configs/accelerate/single_gpu.yaml \
  scripts/train_lora.py --config configs/experiment/r1_attn_only.yaml