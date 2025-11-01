#!/bin/bash
#SBATCH --job-name=r1_attn_only
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --time=2-00:00:00
#SBATCH --output=runs/r1_attn_only/slurm.out

source ~/.bashrc
conda activate YOUR_ENV  # or source venv
export HF_TOKEN=${HF_TOKEN}
export WANDB_PROJECT=llama2-dolly

accelerate launch --config_file configs/accelerate/single_gpu.yaml \
  scripts/train_lora.py --config configs/experiment/r1_attn_only.yaml