#!/bin/bash
#SBATCH --job-name=r2_attn_mlp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --time=2-00:00:00
#SBATCH --output=runs/r2_attn_mlp/slurm.out

source ~/.bashrc
conda activate YOUR_ENV
export HF_TOKEN=${HF_TOKEN}
export WANDB_PROJECT=llama2-dolly

accelerate launch --config_file configs/accelerate/single_gpu.yaml \
  scripts/train_lora.py --config configs/experiment/r2_attn_mlp.yaml