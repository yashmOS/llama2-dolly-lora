.PHONY: data train-r1 train-r2 train-r3 merge-r2 ae-gen-base ae-gen-r2 plots

data:
	python scripts/prepare_dolly.py --config configs/experiment/data_dolly.yaml

train-r1:
	accelerate launch --config_file configs/accelerate/single_gpu.yaml \
		scripts/train_lora.py --config configs/experiment/r1_attn_only.yaml

train-r2:
	accelerate launch --config_file configs/accelerate/single_gpu.yaml \
		scripts/train_lora.py --config configs/experiment/r2_attn_mlp.yaml

train-r3:
	accelerate launch --config_file configs/accelerate/fsdp_2gpu.yaml \
		scripts/train_lora_fsdp.py --config configs/experiment/r3_fsdp_scaling.yaml

merge-r2:
	python scripts/merge_and_export.py \
		--base meta-llama/Llama-2-7b-hf \
		--adapter runs/r2_attn_mlp/checkpoints/best \
		--out runs/r2_attn_mlp/merged

ae-gen-base:
	python scripts/gen_alpacaeval.py \
		--model meta-llama/Llama-2-7b-hf \
		--out runs/alpacaeval/base_outputs.jsonl

ae-gen-r2:
	python scripts/gen_alpacaeval.py \
		--model runs/r2_attn_mlp/merged \
		--out runs/alpacaeval/r2_outputs.jsonl

plots:
	python scripts/plot_training.py
