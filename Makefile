PY=python
ACC=accelerate

# === Data ===
data:  ## Prepare Dolly-15K splits
	$(PY) scripts/prepare_dolly.py --config configs/experiment/data_dolly.yaml

# === Training ===
train-r1:  ## QLoRA, LoRA=attn only
	$(ACC) launch --config_file configs/accelerate/single_gpu.yaml \
		scripts/train_lora.py --config configs/experiment/r1_attn_only.yaml

train-r2:  ## QLoRA, LoRA=attn+MLP (final)
	$(ACC) launch --config_file configs/accelerate/single_gpu.yaml \
		scripts/train_lora.py --config configs/experiment/r2_attn_mlp.yaml

train-r3-fsdp:  ## QLoRA + FSDP (throughput demo)
	$(ACC) launch --config_file configs/accelerate/fsdp_2gpu.yaml \
		scripts/train_lora_fsdp.py --config configs/experiment/r3_fsdp_scaling.yaml

# === Merge LoRA (for MT-Bench FastChat) ===
merge-r2:
	$(PY) scripts/merge_and_export.py \
		--base meta-llama/Llama-2-7b-hf \
		--adapter runs/r2_attn_mlp/checkpoints/best \
		--out runs/r2_attn_mlp/merged

# === AlpacaEval 2 ===
AE_DIR=eval/alpacaeval

ae-gen-base:
	$(PY) scripts/gen_alpacaeval.py \
		--model meta-llama/Llama-2-7b-hf \
		--out $(AE_DIR)/answers_base.jsonl

ae-gen-r2:
	$(PY) scripts/gen_alpacaeval.py \
		--model runs/r2_attn_mlp/merged \
		--out $(AE_DIR)/answers_r2.jsonl

ae-judge-base:
	$(PY) scripts/run_alpacaeval_judge.py \
		--outputs $(AE_DIR)/answers_base.jsonl \
		--judge configs/judges/openai_gpt4.1mini.yaml \
		--out $(AE_DIR)/scores_base_gpt4.1mini.json

ae-judge-r2:
	$(PY) scripts/run_alpacaeval_judge.py \
		--outputs $(AE_DIR)/answers_r2.jsonl \
		--judge configs/judges/openai_gpt4.1mini.yaml \
		--out $(AE_DIR)/scores_r2_gpt4.1mini.json

# === MT-Bench (FastChat) ===
MT_DIR=eval/mtbench

mt-gen-base:
	$(PY) scripts/gen_mtbench.py \
		--model-path meta-llama/Llama-2-7b-hf \
		--model-id base_llama2_7b \
		--out $(MT_DIR)/answers_base.jsonl

mt-gen-r2:
	$(PY) scripts/gen_mtbench.py \
		--model-path runs/r2_attn_mlp/merged \
		--model-id r2_attn_mlp \
		--out $(MT_DIR)/answers_r2.jsonl

mt-judge-base:
	$(PY) scripts/run_mtbench_judge.py \
		--model-id base_llama2_7b \
		--judge gpt-4.1-mini \
		--out $(MT_DIR)/scores_base_gpt4.1mini.json

mt-judge-r2:
	$(PY) scripts/run_mtbench_judge.py \
		--model-id r2_attn_mlp \
		--judge gpt-4.1-mini \
		--out $(MT_DIR)/scores_r2_gpt4.1mini.json

# === Plots ===
plots:
	$(PY) scripts/plot_losses.py --runs runs/r1_attn_only runs/r2_attn_mlp runs/r3_fsdp_scaling \
		--out report/figs/loss_curves.png

.PHONY: data train-r1 train-r2 train-r3-fsdp merge-r2 ae-gen-base ae-gen-r2 ae-judge-base ae-judge-r2 mt-gen-base mt-gen-r2 mt-judge-base mt-judge-r2 plots