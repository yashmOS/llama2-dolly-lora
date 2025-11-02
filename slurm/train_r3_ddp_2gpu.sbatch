PY=python
ACC=accelerate

# === Paths & IDs ===
BASE_MODEL=meta-llama/Llama-2-7b-hf
RUN_R1=runs/r1_attn_only
RUN_R2=runs/r2_attn_mlp
RUN_R3=runs/r3_ddp_qlora

# Adapter dirs (assumes you keep a "checkpoints/best" symlink or folder)
ADAPTER_R1=$(RUN_R1)/checkpoints/best
ADAPTER_R2=$(RUN_R2)/checkpoints/best
ADAPTER_R3=$(RUN_R3)/checkpoints/best

# === Data ===
data:  ## Prepare Dolly-15K splits
	$(PY) scripts/prepare_dolly.py --config configs/experiment/data_dolly.yaml

# === Training ===
train-r1:  ## QLoRA, LoRA=attn only (single GPU)
	$(ACC) launch --config_file configs/accelerate/single_gpu.yaml \
		scripts/train_lora.py --config configs/experiment/r1_attn_only.yaml

train-r2:  ## QLoRA, LoRA=attn+MLP (single GPU)
	$(ACC) launch --config_file configs/accelerate/single_gpu.yaml \
		scripts/train_lora.py --config configs/experiment/r2_attn_mlp.yaml

train-r3:  ## QLoRA, DDP (multi-GPU)
	$(ACC) launch --config_file configs/accelerate/ddp_2gpu.yaml \
		scripts/train_lora.py --config configs/experiment/r3_ddp_qlora.yaml

# === AlpacaEval 2 === (PEFT-aware; no merge needed)
AE_DIR=eval/alpacaeval

ae-gen-base:
	$(PY) scripts/gen_alpacaeval.py \
		--model $(BASE_MODEL) \
		--out $(AE_DIR)/answers_base.jsonl

ae-gen-r2:
	$(PY) scripts/gen_alpacaeval.py \
		--base $(BASE_MODEL) \
		--adapter $(ADAPTER_R2) \
		--out $(AE_DIR)/answers_r2.jsonl

ae-gen-r3:
	$(PY) scripts/gen_alpacaeval.py \
		--base $(BASE_MODEL) \
		--adapter $(ADAPTER_R3) \
		--out $(AE_DIR)/answers_r3.jsonl

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

ae-judge-r3:
	$(PY) scripts/run_alpacaeval_judge.py \
		--outputs $(AE_DIR)/answers_r3.jsonl \
		--judge configs/judges/openai_gpt4.1mini.yaml \
		--out $(AE_DIR)/scores_r3_gpt4.1mini.json

# === MT-Bench (FastChat) === (PEFT-aware; no merge needed)
MT_DIR=eval/mtbench

mt-gen-base:
	$(PY) scripts/gen_mtbench.py \
		--model-path $(BASE_MODEL) \
		--model-id base_llama2_7b \
		--out $(MT_DIR)/answers_base.jsonl

mt-gen-r2:
	$(PY) scripts/gen_mtbench.py \
		--base $(BASE_MODEL) \
		--adapter $(ADAPTER_R2) \
		--model-id r2_attn_mlp \
		--out $(MT_DIR)/answers_r2.jsonl

mt-gen-r3:
	$(PY) scripts/gen_mtbench.py \
		--base $(BASE_MODEL) \
		--adapter $(ADAPTER_R3) \
		--model-id r3_ddp_qlora \
		--out $(MT_DIR)/answers_r3.jsonl

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

mt-judge-r3:
	$(PY) scripts/run_mtbench_judge.py \
		--model-id r3_ddp_qlora \
		--judge gpt-4.1-mini \
		--out $(MT_DIR)/scores_r3_gpt4.1mini.json

# === Plots ===
plots:
	$(PY) scripts/plot_losses.py --runs $(RUN_R1) $(RUN_R2) $(RUN_R3) \
		--out report/figs/loss_curves.png

.PHONY: data train-r1 train-r2 train-r3 \
	ae-gen-base ae-gen-r2 ae-gen-r3 ae-judge-base ae-judge-r2 ae-judge-r3 \
	mt-gen-base mt-gen-r2 mt-gen-r3 mt-judge-base mt-judge-r2 mt-judge-r3 \
	plots
