import os, json, argparse, yaml, math
import torch
from dataclasses import dataclass
from typing import Dict, List
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer
import matplotlib.pyplot as plt
import pandas as pd

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_tokenizer(model_name, cache_dir):
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok

def load_model(model_name, cfg):
    bnb_cfg = None
    if cfg.get("load_in_4bit", True):
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=torch.bfloat16 if cfg.get("bnb_4bit_compute_dtype","bfloat16")=="bfloat16" else torch.float16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cfg.get("cache_dir"),
        device_map="auto",
        quantization_config=bnb_cfg
    )
    model.config.use_cache = False
    return model

def build_lora_config(cfg):
    return LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

def load_jsonl_dataset(path, split_name):
    return load_dataset("json", data_files={split_name: path})[split_name]

def plot_losses(trainer, out_dir):
    logs = trainer.state.log_history
    df = pd.DataFrame(logs)
    df.to_csv(os.path.join(out_dir, "training_logs.csv"), index=False)

    # Smooth-ish curves for train/eval loss
    plt.figure()
    if "loss" in df.columns:
        plt.plot(df.dropna(subset=["loss"])["loss"].values, label="train loss")
    if "eval_loss" in df.columns:
        plt.plot(df.dropna(subset=["eval_loss"])["eval_loss"].values, label="eval loss")
    plt.xlabel("logging steps / eval events")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Training / Validation Loss")
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    plt.savefig(os.path.join(out_dir, "plots", "loss_curves.png"), bbox_inches="tight")
    plt.close()

def save_hardware_meta(out_dir):
    meta = {
        "num_gpus": torch.cuda.device_count(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "torch_cuda_available": torch.cuda.is_available()
    }
    with open(os.path.join(out_dir, "hardware.json"), "w") as f:
        json.dump(meta, f, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fsdp", action="store_true", help="Enable FSDP (used by multi-GPU Slurm script)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg["output_dir"], exist_ok=True)

    tokenizer = load_tokenizer(cfg["model_name"], cfg.get("cache_dir"))
    model = load_model(cfg["model_name"], cfg)

    # Attach LoRA
    lora_cfg = build_lora_config(cfg)
    model = get_peft_model(model, lora_cfg)

    # Load data
    train_ds = load_jsonl_dataset(cfg["train_file"], "train")
    eval_ds  = load_jsonl_dataset(cfg["eval_file"], "validation")

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        num_train_epochs=cfg["num_train_epochs"],
        bf16=cfg.get("bf16", False),
        fp16=not cfg.get("bf16", False),
        logging_steps=cfg["logging_steps"],
        save_strategy=cfg["save_strategy"],
        evaluation_strategy=cfg["evaluation_strategy"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        seed=cfg["seed"],
        report_to=["tensorboard"],
        dataloader_drop_last=False,
    )

    if args.fsdp:
        # Enable sharded training when launched via the multi-GPU Slurm script
        training_args.fsdp = "full_shard auto_wrap"
        training_args.fsdp_transformer_layer_cls_to_wrap = "LlamaDecoderLayer"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
        max_seq_length=cfg["max_seq_length"],
        packing=True,  # on-the-fly packing for throughput
        dataset_text_field="text",
    )

    trainer.train()
    trainer.save_model()                # saves the PEFT adapter
    tokenizer.save_pretrained(cfg["output_dir"])

    # Export curves + hardware meta for the report
    plot_losses(trainer, cfg["output_dir"])
    save_hardware_meta(cfg["output_dir"])

    # Also save final eval loss to a tiny JSON for easy tabling
    metrics = trainer.evaluate()
    with open(os.path.join(cfg["output_dir"], "final_eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete. Adapter saved at:", cfg["output_dir"])

if __name__ == "__main__":
    main()
