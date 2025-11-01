import argparse, os, yaml, math, json
import torch
from dataclasses import dataclass
from typing import Dict, List

from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_bnb(cfg):
    q = cfg["quantization"]
    if not q.get("qlora_4bit", True):
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=q.get("double_quant", True),
        bnb_4bit_quant_type="nf4" if q.get("nf4", True) else "fp4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def format_example(ex: Dict[str, str], template: str) -> str:
    return template.format(
        instruction=ex.get("instruction", ""),
        context=ex.get("context", ""),
        response=ex.get("response", ""),
    )


def main(cfg_path: str):
    cfg = load_yaml(cfg_path)
    data_cfg = load_yaml(cfg["paths"]["data_config"])

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Model
    bnb = build_bnb(cfg)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype=torch.bfloat16 if cfg["train"].get("bf16", True) else torch.float16,
        quantization_config=bnb,
        device_map="auto",
    )
    model.config.use_cache = False  # important for grad checkpointing
    if bnb is not None:
        model = prepare_model_for_kbit_training(model)

    # PEFT LoRA
    lora = LoraConfig(
        r=cfg["peft"]["r"],
        lora_alpha=cfg["peft"]["alpha"],
        lora_dropout=cfg["peft"].get("dropout", 0.0),
        target_modules=cfg["peft"]["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    # Datasets (processed JSONL)
    data_files = {
        "train": "data/processed/train.jsonl",
        "validation": "data/processed/val.jsonl",
    }
    ds = load_dataset("json", data_files=data_files)

    template = data_cfg["template"]

    def fmt(batch):
        return {
            "text": [format_example(ex, template) for ex in batch]
        }

    # Map to a single "text" field
    train_text = ds["train"].map(fmt, batched=False)
    val_text = ds["validation"].map(fmt, batched=False)

    # Only compute loss on the response portion
    response_template = "### Response:"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tok,
    )

    # Training args
    ta = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["train"]["epochs"],
        per_device_train_batch_size=cfg["train"]["per_device_batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg["train"]["grad_accum_steps"],
        learning_rate=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
        logging_steps=cfg["train"]["logging_steps"],
        lr_scheduler_type=cfg["train"]["lr_schedule"],
        warmup_steps=cfg["train"]["warmup_steps"],
        bf16=cfg["train"].get("bf16", True),
        gradient_checkpointing=cfg["train"].get("gradient_checkpointing", True),
        evaluation_strategy=cfg["train"]["eval_strategy"],
        save_strategy=cfg["train"]["save_strategy"],
        save_total_limit=2,
        report_to=["wandb"],
        run_name=cfg["run_name"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=train_text,
        eval_dataset=val_text,
        formatting_func=None,
        dataset_text_field="text",
        max_seq_length=cfg["train"]["max_seq_len"],
        data_collator=collator,
        args=ta,
    )

    out = trainer.train()

    # Save PEFT adapter (best checkpoint dir mirrors HF)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    trainer.model.save_pretrained(os.path.join(cfg["output_dir"], "best"))
    trainer.tokenizer.save_pretrained(os.path.join(cfg["output_dir"], "best"))

    # Dump training history to CSV for plotting later
    hist = trainer.state.log_history
    os.makedirs(os.path.dirname(cfg["paths"]["history_csv"]), exist_ok=True)
    import pandas as pd
    pd.DataFrame(hist).to_csv(cfg["paths"]["history_csv"], index=False)

    print("Training complete. History saved to:", cfg["paths"]["history_csv"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)