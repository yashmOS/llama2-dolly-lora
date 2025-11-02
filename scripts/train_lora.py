import argparse, os, yaml, textwrap
import torch
from typing import Dict, Union, List
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import csv

# if torch.cuda.is_available():
#     if "LOCAL_RANK" in os.environ:            # Accelerate/DDP per-rank
#         torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
#     else:                                      # single-GPU or non-DDP
#         torch.cuda.set_device(0)

# ----------------------------- Logging ---------------------------------
class CSVLoggerCallback(TrainerCallback):
    """Write live logs to CSV (runs/<run>/logs/history.csv)."""
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self._file = open(csv_path, "w", newline="", encoding="utf-8")
        self._writer = None
        self._header = [
            "step", "epoch", "loss", "eval_loss", "learning_rate", "grad_norm",
            "train_runtime", "train_samples_per_second", "train_steps_per_second"
        ]

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if self._writer is None:
            extras = [k for k in logs.keys() if k not in self._header]
            self._header += extras
            self._writer = csv.DictWriter(self._file, fieldnames=self._header)
            self._writer.writeheader()
        row = {k: logs.get(k, "") for k in self._header}
        row["step"] = getattr(state, "global_step", row.get("step", ""))
        row["epoch"] = getattr(state, "epoch", row.get("epoch", ""))
        self._writer.writerow(row)
        self._file.flush()

    def on_train_end(self, *args, **kwargs):
        try:
            self._file.close()
        except Exception:
            pass


# ----------------------------- Utilities --------------------------------
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_bnb(cfg):
    q = cfg["quantization"]
    if not q.get("qlora_4bit", True):
        return None
    bnb_4bit_compute_dtype = torch.bfloat16 if cfg["train"].get("bf16", True) else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=q.get("double_quant", True),
        bnb_4bit_quant_type="nf4" if q.get("nf4", True) else "fp4",
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
    )


def format_example(ex: Dict[str, str], template: str) -> str:
    return template.format(
        instruction=ex.get("instruction", ""),
        context=ex.get("context", ""),
        response=ex.get("response", ""),
    )


def _strip_leading_ws_ids(tok, ids: List[int]) -> List[int]:
    def _is_ws(tid: int) -> bool:
        s = tok.decode([tid])
        return s.strip() == ""
    i = 0
    while i < len(ids) and _is_ws(ids[i]):
        i += 1
    return ids[i:]


def build_response_anchor(tok, normalized_template: str) -> Union[str, List[int]]:
    """
    Build a robust anchor for TRL's DataCollatorForCompletionOnlyLM:
      1) Extract the exact label line before {response} (e.g., '### Response:').
      2) Prepend a newline to mimic in-text context.
      3) Tokenize and strip leading whitespace token IDs.
      4) If result is empty (edge case), FALL BACK to string anchor '### Response:'.
    Returns either a non-empty List[int] or the fallback string.
    """
    before_resp = normalized_template.split("{response}", 1)[0]
    last_nl = before_resp.rfind("\n")
    label_line = before_resp[last_nl + 1:] if last_nl != -1 else before_resp
    # ensure it actually contains something human-readable like '### Response:'
    if "Response:" not in label_line:
        label_line = "### Response:\n"
    elif not label_line.endswith("\n"):
        label_line = label_line + "\n"

    contextual = "\n" + label_line  # e.g., "\n### Response:\n"
    ids = tok.encode(contextual, add_special_tokens=False)
    ids = _strip_leading_ws_ids(tok, ids)

    if len(ids) == 0:
        # Fallback: string anchor is more forgiving than empty token list
        return "### Response:"
    return ids


# ----------------------------- Main -------------------------------------
def main(cfg_path: str):
    cfg = load_yaml(cfg_path)
    data_cfg = load_yaml(cfg["paths"]["data_config"])

    # Normalize the YAML block to a consistent template
    template = textwrap.dedent(data_cfg["template"]).lstrip("\n")

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Model (QLoRA 4-bit)
    bnb = build_bnb(cfg)
    dtype = torch.bfloat16 if cfg["train"].get("bf16", True) else torch.float16

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # Only set device_map for multi-GPU + QLoRA (bitsandbytes) runs.
    # QLoRA per-rank device mapping (DDP only)
    device_map = None
    if world_size > 1 and bnb is not None:
        device_map = {"": local_rank}

    # ensure CUDA device is bound before TrainingArguments BF16 probe
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        _ = torch.cuda.current_device() 

    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["base_model"],
            dtype=dtype,
            quantization_config=bnb,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["base_model"],
            torch_dtype=dtype,
            quantization_config=bnb,
            device_map=device_map,
            low_cpu_mem_usage=True,
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
    data_files = {"train": "data/processed/train.jsonl", "validation": "data/processed/val.jsonl"}
    ds = load_dataset("json", data_files=data_files)

    def to_text(example):
        return {"text": format_example(example, template)}

    train_text = ds["train"].map(to_text, remove_columns=ds["train"].column_names)
    val_text   = ds["validation"].map(to_text, remove_columns=ds["validation"].column_names)

    # Robust anchor for response-only loss
    response_template = build_response_anchor(tok, template)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,  # accepts List[int] or str
        tokenizer=tok,
    )

    # Training args (stable optimizer for QLoRA; gradient clipping)
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
        save_total_limit=3,
        report_to=["wandb"] if os.environ.get("WANDB_MODE", "online") != "disabled" else [],
        run_name=cfg["run_name"],
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        ddp_find_unused_parameters=False,
        load_best_model_at_end=cfg["train"].get("load_best_model_at_end", True),
        metric_for_best_model=cfg["train"].get("metric_for_best_model", "eval_loss"),
        greater_is_better=cfg["train"].get("greater_is_better", False),
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
    trainer.add_callback(CSVLoggerCallback(cfg["paths"]["history_csv"]))

    trainer.train()

    # Save PEFT adapter + tokenizer
    os.makedirs(cfg["output_dir"], exist_ok=True)
    trainer.model.save_pretrained(os.path.join(cfg["output_dir"], "best"))
    trainer.tokenizer.save_pretrained(os.path.join(cfg["output_dir"], "best"))
    print("Training complete. Live CSV at:", cfg["paths"]["history_csv"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
