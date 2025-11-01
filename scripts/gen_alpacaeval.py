"""
Generate outputs for AlpacaEval 2 using a local HF model (base or merged R2).
Saves JSONL with keys: instruction, output, generator.
"""
import argparse, json, os
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate(model_path: str, out_path: str, max_new_tokens: int = 512, temperature: float = 0.2):
    eval_set = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, device_map="auto")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map="auto")
    model.eval()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in eval_set:
            instr = ex["instruction"].strip()
            prompt = instr  # AlpacaEval eval set already merges input
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0),
                    temperature=temperature,
                    eos_token_id=tok.eos_token_id,
                )
            out = tok.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            rec = {"instruction": instr, "output": out, "generator": os.path.basename(model_path)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("Wrote:", out_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model path (e.g., base or merged R2)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()
    generate(args.model, args.out, args.max_new_tokens, args.temperature)
