import argparse, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main(base: str, adapter: str, out: str):
    os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(base, use_fast=True)

    dtype = torch.bfloat16
    try:
        model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype)

    model = PeftModel.from_pretrained(model, adapter)
    merged = model.merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    print("Merged model saved to:", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.base, args.adapter, args.out)
