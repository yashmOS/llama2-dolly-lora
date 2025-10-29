import argparse, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="e.g. meta-llama/Llama-2-7b-hf")
    ap.add_argument("--adapter_dir", required=True, help="outputs/checkpoints")
    ap.add_argument("--out_dir", default="outputs/merged")
    ap.add_argument("--cache_dir", default="./hf_cache")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load base in 16-bit for clean merge-to-fp16
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model = model.merge_and_unload()
    model.to(torch.float16)

    tok = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print("Merged model saved to:", args.out_dir)

if __name__ == "__main__":
    main()
