import argparse, os, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def load_model_and_tok(base_model, adapter_dir=None, cache_dir="./hf_cache"):
    tok = AutoTokenizer.from_pretrained(base_model, cache_dir=cache_dir, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Load 4-bit for inference to reduce VRAM
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(base_model, cache_dir=cache_dir, quantization_config=bnb_cfg, device_map="auto")
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_dir", required=True, help="Path to LoRA adapter dir (outputs/checkpoints)")
    ap.add_argument("--input_jsonl", required=True, help='JSONL with {"id":..., "prompt":...}')
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--cache_dir", default="./hf_cache")
    args = ap.parse_args()

    model, tok = load_model_and_tok(args.base_model, args.adapter_dir, args.cache_dir)

    with open(args.input_jsonl, "r", encoding="utf-8") as f_in, open(args.output_jsonl, "w", encoding="utf-8") as f_out:
        for line in f_in:
            rec = json.loads(line)
            prompt = rec["prompt"]
            ids = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **ids,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tok.eos_token_id
                )
            out_text = tok.decode(gen[0], skip_special_tokens=True)
            f_out.write(json.dumps({"id": rec["id"], "prompt": prompt, "response": out_text}, ensure_ascii=False) + "\n")

    print("Saved generations to", args.output_jsonl)

if __name__ == "__main__":
    main()
