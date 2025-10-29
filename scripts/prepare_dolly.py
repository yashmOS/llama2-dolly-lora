import argparse, os, json, random
from datasets import load_dataset, Dataset
from tqdm import tqdm

TEMPLATE = """### Instruction:
{instruction}

### Context:
{context}

### Response:
{response}
"""

def format_row(ex):
    inst = (ex.get("instruction") or "").strip()
    ctx = (ex.get("context") or "").strip()
    rsp = (ex.get("response") or "").strip()
    return {
        "text": TEMPLATE.format(instruction=inst, context=ctx, response=rsp)
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset_ratio", type=float, default=0.4, help="<=0.5 per requirement")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="data/processed")
    args = ap.parse_args()

    assert 0 < args.subset_ratio <= 0.5, "subset_ratio must be in (0, 0.5]"

    os.makedirs(args.outdir, exist_ok=True)
    ds = load_dataset("databricks/databricks-dolly-15k")
    base = ds["train"]  # dolly-15k is provided as a single split

    # Drop empty responses
    base = base.filter(lambda ex: ex.get("response") is not None and len(ex["response"].strip()) > 0)

    # Shuffle deterministically
    base = base.shuffle(seed=args.seed)

    # 80/20 split, then 20 -> 10/10 val/test
    split = base.train_test_split(test_size=0.2, seed=args.seed)
    train_full = split["train"]
    temp = split["test"]
    more = temp.train_test_split(test_size=0.5, seed=args.seed)
    val_full = more["train"]
    test_full = more["test"]

    def subsample(dset, ratio):
        n = int(len(dset) * ratio)
        idx = list(range(len(dset)))
        random.Random(args.seed).shuffle(idx)
        idx = idx[:n]
        return dset.select(idx)

    train = subsample(train_full, args.subset_ratio)
    val = subsample(val_full, min(args.subset_ratio * 1.0, 0.5))  # keep ≤50% consistently
    test = subsample(test_full, min(args.subset_ratio * 1.0, 0.5))

    # Format with template
    train = train.map(format_row, remove_columns=[c for c in train.column_names if c != "text"])
    val   = val.map(format_row,   remove_columns=[c for c in val.column_names if c != "text"])
    test  = test.map(format_row,  remove_columns=[c for c in test.column_names if c != "text"])

    # Save jsonl
    def save_jsonl(dset, path):
        with open(path, "w", encoding="utf-8") as f:
            for r in dset:
                f.write(json.dumps({"text": r["text"]}, ensure_ascii=False) + "\n")

    save_jsonl(train, os.path.join(args.outdir, "train.jsonl"))
    save_jsonl(val,   os.path.join(args.outdir, "val.jsonl"))
    save_jsonl(test,  os.path.join(args.outdir, "test.jsonl"))

    # Small manifest to help the report
    manifest = {
        "seed": args.seed,
        "subset_ratio": args.subset_ratio,
        "sizes": {"train": len(train), "val": len(val), "test": len(test)}
    }
    with open(os.path.join(args.outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("Prepared Dolly-15K with template and subsampling.")
    print("Saved:", os.path.join(args.outdir, "train.jsonl"),
                 os.path.join(args.outdir, "val.jsonl"),
                 os.path.join(args.outdir, "test.jsonl"))

if __name__ == "__main__":
    main()
