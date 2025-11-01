import argparse, json, os, random, yaml
from datasets import load_dataset


def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    ds_name = cfg["dataset"]
    seed = cfg["split"]["seed"]
    random.seed(seed)

    os.makedirs("data/processed", exist_ok=True)

    ds = load_dataset(ds_name)["train"]

    rows = []
    for ex in ds:
        instr = (ex.get("instruction") or "").strip()
        ctx = (ex.get("context") or "").strip()
        resp = (ex.get("response") or "").strip()
        if cfg.get("filters", {}).get("drop_empty_response", True) and (not resp):
            continue
        rows.append({"instruction": instr, "context": ctx, "response": resp})

    random.shuffle(rows)
    n = len(rows)
    n_train = int(n * cfg["split"]["train"])
    n_val = int(n * cfg["split"]["val"])

    splits = {
        "train": rows[:n_train],
        "val": rows[n_train:n_train + n_val],
        "test": rows[n_train + n_val :],
    }

    for name, data in splits.items():
        path = f"data/processed/{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(data)} -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)