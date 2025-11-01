import argparse, json, os, subprocess, sys


def judge(model_id: str, judge_model: str, out_path: str):
    # Run FastChat MT-bench judgment (single-answer grading)
    cmd = [
        sys.executable, "-m", "fastchat.llm_judge.gen_judgment",
        "--model-list", model_id,
        "--bench-name", "mt_bench",
        "--judge-model", judge_model,
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    # FastChat writes to data/mt_bench/model_judgment/{model_id}_{judge_model}.jsonl
    src = os.path.join("data", "mt_bench", "model_judgment", f"{model_id}_{judge_model}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # For simplicity, just copy raw judgments; FastChat also has a result summarizer.
    import shutil
    shutil.copyfile(src, out_path)
    print("Copied judge output to:", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--judge", required=True, help="e.g., gpt-4.1-mini or gpt-4.1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    judge(args.model_id, args.judge, args.out)