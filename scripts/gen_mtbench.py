"""Thin wrapper around FastChat's MT-Bench generator.
Requires: pip install fastchat
Outputs jsonl to eval/mtbench/answers_*.jsonl by symlinking/copying FastChat's default output.
"""
import argparse, os, shutil, subprocess, sys


def gen(model_path: str, model_id: str, out_path: str):
    cmd = [
        sys.executable, "-m", "fastchat.llm_judge.gen_model_answer",
        "--model-path", model_path,
        "--model-id", model_id,
        "--bench-name", "mt_bench",
        "--max-new-token", "512",
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    # FastChat writes to data/mt_bench/model_answer/{model_id}.jsonl
    src = os.path.join("data", "mt_bench", "model_answer", f"{model_id}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    shutil.copyfile(src, out_path)
    print("Copied:", src, "->", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    gen(args.model_path, args.model_id, args.out)