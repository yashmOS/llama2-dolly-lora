import argparse, json, os, subprocess, sys, yaml


def run(outputs_path: str, judge_cfg_path: str, out_json: str):
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    # Call the AlpacaEval CLI
    cmd = [
        "alpaca_eval",
        "--model_outputs", outputs_path,
        "--annotators_config", judge_cfg_path,
    ]
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr, file=sys.stderr)
        sys.exit(res.returncode)
    # The CLI writes results next to outputs. For convenience, copy the last line JSON to a file.
    # Users can also inspect the printed leaderboard.
    with open(out_json, "w") as f:
        f.write(res.stdout)
    print("Saved judge console output to:", out_json)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run(args.outputs, args.judge, args.out)