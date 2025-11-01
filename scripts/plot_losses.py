import argparse, os
import pandas as pd
import matplotlib.pyplot as plt


def plot(paths, out_png):
    plt.figure(figsize=(8,5))
    for run in paths:
        csv = os.path.join(run, "logs", "history.csv")
        if not os.path.exists(csv):
            print("Skip, not found:", csv)
            continue
        df = pd.read_csv(csv)
        # filter training/eval losses
        if "loss" in df:
            plt.plot(df.index, df["loss"].interpolate(), label=f"{os.path.basename(run)}: train")
        if "eval_loss" in df:
            plt.plot(df.index, df["eval_loss"].interpolate(), label=f"{os.path.basename(run)}: eval")
    plt.title("Training / Validation Loss")
    plt.xlabel("log step")
    plt.ylabel("loss")
    plt.legend()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print("Saved:", out_png)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    plot(args.runs, args.out)