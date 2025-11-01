# Same trainer as train_lora.py, just a thin wrapper so using a separate sbatch.
# FSDP is controlled entirely by the accelerate config used at launch.
from train_lora import main

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)