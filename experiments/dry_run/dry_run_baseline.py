# experiments/dry_run/dry_run_baseline.py
import argparse, csv, os
from cfg.load_config import load as load_cfg

# ---- config ----
CFG = load_cfg()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=CFG.get("clients", 10))
    ap.add_argument("--rounds", type=int, default=CFG.get("rounds", 5))
    ap.add_argument("--out", type=str, default=os.path.join(CFG.get("results_dir", "experiments/results"), "baseline.csv"))
    args = ap.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Dummy baseline: write simple accuracy/loss that changes per round
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "avg_acc", "avg_loss"])
        for r in range(1, args.rounds + 1):
            avg_acc = 0.50 + 0.02 * r     # e.g., improves 2% per round
            avg_loss = 1.00 - 0.10 * r    # e.g., drops 0.1 per round
            w.writerow([r, avg_acc, avg_loss])

    print(f"✅ Wrote results CSV → {args.out}")

if __name__ == "__main__":
    main()
