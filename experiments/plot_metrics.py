# Teammate C

# experiments/dry_run/plot_metrics.py
import argparse, csv, os
import matplotlib.pyplot as plt


def load_metrics(csv_path):
    """Load per-client and average metrics from baseline_fl CSV."""
    per_client = {}
    avg = {}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rnd = int(row["round"])
            cid = row["client_id"]
            is_mal = int(row["is_malicious"])
            train_loss = float(row["train_loss"])
            eval_loss = float(row["eval_loss"])
            acc = float(row["accuracy"])

            if cid == "avg":
                avg.setdefault("round", []).append(rnd)
                avg.setdefault("train_loss", []).append(train_loss)
                avg.setdefault("eval_loss", []).append(eval_loss)
                avg.setdefault("accuracy", []).append(acc)
            else:
                cid_int = int(cid)
                if cid_int not in per_client:
                    per_client[cid_int] = {
                        "round": [],
                        "train_loss": [],
                        "eval_loss": [],
                        "accuracy": [],
                        "is_malicious": is_mal,
                    }
                per_client[cid_int]["round"].append(rnd)
                per_client[cid_int]["train_loss"].append(train_loss)
                per_client[cid_int]["eval_loss"].append(eval_loss)
                per_client[cid_int]["accuracy"].append(acc)

    return per_client, avg


def plot_per_client(per_client, out_dir, metric_key, title):
    """Plot per-client curves, highlighting malicious clients."""
    plt.figure()
    for cid, data in per_client.items():
        rounds = data["round"]
        values = data[metric_key]
        is_mal = data["is_malicious"] == 1
        style = "r--" if is_mal else "b-"
        label = f"client {cid}{' (malicious)' if is_mal else ''}"
        plt.plot(rounds, values, style, alpha=0.7, label=label)

    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel(metric_key.replace("_", " ").title())
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.tight_layout()

    fname = f"per_client_{metric_key}.png"
    path = os.path.join(out_dir, fname)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def plot_average(avg, out_dir, metric_key, title):
    """Plot average metric over rounds."""
    plt.figure()
    rounds = avg.get("round", [])
    values = avg.get(metric_key, [])
    plt.plot(rounds, values, "k-o")
    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel(metric_key.replace("_", " ").title())
    plt.grid(True)
    plt.tight_layout()

    fname = f"avg_{metric_key}.png"
    path = os.path.join(out_dir, fname)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def main():
    ap = argparse.ArgumentParser(description="Plot FL metrics from baseline_fl CSV")
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV path")
    ap.add_argument("--out", required=True, help="Output directory for plots")
    args = ap.parse_args()

    # Ensure output directory exists
    os.makedirs(args.out, exist_ok=True)

    per_client, avg = load_metrics(args.inp)

    # Per-client plots
    pc_acc = plot_per_client(per_client, args.out, "accuracy", "Per-client Accuracy vs Rounds")
    pc_train = plot_per_client(per_client, args.out, "train_loss", "Per-client Train Loss vs Rounds")
    pc_eval = plot_per_client(per_client, args.out, "eval_loss", "Per-client Eval Loss vs Rounds")

    # Average plots
    avg_acc = plot_average(avg, args.out, "accuracy", "Average Accuracy vs Rounds")
    avg_train = plot_average(avg, args.out, "train_loss", "Average Train Loss vs Rounds")
    avg_eval = plot_average(avg, args.out, "eval_loss", "Average Eval Loss vs Rounds")

    print("Saved plots:")
    for p in [pc_acc, pc_train, pc_eval, avg_acc, avg_train, avg_eval]:
        print("  -", p)

if __name__ == "__main__":
    main()
