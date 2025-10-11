# Teammate C

# experiments/dry_run/plot_metrics.py
import argparse, csv, os
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Ensure output directory exists
    os.makedirs(args.out, exist_ok=True)

    rounds, acc, loss = [], [], []
    with open(args.inp) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rounds.append(int(row["round"]))
            acc.append(float(row["avg_acc"]))
            loss.append(float(row["avg_loss"]))

    # Accuracy plot
    plt.figure()
    plt.plot(rounds, acc)
    plt.title("Accuracy vs Rounds")
    plt.xlabel("Round")
    plt.ylabel("Average Accuracy")
    plt.grid(True)
    acc_path = os.path.join(args.out, "acc.png")
    plt.savefig(acc_path, bbox_inches="tight")
    plt.close()

    # Loss plot
    plt.figure()
    plt.plot(rounds, loss)
    plt.title("Loss vs Rounds")
    plt.xlabel("Round")
    plt.ylabel("Average Loss")
    plt.grid(True)
    loss_path = os.path.join(args.out, "loss.png")
    plt.savefig(loss_path, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved plots → {acc_path} and {loss_path}")

if __name__ == "__main__":
    main()
