"""
Baseline Federated Learning experiment with FEMNIST.

This script runs a complete FL experiment with:
- Configurable number of clients and rounds
- Optional data poisoning attack
- Per-round metric collection
- CSV output for plotting

Usage:
    python experiments/baseline_fl.py --clients 10 --rounds 10
    python experiments/baseline_fl.py --clients 10 --rounds 10 --attack --malicious 2
"""

import argparse
import csv
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import flwr as fl
from cfg.load_config import load as load_cfg
from framework.client import create_client_fn
from framework.server import create_strategy


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Baseline FL experiment with FEMNIST")
    
    parser.add_argument(
        "--clients",
        type=int,
        default=None,
        help="Number of clients (default: from config)"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Number of FL rounds (default: from config)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Local training epochs per round (default: 1)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--attack",
        action="store_true",
        help="Enable data poisoning attack"
    )
    parser.add_argument(
        "--malicious",
        type=int,
        default=2,
        help="Number of malicious clients (default: 2)"
    )
    parser.add_argument(
        "--poison-rate",
        type=float,
        default=0.1,
        help="Poisoning rate for malicious clients (default: 0.1)"
    )
    parser.add_argument(
        "--defense",
        type=str,
        default="fedavg",
        choices=["fedavg", "pid", "krum", "bulyan", "rfa"],
        help="Defense strategy (default: fedavg)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV file (default: results/<experiment_name>.csv)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: from config)"
    )
    
    return parser.parse_args()


def save_metrics_to_csv(metrics_list, output_path):
    """
    Save collected metrics to CSV file.
    
    Args:
        metrics_list: List of metrics dictionaries from server
        output_path: Path to output CSV file
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Flatten metrics for CSV
    rows = []
    for round_data in metrics_list:
        round_summary = round_data["round_summary"]
        client_metrics = round_data.get("client_metrics", [])
        eval_metrics = round_data.get("eval_metrics", [])
        
        # Create a map of client_id -> eval metrics
        eval_map = {m["client_id"]: m for m in eval_metrics}
        
        # Write per-client metrics
        for cm in client_metrics:
            client_id = cm["client_id"]
            eval_data = eval_map.get(client_id, {})
            
            rows.append({
                "round": round_summary["round"],
                "client_id": client_id,
                "is_malicious": cm["is_malicious"],
                "train_loss": cm["train_loss"],
                "eval_loss": eval_data.get("eval_loss", 0.0),
                "accuracy": eval_data.get("accuracy", 0.0),
                "num_samples": cm["num_samples"]
            })
        
        # Also add round-wise averages
        rows.append({
            "round": round_summary["round"],
            "client_id": "avg",
            "is_malicious": 0,
            "train_loss": round_summary["avg_train_loss"],
            "eval_loss": round_summary.get("avg_eval_loss", 0.0),
            "accuracy": round_summary.get("avg_accuracy", 0.0),
            "num_samples": round_summary["total_samples"]
        })
    
    # Write to CSV
    if rows:
        fieldnames = ["round", "client_id", "is_malicious", "train_loss", "eval_loss", "accuracy", "num_samples"]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"✓ Metrics saved to {output_path}")


def main():
    """Run baseline FL experiment."""
    args = parse_args()
    
    # Load configuration
    cfg = load_cfg()
    
    # Use CLI args or fall back to config
    num_clients = args.clients if args.clients is not None else cfg.get("clients", 10)
    num_rounds = args.rounds if args.rounds is not None else cfg.get("rounds", 5)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    
    # Determine malicious clients
    malicious_clients = []
    if args.attack:
        malicious_clients = list(range(args.malicious))  # First N clients are malicious
        print(f"⚠️  Attack enabled: {len(malicious_clients)} malicious clients")
    
    # Determine output path
    if args.out:
        output_path = args.out
    else:
        experiment_name = "baseline_attack" if args.attack else "baseline_noattack"
        output_path = os.path.join(cfg["results_dir"], f"{experiment_name}.csv")
    
    print(f"\n{'='*60}")
    print(f"Federated Learning Experiment")
    print(f"{'='*60}")
    print(f"Clients: {num_clients}")
    print(f"Rounds: {num_rounds}")
    print(f"Epochs per round: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Defense: {args.defense}")
    print(f"Malicious clients: {len(malicious_clients)}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")
    
    # Create client factory
    client_fn = create_client_fn(
        num_partitions=num_clients,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        device="cpu",  # TODO: Add GPU support
        malicious_clients=malicious_clients,
        poison_rate=args.poison_rate,
        seed=seed
    )
    
    # Create strategy
    strategy = create_strategy(
        defense_type=args.defense,
        metrics_file=output_path
    )
    
    # Configure and start simulation
    print("Starting Flower simulation...")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0}
    )
    
    print("\n✓ Simulation complete!")
    
    # Save metrics
    metrics = strategy.get_metrics()
    save_metrics_to_csv(metrics, output_path)
    
    # Print final summary
    if metrics:
        final_round = metrics[-1]["round_summary"]
        print(f"\nFinal Results (Round {final_round['round']}):")
        print(f"  Average accuracy: {final_round.get('avg_accuracy', 0.0):.4f}")
        print(f"  Average eval loss: {final_round.get('avg_eval_loss', 0.0):.4f}")
        print(f"  Average train loss: {final_round.get('avg_train_loss', 0.0):.4f}")


if __name__ == "__main__":
    main()
