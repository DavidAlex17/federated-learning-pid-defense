"""
Federated Learning server implementation using Flower.

This module defines the server-side aggregation strategies:
- FedAvg (baseline)
- PID-based defense (future)
- Krum, Bulyan, RFA (future)
- Trust-based filtering (future)
"""

from typing import List, Tuple, Dict, Optional, Callable
import numpy as np
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import (
    Parameters,
    Scalar,
    FitRes,
    EvaluateRes,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)


class FedAvgWithMetrics(FedAvg):
    """
    FedAvg strategy with enhanced metric collection.
    
    This is the baseline strategy that will be extended with
    Byzantine-robust defenses (PID, Krum, Bulyan, etc.) in future milestones.
    """
    
    def __init__(
        self,
        *args,
        metrics_file: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize FedAvg with metric collection.
        
        Args:
            metrics_file: Path to save per-round metrics (CSV)
            *args, **kwargs: Arguments for base FedAvg strategy
        """
        super().__init__(*args, **kwargs)
        self.metrics_file = metrics_file
        self.round_metrics = []
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[any, FitRes]],
        failures: List[Tuple[any, FitRes] | BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model updates from clients using FedAvg.
        
        This is where Byzantine-robust defenses will be added in future milestones:
        - PID-based filtering
        - Krum selection
        - Bulyan aggregation
        - RFA (Robust Federated Aggregation)
        - Trust-based filtering
        
        Args:
            server_round: Current round number
            results: List of (client_proxy, fit_result) tuples
            failures: List of failed clients
            
        Returns:
            Tuple of (aggregated_parameters, metrics_dict)
        """
        if not results:
            return None, {}
        
        # Extract metrics from clients
        client_metrics = []
        for client, fit_res in results:
            metrics = fit_res.metrics
            client_metrics.append({
                "round": server_round,
                "client_id": metrics.get("client_id", -1),
                "train_loss": metrics.get("train_loss", 0.0),
                "is_malicious": metrics.get("is_malicious", 0),
                "num_samples": fit_res.num_examples
            })
        
        # Compute round-wise aggregates
        total_samples = sum(m["num_samples"] for m in client_metrics)
        avg_train_loss = sum(
            m["train_loss"] * m["num_samples"] for m in client_metrics
        ) / total_samples if total_samples > 0 else 0.0
        
        round_summary = {
            "round": server_round,
            "num_clients": len(results),
            "num_failures": len(failures),
            "avg_train_loss": avg_train_loss,
            "total_samples": total_samples
        }
        
        # Store for later export
        self.round_metrics.append({
            "round_summary": round_summary,
            "client_metrics": client_metrics
        })
        
        # Call parent FedAvg aggregation
        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Add our metrics to the aggregated metrics
        aggregated_metrics["avg_train_loss"] = avg_train_loss
        
        return aggregated_params, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[any, EvaluateRes]],
        failures: List[Tuple[any, EvaluateRes] | BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation results from clients.
        
        Args:
            server_round: Current round number
            results: List of (client_proxy, evaluate_result) tuples
            failures: List of failed evaluations
            
        Returns:
            Tuple of (aggregated_loss, metrics_dict)
        """
        if not results:
            return None, {}
        
        # Extract evaluation metrics
        eval_metrics = []
        for client, eval_res in results:
            metrics = eval_res.metrics
            eval_metrics.append({
                "round": server_round,
                "client_id": metrics.get("client_id", -1),
                "accuracy": metrics.get("accuracy", 0.0),
                "eval_loss": eval_res.loss,
                "is_malicious": metrics.get("is_malicious", 0),
                "num_samples": eval_res.num_examples
            })
        
        # Compute weighted averages
        total_samples = sum(m["num_samples"] for m in eval_metrics)
        avg_accuracy = sum(
            m["accuracy"] * m["num_samples"] for m in eval_metrics
        ) / total_samples if total_samples > 0 else 0.0
        
        avg_eval_loss = sum(
            m["eval_loss"] * m["num_samples"] for m in eval_metrics
        ) / total_samples if total_samples > 0 else 0.0
        
        # Update stored metrics with evaluation results
        if self.round_metrics and self.round_metrics[-1]["round_summary"]["round"] == server_round:
            self.round_metrics[-1]["eval_metrics"] = eval_metrics
            self.round_metrics[-1]["round_summary"]["avg_accuracy"] = avg_accuracy
            self.round_metrics[-1]["round_summary"]["avg_eval_loss"] = avg_eval_loss
        
        # Call parent aggregation
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        # Add our metrics
        aggregated_metrics["avg_accuracy"] = avg_accuracy
        
        return aggregated_loss, aggregated_metrics
    
    def get_metrics(self) -> List[Dict]:
        """
        Get collected metrics from all rounds.
        
        Returns:
            List of per-round metrics dictionaries
        """
        return self.round_metrics


def create_strategy(
    defense_type: str = "fedavg",
    metrics_file: Optional[str] = None,
    **defense_params
) -> FedAvgWithMetrics:
    """
    Create a server aggregation strategy.
    
    Args:
        defense_type: Type of defense ('fedavg', 'pid', 'krum', 'bulyan', 'rfa')
        metrics_file: Path to save metrics
        **defense_params: Additional defense-specific parameters
        
    Returns:
        Server strategy instance
    """
    if defense_type == "fedavg":
        return FedAvgWithMetrics(
            fraction_fit=1.0,  # Use all available clients each round
            fraction_evaluate=1.0,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
            metrics_file=metrics_file
        )
    elif defense_type == "pid":
        # TODO: Implement PID-based strategy in future milestone
        raise NotImplementedError("PID defense not yet implemented")
    elif defense_type == "krum":
        # TODO: Implement Krum strategy in future milestone
        raise NotImplementedError("Krum defense not yet implemented")
    elif defense_type == "bulyan":
        # TODO: Implement Bulyan strategy in future milestone
        raise NotImplementedError("Bulyan defense not yet implemented")
    elif defense_type == "rfa":
        # TODO: Implement RFA strategy in future milestone
        raise NotImplementedError("RFA defense not yet implemented")
    else:
        raise ValueError(f"Unknown defense type: {defense_type}")