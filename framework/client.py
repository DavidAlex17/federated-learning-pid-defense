"""
Federated Learning client implementation using Flower.

This module defines the FlowerClient class that handles:
- Local training on client data
- Model evaluation
- Data poisoning (when enabled)
"""

from typing import Dict, Tuple, List
import torch
from flwr.client import NumPyClient
from model.cnn_model import train_model, evaluate_model
# Use subset loader for fast testing, femnist_loader for production
from data.femnist_subset_loader import load_femnist_partition
from data.femnist_loader import apply_label_flipping_poison


class FEMNISTClient(NumPyClient):
    """
    Flower client for FEMNIST federated learning.
    
    Each client:
    - Has a unique partition of FEMNIST data
    - Trains locally for a specified number of epochs
    - Can optionally apply data poisoning
    """
    
    def __init__(
        self,
        client_id: int,
        model: torch.nn.Module,
        num_partitions: int = 10,
        epochs: int = 1,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        device: str = "cpu",
        is_malicious: bool = False,
        poison_rate: float = 0.0,
        seed: int = 42
    ):
        """
        Initialize a Flower client.
        
        Args:
            client_id: Unique client identifier
            model: PyTorch model to train
            num_partitions: Total number of clients in federation
            epochs: Number of local training epochs per round
            learning_rate: Learning rate for local training
            batch_size: Batch size for data loaders
            device: Device to use ('cpu' or 'cuda')
            is_malicious: Whether this client is malicious/poisoned
            poison_rate: Rate of poisoning (if malicious)
            seed: Random seed for reproducibility
        """
        self.client_id = client_id
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device
        self.is_malicious = is_malicious
        self.poison_rate = poison_rate
        
        # Load client's data partition
        self.train_loader, self.test_loader = load_femnist_partition(
            partition_id=client_id,
            num_partitions=num_partitions,
            batch_size=batch_size,
            seed=seed
        )

        # Apply simple label-flipping poisoning for malicious clients
        if self.is_malicious and self.poison_rate > 0.0:
            # Flip a subset of labels from class 0 to class 1 by default.
            self.train_loader = apply_label_flipping_poison(
                self.train_loader,
                poison_rate=self.poison_rate,
                source_class=0,
                target_class=1,
                seed=seed + client_id
            )
    
    def get_parameters(self, config: Dict) -> List[bytes]:
        """
        Get model parameters as a list of numpy arrays.
        
        Args:
            config: Configuration dictionary from server
            
        Returns:
            List of model parameters
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[bytes]) -> None:
        """
        Set model parameters from a list of numpy arrays.
        
        Args:
            parameters: List of model parameters from server
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(
        self, parameters: List[bytes], config: Dict
    ) -> Tuple[List[bytes], int, Dict]:
        """
        Train the model on local data.
        
        Args:
            parameters: Model parameters from server
            config: Configuration dictionary from server
            
        Returns:
            Tuple of (updated_parameters, num_samples, metrics)
        """
        # Update local model with global parameters
        self.set_parameters(parameters)
        
        # Train locally
        train_metrics = train_model(
            model=self.model,
            train_loader=self.train_loader,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            device=self.device
        )
        
        # Return updated model parameters and metrics
        updated_parameters = self.get_parameters(config={})
        num_samples = train_metrics["samples"]
        
        metrics = {
            "client_id": self.client_id,
            "train_loss": train_metrics["loss"],
            "is_malicious": int(self.is_malicious)
        }
        
        return updated_parameters, num_samples, metrics
    
    def evaluate(
        self, parameters: List[bytes], config: Dict
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate the model on local test data.
        
        Args:
            parameters: Model parameters from server
            config: Configuration dictionary from server
            
        Returns:
            Tuple of (loss, num_samples, metrics)
        """
        # Update local model with global parameters
        self.set_parameters(parameters)
        
        # Evaluate on local test set
        eval_metrics = evaluate_model(
            model=self.model,
            test_loader=self.test_loader,
            device=self.device
        )
        
        loss = eval_metrics["loss"]
        num_samples = eval_metrics["samples"]
        
        metrics = {
            "client_id": self.client_id,
            "accuracy": eval_metrics["accuracy"],
            "is_malicious": int(self.is_malicious)
        }
        
        return loss, num_samples, metrics


def create_client_fn(
    num_partitions: int,
    epochs: int = 1,
    learning_rate: float = 0.01,
    batch_size: int = 32,
    device: str = "cpu",
    malicious_clients: List[int] = None,
    poison_rate: float = 0.1,
    seed: int = 42
):
    """
    Factory function to create client instances for Flower simulation.
    
    Args:
        num_partitions: Total number of clients
        epochs: Local training epochs per round
        learning_rate: Learning rate
        batch_size: Batch size
        device: Device to use
        malicious_clients: List of client IDs that should be malicious
        poison_rate: Poisoning rate for malicious clients
        seed: Random seed
        
    Returns:
        Client factory function
    """
    if malicious_clients is None:
        malicious_clients = []
    
    def client_fn(cid: str) -> FEMNISTClient:
        """Create a client instance."""
        from model.cnn_model import FEMNISTNet
        
        client_id = int(cid)
        model = FEMNISTNet(num_classes=62)
        is_malicious = client_id in malicious_clients
        
        return FEMNISTClient(
            client_id=client_id,
            model=model,
            num_partitions=num_partitions,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            device=device,
            is_malicious=is_malicious,
            poison_rate=poison_rate if is_malicious else 0.0,
            seed=seed
        )
    
    return client_fn