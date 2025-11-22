"""
FEMNIST subset loader for faster experimentation.

Uses a small subset of FEMNIST for quick testing and development.
Switch to full femnist_loader.py for production runs.
"""

from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader, Subset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
import numpy as np
from PIL import Image


def load_femnist_partition(
    partition_id: int,
    num_partitions: int = 10,
    batch_size: int = 32,
    seed: int = 42,
    samples_per_client: int = 500  # Limit samples for faster training
) -> Tuple[DataLoader, DataLoader]:
    """
    Load a SUBSET of FEMNIST data for a specific client partition.
    
    This is for FAST TESTING. Use full dataset for actual experiments.
    
    Args:
        partition_id: Client ID (0 to num_partitions-1)
        num_partitions: Total number of clients
        batch_size: Batch size for DataLoader
        seed: Random seed for reproducibility
        samples_per_client: Max samples per client (default: 500 for speed)
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Load federated FEMNIST dataset with IID partitioning
    fds = FederatedDataset(
        dataset="flwrlabs/femnist",
        partitioners={
            "train": IidPartitioner(num_partitions=num_partitions),
        }
    )
    
    # Load this client's partition
    partition = fds.load_partition(partition_id, split="train")
    
    # Take only a subset for faster training
    if len(partition) > samples_per_client:
        np.random.seed(seed + partition_id)
        indices = np.random.choice(len(partition), samples_per_client, replace=False)
        partition = partition.select(indices)
    
    # Split into train/test (80/20)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=seed)
    
    # Convert to PyTorch format
    def apply_transforms(batch):
        """Transform batch for PyTorch: normalize images and convert to tensors."""
        transformed_images = []
        for img in batch["image"]:
            # Convert PIL Image to numpy array
            if isinstance(img, Image.Image):
                img_array = np.array(img)
            else:
                img_array = img
            
            # Convert to tensor and normalize
            img_tensor = torch.tensor(img_array, dtype=torch.float32).reshape(1, 28, 28) / 255.0
            transformed_images.append(img_tensor)
        
        batch["image"] = transformed_images
        # FEMNIST uses "character" as the label column name
        batch["label"] = [torch.tensor(label, dtype=torch.long) for label in batch["character"]]
        return batch
    
    # Apply transforms
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    
    # Custom collate function to properly batch tensors
    def collate_fn(batch):
        """Collate function to stack tensors into batches."""
        images = torch.stack([item["image"] for item in batch])
        labels = torch.stack([item["label"] for item in batch])
        return {"image": images, "label": labels}
    
    # Create DataLoaders
    train_loader = DataLoader(
        partition_train_test["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        partition_train_test["test"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, test_loader


def get_femnist_info():
    """
    Get FEMNIST dataset information.
    
    Returns:
        Dictionary with dataset metadata
    """
    return {
        "num_classes": 62,  # 10 digits + 26 uppercase + 26 lowercase
        "input_shape": (1, 28, 28),  # Grayscale 28x28 images
        "dataset_name": "FEMNIST (subset)",
        "description": "Federated Extended MNIST subset for fast testing"
    }


if __name__ == "__main__":
    # Quick test
    print("Testing FEMNIST subset loader...")
    info = get_femnist_info()
    print(f"Dataset: {info['dataset_name']}")
    print(f"Classes: {info['num_classes']}")
    print(f"Input shape: {info['input_shape']}")
    
    print("\nLoading partition 0 of 10 (max 500 samples)...")
    train_loader, test_loader = load_femnist_partition(
        partition_id=0, 
        num_partitions=10,
        samples_per_client=500
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Check first batch
    for batch in train_loader:
        images = batch["image"]
        labels = batch["label"]
        print(f"\nFirst batch - images shape: {images.shape}")
        print(f"First batch - labels shape: {labels.shape}")
        print(f"Sample labels: {labels[:5]}")
        break
    
    print("\n✓ FEMNIST subset loader test complete!")
