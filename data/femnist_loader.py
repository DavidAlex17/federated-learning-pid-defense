"""
FEMNIST data loader for Federated Learning experiments.

Uses flwr-datasets to load and partition FEMNIST data across clients.
FEMNIST has 62 classes (digits + uppercase + lowercase letters) and is
naturally non-IID when partitioned by user.
"""

from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
import numpy as np
from PIL import Image


def load_femnist_partition(
    partition_id: int,
    num_partitions: int = 10,
    batch_size: int = 32,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Load FEMNIST data for a specific client partition.
    
    Args:
        partition_id: Client ID (0 to num_partitions-1)
        num_partitions: Total number of clients
        batch_size: Batch size for DataLoader
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Load federated FEMNIST dataset with IID partitioning
    # Note: FEMNIST is naturally non-IID by writer, but we use IID partitioner
    # to distribute the dataset evenly across simulated clients
    fds = FederatedDataset(
        dataset="flwrlabs/femnist",
        partitioners={
            "train": IidPartitioner(num_partitions=num_partitions),
        }
    )
    
    # Load this client's partition
    partition = fds.load_partition(partition_id, split="train")
    
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


def apply_label_flipping_poison(
    data_loader: DataLoader,
    poison_rate: float = 0.1,
    source_class: int = 0,
    target_class: int = 1,
    seed: Optional[int] = None
) -> DataLoader:
    """Simple label-flip poisoning wrapper around an existing DataLoader.

    This implementation wraps the original ``data_loader`` and, on-the-fly,
    flips a fixed fraction of labels from ``source_class`` to ``target_class``
    for each batch. Images are left unchanged.

    Notes:
        - This keeps the underlying dataset intact and only perturbs labels
          as they are yielded to the training loop.
        - The same seed yields a reproducible mask for each epoch because we
          use ``np.random.RandomState`` with the provided seed.
    """
    if poison_rate <= 0.0:
        return data_loader

    rng = np.random.RandomState(seed) if seed is not None else np.random

    class PoisonedDataLoader:
        def __init__(self, base_loader: DataLoader):
            self.base_loader = base_loader

        def __iter__(self):
            for batch in self.base_loader:
                images = batch["image"]
                labels = batch["label"].clone()

                mask = (labels == source_class)
                if poison_rate < 1.0:
                    mask_indices = mask.nonzero(as_tuple=True)[0]
                    if len(mask_indices) > 0:
                        num_to_flip = int(len(mask_indices) * poison_rate)
                        if num_to_flip > 0:
                            selected = rng.choice(mask_indices.cpu().numpy(), size=num_to_flip, replace=False)
                            selected = torch.tensor(selected, dtype=torch.long, device=labels.device)
                            mask = torch.zeros_like(labels, dtype=torch.bool)
                            mask[selected] = True

                labels[mask] = target_class
                yield {"image": images, "label": labels}

        def __len__(self):
            return len(self.base_loader)

    return PoisonedDataLoader(data_loader)


def get_femnist_info():
    """
    Get FEMNIST dataset information.
    
    Returns:
        Dictionary with dataset metadata
    """
    return {
        "num_classes": 62,  # 10 digits + 26 uppercase + 26 lowercase
        "input_shape": (1, 28, 28),  # Grayscale 28x28 images
        "dataset_name": "FEMNIST",
        "description": "Federated Extended MNIST (handwritten characters)"
    }


if __name__ == "__main__":
    # Quick test
    print("Testing FEMNIST loader...")
    info = get_femnist_info()
    print(f"Dataset: {info['dataset_name']}")
    print(f"Classes: {info['num_classes']}")
    print(f"Input shape: {info['input_shape']}")
    
    print("\nLoading partition 0 of 10...")
    train_loader, test_loader = load_femnist_partition(partition_id=0, num_partitions=10)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Check first batch
    for batch in train_loader:
        images = batch["image"]
        labels = batch["label"]
        print(f"\nFirst batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Sample labels: {labels[:5]}")
        break
    
    print("\n✓ FEMNIST loader test complete!")
