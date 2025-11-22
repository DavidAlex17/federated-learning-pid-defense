"""
CNN model for FEMNIST classification.

FEMNIST has 62 classes (10 digits + 26 uppercase + 26 lowercase letters)
and uses 28x28 grayscale images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FEMNISTNet(nn.Module):
    """
    Convolutional Neural Network for FEMNIST classification.
    
    Architecture:
    - Conv1: 1 -> 32 channels, 5x5 kernel
    - MaxPool: 2x2
    - Conv2: 32 -> 64 channels, 5x5 kernel
    - MaxPool: 2x2
    - FC1: 64*4*4 -> 512
    - FC2: 512 -> 62 (num classes)
    """
    
    def __init__(self, num_classes: int = 62):
        super(FEMNISTNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # After conv1 + pool: 28x28 -> 14x14
        # After conv2 + pool: 14x14 -> 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))  # (N, 32, 14, 14)
        
        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))  # (N, 64, 7, 7)
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)  # (N, 3136)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def train_model(
    model: nn.Module,
    train_loader,
    epochs: int = 1,
    learning_rate: float = 0.01,
    device: str = "cpu"
) -> dict:
    """
    Train the model for one FL round (local training).
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        epochs: Number of local epochs
        learning_rate: Learning rate
        device: Device to train on ('cpu' or 'cuda')
        
    Returns:
        Dictionary with training metrics
    """
    model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    total_loss = 0.0
    total_samples = 0
    
    for epoch in range(epochs):
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    
    return {
        "loss": avg_loss,
        "samples": total_samples
    }


def evaluate_model(
    model: nn.Module,
    test_loader,
    device: str = "cpu"
) -> dict:
    """
    Evaluate the model.
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for test data
        device: Device to evaluate on ('cpu' or 'cuda')
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total_samples += images.size(0)
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "samples": total_samples
    }


if __name__ == "__main__":
    # Quick test
    print("Testing FEMNIST CNN model...")
    
    model = FEMNISTNet(num_classes=62)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    dummy_input = torch.randn(4, 1, 28, 28)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    print("\n✓ Model test complete!")