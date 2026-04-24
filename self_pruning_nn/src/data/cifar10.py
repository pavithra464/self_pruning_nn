import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from typing import Tuple

def get_dataloaders(data_dir: str, batch_size: int, num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    """
    Sets up CIFAR-10 data loaders with standard augmentation and normalization.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Standard CIFAR-10 normalization
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    
    # Assertions for sanity
    assert len(trainset) == 50000, f"Expected 50000 training examples, got {len(trainset)}"
    assert len(testset) == 10000, f"Expected 10000 test examples, got {len(testset)}"

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader

def get_tiny_dataloader(data_dir: str, batch_size: int) -> DataLoader:
    """Creates a very small dataloader for smoke-testing overfit."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    tiny_subset = Subset(trainset, indices=list(range(min(batch_size, len(trainset)))))
    return DataLoader(tiny_subset, batch_size=batch_size, shuffle=False)
