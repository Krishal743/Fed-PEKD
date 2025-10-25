# FILE: data.py

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_data(num_clients, alpha=0.5, max_samples_per_client=1000):
    """
    Downloads FashionMNIST, splits it with a Dirichlet distribution,
    and then subsamples to create a few-shot scenario.
    """
    print(f"Loading data, splitting with Dirichlet (alpha={alpha}), and subsampling to {max_samples_per_client} samples/client...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    num_classes = 10
    labels = np.array(train_dataset.targets)
    client_data_indices = [[] for _ in range(num_clients)]
    
    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        client_split = np.split(idx_k, proportions)
        for i in range(num_clients):
            client_data_indices[i].extend(client_split[i])

    # ** THE KEY CHANGE IS HERE **
    # Subsample the indices for each client to create a few-shot setting
    for i in range(num_clients):
        np.random.shuffle(client_data_indices[i])
        client_data_indices[i] = client_data_indices[i][:max_samples_per_client]

    # Create the final DataLoaders
    train_loaders = []
    for i in range(num_clients):
        client_subset = Subset(train_dataset, client_data_indices[i])
        loader = DataLoader(client_subset, batch_size=32, shuffle=True)
        train_loaders.append(loader)
        
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    print("Data loaded and split successfully.")
    return train_loaders, test_loader