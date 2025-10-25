# FILE: data.py

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_data(num_clients, alpha=0.5, max_samples_per_client=1000):
    """
    Downloads FashionMNIST, splits it with a Dirichlet distribution,
    and then intelligently subsamples to create a few-shot scenario
    that guarantees class representation.
    """
    print(f"Loading data, splitting with Dirichlet (alpha={alpha}), and (SMART) subsampling to {max_samples_per_client} samples/client...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    num_classes = 10
    labels = np.array(train_dataset.targets)
    
    # client_data_indices will be a list of dictionaries.
    # Each dict maps: class_index -> list_of_image_indices
    client_data_indices = [{} for _ in range(num_clients)]
    
    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        client_split = np.split(idx_k, proportions)
        for i in range(num_clients):
            client_data_indices[i][k] = client_split[i]

    # ** THE KEY CHANGE IS HERE **
    # Intelligently build the final subset for each client
    
    final_client_indices = [[] for _ in range(num_clients)]
    
    # We set a max of 100 samples per class per client
    # 100 * 10 classes = 1000 total samples (matching our max_samples_per_client)
    max_samples_per_class = max_samples_per_client // num_classes 
    
    for i in range(num_clients):
        for k in range(num_classes):
            # Get the indices for this class
            class_indices = client_data_indices[i][k]
            
            # If the client has more than our max, subsample
            if len(class_indices) > max_samples_per_class:
                np.random.shuffle(class_indices)
                class_indices = class_indices[:max_samples_per_class]
                
            # Add these indices to the client's final list
            final_client_indices[i].extend(class_indices)

    # Create the final DataLoaders
    train_loaders = []
    for i in range(num_clients):
        client_subset = Subset(train_dataset, final_client_indices[i])
        loader = DataLoader(client_subset, batch_size=32, shuffle=True)
        train_loaders.append(loader)
        
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    print("Data loaded and split successfully.")
    return train_loaders, test_loader