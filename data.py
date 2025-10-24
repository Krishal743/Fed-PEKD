import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_data(num_clients):
    """
    Downloads FashionMNIST and splits it into non-IID partitions for clients.
    """
    # Transformation to normalize the image data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Download the training and test datasets
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # This is a simple way to create a non-IID split.
    # Each client will get data from only a few classes.
    # Example: Client 0 gets digits 0, 1. Client 1 gets 2, 3 etc.
    client_data_dict = {i: [] for i in range(num_clients)}
    labels_per_client = 10 // num_clients
    
    for image_idx, (_, label) in enumerate(train_dataset):
        client_id = label // labels_per_client
        if client_id < num_clients:
            client_data_dict[client_id].append(image_idx)

    train_loaders = []
    for i in range(num_clients):
        subset_indices = client_data_dict[i]
        # We only use a few samples, as described in the paper (10-50 per class) [cite: 9, 14, 34]
        # Let's take a maximum of 500 samples per client for a start
        client_subset = Subset(train_dataset, subset_indices[:500])
        loader = DataLoader(client_subset, batch_size=32, shuffle=True)
        train_loaders.append(loader)
        
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    print("Data loaded and split for", num_clients, "clients.")
    return train_loaders, test_loader