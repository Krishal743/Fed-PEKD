import copy
from data import get_data
from models import SimpleMLP
from client import Client
from server import Server

# In main_fedavg.py

import copy
import torch
from data import get_data
from models import SimpleMLP
from client import Client
from server import Server # We'll just use its evaluate() method

def federated_averaging(server_model, client_models):
    """
    Performs the Federated Averaging algorithm.
    The server's global model is updated by averaging the weights of the client models.
    """
    # Get the state dictionary of the server model
    global_dict = server_model.state_dict()
    
    # Set all weights in the global model to zero
    for key in global_dict:
        global_dict[key] = torch.zeros_like(global_dict[key])
    
    # Sum up the weights from all client models
    for client_model in client_models:
        for key in global_dict:
            global_dict[key] += client_model.state_dict()[key]
            
    # Divide by the number of clients to get the average
    for key in global_dict:
        global_dict[key] = torch.div(global_dict[key], len(client_models))
        
    # Load the new averaged weights into the server model
    server_model.load_state_dict(global_dict)
    return server_model


if __name__ == '__main__':
    print("🚀 Starting FedAvg Baseline Simulation...")
    
    # --- Configuration ---
    NUM_CLIENTS = 5
    NUM_ROUNDS = 10
    
    # --- Setup ---
    train_loaders, test_loader = get_data(NUM_CLIENTS)
    global_model = SimpleMLP()
    
    # We only need the server for evaluation
    server = Server(global_model, test_loader)
    
    clients = []
    for i in range(NUM_CLIENTS):
        client_model = copy.deepcopy(global_model)
        client = Client(client_id=i, model=client_model, data_loader=train_loaders[i])
        clients.append(client)

    # --- Federated Averaging Training ---
    print("\n--- Starting FedAvg Training ---")
    print("Round 0 (Initial Model):")
    server.evaluate()

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n--- Round {round_num} ---")
        
        # 1. Clients train locally
        for client in clients:
            client.local_train(epochs=2)
            
        # 2. Server aggregates the model weights
        client_models = [client.model for client in clients]
        global_model = federated_averaging(global_model, client_models)
        
        # 3. Server distributes the updated global model
        for client in clients:
            client.model.load_state_dict(global_model.state_dict())
            
        # 4. Evaluate the improved global model
        print(f"End of Round {round_num}:")
        server.evaluate()
        
    print("\n✅ FedAvg Baseline Simulation Complete!")