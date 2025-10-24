# FILE: main.py

import copy
from data import get_data
from models import SimpleMLP, SimpleCNN
from client import Client
from server import Server

if __name__ == '__main__':
    print("🚀 Starting Fed-PEKD Project - Phase 6: Model Heterogeneity...")
    
    # --- Configuration ---
    NUM_CLIENTS = 5
    NUM_ROUNDS = 10
    
    # --- Setup ---
    train_loaders, test_loader = get_data(NUM_CLIENTS)
    
    # The global model on the server will be a CNN (as it's generally stronger)
    global_model = SimpleCNN()
    server = Server(global_model, test_loader)
    
    # Create clients with HETEROGENEOUS models
    clients = []
    for i in range(NUM_CLIENTS):
        # Let's make the first 2 clients use a CNN, and the rest use an MLP
        if i < 2:
            print(f"Client {i} is using a CNN model.")
            client_model = SimpleCNN()
        else:
            print(f"Client {i} is using an MLP model.")
            client_model = SimpleMLP()
            
        client = Client(client_id=i, model=client_model, data_loader=train_loaders[i])
        clients.append(client)
        
    # --- Federated Training with Fed-PEKD ---
    print("\n--- Starting Fed-PEKD Training with Heterogeneous Clients ---")
    print("Round 0 (Initial Model):")
    server.evaluate()

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n--- Round {round_num} ---")
        
        client_summaries = []
        
        # 1. Clients train locally and extract knowledge
        for client in clients:
            client.local_train(epochs=2)
            knowledge = client.extract_knowledge()
            client_summaries.append(knowledge)
            
        # 2. Server aggregates knowledge and distills it into the global model
        server.aggregate_and_distill(client_summaries)
        
        # 3. Server distributes the updated global model to clients for the next round
        # NOTE: This step is tricky. A client with an MLP can't load a CNN's weights.
        # In Fed-PEKD, the clients can actually continue training their own model type.
        # The global model is only for the server's evaluation. Let's comment out the load_state_dict line
        # to more accurately reflect how KD-based FL works with heterogeneity.
        
        # for client in clients:
        #     client.model.load_state_dict(server.global_model.state_dict())
            
        # 4. Evaluate the improved global model
        print(f"End of Round {round_num}:")
        server.evaluate()
        
    print("\n✅ Phase 6 Fed-PEKD Training Complete!")
    print("The model learned even with clients using different architectures!")