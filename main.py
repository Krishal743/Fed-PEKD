import copy
from data import get_data
from models import SimpleMLP
from client import Client
from server import Server

if __name__ == '__main__':
    print("🚀 Starting Fed-PEKD Project - Phase 3...")
    
    # --- Configuration ---
    NUM_CLIENTS = 5
    NUM_ROUNDS = 10 # Let's run for more rounds to see the learning
    
    # --- Setup ---
    train_loaders, test_loader = get_data(NUM_CLIENTS)
    global_model = SimpleMLP()
    server = Server(global_model, test_loader)
    
    clients = []
    for i in range(NUM_CLIENTS):
        client_model = copy.deepcopy(global_model)
        client = Client(client_id=i, model=client_model, data_loader=train_loaders[i])
        clients.append(client)
        
    # --- Federated Training with Fed-PEKD ---
    print("\n--- Starting Fed-PEKD Training ---")
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
        for client in clients:
            client.model.load_state_dict(server.global_model.state_dict())
            
        # 4. Evaluate the improved global model
        print(f"End of Round {round_num}:")
        server.evaluate()
        
    print("\n✅ Phase 3 Fed-PEKD Training Complete!")
    print("Hooray! The global model's accuracy is now improving with each round!")