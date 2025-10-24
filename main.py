# FILE: main.py

import copy
import torch
from data import get_data
from models import SimpleCNN, Classifier
from client import Client
from server import Server

def evaluate_client(client, test_loader):
    """Helper function to evaluate a client's model."""
    client.model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = client.model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"🌍 Global Accuracy on Test Set: {accuracy:.2f}%")
    return accuracy

if __name__ == '__main__':
    print("🚀 Starting Fed-PEKD Project - Final Version...")
    
    # --- Configuration ---
    NUM_CLIENTS = 5
    NUM_ROUNDS = 40
    
    # --- Setup ---
    train_loaders, test_loader = get_data(NUM_CLIENTS)
    
    # The server now manages a standalone Classifier model
    global_classifier = Classifier()
    server = Server(global_classifier)
    
    # Create clients, all with the SimpleCNN model
    clients = []
    for i in range(NUM_CLIENTS):
        client = Client(client_id=i, model=SimpleCNN(), data_loader=train_loaders[i])
        clients.append(client)
        
    # --- Federated Training with Corrected Logic ---
    print("\n--- Starting Fed-PEKD Training ---")
    print("Round 0 (Initial Model):")
    evaluate_client(clients[0], test_loader)

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n--- Round {round_num} ---")
        
        client_summaries = []
        
        # 1. Clients train locally and extract knowledge
        for client in clients:
            client.local_train(epochs=2)
            knowledge = client.extract_knowledge()
            client_summaries.append(knowledge)
            
        # 2. Server aggregates and trains the global classifier
        server.aggregate_and_distill(client_summaries)
        
        # 3. Server distributes the updated classifier back to ALL clients
        for client in clients:
            client.model.classifier.load_state_dict(server.global_classifier.state_dict())
            
        # 4. Evaluate by testing one of the updated clients
        print(f"End of Round {round_num}:")
        evaluate_client(clients[0], test_loader)
        
    print("\n✅ Fed-PEKD Training Complete!")