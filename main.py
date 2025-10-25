# FILE: main.py

import copy
import torch
import argparse # New import
import pandas as pd # New import for logging
import os # New import for saving results
from data import get_data
from models import SimpleCNN, Classifier
from client import Client
from server import Server

# --- Evaluation Helper ---
def evaluate_client(client, test_loader):
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
    return accuracy

# --- FedAvg Helper ---
def federated_averaging(server_model, client_models):
    global_dict = server_model.state_dict()
    for key in global_dict:
        global_dict[key] = torch.zeros_like(global_dict[key])
    for client_model in client_models:
        for key in global_dict:
            global_dict[key] += client_model.state_dict()[key]
    for key in global_dict:
        global_dict[key] = torch.div(global_dict[key], len(client_models))
    server_model.load_state_dict(global_dict)
    return server_model

# --- Main Experiment Function ---
def run_experiment(args):
    print(f"🚀 Starting Experiment: {args.exp_name}")
    print(f"--- Configuration ---")
    print(f"Method: {args.method}")
    print(f"Rounds: {args.num_rounds}, Local Epochs: {args.local_epochs}")
    print(f"Data: Dirichlet alpha={args.alpha}, Samples/Client: {args.max_samples}")
    if args.method == "fed-pekd":
        print(f"Fed-PEKD Samples: {args.pekd_samples}, Distill Epochs: {args.distill_epochs}")
    print("---------------------")

    # --- Setup ---
    train_loaders, test_loader = get_data(args.num_clients, args.alpha, args.max_samples)
    
    # --- Logging Setup ---
    results = [] # To store (round, accuracy) tuples
    
    # --- Model Setup ---
    if args.method == "fedavg":
        global_model = SimpleCNN()
        clients = [Client(i, copy.deepcopy(global_model), train_loaders[i]) for i in range(args.num_clients)]
    elif args.method == "fed-pekd":
        global_classifier = Classifier()
        server = Server(global_classifier)
        clients = [Client(i, SimpleCNN(), train_loaders[i]) for i in range(args.num_clients)]
    
    # --- Initial Evaluation ---
    initial_acc = evaluate_client(clients[0], test_loader)
    print(f"Round 0 (Initial Model): 🌍 Global Accuracy on Test Set: {initial_acc:.2f}%")
    results.append({'round': 0, 'accuracy': initial_acc, 'method': args.exp_name})

    # --- Federated Training Loop ---
    for round_num in range(1, args.num_rounds + 1):
        print(f"\n--- Round {round_num} ---")
        
        if args.method == "fedavg":
            # 1. Clients train locally
            for client in clients:
                client.optimizer = torch.optim.Adam(client.model.parameters(), lr=args.lr)
                client.local_train(epochs=args.local_epochs)
            
            # 2. Server aggregates weights
            client_models = [client.model for client in clients]
            global_model = federated_averaging(global_model, client_models)
            
            # 3. Server distributes updated model
            for client in clients:
                client.model.load_state_dict(global_model.state_dict())

        elif args.method == "fed-pekd":
            client_knowledge = []
            # 1. Clients train locally and extract knowledge
            for client in clients:
                client.optimizer = torch.optim.Adam(client.model.feature_extractor.parameters(), lr=args.lr)
                client.local_train(epochs=args.local_epochs)
                knowledge = client.extract_knowledge(num_samples=args.pekd_samples)
                client_knowledge.append(knowledge)
            
            # 2. Server aggregates and trains classifier
            server.aggregate_and_distill(client_knowledge, distill_epochs=args.distill_epochs)
            
            # 3. Server distributes updated classifier
            for client in clients:
                client.model.classifier.load_state_dict(server.global_classifier.state_dict())
        
        # 4. Evaluate and Log
        acc = evaluate_client(clients[0], test_loader)
        print(f"End of Round {round_num}: 🌍 Global Accuracy on Test Set: {acc:.2f}%")
        results.append({'round': round_num, 'accuracy': acc, 'method': args.exp_name})

    print(f"\n✅ Experiment '{args.exp_name}' Complete!")
    
    # --- Save Results to CSV ---
    if not os.path.exists('results'):
        os.makedirs('results')
    
    results_df = pd.DataFrame(results)
    output_path = f"results/{args.exp_name}.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

# --- Argument Parsing ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fed-PEKD Research Experiments")
    
    # Experiment Naming
    parser.add_argument('--exp_name', type=str, default='fedpekd_run', help='Name of the experiment')
    
    # Core Method
    parser.add_argument('--method', type=str, required=True, choices=['fedavg', 'fed-pekd'], help='Federated learning method')
    
    # FL Parameters
    parser.add_argument('--num_rounds', type=int, default=40)
    parser.add_argument('--num_clients', type=int, default=5)
    parser.add_argument('--local_epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    
    # Data Parameters
    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet distribution alpha')
    parser.add_argument('--max_samples', type=int, default=1000, help='Max samples per client')
    
    # Fed-PEKD Specific Parameters
    parser.add_argument('--pekd_samples', type=int, default=500, help='Number of embeddings to send')
    parser.add_argument('--distill_epochs', type=int, default=30, help='Server distillation epochs')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility (as per your plan)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    run_experiment(args)