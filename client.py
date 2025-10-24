# FILE: client.py

import torch
import torch.nn as nn
import torch.optim as optim

class Client:
    def __init__(self, client_id, model, data_loader):
        self.client_id = client_id
        self.model = model
        self.data_loader = data_loader
        # Let's revert to the original learning rate for now
        self.optimizer = optim.Adam(self.model.feature_extractor.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def local_train(self, epochs=1):
        self.model.train()
        for epoch in range(epochs):
            for images, labels in self.data_loader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        
    def extract_knowledge(self, num_samples=100):
        """
        Extracts a random sample of REAL feature embeddings and their logits.
        """
        self.model.eval()
        all_features = []
        all_logits = []
        
        with torch.no_grad():
            for images, _ in self.data_loader:
                # Pass images through the model to get features and logits
                features = self.model.get_features(images)
                logits = self.model(images)
                
                all_features.append(features)
                all_logits.append(logits)
        
        # Concatenate all features and logits from this client
        all_features = torch.cat(all_features)
        all_logits = torch.cat(all_logits)

        # Randomly sample a subset of the features and logits
        num_total_samples = all_features.shape[0]
        sample_indices = torch.randperm(num_total_samples)[:num_samples]
        
        sampled_features = all_features[sample_indices]
        sampled_logits = all_logits[sample_indices]

        # The new "knowledge" is this small batch of real data
        knowledge = {
            'features': sampled_features,
            'logits': sampled_logits
        }
        
        return knowledge