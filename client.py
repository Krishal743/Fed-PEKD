# FILE: client.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Client:
    def __init__(self, client_id, model, data_loader):
        self.client_id = client_id
        self.model = model
        self.data_loader = data_loader
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
        
    def extract_knowledge(self, num_samples=500):
        """
        Extracts a random sample of REAL feature embeddings and their logits,
        plus a new confidence score.
        """
        self.model.eval()
        all_features = []
        all_logits = []
        
        with torch.no_grad():
            for images, _ in self.data_loader:
                features = self.model.get_features(images)
                logits = self.model(images)
                
                all_features.append(features)
                all_logits.append(logits)
        
        all_features = torch.cat(all_features)
        all_logits = torch.cat(all_logits)

        num_total_samples = all_features.shape[0]
        sample_indices = torch.randperm(num_total_samples)[:num_samples]
        
        sampled_features = all_features[sample_indices]
        sampled_logits = all_logits[sample_indices]

        # ** NEW: Calculate Confidence Score **
        # Apply softmax to the logits to get probabilities
        probabilities = F.softmax(sampled_logits, dim=1)
        # Get the max probability for each sample
        max_probabilities, _ = torch.max(probabilities, dim=1)
        # The confidence is the average of these max probabilities
        confidence_score = max_probabilities.mean().item()
        
        knowledge = {
            'features': sampled_features,
            'logits': sampled_logits,
            'confidence': confidence_score
        }
        
        return knowledge