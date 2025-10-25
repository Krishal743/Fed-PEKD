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
        # Optimizer only trains the feature extractor
        self.optimizer = optim.Adam(self.model.feature_extractor.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def local_train(self, epochs=1):
        """
        Trains the client's feature extractor on its own data.
        The classifier remains frozen with the weights from the global model.
        """
        self.model.train()
        for epoch in range(epochs):
            for images, labels in self.data_loader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        
    def extract_knowledge(self, num_samples=500, agg_method="real-samples"):
        """
        Extracts knowledge based on the aggregation method.
        - "real-samples" / "gmm-pseudo": A random sample of embeddings + logits + confidence
        - "prototypes": Class-mean prototypes and logits
        """
        self.model.eval()
        all_features = []
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.data_loader:
                features = self.model.get_features(images)
                logits = self.model(images)
                
                all_features.append(features)
                all_logits.append(logits)
                all_labels.append(labels)
        
        all_features = torch.cat(all_features)
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        if agg_method == "prototypes":
            prototypes = []
            prototype_logits = []
            for c in range(10): # 10 classes in FashionMNIST
                class_mask = (all_labels == c)
                # Only create a prototype if the client has data for this class
                if class_mask.sum() > 0:
                    prototypes.append(all_features[class_mask].mean(dim=0))
                    prototype_logits.append(all_logits[class_mask].mean(dim=0))
            
            knowledge = {
                'features': torch.stack(prototypes) if prototypes else torch.empty(0, all_features.shape[1]),
                'logits': torch.stack(prototype_logits) if prototype_logits else torch.empty(0, all_logits.shape[1]),
                'confidence': 1.0 # Confidence weighting is not used here, but we keep the key
            }

        else: # "real-samples" or "gmm-pseudo"
            num_total_samples = all_features.shape[0]
            # Ensure we don't try to sample more than we have
            num_samples = min(num_samples, num_total_samples)
            
            sample_indices = torch.randperm(num_total_samples)[:num_samples]
            
            sampled_features = all_features[sample_indices]
            sampled_logits = all_logits[sample_indices]

            probabilities = F.softmax(sampled_logits, dim=1)
            max_probabilities, _ = torch.max(probabilities, dim=1)
            confidence_score = max_probabilities.mean().item()
            
            knowledge = {
                'features': sampled_features,
                'logits': sampled_logits,
                'confidence': confidence_score
            }
        
        return knowledge