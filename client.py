import torch
import torch.nn as nn
import torch.optim as optim

class Client:
    def __init__(self, client_id, model, data_loader):
        self.client_id = client_id
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def local_train(self, epochs=1):
        """
        Trains the client's model on its own data for a few epochs.
        """
        self.model.train() # Set the model to training mode
        
        for epoch in range(epochs):
            for images, labels in self.data_loader:
                # 1. Clear old gradients
                self.optimizer.zero_grad()
                
                # 2. Make a prediction (forward pass)
                outputs = self.model(images)
                
                # 3. Calculate the loss
                loss = self.criterion(outputs, labels)
                
                # 4. Calculate gradients (backward pass)
                loss.backward()
                
                # 5. Update the model's weights
                self.optimizer.step()
        
        # This is just for us to see the progress
        # print(f"Client {self.client_id} finished local training.")
        # (Keep the __init__ and local_train methods as they are)
# Add this new method to the Client class:

    def extract_knowledge(self):
        """
        Extracts feature statistics (mean, variance) and average logits.
        """
        self.model.eval() # Set the model to evaluation mode
        feature_list = []
        logit_list = []
        
        with torch.no_grad():
            for images, _ in self.data_loader:
                # Get features and logits from the model
                features = self.model.get_features(images)
                logits = self.model(images)
                
                feature_list.append(features)
                logit_list.append(logits)
        
        # Stack all features and logits from this client
        all_features = torch.cat(feature_list)
        all_logits = torch.cat(logit_list)
        
        # Calculate statistics
        feature_mean = all_features.mean(dim=0)
        feature_variance = all_features.var(dim=0)
        avg_logits = all_logits.mean(dim=0)
        
        # This is the "knowledge summary" the client will share
        knowledge = {
            'mean': feature_mean,
            'variance': feature_variance,
            'logits': avg_logits
        }
        
        # print(f"Client {self.client_id} extracted knowledge.")
        return knowledge