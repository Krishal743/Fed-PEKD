import torch
import torch.nn as nn
import torch.nn.functional as F

# A simple MLP (Multi-Layer Perceptron) model for FashionMNIST
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256) # Input is a 28x28 image
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)      # Output is 10 classes (for 10 types of clothes)

    def forward(self, x):
        # This function defines how data flows through the model
        x = x.view(-1, 28 * 28)  # Flatten the image
        
        # We'll call this part the "feature extractor"
        features = F.relu(self.fc1(x))
        features = F.relu(self.fc2(features))
        
        # This part is the "classifier"
        logits = self.fc3(features)
        return logits
        
    def get_features(self, x):
        # A helper function to get the features from the middle of the model
        x = x.view(-1, 28 * 28)
        features = F.relu(self.fc1(x))
        features = F.relu(self.fc2(features))
        return features