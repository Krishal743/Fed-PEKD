# FILE: models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# A simple MLP (Multi-Layer Perceptron) model for FashionMNIST
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256) # Input is a 28x28 image
        self.fc2 = nn.Linear(256, 128)
        self.classifier = nn.Linear(128, 10)      # Output is 10 classes

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        features = self.get_features(x)
        logits = self.classifier(features)
        return logits
        
    def get_features(self, x):
        x = x.view(-1, 28 * 28)
        features = F.relu(self.fc1(x))
        features = F.relu(self.fc2(features))
        return features

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        
        self.feature_extractor = nn.Sequential(
            self.conv1, nn.ReLU(), self.pool,
            self.conv2, nn.ReLU(), self.pool
        )
        
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        features = self.get_features(x)
        logits = self.classifier(features)
        return logits

    def get_features(self, x):
        conv_features = self.feature_extractor(x)
        flat_features = conv_features.view(-1, 32 * 7 * 7)
        final_features = F.relu(self.fc1(flat_features))
        return final_features