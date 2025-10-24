# FILE: models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# A standalone classifier that can be trained on the server
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        final_features = F.relu(self.fc1(x))
        logits = self.classifier(final_features)
        return logits

# Note: The SimpleMLP is no longer used in this experiment, but we can leave it.
class SimpleMLP(nn.Module):
    # ... (code for SimpleMLP remains the same) ...
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.classifier = nn.Linear(128, 10)

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
        # This is the "body" or "feature extractor"
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        # This is the "head" or "classifier"
        self.classifier = Classifier()

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(-1, 32 * 7 * 7) # Flatten
        logits = self.classifier(features)
        return logits

    def get_features(self, x):
        # The features we extract are now the direct output of the conv layers
        features = self.feature_extractor(x)
        features = features.view(-1, 32 * 7 * 7) # Flatten
        return features