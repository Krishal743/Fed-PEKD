# FILE: server.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def kd_loss(student_logits, teacher_logits, temperature=4.0):
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature**2)

class Server:
    def __init__(self, global_classifier):
        self.global_classifier = global_classifier
        self.optimizer = optim.Adam(self.global_classifier.parameters(), lr=0.001)

    def aggregate_and_distill(self, client_knowledge, distill_epochs=10):
        """
        Aggregates real embeddings from clients and trains the classifier.
        """
        # --- 1. Aggregate Knowledge ---
        # Gather all features and logits from all clients
        aggregated_features = torch.cat([k['features'] for k in client_knowledge])
        aggregated_logits = torch.cat([k['logits'] for k in client_knowledge])

        # Create a simple dataset and loader for training
        distill_dataset = torch.utils.data.TensorDataset(aggregated_features, aggregated_logits)
        distill_loader = torch.utils.data.DataLoader(distill_dataset, batch_size=64, shuffle=True)

        # --- 2. Distill Knowledge into the Global Classifier ---
        self.global_classifier.train()
        for _ in range(distill_epochs):
            for features, teacher_logits in distill_loader:
                self.optimizer.zero_grad()
                
                student_logits = self.global_classifier(features)
                
                loss = kd_loss(student_logits, teacher_logits)
                loss.backward()
                self.optimizer.step()