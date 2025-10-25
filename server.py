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
        Aggregates embeddings and logits using confidence-based weighting.
        """
        # --- 1. Confidence-Weighted Aggregation ---
        total_confidence = sum(k['confidence'] for k in client_knowledge)
        
        # If total_confidence is zero, fall back to equal weighting
        if total_confidence == 0:
            total_confidence = len(client_knowledge)
        
        weighted_features = []
        weighted_logits = []
        
        for k in client_knowledge:
            weight = k['confidence'] / total_confidence
            # Note: We need to scale the number of samples by the weight
            num_samples = k['features'].shape[0]
            weighted_num_samples = int(num_samples * weight * len(client_knowledge))
            
            if weighted_num_samples > 0:
                 weighted_features.append(k['features'][:weighted_num_samples])
                 weighted_logits.append(k['logits'][:weighted_num_samples])

        if not weighted_features:
            print("Warning: No features to aggregate after weighting. Skipping round.")
            return

        aggregated_features = torch.cat(weighted_features)
        aggregated_logits = torch.cat(weighted_logits)

        # Create a dataset for distillation
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