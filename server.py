# FILE: server.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# Import the new function
from pseudo_gen import generate_gmm_embeddings, generate_gmm_from_prototypes

def kd_loss(student_logits, teacher_logits, temperature=4.0):
    """
    Knowledge Distillation loss function.
    Handles potential NaN inputs gracefully.
    """
    # Prevent NaN propagation in softmax
    teacher_logits = torch.nan_to_num(teacher_logits)
    student_logits = torch.nan_to_num(student_logits)
    
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    
    # Ensure inputs to kl_div are valid
    soft_teacher = torch.nan_to_num(soft_teacher)
    soft_student = torch.nan_to_num(soft_student)
    
    loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature**2)
    
    # Final check for NaN loss
    if torch.isnan(loss):
        print("Warning: NaN detected in kd_loss. Returning zero loss.")
        return torch.tensor(0.0, requires_grad=True)
        
    return loss


class Server:
    def __init__(self, global_classifier):
        self.global_classifier = global_classifier
        self.optimizer = optim.Adam(self.global_classifier.parameters(), lr=0.001)

    def aggregate_and_distill(self, client_knowledge, distill_epochs=10, agg_method="real-samples"):
        
        valid_classes = set(range(10)) # Track classes with valid data

        if agg_method == "real-samples":
            total_confidence = sum(k['confidence'] for k in client_knowledge)
            if total_confidence == 0: total_confidence = len(client_knowledge)
            
            weighted_features, weighted_logits = [], []
            for k in client_knowledge:
                weight = k['confidence'] / total_confidence
                num_samples = k['features'].shape[0]
                weighted_num_samples = int(num_samples * weight * len(client_knowledge))
                if weighted_num_samples > 0:
                     weighted_features.append(k['features'][:weighted_num_samples])
                     weighted_logits.append(k['logits'][:weighted_num_samples])
            
            if not weighted_features:
                print("Warning: No features to aggregate. Skipping round.")
                return
            
            aggregated_features = torch.cat(weighted_features)
            aggregated_logits = torch.cat(weighted_logits)
        
        elif agg_method == "gmm-pseudo":
            aggregated_features, aggregated_logits = generate_gmm_embeddings(client_knowledge)

        elif agg_method == "prototypes":
            aggregated_features = torch.cat([k['features'] for k in client_knowledge])
            aggregated_logits = torch.cat([k['logits'] for k in client_knowledge])
        
        elif agg_method == "prototype-gmm":
            aggregated_features, aggregated_logits, valid_classes = generate_gmm_from_prototypes(client_knowledge) # Now returns valid classes

        else:
            raise ValueError(f"Unknown aggregation method: {agg_method}")

        # Check for failure
        if aggregated_features is None or aggregated_features.shape[0] == 0:
            print(f"Warning: Aggregation method {agg_method} produced no data. Skipping round.")
            return

        # --- Distillation ---
        distill_dataset = torch.utils.data.TensorDataset(aggregated_features, aggregated_logits)
        batch_size = min(64, aggregated_features.shape[0])
        distill_loader = torch.utils.data.DataLoader(distill_dataset, batch_size=batch_size, shuffle=True)

        self.global_classifier.train()
        for _ in range(distill_epochs):
            for features, teacher_logits in distill_loader:
                self.optimizer.zero_grad()
                student_logits = self.global_classifier(features)
                
                # ** THE KEY CHANGE IS HERE **
                # Only calculate loss for classes where we had valid data
                loss = torch.tensor(0.0, requires_grad=True)
                if len(valid_classes) > 0:
                    # Calculate loss only on the output dimensions corresponding to valid classes
                    valid_idx = list(valid_classes)
                    
                    # Ensure teacher_logits doesn't contain NaNs before softmax
                    teacher_logits_valid = torch.nan_to_num(teacher_logits[:, valid_idx])
                    student_logits_valid = student_logits[:, valid_idx]

                    if teacher_logits_valid.shape[0] > 0 and student_logits_valid.shape[0] > 0 :
                         loss = kd_loss(student_logits_valid, teacher_logits_valid)
                    else:
                         print("Warning: Empty batch after filtering for valid classes.")

                # Final check before backward pass
                if not torch.isnan(loss) and loss.requires_grad:
                    loss.backward()
                    # Add gradient clipping as an extra safeguard
                    torch.nn.utils.clip_grad_norm_(self.global_classifier.parameters(), max_norm=1.0)
                    self.optimizer.step()
                else:
                    print(f"Warning: Skipping optimizer step due to invalid loss ({loss}).")