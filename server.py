import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def kd_loss(student_logits, teacher_logits, temperature=4.0):
    """
    Knowledge Distillation loss function.
    """
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature**2)

class Server:
    def __init__(self, global_model, test_loader):
        self.global_model = global_model
        self.test_loader = test_loader
        self.optimizer = optim.Adam(self.global_model.parameters(), lr=0.001)

    def aggregate_and_distill(self, client_summaries, num_samples=500, distill_epochs=2):
        """
        The core Fed-PEKD logic: aggregate, generate, and distill.
        """
        # --- 1. Aggregate Knowledge ---
        # Average the statistics from all clients
        global_mean = torch.stack([s['mean'] for s in client_summaries]).mean(dim=0)
        global_variance = torch.stack([s['variance'] for s in client_summaries]).mean(dim=0)
        global_logits = torch.stack([s['logits'] for s in client_summaries]).mean(dim=0)
        
        # --- 2. Generate Pseudo-Embeddings [cite: 22, 38] ---
        # Create "ghost data" from the global statistics
        # We add a small value to variance to avoid sqrt(0)
        pseudo_embeddings = global_mean + torch.randn(num_samples, global_mean.shape[0]) * torch.sqrt(global_variance + 1e-8)
        
        # The aggregated logits are our "teacher" labels for the ghost data
        teacher_logits = global_logits.unsqueeze(0).repeat(num_samples, 1)

        # --- 3. Distill Knowledge [cite: 23] ---
        # Train the global model on the ghost data
        self.global_model.train()
        for _ in range(distill_epochs):
            self.optimizer.zero_grad()
            
            # We only need to pass the embeddings through the final layers (classifier)
            student_logits = self.global_model.fc3(pseudo_embeddings)
            
            # Calculate the distillation loss
            loss = kd_loss(student_logits, teacher_logits)
            
            loss.backward()
            self.optimizer.step()
        
        # print("Server finished knowledge distillation.")
        
    def evaluate(self):
        """
        Tests the global model on the test dataset.
        """
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                outputs = self.global_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"🌍 Global Model Accuracy on Test Set: {accuracy:.2f}%")
        return accuracy