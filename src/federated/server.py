import torch
import copy
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

class FederatedServer:
    """
    The central coordinator for the Bahrain FNSL network.
    It aggregates model updates without ever seeing raw patient data.
    """
    def __init__(self, global_model):
        self.global_model = global_model
        self.client_weights = []

    def aggregate(self, client_models, dp_noise_multiplier=0.01):
        """
        Performs Federated Averaging (FedAvg) with Differential Privacy.
        """
        print(f"Aggregating updates from {len(client_models)} institutions...")
        
        # Start with a clean slate for the new global weights
        global_dict = self.global_model.state_dict()
        
        # Initialize aggregated weights to zeros
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])

        # Sum up client weights
        for client_model in client_models:
            client_dict = client_model.state_dict()
            for key in global_dict.keys():
                global_dict[key] += client_dict[key].float()
        
        # Average the summed weights
        for key in global_dict.keys():
            global_dict[key] = global_dict[key] / len(client_models)
            
            # Add Differential Privacy noise to prevent reverse-engineering
            noise = torch.randn_like(global_dict[key]) * dp_noise_multiplier
            global_dict[key] += noise
            
        self.global_model.load_state_dict(global_dict)
        print("Global model updated with DP-protected weights.")
        return self.global_model

    def distribute(self):
        """Sends the current global model to all participating hospitals."""
        return copy.deepcopy(self.global_model)

class HospitalClient:
    """
    Represents a local healthcare institution in Bahrain.
    Trains the model on local data and only shares weight updates.
    """
    def __init__(self, client_id, local_data, model, learning_rate=1e-3, batch_size=32):
        self.client_id = client_id
        self.local_data = local_data
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.loss_fn = nn.BCELoss() # Binary Cross-Entropy for binary classification

        # Prepare DataLoader for local data
        # local_data is a list of (snp_data, ehr_sequence, kg_embedding, label)
        snp_tensors = torch.stack([d[0] for d in local_data])
        ehr_tensors = torch.stack([d[1] for d in local_data])
        kg_tensors = torch.stack([d[2] for d in local_data])
        label_tensors = torch.stack([d[3] for d in local_data])

        self.dataset = TensorDataset(snp_tensors, ehr_tensors, kg_tensors, label_tensors)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def train_local(self, epochs=5, kg_embedding=None):
        """
        Trains the model on local sensitive data.
        Returns the trained model and the average loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for epoch in range(epochs):
            for snp, ehr, kg, label in self.dataloader:
                self.optimizer.zero_grad()
                
                # Use the provided kg_embedding or the one from the batch if preferred
                # For now, we'll use the batch's kg_embedding as it might be patient-specific
                output = self.model(snp, ehr, kg) # Assuming kg_embedding is passed directly
                loss = self.loss_fn(output, label)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"Hospital {self.client_id}: Local training complete. Avg Loss: {avg_loss:.4f}")
        return self.model, avg_loss

    def evaluate_local(self, kg_embedding=None):
        """
        Evaluates the model on local data.
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for snp, ehr, kg, label in self.dataloader:
                output = self.model(snp, ehr, kg) # Assuming kg_embedding is passed directly
                predicted = (output > 0.5).float()
                total += label.size(0)
                correct += (predicted == label).sum().item()
        accuracy = correct / total if total > 0 else 0.0
        print(f"Hospital {self.client_id}: Local Accuracy: {accuracy:.4f}")
        return accuracy
