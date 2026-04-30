import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuroSymbolicLayer(nn.Module):
    """
    The core symbolic reasoning component. It takes neural features and a knowledge
    graph embedding, then applies symbolic constraints to refine the prediction.
    """
    def __init__(self, feature_dim, kg_dim):
        super(NeuroSymbolicLayer, self).__init__()
        # A gate to control the influence of symbolic knowledge
        self.knowledge_gate = nn.Linear(feature_dim + kg_dim, feature_dim)
        
        # A layer to combine neural patterns with symbolic insights
        self.fusion_layer = nn.Linear(feature_dim, feature_dim)
        
        # Final prediction layer
        self.output_layer = nn.Linear(feature_dim, 1)

    def forward(self, neural_features, kg_embeddings, symbolic_constraints=None):
        # Concatenate neural features with knowledge graph embeddings
        combined_input = torch.cat([neural_features, kg_embeddings], dim=-1)
        
        # Apply knowledge gate to filter and integrate symbolic information
        gated_features = torch.sigmoid(self.knowledge_gate(combined_input)) * neural_features
        
        # Fuse features
        fused_features = F.relu(self.fusion_layer(gated_features))
        
        # Initial risk prediction
        raw_risk_score = torch.sigmoid(self.output_layer(fused_features))
        
        # Apply explicit symbolic constraints (conceptual)
        if symbolic_constraints is not None:
            # This is where the KG would actively 'correct' or 'modulate' the neural output.
            # For example, if KG says 'patient has protective gene X, so T2DM risk is lower'
            # or 'patient has severe genetic predisposition Y, so T2DM risk is higher'.
            # This would involve more complex logic, potentially a differentiable reasoning module.
            # For now, we'll simulate a simple modulation.
            if 'lower_risk_factor' in symbolic_constraints:
                raw_risk_score = raw_risk_score * (1 - symbolic_constraints['lower_risk_factor'])
            if 'higher_risk_factor' in symbolic_constraints:
                raw_risk_score = raw_risk_score * (1 + symbolic_constraints['higher_risk_factor'])
        
        return raw_risk_score

class FNSLModel(nn.Module):
    """
    The full Federated Neuro-Symbolic Learning model.
    Uses a GRU for temporal EHR data and a symbolic layer for clinical logic.
    """
    def __init__(self, snp_dim, ehr_dim, hidden_dim, kg_dim):
        super(FNSLModel, self).__init__()
        
        # 1. The 'Neuro' part: Feature extraction from SNPs and EHRs
        self.snp_encoder = nn.Linear(snp_dim, hidden_dim)
        self.ehr_gru = nn.GRU(ehr_dim, hidden_dim, batch_first=True)
        
        # 2. The 'Bridge': Combining genomic and temporal data
        self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # 3. The 'Symbolic' part: Grounding predictions in medical knowledge
        self.symbolic_brain = NeuroSymbolicLayer(hidden_dim, kg_dim)

    def forward(self, snp_data, ehr_sequence, kg_embeddings, symbolic_constraints=None):
        # Process genomic data
        snp_features = F.relu(self.snp_encoder(snp_data))
        
        # Process temporal EHR data (last hidden state of GRU)
        # Ensure ehr_sequence has at least one time step
        if ehr_sequence.size(1) == 0:
            ehr_features = torch.zeros(ehr_sequence.size(0), self.ehr_gru.hidden_size, device=ehr_sequence.device)
        else:
            _, ehr_hidden = self.ehr_gru(ehr_sequence)
            ehr_features = ehr_hidden.squeeze(0)
        
        # Fuse the data
        combined_features = torch.cat([snp_features, ehr_features], dim=-1)
        fused = F.relu(self.fusion_layer(combined_features))
        
        # Apply symbolic reasoning with optional constraints
        risk_score = self.symbolic_brain(fused, kg_embeddings, symbolic_constraints)
        
        return risk_score

# Example usage for testing
if __name__ == "__main__":
    # Dummy dimensions for testing
    snp_dim = 1000
    ehr_dim = 50
    hidden_dim = 128
    kg_dim = 64
    batch_size = 4
    time_steps = 10

    model = FNSLModel(snp_dim=snp_dim, ehr_dim=ehr_dim, hidden_dim=hidden_dim, kg_dim=kg_dim)
    print("FNSL Model initialized successfully.")

    # Generate dummy data
    dummy_snp = torch.randn(batch_size, snp_dim)
    dummy_ehr = torch.randn(batch_size, time_steps, ehr_dim)
    dummy_kg_embedding = torch.randn(batch_size, kg_dim)
    
    # Test forward pass without symbolic constraints
    output_no_constraints = model(dummy_snp, dummy_ehr, dummy_kg_embedding)
    print(f"Output shape (no constraints): {output_no_constraints.shape}")

    # Test forward pass with symbolic constraints
    dummy_constraints = {
        'lower_risk_factor': 0.1, # Example: a protective gene reduces risk by 10%
        'higher_risk_factor': 0.05 # Example: a specific comorbidity increases risk by 5%
    }
    output_with_constraints = model(dummy_snp, dummy_ehr, dummy_kg_embedding, dummy_constraints)
    print(f"Output shape (with constraints): {output_with_constraints.shape}")

    # Test with empty EHR sequence
    dummy_ehr_empty = torch.randn(batch_size, 0, ehr_dim)
    output_empty_ehr = model(dummy_snp, dummy_ehr_empty, dummy_kg_embedding)
    print(f"Output shape (empty EHR): {output_empty_ehr.shape}")
