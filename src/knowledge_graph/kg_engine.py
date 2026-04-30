import torch
import numpy as np

class MedicalKnowledgeGraph:
    """
    A symbolic representation of Bahraini medical knowledge.
    Encodes relationships between SNPs, diseases (T2DM, SCD), and clinical biomarkers.
    """
    def __init__(self, kg_dim=64):
        self.kg_dim = kg_dim
        # In a real system, this would load from a graph database (like Neo4j)
        # Here we simulate embeddings for key Bahraini medical entities
        self.entities = {
            "T2DM": np.random.randn(kg_dim),
            "SCD": np.random.randn(kg_dim),
            "HBB_GENE": np.random.randn(kg_dim),
            "INSULIN_RESISTANCE": np.random.randn(kg_dim),
            "BAHRAIN_GENOME_BASE": np.random.randn(kg_dim)
        }

    def get_context_embedding(self, disease_type="T2DM"):
        """
        Returns a symbolic embedding that guides the neural network's reasoning.
        """
        if disease_type in self.entities:
            # Return as a torch tensor for the model
            return torch.tensor(self.entities[disease_type], dtype=torch.float32)
        else:
            return torch.zeros(self.kg_dim)

    def validate_prediction(self, risk_score, biomarkers):
        """
        A symbolic check to ensure the prediction makes clinical sense.
        Example: If blood sugar is low, a high T2DM risk score might be flagged.
        """
        # Simplified symbolic rule
        if biomarkers['blood_sugar'] < 70 and risk_score > 0.8:
            return False, "Conflict: High risk predicted despite low blood sugar."
        return True, "Prediction clinically plausible."
