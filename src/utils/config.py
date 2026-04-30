class FNSLConfig:
    """
    Global configuration settings for the Federated Neuro-Symbolic Learning (FNSL) project.
    Centralizes parameters for data simulation, model architecture, federated training,
    and privacy mechanisms.
    """

    # Data Simulation Settings
    NUM_PATIENTS_TOTAL = 10000        # Total number of synthetic patients
    SNP_DIM = 1000                    # Dimension of SNP data (number of genetic markers)
    EHR_DIM = 50                      # Dimension of EHR features (e.g., blood sugar, BMI)
    TIME_ST    TIME_STEPS = 10                   # Number of time steps in EHR sequences
    T2DM_PREVALENCE = 0.15            # Simulated prevalence of T2DM
    SCD_PREVALENCE = 0.02             # Simulated prevalence of SCD
    HETEROGENEITY_FACTOR = 0.7        # Factor for data heterogeneity across clients (0=IID, 1=max non-IID)
    
    # Model Architecture Settingsttings
    HIDDEN_DIM = 128                  # Hidden dimension for neural networks (SNP encoder, GRU)
    KG_DIM = 64                       # Dimension of Knowledge Graph embeddings

    # Federated Learning Settings
    NUM_CLIENTS = 10                  # Number of participating healthcare institutions (hospitals)
    FEDERATED_ROUNDS = 50             # Total number of federated communication rounds
    LOCAL_EPOCHS = 3                  # Number of local training epochs per client per round
    BATCH_SIZE = 32                   # Batch size for local client training
    LEARNING_RATE = 1e-3              # Learning rate for client optimizers
    
    # Privacy Mechanisms
    DIFFERENTIAL_PRIVACY_NOISE_MULTIPLIER = 0.001 # Noise multiplier for DP (0 for no DP)
    # SECURE_MULTI_PARTY_COMPUTATION_ENABLED = True # Conceptual flag for SMPC

    # Network Slicing (Conceptual)
    NUM_NETWORK_SLICES = 3            # Number of conceptual network slices (e.g., Critical, Standard, Best-Effort)
    SLICE_ALLOCATION_STRATEGY = "dynamic" # "static" or "dynamic"

    # Paths
    MODEL_SAVE_PATH = "checkpoints/fnsl_global_model.pth"
    LOG_DIR = "logs/fnsl_experiments"
    RESULTS_DIR = "results/fnsl_visualizations"
    VISUALIZATION_INTERVAL = 5        # Generate plots every N federated rounds

    def __init__(self):
        pass

    @classmethod
    def print_config(cls):
        print("\n--- FNSL Configuration ---")
        for attr, value in vars(cls).items():
            if not attr.startswith("__") and not callable(value):
                print(f"{attr}: {value}")
        print("-------------------------")

# Example usage:
if __name__ == "__main__":
    FNSLConfig.print_config()
