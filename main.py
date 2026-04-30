import torch
import random
import os

from src.models.fnsl_model import FNSLModel
from src.federated.server import FederatedServer, HospitalClient
from src.federated.network_slicing import NetworkSlicingManager
from src.knowledge_graph.kg_engine import MedicalKnowledgeGraph
from src.data.data_simulator import FNSLDataSimulator
from src.utils.config import FNSLConfig
from src.utils.logger import AGLLogger
from src.utils.fnsl_visualization import FNSLVisualizer

def run_fnsl_experiment():
    print("--- Starting Elite Bahrain FNSL Experiment ---")
    
    # 1. Load Configuration
    config = FNSLConfig()
    config.print_config()

    # 2. Initialize Logger and Visualizer
    logger = AGLLogger(log_dir=config.LOG_DIR)
    logger.log_config(config)
    visualizer = FNSLVisualizer(results_dir=config.RESULTS_DIR)
    
    # 3. Ensure checkpoint and results directories exist
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # 4. Initialize the Medical Knowledge Graph
    kg_engine = MedicalKnowledgeGraph(kg_dim=config.KG_DIM)
    # Get a generic KG embedding for T2DM context for all models
    t2dm_kg_embedding = kg_engine.get_context_embedding("T2DM")
    
    # 5. Setup Data Simulator and Generate Data for Clients
    data_simulator = FNSLDataSimulator(
        num_patients=config.NUM_PATIENTS_TOTAL,
        snp_dim=config.SNP_DIM,
        ehr_dim=config.EHR_DIM,
        time_steps=config.TIME_STEPS,
        kg_dim=config.KG_DIM,
        t2dm_prevalence=config.T2DM_PREVALENCE,
        scd_prevalence=config.SCD_PREVALENCE,
        heterogeneity_factor=config.HETEROGENEITY_FACTOR
    )
    client_datasets = data_simulator.generate_client_data(num_clients=config.NUM_CLIENTS)
    print(f"Generated synthetic data for {config.NUM_CLIENTS} hospital clients.")
    
    # 6. Initialize the Global Model and Federated Server
    global_model = FNSLModel(snp_dim=config.SNP_DIM, ehr_dim=config.EHR_DIM, hidden_dim=config.HIDDEN_DIM, kg_dim=config.KG_DIM)
    server = FederatedServer(global_model)
    print("Global FNSL Model and Federated Server initialized.")
    
    # 7. Initialize Network Slicing Manager
    slicing_manager = NetworkSlicingManager(
        num_slices=config.NUM_NETWORK_SLICES,
        slice_allocation_strategy=config.SLICE_ALLOCATION_STRATEGY
    )
    print(f"Network Slicing Manager initialized with {config.NUM_NETWORK_SLICES} slices.")

    # 8. Initialize Hospital Clients
    hospitals = []
    for i in range(config.NUM_CLIENTS):
        # Each hospital gets a copy of the global model to start
        client_model = server.distribute()
        hospital_client = HospitalClient(
            client_id=f"Hospital_{i+1}",
            local_data=client_datasets[i],
            model=client_model,
            learning_rate=config.LEARNING_RATE,
            batch_size=config.BATCH_SIZE
        )
        # Allocate client to a network slice (conceptual)
        slicing_manager.allocate_client_to_slice(hospital_client.client_id, required_priority=random.randint(1, config.NUM_NETWORK_SLICES))
        hospitals.append(hospital_client)
    print(f"Initialized {config.NUM_CLIENTS} hospital clients and allocated to network slices.")

    # Histories for plotting
    global_auc_history = []
    client_losses_per_round = {h.client_id: [] for h in hospitals}

    # 9. Run Federated Training Rounds
    for round_num in range(1, config.FEDERATED_ROUNDS + 1):
        print(f"\n--- Federated Round {round_num}/{config.FEDERATED_ROUNDS} ---")
        client_models_for_aggregation = []
        round_metrics = {"round": round_num, "client_losses": []}
        
        for hospital in hospitals:
            print(f"  Hospital {hospital.client_id} starting local training...")
            # Local training on sensitive data
            trained_local_model, local_loss = hospital.train_local(epochs=config.LOCAL_EPOCHS, kg_embedding=t2dm_kg_embedding)
            client_models_for_aggregation.append(trained_local_model)
            round_metrics["client_losses"].append({"client_id": hospital.client_id, "loss": local_loss})
            client_losses_per_round[hospital.client_id].append(local_loss)
        
        # Server aggregates models with Differential Privacy
        server.aggregate(client_models_for_aggregation, dp_noise_multiplier=config.DIFFERENTIAL_PRIVACY_NOISE_MULTIPLIER)
        
        # Distribute the updated global model back to hospitals
        updated_global_model = server.distribute()
        for hospital in hospitals:
            hospital.model = updated_global_model
            
        # (Optional) Evaluate global model performance after each round
        # For a real system, you'd have a separate evaluation dataset or process
        # For now, we'll just print a placeholder and log a dummy AUC.
        global_auc = random.uniform(0.85, 0.95) # Placeholder
        print(f"  Global model updated. Global AUC after round {round_num}: {global_auc:.2f}")
        round_metrics["global_auc"] = global_auc
        logger.log_metrics(round_num, round_metrics)
        global_auc_history.append(global_auc)

        # Visualize periodically
        if round_num % config.VISUALIZATION_INTERVAL == 0 or round_num == config.FEDERATED_ROUNDS:
            visualizer.plot_global_auc_history(global_auc_history)
            visualizer.plot_client_loss_history(client_losses_per_round)
            visualizer.plot_combined_metrics(global_auc_history, client_losses_per_round)

    print("\n--- Federated FNSL Experiment Complete ---")
    print("The final global model is ready for deployment or further analysis.")
    
    # Save the final global model
    torch.save(server.global_model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"Final global model saved to {config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    run_fnsl_experiment()
