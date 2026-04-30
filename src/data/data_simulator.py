import numpy as np
import torch
import random

class FNSLDataSimulator:
    """
    Generates synthetic data for the FNSL framework, mimicking Bahraini genomic
    (SNP) and Electronic Health Record (EHR) data, along with associated labels.
    Allows for controlled data heterogeneity across clients.
    """
    def __init__(self, num_patients=1000, snp_dim=1000, ehr_dim=50, time_steps=10, kg_dim=64,
                 t2dm_prevalence=0.15, scd_prevalence=0.02, heterogeneity_factor=0.5):
        self.num_patients = num_patients
        self.snp_dim = snp_dim
        self.ehr_dim = ehr_dim
        self.time_steps = time_steps
        self.kg_dim = kg_dim
        self.t2dm_prevalence = t2dm_prevalence
        self.scd_prevalence = scd_prevalence
        self.heterogeneity_factor = heterogeneity_factor # 0 for IID, 1 for max non-IID

        # Simulate some common Bahraini-specific SNP markers for T2DM/SCD
        # These are conceptual and not based on real genetic data
        self.bahraini_snp_markers = random.sample(range(snp_dim), k=int(snp_dim * 0.05)) # 5% are 'Bahraini specific'
        self.t2dm_snp_markers = random.sample(range(snp_dim), k=int(snp_dim * 0.1)) # 10% are T2DM related
        self.scd_snp_markers = random.sample(range(snp_dim), k=int(snp_dim * 0.03)) # 3% are SCD related

        print(f"FNSL Data Simulator initialized for {num_patients} patients with heterogeneity factor {heterogeneity_factor}.")

    def _generate_snp_data(self, patient_id, client_bias=0.0):
        """
        Generates synthetic SNP data for a single patient, with an optional client bias
        to introduce heterogeneity.
        SNPs are binary (0 or 1) representing presence/absence of a variant.
        """
        snp_data = np.random.randint(0, 2, size=self.snp_dim).astype(np.float32)

        # Introduce some correlation with T2DM/SCD markers
        # Client bias shifts the probability of being T2DM prone
        is_t2dm_prone = random.random() < (self.t2dm_prevalence + client_bias)
        is_scd_carrier = random.random() < self.scd_prevalence

        if is_t2dm_prone:
            for marker in self.t2dm_snp_markers:
                snp_data[marker] = random.choice([0, 1, 1]) # Higher chance of '1'
        if is_scd_carrier:
            for marker in self.scd_snp_markers:
                snp_data[marker] = random.choice([0, 1, 1]) # Higher chance of '1'

        return torch.tensor(snp_data, dtype=torch.float32), is_t2dm_prone

    def _generate_ehr_sequence(self, patient_id, is_t2dm_prone, client_ehr_shift=0.0):
        """
        Generates a synthetic EHR sequence over time for a single patient.
        EHR features could include blood sugar, BMI, blood pressure, etc.
        Client_ehr_shift introduces heterogeneity in baseline biomarker levels.
        """
        ehr_sequence = np.zeros((self.time_steps, self.ehr_dim), dtype=np.float32)
        
        # Simulate a trend for T2DM related biomarkers
        baseline_blood_sugar = random.uniform(80, 100) + client_ehr_shift # mg/dL
        baseline_bmi = random.uniform(20, 25) + client_ehr_shift * 0.5

        for t in range(self.time_steps):
            # Simulate blood sugar trend
            if is_t2dm_prone and t > self.time_steps / 2: # Worsens over time for prone individuals
                ehr_sequence[t, 0] = baseline_blood_sugar + (t - self.time_steps / 2) * random.uniform(1, 3) + np.random.normal(0, 5)
            else:
                ehr_sequence[t, 0] = baseline_blood_sugar + np.random.normal(0, 3)
            ehr_sequence[t, 0] = np.clip(ehr_sequence[t, 0], 60, 250) # Clip values

            # Simulate BMI trend
            if is_t2dm_prone and t > self.time_steps / 3:
                ehr_sequence[t, 1] = baseline_bmi + (t - self.time_steps / 3) * random.uniform(0.1, 0.5) + np.random.normal(0, 1)
            else:
                ehr_sequence[t, 1] = baseline_bmi + np.random.normal(0, 0.5)
            ehr_sequence[t, 1] = np.clip(ehr_sequence[t, 1], 18, 40)

            # Fill other EHR dimensions with random noise for now
            ehr_sequence[t, 2:] = np.random.normal(0, 1, size=self.ehr_dim - 2)

        return torch.tensor(ehr_sequence, dtype=torch.float32)

    def _generate_label(self, snp_data, ehr_sequence, is_t2dm_prone):
        """
        Generates a binary label (0: no T2DM onset, 1: T2DM onset) based on SNP and EHR.
        """
        # Simple rule: if T2DM prone and blood sugar crosses a threshold at some point
        if is_t2dm_prone and np.any(ehr_sequence[:, 0].numpy() > 125): # Fasting blood sugar > 125 mg/dL
            return torch.tensor([[1.0]], dtype=torch.float32)
        return torch.tensor([[0.0]], dtype=torch.float32)

    def generate_client_data(self, num_clients=3):
        """
        Generates a list of datasets, one for each simulated hospital client.
        Each dataset contains (snp_data, ehr_sequence, kg_embedding, label).
        Introduces heterogeneity by varying client_bias and client_ehr_shift.
        """
        all_client_data = []
        patients_per_client = self.num_patients // num_clients

        for client_idx in range(num_clients):
            client_data = []
            # Introduce client-specific bias for heterogeneity
            client_bias = (random.random() - 0.5) * 2 * self.heterogeneity_factor * self.t2dm_prevalence
            client_ehr_shift = (random.random() - 0.5) * 2 * self.heterogeneity_factor * 10 # Shift baseline blood sugar

            for patient_id in range(patients_per_client):
                snp_data, is_t2dm_prone = self._generate_snp_data(patient_id, client_bias)
                ehr_sequence = self._generate_ehr_sequence(patient_id, is_t2dm_prone, client_ehr_shift)
                
                # Dummy KG embedding for now, will be replaced by actual KG engine
                kg_embedding = torch.randn(self.kg_dim)
                
                label = self._generate_label(snp_data, ehr_sequence, is_t2dm_prone)
                client_data.append((snp_data, ehr_sequence, kg_embedding, label))
            all_client_data.append(client_data)
        return all_client_data

# Example usage:
if __name__ == "__main__":
    simulator = FNSLDataSimulator(num_patients=300, snp_dim=1000, ehr_dim=50, time_steps=10, heterogeneity_factor=0.8)
    client_datasets = simulator.generate_client_data(num_clients=3)
    print(f"Generated data for {len(client_datasets)} clients.")
    for i, data in enumerate(client_datasets):
        print(f"Client {i+1} has {len(data)} patients.")
        sample_snp, sample_ehr, sample_kg, sample_label = data[0]
        print(f"  Sample SNP shape: {sample_snp.shape}")
        print(f"  Sample EHR sequence shape: {sample_ehr.shape}")
        print(f"  Sample KG embedding shape: {sample_kg.shape}")
        print(f"  Sample Label: {sample_label.item()}")
