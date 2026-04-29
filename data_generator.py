import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from datetime import datetime, timedelta


class SyntheticCohortGenerator:
    def __init__(self, n_patients: int = 10000, years: int = 10, 
                 t2dm_incidence: float = 0.15, random_seed: int = 42):
        self.n_patients = n_patients
        self.years = years
        self.quarters = years * 4
        self.t2dm_incidence = t2dm_incidence
        np.random.seed(random_seed)
    
    def generate(self) -> Tuple[Dict, Dict]:
        patients = []
        
        for patient_id in range(self.n_patients):
            patient = self._generate_patient_profile(patient_id)
            patients.append(patient)
        
        train_split = int(0.8 * len(patients))
        train_data = self._prepare_dataset(patients[:train_split])
        val_data = self._prepare_dataset(patients[train_split:])
        
        return train_data, val_data
    
    def _generate_patient_profile(self, patient_id: int) -> Dict:
        age = np.random.randint(30, 75)
        gender = np.random.choice(['M', 'F'])
        ethnicity = np.random.choice(['Arab', 'Asian', 'Other'])
        
        prs = np.random.beta(2, 5)
        
        snps = np.random.binomial(2, 0.3, 500).astype(np.float32)
        
        temporal_biomarkers = self._generate_temporal_biomarkers(age, prs)
        
        t2dm_onset = np.random.random() < self.t2dm_incidence
        if t2dm_onset:
            onset_quarter = np.random.randint(10, self.quarters)
        else:
            onset_quarter = self.quarters + 1
        
        return {
            'id': patient_id,
            'age': age,
            'gender': gender,
            'ethnicity': ethnicity,
            'prs': prs,
            'snps': snps,
            'temporal_biomarkers': temporal_biomarkers,
            't2dm_label': 1 if t2dm_onset else 0,
            'onset_quarter': onset_quarter
        }
    
    def _generate_temporal_biomarkers(self, age: int, prs: float) -> np.ndarray:
        biomarkers = np.zeros((self.quarters, 8), dtype=np.float32)
        
        base_glucose = 90 + age * 0.2 + prs * 20
        base_hba1c = 5.0 + prs * 0.8
        base_bmi = 24 + age * 0.1 + prs * 5
        
        for quarter in range(self.quarters):
            progression = quarter / self.quarters
            
            glucose_noise = np.random.normal(0, 5)
            hba1c_noise = np.random.normal(0, 0.2)
            bmi_noise = np.random.normal(0, 0.5)
            
            biomarkers[quarter, 0] = base_glucose + progression * 15 + glucose_noise
            biomarkers[quarter, 1] = base_hba1c + progression * 0.5 + hba1c_noise
            biomarkers[quarter, 2] = base_bmi + progression * 2 + bmi_noise
            biomarkers[quarter, 3] = 180 + progression * 30 + np.random.normal(0, 10)
            biomarkers[quarter, 4] = 40 - progression * 5 + np.random.normal(0, 3)
            biomarkers[quarter, 5] = 150 + progression * 50 + np.random.normal(0, 20)
            biomarkers[quarter, 6] = 120 + progression * 10 + np.random.normal(0, 5)
            biomarkers[quarter, 7] = 80 + progression * 5 + np.random.normal(0, 3)
        
        return biomarkers
    
    def _prepare_dataset(self, patients: List[Dict]) -> Dict:
        snps_list = []
        temporal_list = []
        labels_list = []
        
        for patient in patients:
            snps_list.append(patient['snps'])
            temporal_list.append(patient['temporal_biomarkers'])
            labels_list.append(patient['t2dm_label'])
        
        return {
            'snps': np.array(snps_list, dtype=np.float32),
            'temporal': np.array(temporal_list, dtype=np.float32),
            'labels': np.array(labels_list, dtype=np.float32).reshape(-1, 1)
        }
    
    def generate_patient_for_inference(self) -> Dict:
        patient = self._generate_patient_profile(0)
        
        return {
            'snps': patient['snps'],
            'temporal_biomarkers': patient['temporal_biomarkers'],
            'polygenic_risk_score': patient['prs'],
            'age': patient['age'],
            'gender': patient['gender'],
            'biomarkers': {
                'HbA1c': patient['temporal_biomarkers'][-1, 1],
                'fasting_glucose': patient['temporal_biomarkers'][-1, 0],
                'BMI': patient['temporal_biomarkers'][-1, 2],
                'triglycerides': patient['temporal_biomarkers'][-1, 5]
            },
            'hba1c_trend': 'increasing' if patient['temporal_biomarkers'][-1, 1] > patient['temporal_biomarkers'][0, 1] else 'stable',
            'glucose_trend': 'increasing' if patient['temporal_biomarkers'][-1, 0] > patient['temporal_biomarkers'][0, 0] else 'stable',
            'weight_trend': 'stable'
        }


class DistributedDataPartitioner:
    def __init__(self, train_data: Dict, n_clients: int = 10):
        self.train_data = train_data
        self.n_clients = n_clients
    
    def partition_iid(self) -> List[Dict]:
        n_samples = self.train_data['snps'].shape[0]
        samples_per_client = n_samples // self.n_clients
        
        client_datasets = []
        
        for client_id in range(self.n_clients):
            start_idx = client_id * samples_per_client
            end_idx = start_idx + samples_per_client if client_id < self.n_clients - 1 else n_samples
            
            client_data = {
                'snps': self.train_data['snps'][start_idx:end_idx],
                'temporal': self.train_data['temporal'][start_idx:end_idx],
                'labels': self.train_data['labels'][start_idx:end_idx]
            }
            
            client_datasets.append(client_data)
        
        return client_datasets
    
    def partition_non_iid(self, alpha: float = 0.5) -> List[Dict]:
        n_samples = self.train_data['snps'].shape[0]
        
        label_indices = {}
        for idx, label in enumerate(self.train_data['labels'].flatten()):
            label = int(label)
            if label not in label_indices:
                label_indices[label] = []
            label_indices[label].append(idx)
        
        client_datasets = [{'snps': [], 'temporal': [], 'labels': []} for _ in range(self.n_clients)]
        
        for label, indices in label_indices.items():
            np.random.shuffle(indices)
            
            label_distribution = np.random.dirichlet(np.ones(self.n_clients) * alpha)
            
            start_idx = 0
            for client_id, proportion in enumerate(label_distribution):
                n_samples_for_client = int(len(indices) * proportion)
                end_idx = start_idx + n_samples_for_client
                
                client_indices = indices[start_idx:end_idx]
                
                client_datasets[client_id]['snps'].append(self.train_data['snps'][client_indices])
                client_datasets[client_id]['temporal'].append(self.train_data['temporal'][client_indices])
                client_datasets[client_id]['labels'].append(self.train_data['labels'][client_indices])
                
                start_idx = end_idx
        
        for client_id in range(self.n_clients):
            client_datasets[client_id]['snps'] = np.vstack(client_datasets[client_id]['snps'])
            client_datasets[client_id]['temporal'] = np.vstack(client_datasets[client_id]['temporal'])
            client_datasets[client_id]['labels'] = np.vstack(client_datasets[client_id]['labels'])
        
        return client_datasets


class DataValidator:
    @staticmethod
    def validate_cohort(cohort: Dict) -> bool:
        required_keys = ['snps', 'temporal', 'labels']
        
        if not all(key in cohort for key in required_keys):
            return False
        
        n_samples = cohort['snps'].shape[0]
        
        if cohort['temporal'].shape[0] != n_samples:
            return False
        
        if cohort['labels'].shape[0] != n_samples:
            return False
        
        if np.isnan(cohort['snps']).any() or np.isnan(cohort['temporal']).any():
            return False
        
        return True
    
    @staticmethod
    def get_cohort_statistics(cohort: Dict) -> Dict:
        n_samples = cohort['snps'].shape[0]
        n_positive = int(cohort['labels'].sum())
        
        return {
            'n_samples': n_samples,
            'n_positive_cases': n_positive,
            'prevalence': n_positive / n_samples,
            'snp_mean': float(np.mean(cohort['snps'])),
            'snp_std': float(np.std(cohort['snps'])),
            'temporal_mean': float(np.mean(cohort['temporal'])),
            'temporal_std': float(np.std(cohort['temporal']))
        }
