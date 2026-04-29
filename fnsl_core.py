import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import networkx as nx
from typing import Dict, List, Tuple, Optional
import json


class SNPFeatureExtractor(keras.Model):
    def __init__(self, n_snps: int, hidden_dim: int = 256):
        super().__init__()
        self.dense1 = layers.Dense(hidden_dim, activation='relu')
        self.dense2 = layers.Dense(hidden_dim // 2, activation='relu')
        self.output_layer = layers.Dense(128)
        
    def call(self, snp_data):
        x = self.dense1(snp_data)
        x = self.dense2(x)
        return self.output_layer(x)


class TemporalGRUModule(keras.Model):
    def __init__(self, gru_units: int = 128):
        super().__init__()
        self.gru_layer = layers.GRU(gru_units, return_sequences=False)
        self.dropout = layers.Dropout(0.3)
        
    def call(self, temporal_data, training=False):
        x = self.gru_layer(temporal_data)
        return self.dropout(x, training=training)


class MedicalKnowledgeGraph:
    def __init__(self, kg_path: Optional[str] = None):
        self.graph = nx.DiGraph()
        self.embeddings = {}
        self.entity_to_idx = {}
        self.relationship_types = set()
        
        if kg_path:
            self.load_from_file(kg_path)
        else:
            self._initialize_default_kg()
    
    def _initialize_default_kg(self):
        genes = ['FOXO1', 'TCF7L2', 'PPARG', 'INS', 'GCK', 'ABCC8']
        pathways = ['Insulin_Signaling', 'Glucose_Metabolism', 'Beta_Cell_Function']
        diseases = ['T2DM', 'Prediabetes', 'Metabolic_Syndrome']
        biomarkers = ['HbA1c', 'Fasting_Glucose', 'BMI', 'Triglycerides']
        
        all_entities = genes + pathways + diseases + biomarkers
        for idx, entity in enumerate(all_entities):
            self.entity_to_idx[entity] = idx
            self.graph.add_node(entity)
        
        gene_pathway_edges = [
            ('FOXO1', 'Insulin_Signaling'),
            ('TCF7L2', 'Glucose_Metabolism'),
            ('PPARG', 'Insulin_Signaling'),
            ('INS', 'Beta_Cell_Function'),
        ]
        
        pathway_disease_edges = [
            ('Insulin_Signaling', 'T2DM'),
            ('Glucose_Metabolism', 'T2DM'),
            ('Beta_Cell_Function', 'T2DM'),
            ('Insulin_Signaling', 'Metabolic_Syndrome'),
        ]
        
        disease_biomarker_edges = [
            ('T2DM', 'HbA1c'),
            ('T2DM', 'Fasting_Glucose'),
            ('Metabolic_Syndrome', 'Triglycerides'),
            ('Metabolic_Syndrome', 'BMI'),
        ]
        
        for source, target in gene_pathway_edges + pathway_disease_edges + disease_biomarker_edges:
            self.graph.add_edge(source, target)
            self.relationship_types.add('regulates')
        
        self._generate_embeddings()
    
    def _generate_embeddings(self):
        n_entities = len(self.entity_to_idx)
        embedding_dim = 128
        
        for entity in self.graph.nodes():
            self.embeddings[entity] = np.random.randn(embedding_dim).astype(np.float32)
    
    def load_from_file(self, path: str):
        with open(path, 'r') as f:
            kg_data = json.load(f)
        
        for entity in kg_data.get('entities', []):
            self.graph.add_node(entity)
            self.entity_to_idx[entity] = len(self.entity_to_idx)
        
        for edge in kg_data.get('edges', []):
            self.graph.add_edge(edge['source'], edge['target'])
            self.relationship_types.add(edge.get('type', 'related_to'))
        
        self._generate_embeddings()
    
    def get_entity_embedding(self, entity: str) -> np.ndarray:
        return self.embeddings.get(entity, np.zeros(128, dtype=np.float32))
    
    def get_related_entities(self, entity: str, depth: int = 2) -> List[str]:
        if entity not in self.graph:
            return []
        
        related = set()
        for d in range(1, depth + 1):
            for node in nx.descendants(self.graph, entity):
                related.add(node)
        
        return list(related)


class NeuroSymbolicFusionLayer(keras.Model):
    def __init__(self, kg: MedicalKnowledgeGraph, fusion_dim: int = 256):
        super().__init__()
        self.kg = kg
        self.attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)
        self.fusion_dense = layers.Dense(fusion_dim, activation='relu')
        self.output_projection = layers.Dense(128)
        
    def call(self, neural_features, kg_context):
        attended = self.attention(neural_features, kg_context)
        fused = layers.concatenate([neural_features, attended])
        x = self.fusion_dense(fused)
        return self.output_projection(x)


class ConstraintEngine:
    def __init__(self, kg: MedicalKnowledgeGraph):
        self.kg = kg
        self.constraints = self._initialize_constraints()
    
    def _initialize_constraints(self) -> Dict:
        return {
            'high_hba1c_implies_risk': {
                'condition': lambda biomarkers: biomarkers.get('HbA1c', 0) > 6.5,
                'weight': 0.8
            },
            'genetic_risk_multiplier': {
                'condition': lambda prs: prs > 0.7,
                'weight': 0.6
            },
            'age_interaction': {
                'condition': lambda age: age > 45,
                'weight': 0.4
            },
            'bmi_threshold': {
                'condition': lambda bmi: bmi > 30,
                'weight': 0.5
            }
        }
    
    def apply_constraints(self, prediction: float, patient_data: Dict) -> float:
        adjusted_prediction = prediction
        
        for constraint_name, constraint in self.constraints.items():
            if constraint_name == 'high_hba1c_implies_risk':
                if constraint['condition'](patient_data.get('biomarkers', {})):
                    adjusted_prediction = min(1.0, adjusted_prediction + constraint['weight'] * 0.1)
            
            elif constraint_name == 'genetic_risk_multiplier':
                prs = patient_data.get('polygenic_risk_score', 0)
                if constraint['condition'](prs):
                    adjusted_prediction = min(1.0, adjusted_prediction * (1 + constraint['weight']))
            
            elif constraint_name == 'age_interaction':
                age = patient_data.get('age', 0)
                if constraint['condition'](age):
                    adjusted_prediction = min(1.0, adjusted_prediction + constraint['weight'] * 0.05)
            
            elif constraint_name == 'bmi_threshold':
                bmi = patient_data.get('bmi', 0)
                if constraint['condition'](bmi):
                    adjusted_prediction = min(1.0, adjusted_prediction + constraint['weight'] * 0.08)
        
        return adjusted_prediction


class NeuroSymbolicModel(keras.Model):
    def __init__(self, n_snps: int = 500, temporal_steps: int = 40, 
                 kg_dim: int = 128, privacy_budget: float = 1.0):
        super().__init__()
        self.snp_extractor = SNPFeatureExtractor(n_snps)
        self.temporal_module = TemporalGRUModule()
        self.knowledge_graph = MedicalKnowledgeGraph()
        self.constraint_engine = ConstraintEngine(self.knowledge_graph)
        
        self.kg_embedding_layer = layers.Dense(kg_dim)
        self.fusion_layer = NeuroSymbolicFusionLayer(self.knowledge_graph)
        
        self.classifier = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        self.privacy_budget = privacy_budget
    
    def call(self, inputs, training=False):
        snp_data = inputs['snps']
        temporal_data = inputs['temporal_biomarkers']
        
        snp_features = self.snp_extractor(snp_data)
        temporal_features = self.temporal_module(temporal_data, training=training)
        
        kg_entities = self._get_relevant_kg_embeddings(inputs)
        kg_context = self.kg_embedding_layer(kg_entities)
        
        fused_features = self.fusion_layer(
            layers.concatenate([snp_features, temporal_features]),
            kg_context
        )
        
        prediction = self.classifier(fused_features, training=training)
        
        return prediction
    
    def _get_relevant_kg_embeddings(self, inputs) -> tf.Tensor:
        key_entities = ['FOXO1', 'TCF7L2', 'Insulin_Signaling', 'T2DM']
        embeddings = []
        
        for entity in key_entities:
            emb = self.knowledge_graph.get_entity_embedding(entity)
            embeddings.append(emb)
        
        return tf.constant(np.array(embeddings), dtype=tf.float32)
    
    def predict_with_explanation(self, inputs: Dict) -> Tuple[float, Dict]:
        prediction = self.predict(inputs)
        
        snp_data = inputs.get('snps', {})
        temporal_data = inputs.get('temporal_biomarkers', {})
        
        explanation = {
            'risk_score': float(prediction[0][0]),
            'key_genetic_factors': self._extract_genetic_factors(snp_data),
            'key_biomarker_trends': self._extract_biomarker_trends(temporal_data),
            'activated_pathways': self._get_activated_pathways(inputs),
            'clinical_reasoning': self._generate_clinical_reasoning(inputs, prediction)
        }
        
        return prediction, explanation
    
    def _extract_genetic_factors(self, snp_data: Dict) -> List[str]:
        high_risk_genes = ['FOXO1', 'TCF7L2', 'PPARG']
        return [gene for gene in high_risk_genes if np.random.random() > 0.5]
    
    def _extract_biomarker_trends(self, temporal_data: Dict) -> List[str]:
        trends = []
        if temporal_data.get('hba1c_trend') == 'increasing':
            trends.append('Rising HbA1c levels over past 5 years')
        if temporal_data.get('glucose_trend') == 'increasing':
            trends.append('Elevated fasting glucose trajectory')
        return trends
    
    def _get_activated_pathways(self, inputs: Dict) -> List[str]:
        return ['Insulin_Signaling', 'Glucose_Metabolism', 'Beta_Cell_Function']
    
    def _generate_clinical_reasoning(self, inputs: Dict, prediction: np.ndarray) -> str:
        risk_score = float(prediction[0][0])
        
        if risk_score > 0.7:
            return "High genetic predisposition combined with deteriorating biomarker profiles indicates substantial T2DM risk"
        elif risk_score > 0.4:
            return "Moderate risk profile with multiple contributing factors requiring close monitoring"
        else:
            return "Low to moderate risk with favorable biomarker trends"


class FederatedClient:
    def __init__(self, client_id: int, model: NeuroSymbolicModel, 
                 privacy_epsilon: float = 1.0, privacy_delta: float = 1e-5):
        self.client_id = client_id
        self.model = model
        self.privacy_epsilon = privacy_epsilon
        self.privacy_delta = privacy_delta
        self.local_data = None
    
    def set_local_data(self, data: Dict):
        self.local_data = data
    
    def local_training_step(self, batch_size: int = 32, epochs: int = 1):
        if self.local_data is None:
            return None
        
        x_snps = self.local_data['snps']
        x_temporal = self.local_data['temporal']
        y = self.local_data['labels']
        
        dataset = tf.data.Dataset.from_tensor_slices((
            {'snps': x_snps, 'temporal_biomarkers': x_temporal},
            y
        )).batch(batch_size)
        
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        
        for epoch in range(epochs):
            for batch_x, batch_y in dataset:
                with tf.GradientTape() as tape:
                    predictions = self.model(batch_x, training=True)
                    loss = keras.losses.binary_crossentropy(batch_y, predictions)
                    loss = tf.reduce_mean(loss)
                
                gradients = tape.gradient(loss, self.model.trainable_weights)
                clipped_gradients = self._apply_dp_clipping(gradients)
                optimizer.apply_gradients(zip(clipped_gradients, self.model.trainable_weights))
        
        return self._get_model_update()
    
    def _apply_dp_clipping(self, gradients: List[tf.Tensor]) -> List[tf.Tensor]:
        clipped_gradients = []
        l2_norm_clip = 1.0
        
        for grad in gradients:
            if grad is None:
                clipped_gradients.append(None)
            else:
                grad_norm = tf.norm(grad)
                scale = tf.minimum(1.0, l2_norm_clip / (grad_norm + 1e-8))
                clipped_grad = grad * scale
                
                noise_scale = l2_norm_clip * np.sqrt(2 * np.log(1.25 / self.privacy_delta)) / self.privacy_epsilon
                noise = tf.random.normal(shape=tf.shape(clipped_grad), stddev=noise_scale)
                
                clipped_gradients.append(clipped_grad + noise)
        
        return clipped_gradients
    
    def _get_model_update(self) -> Dict:
        return {
            'client_id': self.client_id,
            'weights': [w.numpy() for w in self.model.trainable_weights]
        }


class FederatedTrainer:
    def __init__(self, model: NeuroSymbolicModel, n_clients: int = 10,
                 privacy_epsilon: float = 1.0, privacy_delta: float = 1e-5):
        self.model = model
        self.n_clients = n_clients
        self.privacy_epsilon = privacy_epsilon
        self.privacy_delta = privacy_delta
        self.clients = [
            FederatedClient(i, model, privacy_epsilon, privacy_delta) 
            for i in range(n_clients)
        ]
        self.training_history = []
    
    def train(self, client_datasets: List[Dict], epochs: int = 10, 
              rounds: int = 100) -> Dict:
        
        for round_num in range(rounds):
            client_updates = []
            
            for client_idx, client in enumerate(self.clients):
                if client_idx < len(client_datasets):
                    client.set_local_data(client_datasets[client_idx])
                    update = client.local_training_step(batch_size=32, epochs=epochs)
                    if update:
                        client_updates.append(update)
            
            if client_updates:
                self._aggregate_updates(client_updates)
            
            if round_num % 10 == 0:
                print(f"Federated Round {round_num + 1}/{rounds} completed")
                self.training_history.append({
                    'round': round_num,
                    'n_clients_updated': len(client_updates)
                })
        
        return {'history': self.training_history, 'final_model': self.model}
    
    def _aggregate_updates(self, client_updates: List[Dict]):
        n_clients = len(client_updates)
        
        aggregated_weights = None
        
        for update in client_updates:
            weights = update['weights']
            
            if aggregated_weights is None:
                aggregated_weights = [w / n_clients for w in weights]
            else:
                aggregated_weights = [
                    agg_w + w / n_clients 
                    for agg_w, w in zip(aggregated_weights, weights)
                ]
        
        if aggregated_weights:
            for model_weight, agg_weight in zip(self.model.trainable_weights, aggregated_weights):
                model_weight.assign(tf.constant(agg_weight, dtype=model_weight.dtype))


class ExplanationGenerator:
    def __init__(self, model: NeuroSymbolicModel):
        self.model = model
    
    def generate_explanation(self, patient_data: Dict, prediction: float) -> Dict:
        explanation = {
            'patient_id': patient_data.get('id'),
            'risk_score': prediction,
            'risk_category': self._categorize_risk(prediction),
            'genetic_profile': self._analyze_genetics(patient_data),
            'biomarker_analysis': self._analyze_biomarkers(patient_data),
            'temporal_trends': self._analyze_trends(patient_data),
            'clinical_summary': self._generate_summary(patient_data, prediction)
        }
        
        return explanation
    
    def _categorize_risk(self, score: float) -> str:
        if score > 0.7:
            return 'High Risk'
        elif score > 0.4:
            return 'Moderate Risk'
        else:
            return 'Low Risk'
    
    def _analyze_genetics(self, patient_data: Dict) -> Dict:
        return {
            'polygenic_risk_score': patient_data.get('prs', 0),
            'high_risk_variants': patient_data.get('variants', []),
            'family_history': patient_data.get('family_history', False)
        }
    
    def _analyze_biomarkers(self, patient_data: Dict) -> Dict:
        biomarkers = patient_data.get('biomarkers', {})
        return {
            'hba1c': biomarkers.get('HbA1c'),
            'fasting_glucose': biomarkers.get('fasting_glucose'),
            'bmi': biomarkers.get('BMI'),
            'triglycerides': biomarkers.get('triglycerides')
        }
    
    def _analyze_trends(self, patient_data: Dict) -> Dict:
        return {
            'hba1c_trend': patient_data.get('hba1c_trend', 'stable'),
            'glucose_trend': patient_data.get('glucose_trend', 'stable'),
            'weight_trend': patient_data.get('weight_trend', 'stable')
        }
    
    def _generate_summary(self, patient_data: Dict, prediction: float) -> str:
        risk_category = self._categorize_risk(prediction)
        
        summary_templates = {
            'High Risk': f"Patient presents high T2DM risk ({prediction:.1%}) with multiple concerning factors requiring immediate intervention",
            'Moderate Risk': f"Patient shows moderate T2DM risk ({prediction:.1%}) with several risk factors warranting close monitoring",
            'Low Risk': f"Patient demonstrates low T2DM risk ({prediction:.1%}) with favorable risk profile"
        }
        
        return summary_templates.get(risk_category, "Risk assessment complete")
