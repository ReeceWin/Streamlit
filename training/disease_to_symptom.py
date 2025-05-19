import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

class SymptomDataProcessor:
    """Process disease-symptom data for model training"""
    
    def __init__(self):
        self.all_symptoms = []
        self.all_diseases = []
        self.symptom_to_idx = {}
        self.disease_to_idx = {}
        self.idx_to_disease = {}
    
    def process_data(self, csv_path):
        """Process the CSV data and return processed datasets"""
        print(f"Loading data from {csv_path}")
        
        # Load data
        df = pd.read_csv(csv_path)
        disease_col = df.columns[0]
        
        # Extract unique diseases and symptoms
        disease_data = {}
        all_symptoms_set = set()
        
        # Process each row
        for _, row in tqdm(df.iterrows(), desc="Processing rows", total=len(df)):
            disease = row[disease_col]
            symptoms = []
            
            # Get all symptoms for this disease instance
            for col in df.columns[1:]:
                if pd.notna(row[col]) and str(row[col]).strip():
                    symptom = str(row[col]).strip()
                    symptoms.append(symptom)
                    all_symptoms_set.add(symptom)
            
            # Add to disease data dictionary
            if disease in disease_data:
                disease_data[disease].append(symptoms)
            else:
                disease_data[disease] = [symptoms]
        
        # Create mappings
        self.all_symptoms = sorted(list(all_symptoms_set))
        self.all_diseases = sorted(list(disease_data.keys()))
        
        self.symptom_to_idx = {symptom: idx for idx, symptom in enumerate(self.all_symptoms)}
        self.disease_to_idx = {disease: idx for idx, disease in enumerate(self.all_diseases)}
        self.idx_to_disease = {idx: disease for disease, idx in self.disease_to_idx.items()}
        
        print(f"Found {len(self.all_symptoms)} unique symptoms")
        print(f"Found {len(self.all_diseases)} unique diseases")
        
        # Create dataset records
        records = []
        
        for disease, symptom_sets in disease_data.items():
            for symptoms in symptom_sets:
                # Create a binary vector for symptoms
                symptom_vector = np.zeros(len(self.all_symptoms))
                for symptom in symptoms:
                    symptom_idx = self.symptom_to_idx[symptom]
                    symptom_vector[symptom_idx] = 1
                
                disease_idx = self.disease_to_idx[disease]
                records.append({
                    'symptom_vector': symptom_vector,
                    'disease_idx': disease_idx,
                    'disease': disease,
                    'symptoms': symptoms
                })
        
        # Use stratified split to ensure each disease is represented
        train_indices, test_indices = train_test_split(
            range(len(records)), 
            test_size=0.2, 
            random_state=42,
            stratify=[r['disease_idx'] for r in records]  # Stratify by disease
        )
        
        train_records = [records[i] for i in train_indices]
        test_records = [records[i] for i in test_indices]
        
        print(f"Training samples: {len(train_records)}")
        print(f"Testing samples: {len(test_records)}")
        
        return train_records, test_records
        
    def save_mappings(self, output_dir="./model_data"):
        """Save symptom and disease mappings"""
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "symptom_to_idx.json"), "w") as f:
            json.dump(self.symptom_to_idx, f)
            
        with open(os.path.join(output_dir, "disease_to_idx.json"), "w") as f:
            json.dump(self.disease_to_idx, f)
            
        with open(os.path.join(output_dir, "idx_to_disease.json"), "w") as f:
            json.dump({str(k): v for k, v in self.idx_to_disease.items()}, f)
            
        print(f"Saved mappings to {output_dir}")


def train_model(train_records, test_records, data_processor):
    """Train a random forest classifier for disease prediction"""
    # Prepare data
    X_train = np.array([record['symptom_vector'] for record in train_records])
    y_train = np.array([record['disease_idx'] for record in train_records])
    
    X_test = np.array([record['symptom_vector'] for record in test_records])
    y_test = np.array([record['disease_idx'] for record in test_records])
    
    # Create and train random forest
    print("\nTraining random forest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,  # Use 100 trees
        max_depth=None,    # Let trees grow fully
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight='balanced'  # Handle any class imbalance
    )
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Print classification report with disease names
    disease_names = [data_processor.idx_to_disease[idx] for idx in range(len(data_processor.idx_to_disease))]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=disease_names))
    
    # Calculate feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nTop 10 most important symptoms:")
    for i in range(min(10, len(data_processor.all_symptoms))):
        print(f"{i+1}. {data_processor.all_symptoms[indices[i]]}: {importances[indices[i]]:.4f}")
    
    return model


def save_model(model, output_dir="./model_data"):
    """Save the trained model"""
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, "disease_model.joblib"))
    print(f"Model saved to {output_dir}/disease_model.joblib")


def main():
    # Path to your CSV file
    csv_path = "dataset/DiseaseAndSymptoms.csv"
    
    # Process data
    data_processor = SymptomDataProcessor()
    train_records, test_records = data_processor.process_data(csv_path)
    
    # Save mappings for later use
    data_processor.save_mappings()
    
    # Train and evaluate model
    model = train_model(train_records, test_records, data_processor)
    
    # Save the model
    save_model(model)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()