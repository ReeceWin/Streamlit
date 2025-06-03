import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import json
import os

class BioBERTProcessor:
    """
    Class to handle BioBERT processing tasks including:
    - Extracting symptoms from natural language
    """
    
    def __init__(self, model_name="dmis-lab/biobert-v1.1", model_data_dir="model_data"):
        print("Initializing BioBERT processor...")
        
        # Load the BioBERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Load symptom mappings
        with open(os.path.join(model_data_dir, "symptom_to_idx.json"), "r") as f:
            self.symptom_to_idx = json.load(f)
        
        self.all_symptoms = list(self.symptom_to_idx.keys())
        print(f"Loaded {len(self.all_symptoms)} symptoms for processing")
        
        # Create symptom variants for better matching
        self.symptom_variants = self._create_symptom_variants()

    def _create_symptom_variants(self):
        """Create mapping of common descriptions to dataset symptoms"""
        variants = {}
        
        # Add common variants
        for symptom in self.all_symptoms:
            clean_symptom = symptom.replace('_', ' ')
            variants[clean_symptom] = symptom
            
            # Add some common variations
            if "pain" in clean_symptom:
                variants[clean_symptom.replace("pain", "ache")] = symptom
            if "headache" in clean_symptom:
                variants["head pain"] = symptom
            if "vomiting" in clean_symptom:
                variants["throwing up"] = symptom
            
        # Add additional common variants
        extra_variants = {
            "runny nose": "continuous_sneezing",
            "stuffy nose": "continuous_sneezing",
            "itchy eyes": "watering_from_eyes",
            "watery eyes": "watering_from_eyes",
            "short of breath": "breathlessness",
            "trouble breathing": "breathlessness",
            "stomach ache": "stomach_pain",
            "tummy pain": "stomach_pain",
            "heartburn": "acidity",
            "acid reflux": "acidity",
            "feeling sick": "nausea",
            "tired": "fatigue",
            "exhausted": "fatigue",
            "dizzy": "dizziness",
            "lightheaded": "dizziness"
        }
        
        # Merge the dictionaries
        variants.update(extra_variants)
        return variants

    def extract_symptoms(self, user_text):
        """
        Extract symptoms using both keyword matching and embeddings
        """
        # Convert to lowercase for matching
        text_lower = user_text.lower()
        
        # Track exact text matches to avoid duplicates
        exact_text_matches = set()
        keyword_matches = []
        
        # First check variants mapping
        for variant, symptom in self.symptom_variants.items():
            if variant in text_lower:
                keyword_matches.append(symptom)
                exact_text_matches.add(variant)  # Track this text pattern
        
        # Then check original symptom names, but only if not already matched by a variant
        for symptom in self.all_symptoms:
            clean_symptom = symptom.replace('_', ' ').lower()
            # Only add if this text wasn't matched by a variant already
            if clean_symptom in text_lower and clean_symptom not in exact_text_matches:
                keyword_matches.append(symptom)
                exact_text_matches.add(clean_symptom)
        
        # Remove duplicates
        keyword_matches = list(set(keyword_matches))
        
        # If we found at least one symptom with keyword matching, return it without using embeddings
        # This prevents embedding matching from adding too many false positives for simple inputs
        if keyword_matches:
            return keyword_matches
        
        # Only fall back to embeddings if we found no symptoms with keyword matching
        # and the user input is reasonably complex (more than a few words)
        words = user_text.split()
        if len(words) > 3:
            embedding_matches = self.extract_symptoms_with_embeddings(user_text)
            return list(set(keyword_matches + embedding_matches))
        else:
            # For very short inputs with no keyword matches, don't use embedding matching
            # to avoid false positives
            return keyword_matches
    
    def get_symptom_embeddings(self):
        """
        Pre-compute embeddings for all symptoms to use for similarity matching
        """
        # Process all symptoms to get embeddings
        all_embeddings = []
        
        # Clean symptoms for better processing (replace underscores with spaces)
        clean_symptoms = [s.replace('_', ' ') for s in self.all_symptoms]
        
        # Compute embeddings in batches
        for i in range(0, len(clean_symptoms), 8):  # Process in small batches
            batch = clean_symptoms[i:i+8]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Use [CLS] token embeddings as symptom representations
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            all_embeddings.append(embeddings)
        
        # Concatenate all embeddings
        self.symptom_embeddings = np.vstack(all_embeddings)
        print(f"Generated embeddings for {len(self.all_symptoms)} symptoms")
    
    def extract_symptoms_with_embeddings(self, user_text):
        """
        Extract symptoms using BioBERT embeddings for more robust matching
        """
        # Make sure we have symptom embeddings
        if not hasattr(self, 'symptom_embeddings'):
            self.get_symptom_embeddings()
        
        # Get embedding for user text
        inputs = self.tokenizer(user_text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use [CLS] token embedding
        text_embedding = outputs.last_hidden_state[:, 0, :].numpy()
        
        # Calculate similarities with all symptoms
        similarities = np.dot(self.symptom_embeddings, text_embedding.T).flatten()
        
        # Get top matching symptoms with a higher threshold for more selective matching
        threshold = 0.85  # Increased from 0.7
        matches = []
        
        for i, similarity in enumerate(similarities):
            if similarity > threshold:
                matches.append((self.all_symptoms[i], similarity))
        
        # Sort by similarity
        matches.sort(key=lambda x: x[1], reverse=True)
        
        # Limit the number of symptoms returned (top 3 most similar)
        max_embedding_matches = 3
        return [match[0] for match in matches[:max_embedding_matches]]


# Test the BioBERT processor
if __name__ == "__main__":
    processor = BioBERTProcessor()
    
    # Test symptoms extraction
    test_text = "I have a headache and my stomach hurts. I'm also feeling very tired."
    symptoms = processor.extract_symptoms(test_text)
    print(f"Hybrid matching extracted symptoms: {symptoms}")
    
    # Test embedding-based extraction
    processor.get_symptom_embeddings()
    symptoms_emb = processor.extract_symptoms_with_embeddings(test_text)
    print(f"Embedding-based extracted symptoms: {symptoms_emb}")