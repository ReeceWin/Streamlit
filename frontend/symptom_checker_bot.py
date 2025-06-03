import joblib
import json
import os
import numpy as np
from biobert_processor import BioBERTProcessor
from medical_advice_provider import MedicalAdviceProvider

class SymptomCheckerBot:
    """
    A chatbot that uses a trained model to diagnose diseases based on symptoms
    and provides medical advice from the NHS conditions database.
    """
    
    def __init__(self, model_data_dir="model_data", medical_db="medical_advice_database.json"):
        print("Initializing Symptom Checker Bot...")
        
        # Load the model
        self.model = joblib.load(os.path.join(model_data_dir, "disease_model.joblib"))
        print("Disease model loaded successfully")
        
        # Load mappings
        with open(os.path.join(model_data_dir, "symptom_to_idx.json"), "r") as f:
            self.symptom_to_idx = json.load(f)
            self.all_symptoms = list(self.symptom_to_idx.keys())
        
        with open(os.path.join(model_data_dir, "idx_to_disease.json"), "r") as f:
            self.idx_to_disease = json.load(f)
        
        # Initialize BioBERT processor for symptom extraction
        self.biobert = BioBERTProcessor(model_data_dir=model_data_dir)
        
        # Initialize Medical Advice Provider for NHS advice
        self.advice_provider = MedicalAdviceProvider(medical_db)
        
        # Track conversation state
        self.current_symptoms = []
        self.conversation_stage = "greeting"
        
        # Track asked symptoms to avoid repetition
        self.asked_symptoms = []
        
        # Track current diagnosis
        self.current_diagnosis = None
        self.diagnosis_confidence = 0.0
        
        # High-value symptoms to ask about based on model
        self.high_value_symptoms = self.get_high_value_symptoms()
        
        # Track user's original symptom descriptions
        self.user_symptom_terms = {}
        
        print("Symptom Checker Bot initialized and ready to help!")
    
    def get_high_value_symptoms(self):
        """Get high-value symptoms for questioning based on model feature importance"""
        try:
            # For Random Forest, get feature importances
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Get the top symptoms
            return [self.all_symptoms[indices[i]] for i in range(min(30, len(self.all_symptoms)))]
        except:
            # Fallback if feature importances aren't available
            return self.all_symptoms[:30]
    
    def get_next_question(self):
        """Determine the next symptom to ask about with improved phrasing"""
        # Only handle the most problematic non-symptom items
        non_symptoms = {
            "family history": "Do you have any family history of similar conditions?",
            "history of alcohol consumption": "Do you have a history of alcohol consumption?",
            "receiving blood transfusion": "Have you received a blood transfusion?",
            "receiving unsterile injections": "Have you received any unsterile injections?",
            "extra marital contacts": "Have you had multiple sexual partners?",
            "obesity": "Would you describe yourself as overweight?",
            "coma": "Have you experienced loss of consciousness?"
        }
        
        # If we already have a likely diagnosis, ask about confirming symptoms
        if self.current_diagnosis and self.diagnosis_confidence >= 0.4:
            # Find symptoms typical for this diagnosis that haven't been asked
            disease_specific_symptoms = self.get_disease_specific_symptoms(self.current_diagnosis)
            for symptom in disease_specific_symptoms:
                if symptom not in self.current_symptoms and symptom not in self.asked_symptoms:
                    self.asked_symptoms.append(symptom)
                    formatted_symptom = symptom.replace("_", " ")
                    
                    # Use improved phrasing for non-symptoms
                    if formatted_symptom in non_symptoms:
                        return non_symptoms[formatted_symptom]
                    else:
                        return f"Are you experiencing {formatted_symptom}?"
        
        # Check high-value symptoms
        for symptom in self.high_value_symptoms:
            if symptom not in self.current_symptoms and symptom not in self.asked_symptoms:
                self.asked_symptoms.append(symptom)
                formatted_symptom = symptom.replace("_", " ")
                
                if formatted_symptom in non_symptoms:
                    return non_symptoms[formatted_symptom]
                else:
                    return f"Are you experiencing {formatted_symptom}?"
        
        # Check remaining symptoms
        for symptom in self.all_symptoms:
            if symptom not in self.current_symptoms and symptom not in self.asked_symptoms:
                self.asked_symptoms.append(symptom)
                formatted_symptom = symptom.replace("_", " ")
                
                if formatted_symptom in non_symptoms:
                    return non_symptoms[formatted_symptom]
                else:
                    return f"Are you experiencing {formatted_symptom}?"
        
        return None
    
    def get_disease_specific_symptoms(self, disease):
        """Get symptoms that are specific to a disease for targeted questioning"""
        # Common conditions and their symptoms
        disease_symptoms = {
            "Fungal infection": ["itching", "skin_rash", "nodal_skin_eruptions", "dischromic_patches"],
            "Allergy": ["continuous_sneezing", "shivering", "chills", "watering_from_eyes"],
            "GERD": ["stomach_pain", "acidity", "ulcers_on_tongue", "vomiting", "cough", "chest_pain"],
            "Chronic cholestasis": ["itching", "vomiting", "yellowish_skin", "nausea", "loss_of_appetite", "abdominal_pain"],
            "Diabetes": ["fatigue", "weight_loss", "restlessness", "lethargy", "irregular_sugar_level", "excessive_hunger"],
            "Bronchial Asthma": ["fatigue", "cough", "high_fever", "breathlessness", "mucoid_sputum"],
            "Hypertension": ["headache", "chest_pain", "dizziness", "loss_of_balance", "lack_of_concentration"]
        }
        
        return disease_symptoms.get(disease, self.high_value_symptoms[:5])
    
    def diagnose(self):
        """Make a diagnosis based on current symptoms"""
        if not self.current_symptoms:
            return "Unknown", 0.0, []
            
        # Create symptom vector
        symptom_vector = np.zeros(len(self.symptom_to_idx))
        for symptom in self.current_symptoms:
            if symptom in self.symptom_to_idx:
                symptom_vector[self.symptom_to_idx[symptom]] = 1
        
        # For Random Forest, we can get probability estimates
        probabilities = self.model.predict_proba([symptom_vector])[0]
        disease_idx = np.argmax(probabilities)
        confidence = probabilities[disease_idx]
        
        disease = self.idx_to_disease.get(str(disease_idx), "Unknown condition")
        
        # Get top 3 diagnoses with probabilities
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_diseases = [(self.idx_to_disease.get(str(idx), "Unknown"), probabilities[idx]) 
                       for idx in top_indices if probabilities[idx] > 0.05]
        
        return disease, confidence, top_diseases
    
    def process_yes_no(self, user_input):
        """Process yes/no answers to symptom questions"""
        # Get the last symptom we asked about
        if not self.asked_symptoms:
            return False
        
        last_symptom = self.asked_symptoms[-1]
        
        # Check if user said yes
        if any(word in user_input.lower() for word in ["yes", "yeah", "yep", "correct", "right", "true", "i do", "i am"]):
            if last_symptom not in self.current_symptoms:
                self.current_symptoms.append(last_symptom)
                # For direct questions, we use the standard symptom name
                self.user_symptom_terms[last_symptom] = last_symptom.replace("_", " ")
            return True
            
        # Check if user said no - no action needed as we didn't add the symptom
        elif any(word in user_input.lower() for word in ["no", "nope", "not", "don't", "doesn't", "isn't", "aren't", "i'm not"]):
            return True
            
        # Unclear response
        return False
    
    def find_original_terms(self, user_text, symptoms):
        """Find the original terms user used for symptoms"""
        text_lower = user_text.lower()
        original_terms = {}
        
        # Check variants first as they're more specific
        for variant, symptom in self.biobert.symptom_variants.items():
            if variant in text_lower and symptom in symptoms:
                original_terms[symptom] = variant
        
        # Check standard symptom names
        for symptom in symptoms:
            clean_symptom = symptom.replace('_', ' ').lower()
            if clean_symptom in text_lower and symptom not in original_terms:
                original_terms[symptom] = clean_symptom
                
        return original_terms
    
    def generate_response(self, user_input):
        """Generate a response based on the current conversation stage"""
        if self.conversation_stage == "greeting":
            # The app.py has already prompted for symptoms.
            # user_input here is the first set of symptoms.
            self.conversation_stage = "collecting_symptoms"
            # No return statement here; processing will fall through to the
            # "collecting_symptoms" stage with the current user_input.

        if self.conversation_stage == "collecting_symptoms": # Changed from elif to if
            # Check if this is a yes/no response to our question
            if self.process_yes_no(user_input):
                # User answered our question, acknowledge and continue
                response = "I understand. "
            else:
                # Extract symptoms from user input
                new_symptoms = self.biobert.extract_symptoms(user_input)
                
                # Find original terms user used for symptoms
                original_terms = self.find_original_terms(user_input, new_symptoms)
                
                # Add new symptoms to our list and track original terms
                for symptom in new_symptoms:
                    if symptom not in self.current_symptoms:
                        self.current_symptoms.append(symptom)
                        # Store user's original term if found
                        if symptom in original_terms:
                            self.user_symptom_terms[symptom] = original_terms[symptom]
                        else:
                            # Default to standard format
                            self.user_symptom_terms[symptom] = symptom.replace('_', ' ')
                
                # If we found symptoms in this input, acknowledge them with user's terms
                if new_symptoms:
                    # Display symptoms using user's original terms
                    symptom_display = [self.user_symptom_terms[s] for s in new_symptoms]
                    response = f"I understand you have the following symptoms: {', '.join(symptom_display)}. "
                    
                    # Explain any differences between user terms and medical terms
                    mappings = []
                    
                    if mappings:
                        response += f"In medical terms, I'm recognizing {', '.join(mappings)}. "
                else:
                    response = "I'll need to ask you some questions to better understand your condition. "
            
            # Update current diagnosis
            if self.current_symptoms:
                self.current_diagnosis, self.diagnosis_confidence, top_diseases = self.diagnose()
            
            # Check if we have enough symptoms to make a diagnosis
            if len(self.current_symptoms) >= 3:
                # Get potential diagnosis
                disease, confidence, top_diseases = self.diagnose()
                
                # Set a maximum number of symptoms to collect before making a diagnosis regardless of confidence
                max_symptoms_to_collect = 12
                
                # More strict confidence thresholds
                if confidence > 0.7 and len(self.current_symptoms) >= 3:
                    # Very confident diagnosis
                    self.conversation_stage = "giving_advice"
                    
                    # Format top diseases
                    disease_list = ", ".join([f"{d} ({c:.0%})" for d, c in top_diseases if d != disease])
                    
                    # Display user's original terms for symptoms in the diagnosis
                    user_symptom_list = [self.user_symptom_terms[s] for s in self.current_symptoms]
                    
                    response += f"Based on your symptoms, you most likely have {disease} (confidence: {confidence:.0%}).\n\n"
                    
                    if disease_list:
                        response += f"Other possibilities include: {disease_list}.\n\n"
                        
                    # Get NHS advice for this condition
                    advice = self.advice_provider.get_advice(disease, self.current_symptoms)
                    response += f"Based on your symptoms ({', '.join(user_symptom_list)}), here's advice for {disease}:\n{advice}\n\n"
                    response += "Would you like more information about this condition?"
                    
                elif confidence > 0.5 and len(self.current_symptoms) >= 4:
                    # Moderately confident diagnosis with more symptoms
                    self.conversation_stage = "giving_advice"
                    
                    # Format top diseases
                    disease_list = ", ".join([f"{d} ({c:.0%})" for d, c in top_diseases if d != disease])
                    
                    # Display user's original terms for symptoms in the diagnosis
                    user_symptom_list = [self.user_symptom_terms[s] for s in self.current_symptoms]
                    
                    response += f"Based on your symptoms, you likely have {disease} (confidence: {confidence:.0%}).\n\n"
                    
                    if disease_list:
                        response += f"Other possibilities include: {disease_list}.\n\n"
                        
                    # Get NHS advice for this condition
                    advice = self.advice_provider.get_advice(disease, self.current_symptoms)
                    response += f"Based on your symptoms ({', '.join(user_symptom_list)}), here's advice for {disease}:\n{advice}\n\n"
                    response += "Would you like more information about this condition?"
                    
                elif len(self.asked_symptoms) >= max_symptoms_to_collect:
                    # If we've asked enough questions but still have low confidence, give best guess diagnosis
                    self.conversation_stage = "giving_advice"
                    
                    # Display user's original terms for symptoms in the diagnosis
                    user_symptom_list = [self.user_symptom_terms[s] for s in self.current_symptoms]
                    
                    response += f"Based on the symptoms you've reported, the most likely condition is {disease}, with a confidence of {confidence:.0%}.\n\n"
                    
                    # Format top diseases
                    disease_list = ", ".join([f"{d} ({c:.0%})" for d, c in top_diseases])
                    response += f"Given the variety of symptoms, other possibilities include: {disease_list}.\n\n"
                    
                    # Get NHS advice for this condition
                    advice = self.advice_provider.get_advice(disease, self.current_symptoms)
                    response += f"Based on your symptoms ({', '.join(user_symptom_list)}), here's advice for {disease}:\n{advice}\n\n"
                    response += "Would you like more information about this condition?"
                    
                else:
                    # Need more information for a confident diagnosis
                    next_question = self.get_next_question()
                    if next_question:
                        response += next_question
                    else:
                        # If no more questions but low confidence, give tentative diagnosis
                        self.conversation_stage = "giving_advice"
                        
                        # Display user's original terms for symptoms
                        user_symptom_list = [self.user_symptom_terms[s] for s in self.current_symptoms]
                        
                        response += f"Based on limited information, you might have {disease}, but I'm not very confident ({confidence:.0%}).\n\n"
                        
                        # Format top diseases
                        disease_list = ", ".join([f"{d} ({c:.0%})" for d, c in top_diseases])
                        response += f"Possibilities include: {disease_list}.\n\n"
                        
                        # Get NHS advice for this condition
                        advice = self.advice_provider.get_advice(disease, self.current_symptoms)
                        response += f"Based on your symptoms ({', '.join(user_symptom_list)}), here's advice for {disease}:\n{advice}"
            else:
                # Need more symptoms
                next_question = self.get_next_question()
                if next_question:
                    response += next_question
                else:
                    # If no specific questions but too few symptoms
                    response += "Could you tell me more about how you're feeling? What symptoms are bothering you the most?"
            
            return response
        
        elif self.conversation_stage == "giving_advice":
            # Check if user wants more information
            if "yes" in user_input.lower() or "more" in user_input.lower() or "information" in user_input.lower():
                disease, _, _ = self.diagnose()
                info = self.advice_provider.get_disease_info(disease)
                return f"{info}\n\nIs there anything else you'd like to know about this condition?"
            else:
                # Reset for new conversation
                self.reset_conversation()
                return "Is there anything else I can help you with? You can describe new symptoms if you'd like."
    
    def reset_conversation(self):
        """Reset the conversation to start fresh"""
        self.current_symptoms = []
        self.asked_symptoms = []
        self.conversation_stage = "collecting_symptoms"
        self.current_diagnosis = None
        self.diagnosis_confidence = 0.0
        self.user_symptom_terms = {}
    
    def start_conversation(self):
        """Start an interactive conversation in the console"""
        print("Starting Symptom Checker Bot...")
        
        # Don't print greeting here, let generate_response handle it
        response = self.generate_response("")
        print(f"Bot: {response}")
        
        while True:
            # Get user input
            user_input = input("You: ")
            
            # Check for exit command
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Bot: Thank you for using the Symptom Checker Bot. Take care!")
                break
            
            # Generate and print response
            response = self.generate_response(user_input)
            print(f"Bot: {response}")


# Main function to run the bot
if __name__ == "__main__":
    bot = SymptomCheckerBot()
    bot.start_conversation()