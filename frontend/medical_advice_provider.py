import json
import os

class MedicalAdviceProvider:
    """
    Provides medical advice from NHS database for the symptom checker chatbot
    """
    
    def __init__(self, database_path="medical_advice_database.json"):
        print("Initializing Medical Advice Provider...")
        self.database_path = database_path
        
        # Check if database exists
        if not os.path.exists(database_path):
            print(f"Warning: Database file {database_path} not found!")
            self.advice_db = {}
        else:
            # Load advice database
            with open(database_path, 'r') as f:
                self.advice_db = json.load(f)
            print(f"Loaded medical advice for {len(self.advice_db)} conditions")
    
    def get_advice(self, disease, symptoms):
        """
        Get advice for a disease, personalized for symptoms
        """
        if not self.advice_db:
            return self.get_fallback_advice(disease)
        
        # Check if disease exists in database
        if disease in self.advice_db:
            condition_data = self.advice_db[disease]
            
            # Format symptoms for display
            symptom_text = ", ".join([s.replace("_", " ") for s in symptoms])
            
            # Create output with advice
            output = f"Based on your symptoms ({symptom_text}), here's advice for {disease}:\n\n"
            
            # Add advice as bullet points
            if condition_data["advice"]:
                for advice in condition_data["advice"]:
                    output += f"• {advice}\n"
            
            # Add source attribution
            if "source" in condition_data and "url" in condition_data:
                output += f"\nSource: {condition_data['source']}"
                output += f"\nMore information: {condition_data['url']}"
            
            # Add disclaimer
            output += "\n\nDisclaimer: This is not a substitute for professional medical advice, diagnosis, or treatment."
            
            return output
        else:
            return self.get_fallback_advice(disease)
    
    def get_fallback_advice(self, disease):
        """Generate fallback advice when disease not in database"""
        return (f"For {disease}, please consult a healthcare professional for proper diagnosis and treatment.\n\n"
                f"• Rest and stay hydrated\n"
                f"• Monitor your symptoms\n"
                f"• Seek medical attention if symptoms worsen\n\n"
                f"Disclaimer: This is not a substitute for professional medical advice, diagnosis, or treatment.")
    
    def get_disease_info(self, disease):
        """Get general information about a disease"""
        if disease in self.advice_db:
            condition_data = self.advice_db[disease]
            
            output = f"Information about {disease}:\n\n"
            
            # Add symptoms if available
            if "symptoms" in condition_data and condition_data["symptoms"]:
                output += "Common symptoms include:\n"
                for symptom in condition_data["symptoms"]:
                    output += f"• {symptom}\n"
                output += "\n"
            
            # Add source information
            if "url" in condition_data:
                output += f"For more information, visit: {condition_data['url']}"
            
            return output
        else:
            return f"Information about {disease} is not available in my database. Please consult a healthcare professional for accurate information."


# Test function
if __name__ == "__main__":
    provider = MedicalAdviceProvider()
    
    # Test getting advice for a condition
    disease = "Allergy"
    symptoms = ["continuous_sneezing", "watering_from_eyes", "headache"]
    
    advice = provider.get_advice(disease, symptoms)
    print("==== ADVICE TEST ====")
    print(advice)
    print("\n")
    
    # Test getting disease info
    info = provider.get_disease_info(disease)
    print("==== DISEASE INFO TEST ====")
    print(info)