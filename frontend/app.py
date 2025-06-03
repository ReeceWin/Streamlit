import streamlit as st
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import joblib
import json
import io
from datetime import datetime
import re


if not os.path.exists("symptom_checker_bot.py"):
    with open("symptom_checker_bot.py", "w") as f:
        f.write("""
class SymptomCheckerBot:
    def __init__(self, model_data_dir=None, medical_db=None):
        self.greeted = True  # Changed from False to True
        print("SymptomCheckerBot initialized")
    def generate_response(self, user_input):
        if not self.greeted: # This block will now be skipped on the first call
            self.greeted = True
            return "Hello from SymptomCheckerBot! How can I help you with your symptoms?"
        if "done" in user_input.lower():
            return "Okay, understood. Let me know if you need anything else."
        return f"SymptomCheckerBot received: {user_input}. Please tell me more or type 'done'."
    def reset_conversation(self):
        self.greeted = False
        print("SymptomCheckerBot conversation reset")
""")
if not os.path.exists("biobert_processor.py"):
    with open("biobert_processor.py", "w") as f:
        f.write("""
class BioBERTProcessor:
    def __init__(self):
        print("BioBERTProcessor initialized")
    def process(self, text):
        return f"Processed by BioBERT: {text}"
""")
if not os.path.exists("medical_advice_provider.py"):
    with open("medical_advice_provider.py", "w") as f:
        f.write("""
class MedicalAdviceProvider:
    def __init__(self):
        print("MedicalAdviceProvider initialized")
    def get_advice(self, condition):
        return f"General advice for {condition}: Consult a doctor."
""")
if not os.path.exists("medical_advice_database.json"):
    with open("medical_advice_database.json", "w") as f:
        json.dump({"conditions": []}, f)


from symptom_checker_bot import SymptomCheckerBot
from biobert_processor import BioBERTProcessor
from medical_advice_provider import MedicalAdviceProvider

# Set page configuration
st.set_page_config(
    page_title="Medical Symptom & Skin Disease Checker",
    page_icon="🩺",
    layout="wide"
)

# ====================================
# SKIN DISEASE MODEL COMPONENTS
# ====================================

class CombinedModel(nn.Module):
    """
    Dual-pathway model that processes images and tabular data separately
    before combining them for classification.
    """
    def __init__(self, num_classes, tabular_dim):
        super(CombinedModel, self).__init__()
        
        # Image feature extraction using EfficientNet-B3
        self.image_model = models.efficientnet_b3(weights='IMAGENET1K_V1')
        # EfficientNet-B3 feature dimension before the final classifier layer
        # For efficientnet_b3, the classifier is nn.Linear(1536, num_classes)
        # So, the features fed into it are of size 1536
        self.image_features_dim = 1536 
        self.image_model.classifier = nn.Identity()  # Remove classifier
        
        # Image-specific processing path
        self.image_fc = nn.Sequential(
            nn.Linear(self.image_features_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Tabular data processing path
        self.tabular_model = nn.Sequential(
            nn.BatchNorm1d(tabular_dim),
            nn.Linear(tabular_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, images, tabular):
        # Process image pathway
        img_features = self.image_model(images) # Output shape: (batch_size, 1536)
        img_features = self.image_fc(img_features)
        
        # Process tabular path
        tab_features = self.tabular_model(tabular)
        
        # Combine features
        combined = torch.cat((img_features, tab_features), dim=1)
        
        # Final classification
        return self.classifier(combined)

@st.cache_resource
def load_skin_model_and_metadata():

    # Define the possible locations where your trained model files might be
    possible_paths = [
        ("", "model_metadata.joblib", "final_skin_model.pth"), # Current frontend directory
        ("models", "model_metadata.joblib", "final_skin_model.pth"), # 'models' subdirectory in frontend
        ("dummy_models", "model_metadata.joblib", "final_skin_model.pth"), # 'dummy_models' subdirectory in frontend
        ("../", "model_metadata.joblib", "final_skin_model.pth"), # Parent directory (Artifact root)
    ]

    for base_p, metadata_fn, model_fn in possible_paths:
        current_metadata_path = os.path.join(base_p, metadata_fn)
        current_model_path = os.path.join(base_p, model_fn)
        

        metadata_exists = os.path.exists(current_metadata_path)
        model_exists = os.path.exists(current_model_path)


        if metadata_exists and model_exists:
            try:
                metadata = joblib.load(current_metadata_path)

                # Validate crucial metadata parts
                if not isinstance(metadata, dict) or \
                   'class_names' not in metadata or \
                   'tabular_dim' not in metadata or \
                   'normalize_mean' not in metadata or \
                   'normalize_std' not in metadata:
                    
                    continue # Try next path
                
                if not isinstance(metadata['class_names'], list) or not isinstance(metadata['tabular_dim'], int):
                    continue # Try next path

                # Ensure CombinedModel is defined globally in your script
                model = CombinedModel(
                    num_classes=len(metadata['class_names']),
                    tabular_dim=metadata['tabular_dim']
                )
                
                model.load_state_dict(torch.load(current_model_path, map_location=torch.device('cpu')))
    
                model.eval()
                st.write("--- Model loading successful ---")
                return model, metadata, True
            
            except Exception as e:
                st.error(f"  Error during loading or processing files from '{base_p}': {str(e)}. Trying next path.")
                # Deleting potentially corrupt model/metadata variables so they don't persist
                if 'model' in locals(): del model
                if 'metadata' in locals(): del metadata
                continue # Try next path
        else:
            st.write(f"  Model files (or parts) not found or incomplete in '{base_p}'.")
    
    st.error("--- CRITICAL: Trained model files were not found in any of the specified locations. ---")
    st.warning("The skin analysis feature requires the trained model and its metadata. Please ensure 'model_metadata.joblib' and 'final_skin_model.pth' are correctly placed in one of the searched directories.")
    st.warning("Searched locations were: " + ", ".join([f"'{p[0]}'" for p in possible_paths]))
    return None, None, False

@st.cache_resource
def load_symptom_checker():
    """Load symptom checker model and resources"""
    try:
        # Initialize components
        # biobert = BioBERTProcessor() # Not directly used by SymptomCheckerBot in dummy
        # medical_advice = MedicalAdviceProvider() # Not directly used by SymptomCheckerBot in dummy
          # Use relative path for model_data directory
        model_data_dir_path = "model_data" # Relative path
        if not os.path.exists(model_data_dir_path):
            try:
                os.makedirs(model_data_dir_path, exist_ok=True)
                print(f"Created directory for symptom checker: {model_data_dir_path}")
            except PermissionError:
                st.error(f"Permission denied to create directory: {model_data_dir_path}. Ensure this path is writable or already exists with your models.")
                return None, False

        checker = SymptomCheckerBot(model_data_dir=model_data_dir_path, medical_db="medical_advice_database.json")
        st.success("Symptom checker loaded.")
        return checker, True
    except Exception as e:
        st.error(f"Failed to load symptom checker: {str(e)}")
        return None, False

# Function to extract tabular features for skin disease model
def extract_tabular_features(form_data):
    """Extract tabular features with same structure as training time"""
    # 1. Demographics (3 features)
    age_value = min(max(0, form_data['age'] // 10), 7)  # Simplified age binning
    
    sex_mapping = {'Female': 2.0, 'Male': 1.0, 'Intersex / Non-binary': 0.0, 'Prefer not to say': 0.0}
    sex_value = sex_mapping.get(form_data['sex'], 0.0)
    
    skin_type_mapping = {
        "Always burns, never tans": 1.0,
        "Usually burns, lightly tans": 2.0,
        "Sometimes burns, evenly tans": 3.0,
        "Rarely burns, tans well": 4.0,
        "Very rarely burns, easily tans": 5.0,
        "Never burns, always tans": 6.0,
        "None of the above": 0.0
    }
    fst_value = skin_type_mapping.get(form_data['skin_type'], 0.0)
    
    demographics = [age_value, sex_value, fst_value]
    
    # 2. Race/ethnicity (5 binary features)
    ethnicity_features = [
        1.0 if 'White' in form_data['race_ethnicity'] else 0.0,
        1.0 if 'Black or African American' in form_data['race_ethnicity'] else 0.0,
        1.0 if 'Asian' in form_data['race_ethnicity'] else 0.0,
        1.0 if 'Hispanic, Latino, or Spanish Origin' in form_data['race_ethnicity'] else 0.0,
        1.0 if 'Middle Eastern or North African' in form_data['race_ethnicity'] else 0.0
    ]
    
    # 3. Duration of condition (1 feature)
    duration_mapping = {
        "1 day": 1.0,
        "Less than 1 week": 2.0,
        "1-4 weeks": 3.0,
        "1-3 months": 4.0,
        "More than 3 months": 5.0,
        "More than 1 year": 5.0, # Grouped for simplicity based on original code
        "More than 5 years": 5.0, # Grouped
        "Since childhood": 5.0, # Grouped
        "None of the above": 0.0
    }
    duration_value = duration_mapping.get(form_data['condition_duration'], 0.0)
    
    # 4. Body parts (11 binary features)
    body_parts_options = [
        "Head / Neck", "Arm", "Palm of hand", "Back of hand", "Front torso", 
        "Back torso", "Genitalia / Groin", "Buttocks", "Leg", 
        "Top / side of foot", "Sole of foot"
    ]
    body_parts = [1.0 if part in form_data['body_parts'] else 0.0 for part in body_parts_options]

    # 5. Symptoms (13 binary features) - 7 condition_symptoms + 6 other_symptoms
    condition_symptoms_options = [
        "Concerning in appearance", "Bleeding", "Increasing in size", "Darkening", 
        "Itching", "Burning", "Pain"
    ]
    other_symptoms_options = [
        "Fever", "Chills", "Fatigue", "Joint pain", "Mouth sores", "Shortness of breath"
    ]
    
    symptoms_condition = [1.0 if symp in form_data['condition_symptoms'] else 0.0 for symp in condition_symptoms_options]
    symptoms_other = [1.0 if symp in form_data['other_symptoms'] else 0.0 for symp in other_symptoms_options]
    symptoms = symptoms_condition + symptoms_other
    
    # 6. Textures (4 binary features)
    texture_options = ["Raised or bumpy", "Flat", "Rough or flaky", "Filled with fluid"]
    textures = [1.0 if tex in form_data['textures'] else 0.0 for tex in texture_options]
    
    # 7. Shot type (1 feature)
    shot_type = 1.0  # Default to CLOSE_UP
    
    # Combine all features
    # Order: demographics (3) + duration (1) + ethnicity (5) + body_parts (11) + symptoms (13) + textures (4) + shot_type (1) = 38 features
    all_features = demographics + [duration_value] + ethnicity_features + body_parts + symptoms + textures + [shot_type]
    
    if len(all_features) != 38:
        st.error(f"Mismatch in tabular feature dimension. Expected 38, got {len(all_features)}. Check extract_tabular_features.")
        # Provide details for debugging
        st.error(f"Demographics: {len(demographics)}, Duration: 1, Ethnicity: {len(ethnicity_features)}, Body Parts: {len(body_parts)}, Symptoms: {len(symptoms)}, Textures: {len(textures)}, Shot Type: 1")
        return None

    # Convert to tensor
    feature_tensor = torch.tensor(all_features, dtype=torch.float32).unsqueeze(0)
    
    return feature_tensor

def preprocess_skin_image(image, metadata):
    """Preprocess image for skin disease model"""
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=metadata['normalize_mean'],
            std=metadata['normalize_std']
        )
    ])
    
    # Apply transforms
    img_tensor = preprocess(image)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor

def predict_skin_condition(model, img_tensor, tabular_tensor, metadata, model_loaded):
    """Make prediction for skin disease using model"""
    if not model_loaded or model is None:
        st.warning("Skin model not loaded, cannot predict.")
        return None
    if tabular_tensor is None: # Tabular features failed to extract
        st.error("Tabular features are missing or incorrect, cannot predict.")
        return None
    
    try:
        with torch.no_grad():
            outputs = model(img_tensor, tabular_tensor)
            probabilities = torch.sigmoid(outputs)[0].cpu().numpy()
        
        threshold = 0.5
        class_names = metadata['class_names']
        predictions = [
            {
                "condition": class_names[i],  
                "probability": float(probabilities[i]),
                "above_threshold": probabilities[i] >= threshold
            }
            for i in range(len(class_names))
        ]
        
        # Sort by probability (descending)
        predictions.sort(key=lambda x: x["probability"], reverse=True)
        
        return predictions
    except Exception as e:
        st.error(f"Error during skin prediction: {e}")
        return None

# ====================================
# CHATBOT UI COMPONENTS
# ====================================

def initialize_chat_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I can help you with health concerns. Would you like to:\n\n1. Check your symptoms for possible conditions\n2. Analyze a skin image for potential skin diseases"}
        ]
    if "current_mode" not in st.session_state:
        st.session_state.current_mode = None # Explicitly None initially
    if "images" not in st.session_state:
        st.session_state.images = [] # List to store uploaded image data
    if "form_data" not in st.session_state:
        st.session_state.form_data = None # To store submitted questionnaire

def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def add_chat_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})
    with st.chat_message(role):
        st.markdown(content)

def process_symptom_chat(user_input, symptom_checker):
    add_chat_message("user", user_input) # Re-added this line
    if symptom_checker:
        response = symptom_checker.generate_response(user_input)
        add_chat_message("assistant", response)
    else:
        add_chat_message("assistant", "Symptom checker is currently unavailable.")


def reset_chat():
    keys_to_delete = [key for key in st.session_state.keys() if key != 'show_debug']
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]
    
    # Re-initialize chat state for UI
    initialize_chat_state()

    # Reset the internal state of the symptom checker bot
    symptom_checker, symptom_checker_loaded = load_symptom_checker() # Get the bot instance
    if symptom_checker_loaded and hasattr(symptom_checker, 'reset_conversation'):
        symptom_checker.reset_conversation()
        st.toast("Symptom checker bot has been reset.")

    st.rerun()


def handle_mode_selection(user_input, symptom_checker):
    add_chat_message("user", user_input)

    if re.search(r'\b1\b|symptom|check symptoms', user_input.lower()):
        st.session_state.current_mode = "symptom_checker"
        st.session_state.images = [] # Clear any images if switching
        st.session_state.form_data = None # Clear form data
        response = "I'll help you check your symptoms. Please describe what symptoms you're experiencing."
        add_chat_message("assistant", response)
        
    elif re.search(r'\b2\b|skin|image|analyze skin', user_input.lower()):
        st.session_state.current_mode = "skin_analyzer"
        st.session_state.images = [] # Clear previous images
        st.session_state.form_data = None # Clear previous form data
        response = "I'll help you analyze a skin condition. Please upload an image of the affected area using the uploader below."
        add_chat_message("assistant", response)
        # The uploader will be shown in the main app logic now
            
    else: # Not a clear mode selection, treat as continuation of current mode or default to symptom checker if no mode.
        if st.session_state.current_mode == "symptom_checker":
            process_symptom_chat(user_input, symptom_checker)
        elif st.session_state.current_mode == "skin_analyzer":
             add_chat_message("assistant", "Please upload an image or fill out the questionnaire if an image is already uploaded. You can also type 'restart' or 'switch mode'.")
        else: # No mode set, and input wasn't a mode selection
            response = "I'm not sure how to help with that. Please choose an option:\n\n1. Check symptoms\n2. Analyze skin image"
            add_chat_message("assistant", response)

def show_skin_questionnaire():
    """Show questionnaire for skin analysis."""
    with st.form("skin_questionnaire_form"): # Added a unique key for the form
        st.write("Please provide some additional information about your skin condition:")
        
        age = st.number_input("Age", min_value=0, max_value=120, value=st.session_state.form_data.get('age', 30) if st.session_state.form_data else 30)
        
        sex_options = ["Female", "Male", "Intersex / Non-binary", "Prefer not to say"]
        sex = st.selectbox(
            "Sex at birth",
            sex_options,
            index=sex_options.index(st.session_state.form_data.get('sex', "Female")) if st.session_state.form_data and st.session_state.form_data.get('sex') in sex_options else 0
        )
        
        skin_type_options = [
            "Always burns, never tans", "Usually burns, lightly tans", "Sometimes burns, evenly tans",
            "Rarely burns, tans well", "Very rarely burns, easily tans", "Never burns, always tans",
            "None of the above"
        ]
        skin_type = st.selectbox(
            "How does your skin react to sun exposure?",
            skin_type_options,
            index=skin_type_options.index(st.session_state.form_data.get('skin_type', skin_type_options[0])) if st.session_state.form_data and st.session_state.form_data.get('skin_type') in skin_type_options else 0
        )
        
        st.write("With which racial or ethnic groups do you identify? (Select all that apply)")
        ethnicity_options = {
            "White": "White", "Black or African American": "Black or African American", "Asian": "Asian",
            "Hispanic, Latino, or Spanish Origin": "Hispanic, Latino, or Spanish Origin",
            "Middle Eastern or North African": "Middle Eastern or North African"
        }
        ethnicity_selected = st.session_state.form_data.get('race_ethnicity', []) if st.session_state.form_data else []
        ethnicity = {key: st.checkbox(label, value=(key in ethnicity_selected), key=f"eth_{key.replace(' ', '_')}") for key, label in ethnicity_options.items()}
        
        st.write("Describe how the affected skin area feels (Select all that apply)")
        texture_options = {
            "Raised or bumpy": "Raised or bumpy", "Flat": "Flat", 
            "Rough or flaky": "Rough or flaky", "Filled with fluid": "Filled with fluid"
        }
        textures_selected = st.session_state.form_data.get('textures', []) if st.session_state.form_data else []
        textures = {key: st.checkbox(label, value=(key in textures_selected), key=f"tex_{key.replace(' ', '_')}") for key, label in texture_options.items()}

        st.write("Where on your body is the issue? (Select all that apply)")
        body_part_options = {
            "Head / Neck": "Head / Neck", "Arm": "Arm", "Palm of hand": "Palm of hand", 
            "Back of hand": "Back of hand", "Front torso": "Front torso", "Back torso": "Back torso",
            "Genitalia / Groin": "Genitalia / Groin", "Buttocks": "Buttocks", "Leg": "Leg",
            "Top / side of foot": "Top / side of foot", "Sole of foot": "Sole of foot"
        }
        body_parts_selected = st.session_state.form_data.get('body_parts', []) if st.session_state.form_data else []
        body_parts = {key: st.checkbox(label, value=(key in body_parts_selected), key=f"bp_{key.replace(' ', '_')}") for key, label in body_part_options.items()}

        st.write("Are you experiencing any of the following with your skin issue? (Select all that apply)")
        condition_symptom_options = {
            "Concerning in appearance": "Concerning in appearance", "Bleeding": "Bleeding", 
            "Increasing in size": "Increasing in size", "Darkening": "Darkening", "Itching": "Itching",
            "Burning": "Burning", "Pain": "Pain"
        }
        condition_symptoms_selected = st.session_state.form_data.get('condition_symptoms', []) if st.session_state.form_data else []
        condition_symptoms = {key: st.checkbox(label, value=(key in condition_symptoms_selected), key=f"cs_{key.replace(' ', '_')}") for key, label in condition_symptom_options.items()}

        st.write("Do you have any of these symptoms? (Select all that apply)")
        other_symptom_options = {
            "Fever": "Fever", "Chills": "Chills", "Fatigue": "Fatigue", "Joint pain": "Joint pain",
            "Mouth sores": "Mouth sores", "Shortness of breath": "Shortness of breath"
        }
        other_symptoms_selected = st.session_state.form_data.get('other_symptoms', []) if st.session_state.form_data else []
        other_symptoms = {key: st.checkbox(label, value=(key in other_symptoms_selected), key=f"os_{key.replace(' ', '_')}") for key, label in other_symptom_options.items()}
        
        duration_options = [
            "1 day", "Less than 1 week", "1-4 weeks", "1-3 months", "More than 3 months",
            "More than 1 year", "More than 5 years", "Since childhood", "None of the above"
        ]
        condition_duration = st.selectbox(
            "For how long have you had this skin issue?",
            duration_options,
            index=duration_options.index(st.session_state.form_data.get('condition_duration', duration_options[0])) if st.session_state.form_data and st.session_state.form_data.get('condition_duration') in duration_options else 0
        )
        
        submit_button = st.form_submit_button("Analyze Skin Condition")
        
        if submit_button:
            st.session_state.form_data = {
                "age": age,
                "sex": sex,
                "skin_type": skin_type,
                "race_ethnicity": [k for k, v in ethnicity.items() if v],
                "textures": [k for k, v in textures.items() if v],
                "body_parts": [k for k, v in body_parts.items() if v],
                "condition_symptoms": [k for k, v in condition_symptoms.items() if v],
                "other_symptoms": [k for k, v in other_symptoms.items() if v],
                "condition_duration": condition_duration
            }
            # Add a message to confirm form submission before analysis
            add_chat_message("assistant", "Thank you for the information. Analyzing your skin condition now...")
            run_skin_analysis()
            st.rerun() # Rerun to clear the form and show results if any

def run_skin_analysis():
    try:
        model, metadata, model_loaded = load_skin_model_and_metadata()
        
        if not model_loaded:
            response = "I'm sorry, but I'm unable to analyze skin images at this moment. My skin disease recognition model isn't available. Would you like to check your symptoms instead?"
            add_chat_message("assistant", response)
            return
        
        if not st.session_state.images: # Should have an image by now
            response = "I need an image to analyze your skin condition. Please upload an image of the affected area."
            add_chat_message("assistant", response)
            return

        if not st.session_state.form_data: # Should have form data by now
            response = "I need the questionnaire to be filled out to analyze your skin condition."
            add_chat_message("assistant", response)
            return

        image_data = st.session_state.images[0] # Use the first uploaded image
        pil_image = image_data['image']
        
        img_tensor = preprocess_skin_image(pil_image, metadata)
        tabular_tensor = extract_tabular_features(st.session_state.form_data)
        
        if tabular_tensor is None: # Error during feature extraction
             add_chat_message("assistant", "There was an issue preparing your information for analysis. Please check your inputs or try restarting.")
             return

        predictions = predict_skin_condition(model, img_tensor, tabular_tensor, metadata, model_loaded)
        
        if predictions:
            confirmed_conditions = [p for p in predictions if p['above_threshold']]
            
            response_text = "**Analysis Complete**\n\n"
            if confirmed_conditions:
                response_text += "**Likely Conditions (Confidence >= 50%):**\n"
                for i, cond in enumerate(confirmed_conditions[:3]): # Show top 3 confirmed
                    response_text += f"{i+1}. {cond['condition']} ({cond['probability']*100:.1f}% confidence)\n"
            elif predictions: # No confirmed, show top 3 possible
                response_text += "**Possible Conditions (Top 3):**\n"
                for i, cond in enumerate(predictions[:3]):
                    response_text += f"{i+1}. {cond['condition']} ({cond['probability']*100:.1f}% confidence)\n"
            else: # Should not happen if predict_skin_condition returns a list
                response_text += "No specific conditions identified with high confidence.\n"

            response_text += "\n**Important:** This is not a medical diagnosis. Please consult a healthcare professional for proper evaluation and treatment."
            
            response_text += "\n\n**Patient Information Summary (as provided):**\n"
            response_text += f"- Age: {st.session_state.form_data['age']}\n"
            response_text += f"- Main symptoms: {', '.join(st.session_state.form_data['condition_symptoms']) if st.session_state.form_data['condition_symptoms'] else 'Not specified'}\n"
            response_text += f"- Body location: {', '.join(st.session_state.form_data['body_parts']) if st.session_state.form_data['body_parts'] else 'Not specified'}\n"
            response_text += f"- Duration: {st.session_state.form_data['condition_duration']}\n"
            
            response_text += "\n**What would you like to do next?**\n1. Learn more about one of these conditions (please specify which).\n2. Start a new analysis (type 'restart' or 'new').\n3. Switch to symptom checking (type 'symptoms' or 'check symptoms')."
            add_chat_message("assistant", response_text)
        else:
            add_chat_message("assistant", "I'm sorry, but I couldn't analyze your skin condition with the provided information. Please ensure the image is clear and all questions are answered, then try again.")
            
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}") # Also show in main UI
        add_chat_message("assistant", f"An unexpected error occurred during analysis: {str(e)}. Please try restarting.")
    finally:
        # Clear form data and image for next round if needed, or user can restart
        # st.session_state.form_data = None 
        # st.session_state.images = []
        pass


def main():
    st.title("Medical Assistant Chatbot")
    
    # Sidebar
    with st.sidebar:
        st.title("Options")
        if st.button("Reset Conversation"):
            reset_chat()
            # No need to st.rerun() here, reset_chat() does it.
        
        st.session_state.show_debug = st.checkbox("Show Technical Details", value=st.session_state.get("show_debug", False))

    # Load models & initialize state
    initialize_chat_state() # Ensure all session_state variables are initialized
    symptom_checker, symptom_checker_loaded = load_symptom_checker()
    skin_model, skin_metadata, skin_model_loaded = load_skin_model_and_metadata()


    # Display chat history (all messages up to now)
    display_chat_history()

    # Skin Analyzer UI (File Uploader and Questionnaire)
    # This section is now managed based on current_mode and state
    if st.session_state.current_mode == "skin_analyzer":
        if not st.session_state.images: # If no image uploaded yet for current skin_analyzer session
            uploaded_file = st.file_uploader(
                "Upload skin condition image for analysis", 
                type=["jpg", "jpeg", "png"], 
                key="skin_image_uploader" # Unique key for the uploader
            )
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file).convert('RGB')
                    st.image(image, caption="Uploaded Image Preview", width=200) # Show a small preview
                    
                    # Store image data in session state
                    st.session_state.images = [{
                        'type': 'close_up', # Assuming close_up, adjust if needed
                        'file_name': uploaded_file.name, # Store name for reference
                        'image': image # Store PIL image object
                    }]
                    st.session_state.form_data = None # Reset form data when new image uploaded
                    add_chat_message("assistant", "Image uploaded successfully. Please fill out the questionnaire below to proceed.")
                    st.rerun() # Rerun to make questionnaire appear immediately
                except Exception as e:
                    st.error(f"Error processing uploaded image: {e}")
                    st.session_state.images = [] # Clear if error

        # If image is uploaded but form not yet submitted for skin_analyzer mode
        elif st.session_state.images and not st.session_state.form_data:
            show_skin_questionnaire()
        
        # If form_data is present, it means run_skin_analysis was called by the form's submit.
        # Results would have been added to chat. The form itself is gone after rerun from submit.

    # Chat input
    user_input = st.chat_input("Type your message, 'restart', or 'switch mode'...")

    if user_input:
        # Universal commands
        if user_input.lower() in ["restart", "reset", "start over", "new"]:
            reset_chat() # This will also rerun
        elif user_input.lower() in ["switch mode", "switch"]:
            add_chat_message("user", user_input)
            st.session_state.current_mode = None
            st.session_state.images = []
            st.session_state.form_data = None
            add_chat_message("assistant", "Mode switched. Would you like to:\n\n1. Check your symptoms\n2. Analyze a skin image")
            st.rerun()

        # Mode-specific handling or initial mode selection
        elif st.session_state.current_mode is None:
            handle_mode_selection(user_input, symptom_checker)
            st.rerun() # Rerun to update UI based on mode selection
        
        elif st.session_state.current_mode == "symptom_checker":
            process_symptom_chat(user_input, symptom_checker)
            st.rerun()

        elif st.session_state.current_mode == "skin_analyzer":
            # Handle follow-up questions after skin analysis, or if user types instead of using UI
            add_chat_message("user", user_input)
            
            # Simplified: if user types after analysis, assume it might be a follow-up or confusion
            # More sophisticated NLP could go here to understand intent (e.g., asking about a condition)
            if 'form_data' in st.session_state and st.session_state.form_data and \
               ('learn more' in user_input.lower() or 'tell me about' in user_input.lower()):
                # Placeholder for fetching condition info
                # You'd need to parse which condition the user is asking about
                condition_match = re.search(r"(?:condition|about|more on)\s+([\w\s]+)", user_input, re.IGNORECASE)
                if condition_match and condition_match.group(1):
                    condition_name = condition_match.group(1).strip()
                    # Dummy response, integrate with your MedicalAdviceProvider
                    advice = f"Fetching information for {condition_name}... (Details would appear here from a medical database). For now, please consult a healthcare professional."
                    add_chat_message("assistant", advice)
                else:
                    add_chat_message("assistant", "Please specify which condition you'd like to learn more about from the analysis results.")
            elif not st.session_state.images:
                 add_chat_message("assistant", "Please upload an image using the uploader above to start the skin analysis.")
            elif st.session_state.images and not st.session_state.form_data:
                 add_chat_message("assistant", "Please fill out the questionnaire above to proceed with the skin analysis.")
            else: # General message if in skin_analyzer mode and not a recognized follow-up
                add_chat_message("assistant", "I'm ready for skin analysis. If you've uploaded an image and filled the form, the analysis should have run. You can also 'restart' or 'switch mode'.")
            st.rerun()

if __name__ == "__main__":
    main()