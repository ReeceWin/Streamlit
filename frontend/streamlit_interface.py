import streamlit as st
import os
import sys
# This prevents Streamlit from watching the torch module
os.environ["STREAMLIT_WATCH_MODULES"] = "false"
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import joblib
import json
import io  # Added for saving images
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Skin Disease Classification Tool",
    page_icon="🔬",
    layout="wide"
)

# Define the model architecture
class CombinedModel(nn.Module):
    """
    Dual-pathway model that processes images and tabular data separately
    before combining them for classification.
    """
    def __init__(self, num_classes, tabular_dim):
        super(CombinedModel, self).__init__()
        
        # Image feature extraction using EfficientNet-B3
        self.image_model = models.efficientnet_b3(weights='IMAGENET1K_V1')
        self.image_features = 1536  # EfficientNet-B3 feature dimension
        self.image_model.classifier = nn.Identity()  # Remove classifier
        
        # Image-specific processing path
        self.image_fc = nn.Sequential(
            nn.Linear(self.image_features, 512),
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
        img_features = self.image_model(images)
        img_features = self.image_fc(img_features)
        
        # Process tabular path
        tab_features = self.tabular_model(tabular)
        
        # Combine features
        combined = torch.cat((img_features, tab_features), dim=1)
        
        # Final classification
        return self.classifier(combined)

# Use st.cache_resource for model loading to avoid performance issues
@st.cache_resource
def load_model_and_metadata():
    """Load model and metadata with extensive path checking and diagnostics"""
    try:        
        # Possible locations to check for model files
        possible_paths = [
            # Current directory
            ("./", "model_metadata.joblib", "final_skin_model.pth"),
            # Frontend directory
            ("./frontend/", "model_metadata.joblib", "final_skin_model.pth"),
            # Absolute paths
            ("/mount/src/artifact/", "model_metadata.joblib", "final_skin_model.pth"),
            ("/mount/src/artifact/frontend/", "model_metadata.joblib", "final_skin_model.pth"),
            # Try different filenames (lowercase, etc)
            ("./", "model_metadata.joblib", "final_skin_model.pt"),
            ("./frontend/", "model_metadata.joblib", "final_skin_model.pt")
        ]
        
        # Try each possible path combination
        for base_path, metadata_file, model_file in possible_paths:
            metadata_path = os.path.join(base_path, metadata_file)
            model_path = os.path.join(base_path, model_file)
            
            if os.path.exists(metadata_path) and os.path.exists(model_path):                
                # Try to load metadata file
                try:
                    metadata = joblib.load(metadata_path)
                except Exception as e:
                    st.error(f"Failed to load metadata from {metadata_path}: {str(e)}")
                    continue  # Try next path
                
                # Try to create model
                try:
                    model = CombinedModel(
                        num_classes=len(metadata['class_names']),
                        tabular_dim=metadata['tabular_dim']
                    )
                except Exception as e:
                    st.error(f"Failed to create model: {str(e)}")
                    continue  # Try next path
                
                # Try to load weights
                try:
                    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                except Exception as e:
                    st.error(f"Failed to load weights from {model_path}: {str(e)}")
                    continue  # Try next path
                
                # If we get here, everything worked
                model.eval()
                return model, metadata, True
        
        # If we get here, none of the paths worked
        st.error("Couldn't find or load model files at any expected location")
        return None, None, False
        
    except Exception as e:
        st.error(f"Unexpected error during model loading: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, False

# Function to convert tensor back to viewable image
def tensor_to_image(tensor, metadata):
    """
    Convert a normalized tensor back to a PIL Image for visualization.
    
    Args:
        tensor: PyTorch tensor with shape [1, C, H, W]
        metadata: Model metadata containing normalization values
        
    Returns:
        PIL Image
    """
    # Create a copy of the tensor
    img_tensor = tensor.clone().detach()
    
    # Remove batch dimension
    img_tensor = img_tensor.squeeze(0)
    
    # Denormalize
    mean = torch.tensor(metadata['normalize_mean']).view(3, 1, 1)
    std = torch.tensor(metadata['normalize_std']).view(3, 1, 1)
    img_tensor = img_tensor * std + mean
    
    # Convert to PIL Image (clamp values to valid range, convert to numpy, rearrange channels)
    img_tensor = img_tensor.mul(255).clamp(0, 255).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    img = Image.fromarray(img_tensor)
    
    return img

# Use st.cache_data for image preprocessing to improve performance
@st.cache_data
def preprocess_image(image, metadata):
    """
    Preprocess the image using the same transformations used during training.
    """
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

def extract_tabular_features(form_data):
    """
    ALIGNED WITH TRAINING: Extract tabular features exactly as done during training.
    This function must match the feature extraction logic used in ImprovedSkinDataset.extract_enhanced_tabular_features
    from the training script to ensure consistent features between training and inference.
    """
    # 1. Demographics (3 features)
    # --- Age group ---
    age = form_data['age']
    # Convert numeric age to training age groups (exactly matching training)
    age_value = 0  # Default for unknown
    if age <= 2:
        age_value = 0  # AGE_0_TO_2
    elif age <= 11:
        age_value = 1  # AGE_3_TO_11
    elif age <= 17:
        age_value = 2  # AGE_12_TO_17
    elif age <= 29:
        age_value = 3  # AGE_18_TO_29
    elif age <= 39:
        age_value = 4  # AGE_30_TO_39
    elif age <= 49:
        age_value = 5  # AGE_40_TO_49
    elif age <= 64:
        age_value = 6  # AGE_50_TO_64
    else:
        age_value = 7  # AGE_65_PLUS
    
    # --- Sex at birth ---
    # Exact same encoding as in training
    sex_mapping = {'Female': 2.0, 'Male': 1.0, 'Intersex / Non-binary': 0.0, 'Prefer not to say': 0.0}
    sex_value = sex_mapping.get(form_data['sex'], 0.0)
    
    # --- Fitzpatrick skin type ---
    # Match exactly with training FST encoding
    skin_type_mapping = {
        "Always burns, never tans": 1.0,          # FST1 in training
        "Usually burns, lightly tans": 2.0,       # FST2 in training
        "Sometimes burns, evenly tans": 3.0,      # FST3 in training
        "Rarely burns, tans well": 4.0,           # FST4 in training
        "Very rarely burns, easily tans": 5.0,    # FST5 in training
        "Never burns, always tans": 6.0,          # FST6 in training
        "None of the above": 0.0                  # Unknown in training
    }
    fst_value = skin_type_mapping.get(form_data['skin_type'], 0.0)
    
    # Add demographics in exact same order as training
    demographics = [age_value, sex_value, fst_value]
    
    # 2. Race/ethnicity (5 binary features)
    # Must match exact same order and encoding as training
    ethnicity_features = [
        1.0 if 'White' in form_data['race_ethnicity'] else 0.0,
        1.0 if 'Black or African American' in form_data['race_ethnicity'] else 0.0,
        1.0 if 'Asian' in form_data['race_ethnicity'] else 0.0,
        1.0 if 'Hispanic, Latino, or Spanish Origin' in form_data['race_ethnicity'] else 0.0,
        1.0 if 'Middle Eastern or North African' in form_data['race_ethnicity'] else 0.0
    ]
    
    # 3. Duration of condition (1 feature)
    # Match exact duration encoding from training
    duration_mapping = {
        "1 day": 1.0,                    # ONE_DAY in training
        "Less than 1 week": 2.0,         # LESS_THAN_ONE_WEEK in training
        "1-4 weeks": 3.0,                # ONE_TO_FOUR_WEEKS in training
        "1-3 months": 4.0,               # ONE_TO_THREE_MONTHS in training
        "More than 3 months": 5.0,       # MORE_THAN_THREE_MONTHS in training
        "More than 1 year": 5.0,         # Mapped to MORE_THAN_THREE_MONTHS in training
        "More than 5 years": 5.0,        # Mapped to MORE_THAN_THREE_MONTHS in training
        "Since childhood": 5.0,          # Mapped to MORE_THAN_THREE_MONTHS in training
        "None of the above": 0.0         # Unknown in training
    }
    duration_value = duration_mapping.get(form_data['condition_duration'], 0.0)
    
    # 4. Body parts (11 binary features)
    # Must be in exact same order as training
    body_parts = [
        1.0 if 'Head / Neck' in form_data['body_parts'] else 0.0,
        1.0 if 'Arm' in form_data['body_parts'] else 0.0,
        1.0 if 'Palm of hand' in form_data['body_parts'] else 0.0,
        1.0 if 'Back of hand' in form_data['body_parts'] else 0.0,
        1.0 if 'Front torso' in form_data['body_parts'] else 0.0,
        1.0 if 'Back torso' in form_data['body_parts'] else 0.0,
        1.0 if 'Genitalia / Groin' in form_data['body_parts'] else 0.0,
        1.0 if 'Buttocks' in form_data['body_parts'] else 0.0,
        1.0 if 'Leg' in form_data['body_parts'] else 0.0,
        1.0 if 'Top / side of foot' in form_data['body_parts'] else 0.0,
        1.0 if 'Sole of foot' in form_data['body_parts'] else 0.0
    ]
    
    # 5. Symptoms (13 binary features)
    # Must match exact order from training
    symptoms = [
        1.0 if 'Concerning in appearance' in form_data['condition_symptoms'] else 0.0,  # condition_symptoms_bothersome_appearance
        1.0 if 'Bleeding' in form_data['condition_symptoms'] else 0.0,                  # condition_symptoms_bleeding
        1.0 if 'Increasing in size' in form_data['condition_symptoms'] else 0.0,        # condition_symptoms_increasing_size
        1.0 if 'Darkening' in form_data['condition_symptoms'] else 0.0,                 # condition_symptoms_darkening
        1.0 if 'Itching' in form_data['condition_symptoms'] else 0.0,                   # condition_symptoms_itching
        1.0 if 'Burning' in form_data['condition_symptoms'] else 0.0,                   # condition_symptoms_burning
        1.0 if 'Pain' in form_data['condition_symptoms'] else 0.0,                      # condition_symptoms_pain
        1.0 if 'Fever' in form_data['other_symptoms'] else 0.0,                         # other_symptoms_fever
        1.0 if 'Chills' in form_data['other_symptoms'] else 0.0,                        # other_symptoms_chills
        1.0 if 'Fatigue' in form_data['other_symptoms'] else 0.0,                       # other_symptoms_fatigue
        1.0 if 'Joint pain' in form_data['other_symptoms'] else 0.0,                    # other_symptoms_joint_pain
        1.0 if 'Mouth sores' in form_data['other_symptoms'] else 0.0,                   # other_symptoms_mouth_sores
        1.0 if 'Shortness of breath' in form_data['other_symptoms'] else 0.0            # other_symptoms_shortness_of_breath
    ]
    
    # 6. Textures (4 binary features)
    # Must match exact order from training
    textures = [
        1.0 if 'Raised or bumpy' in form_data['textures'] else 0.0,          # textures_raised_or_bumpy
        1.0 if 'Flat' in form_data['textures'] else 0.0,                     # textures_flat
        1.0 if 'Rough or flaky' in form_data['textures'] else 0.0,           # textures_rough_or_flaky
        1.0 if 'Filled with fluid' in form_data['textures'] else 0.0         # textures_fluid_filled
    ]
    
    # 7. Shot type (1 feature)
    # This must match the image metadata from training
    shot_type_mapping = {
        'close_up': 1.0,  # CLOSE_UP in training
        'angle': 2.0,     # AT_AN_ANGLE in training
        'distance': 3.0   # AT_DISTANCE in training
    }
    
    # Default to close-up if image type is unavailable
    if 'images' in st.session_state and st.session_state.images:
        shot_type = shot_type_mapping.get(st.session_state.images[0]['type'], 1.0)
    else:
        shot_type = 1.0  # Default to CLOSE_UP
    
    # Combine all features in EXACT same order as training
    # demographics + [duration_value] + ethnicity_features + body_parts + symptoms + textures + [shot_type]
    all_features = demographics + [duration_value] + ethnicity_features + body_parts + symptoms + textures + [shot_type]
    
    # Debug information
    if st.session_state.get('show_debug', False):
        with st.expander("Feature Vector Details", expanded=False):
            feature_names = [
                "Age Group", "Biological Sex", "Skin Type", "Condition Duration",
                "Ethnicity - White", "Ethnicity - Black", "Ethnicity - Asian", 
                "Ethnicity - Hispanic", "Ethnicity - Middle Eastern",
                "Body - Head/Neck", "Body - Arm", "Body - Palm", "Body - Back of Hand",
                "Body - Front Torso", "Body - Back Torso", "Body - Genitalia",
                "Body - Buttocks", "Body - Leg", "Body - Top of Foot", "Body - Sole of Foot",
                "Symptom - Concerning Appearance", "Symptom - Bleeding", 
                "Symptom - Increasing Size", "Symptom - Darkening", "Symptom - Itching",
                "Symptom - Burning", "Symptom - Pain", "Symptom - Fever", 
                "Symptom - Chills", "Symptom - Fatigue", "Symptom - Joint Pain",
                "Symptom - Mouth Sores", "Symptom - Shortness of Breath",
                "Texture - Raised", "Texture - Flat", "Texture - Rough", 
                "Texture - Fluid Filled", "Image Shot Type"
            ]
            for i, (name, value) in enumerate(zip(feature_names, all_features)):
                st.write(f"{i}: {name} = {value}")
    
    # Verify feature dimension matches expected
    if len(all_features) != 38:
        st.error(f"CRITICAL ERROR: Feature dimension mismatch. Expected 38, got {len(all_features)}.")
        
    # Convert to tensor
    feature_tensor = torch.tensor(all_features, dtype=torch.float32).unsqueeze(0)
    
    return feature_tensor

def predict_skin_condition(model, img_tensor, tabular_tensor, metadata, model_loaded):
    """
    Make prediction using both image and tabular data.
    Returns None if model is not loaded properly.
    """
    if not model_loaded or model is None:
        return None
    
    try:
        # For actual model prediction:
        with torch.no_grad():
            outputs = model(img_tensor, tabular_tensor)
            # Use sigmoid for multi-label predictions
            probabilities = torch.sigmoid(outputs)[0].cpu().numpy()
        
        # Create predictions with threshold indicator
        threshold = 0.5  # Standard threshold for multi-label binary classification
        class_names = metadata['class_names']
        predictions = [
            {
                "condition": class_names[i], 
                "probability": float(probabilities[i]),
                "above_threshold": probabilities[i] >= threshold  # Flag if condition is likely present
            }
            for i in range(len(class_names))
        ]
        
        # Sort by probability (descending)
        predictions.sort(key=lambda x: x["probability"], reverse=True)
        
        return predictions
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        # Return None if an error occurs
        return None

def main():
    st.title("Skin Disease Classification Tool")
    
    # Add debug mode toggle
    with st.sidebar:
        st.session_state.show_debug = st.checkbox("Show Technical Details", value=False)
        
        if st.session_state.show_debug:
            st.info("Debug mode is ON. You'll see detailed feature information during analysis.")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Image Analysis", "About"])
    
    if page == "About":
        st.header("About This Tool")
        st.write("""
        This tool uses a dual-pathway deep learning model to analyze skin conditions. 
        It combines analysis of clinical images with patient information for a more comprehensive assessment.
        
        ### How it works:
        1. Upload at least one image of your skin condition
        2. Complete the questionnaire with additional information
        3. Our AI model analyzes both the images and your information
        4. Receive an assessment showing potential skin conditions with confidence scores
        
        ### Model Details:
        - **Architecture**: Dual-pathway neural network with EfficientNet-B3 backbone
        - **Approach**: Multi-label classification to identify multiple possible conditions
        - **Performance**: 78.4% recall, 88.4% precision across 20 skin conditions
        - **Training**: Trained on thousands of dermatologist-labeled images
        
        ### What is Multi-Label Classification?
        Unlike traditional classification that selects just one diagnosis, our multi-label approach:
        
        - Can detect multiple skin conditions that may be present simultaneously
        - Mimics how dermatologists often consider several conditions in their differential diagnosis
        - Provides a more complete picture of possible conditions
        
        ### Important Note:
        This tool is for informational purposes only and is not a substitute for professional medical advice. 
        Always consult with a qualified healthcare provider for diagnosis and treatment.
                 
        ### Future Improvements:
        This web application will be attached to a MongoDB database to store user data and images. This will allow for further training of the model and improve the accuracy of the predictions.
        Symptom - Disease prediction will also be added at a later date once the model has completed training.
        """)
        
        # Get class names for the "About" page
        # We'll load the metadata just to get class names
        try:
            metadata = joblib.load('model_metadata.joblib')
            class_names = metadata['class_names']
        except:
            # Fallback to a pre-defined list if metadata can't be loaded
            class_names = [
                'Acne', 'Acute dermatitis, NOS', 'Allergic Contact Dermatitis', 
                'CD - Contact dermatitis', 'Drug Rash', 'Eczema', 'Folliculitis', 
                'Herpes Zoster', 'Impetigo', 'Insect Bite', 'Keratosis pilaris',
                'Lichen planus/lichenoid eruption', 'Pigmented purpuric eruption',
                'Pityriasis rosea', 'Psoriasis', 'Scabies', 'Stasis Dermatitis',
                'Tinea', 'Urticaria', 'Viral Exanthem'
            ]
        
        # Show list of conditions the model can identify
        st.subheader("Conditions This Tool Can Identify:")
        col1, col2 = st.columns(2)
        half = len(class_names) // 2
        with col1:
            for name in class_names[:half]:
                st.write(f"• {name}")
        with col2:
            for name in class_names[half:]:
                st.write(f"• {name}")
        
    else:  # Image Analysis page
        st.header("Skin Condition Analysis")
        
        # Initialize session state
        if 'page' not in st.session_state:
            st.session_state.page = 'upload'
        if 'images' not in st.session_state:
            st.session_state.images = []
        if 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False
        if 'form_data' not in st.session_state:
            st.session_state.form_data = {}
        if 'predictions' not in st.session_state:
            st.session_state.predictions = []
        if 'model_warning_shown' not in st.session_state:
            st.session_state.model_warning_shown = False
            
        # Check if model can be loaded
        # We do this check upfront to prevent users from wasting time if model isn't available
        if not st.session_state.model_warning_shown:
            with st.spinner("Checking model availability..."):
                model, metadata, model_loaded = load_model_and_metadata()
                
                if not model_loaded:
                    st.error("""
                    **Model not available!** 
                    
                    The required model files could not be loaded. Please ensure that:
                    1. The model files (final_skin_model.pth and model_metadata.joblib) exist in the application directory
                    2. The files are accessible to the application
                    3. Restart the application after fixing the issues
                    
                    The application will not be able to make predictions until this is resolved.
                    """)
                    st.session_state.model_warning_shown = True
        
        # Page 1: Image Upload
        if st.session_state.page == 'upload':
            st.subheader("Step 1: Upload Images")
            
            st.write("Upload at least one clear image of your skin condition:")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("1. Close up (Required)")
                st.write("Take a clear, well-lit photo focusing directly on the affected area")
                image1 = st.file_uploader("Choose close-up image...", type=["jpg", "jpeg", "png"], key="image1")
                if image1:
                    img1 = Image.open(image1).convert('RGB')
                    st.image(img1, width=200)
                    # Update or add to images list
                    if not any(img.get('type') == 'close_up' for img in st.session_state.images):
                        st.session_state.images.append({'type': 'close_up', 'file': image1, 'image': img1})
                    else:
                        # Update existing image
                        for i, img in enumerate(st.session_state.images):
                            if img.get('type') == 'close_up':
                                st.session_state.images[i] = {'type': 'close_up', 'file': image1, 'image': img1}
            
            with col2:
                st.write("2. At an angle (Optional)")
                st.write("Capture the condition from a different angle to show texture and elevation")
                image2 = st.file_uploader("Choose angle image...", type=["jpg", "jpeg", "png"], key="image2")
                if image2:
                    img2 = Image.open(image2).convert('RGB')
                    st.image(img2, width=200)
                    # Update or add to images list
                    if not any(img.get('type') == 'angle' for img in st.session_state.images):
                        st.session_state.images.append({'type': 'angle', 'file': image2, 'image': img2})
                    else:
                        # Update existing image
                        for i, img in enumerate(st.session_state.images):
                            if img.get('type') == 'angle':
                                st.session_state.images[i] = {'type': 'angle', 'file': image2, 'image': img2}
            
            with col3:
                st.write("3. At a distance (Optional)")
                st.write("Shows the broader context of where the condition appears on your body")
                image3 = st.file_uploader("Choose distance image...", type=["jpg", "jpeg", "png"], key="image3")
                if image3:
                    img3 = Image.open(image3).convert('RGB')
                    st.image(img3, width=200)
                    # Update or add to images list
                    if not any(img.get('type') == 'distance' for img in st.session_state.images):
                        st.session_state.images.append({'type': 'distance', 'file': image3, 'image': img3})
                    else:
                        # Update existing image
                        for i, img in enumerate(st.session_state.images):
                            if img.get('type') == 'distance':
                                st.session_state.images[i] = {'type': 'distance', 'file': image3, 'image': img3}
            
            # Continue button - require at least one image
            if st.session_state.images and st.button("Continue to Questionnaire"):
                # Check if model is available before proceeding
                model, metadata, model_loaded = load_model_and_metadata()
                if not model_loaded:
                    st.error("""
                    **Cannot proceed!** The required model is not available.
                    
                    This application requires specific model files to function properly. 
                    Please contact the administrator to resolve this issue.
                    """)
                else:
                    # Move to questionnaire page
                    st.session_state.page = 'questionnaire'
                    st.rerun()
        
        # Page 2: Questionnaire
        elif st.session_state.page == 'questionnaire':
            st.subheader("Step 2: Patient Information")
            
            # Display a small thumbnail of the uploaded image(s)
            if st.session_state.images:
                st.write("Uploaded Images:")
                cols = st.columns(len(st.session_state.images))
                for i, (col, img_data) in enumerate(zip(cols, st.session_state.images)):
                    with col:
                        st.image(img_data['image'], width=100)
                        st.caption(f"Image {i+1}: {img_data['type'].replace('_', ' ').title()}")
            
            # Create form for questionnaire
            with st.form("patient_questionnaire"):
                st.write("Please complete the following information to help improve the accuracy of the analysis.")
                
                # Age
                st.subheader("Age")
                age = st.number_input("Enter your age", min_value=0, max_value=120, step=1)
                
                # Sex at birth
                st.subheader("Sex at birth")
                sex = st.selectbox(
                    "Select option",
                    ["Female", "Male", "Intersex / Non-binary", "Prefer not to say"]
                )
                
                # Skin type
                st.subheader("How does your skin react to sun exposure?")
                skin_type = st.selectbox(
                    "Select option",
                    [
                        "Always burns, never tans",
                        "Usually burns, lightly tans",
                        "Sometimes burns, evenly tans",
                        "Rarely burns, tans well",
                        "Very rarely burns, easily tans",
                        "Never burns, always tans",
                        "None of the above"
                    ]
                )
                
                # Race/Ethnicity
                st.subheader("With which racial or ethnic groups do you identify?")
                st.write("Mark all that apply")
                race_ethnicity = {
                    "American Indian or Alaska Native": st.checkbox("American Indian or Alaska Native"),
                    "Asian": st.checkbox("Asian"),
                    "Black or African American": st.checkbox("Black or African American"),
                    "Hispanic, Latino, or Spanish Origin": st.checkbox("Hispanic, Latino, or Spanish Origin"),
                    "Middle Eastern or North African": st.checkbox("Middle Eastern or North African"),
                    "Native Hawaiian or Pacific Islander": st.checkbox("Native Hawaiian or Pacific Islander"),
                    "White": st.checkbox("White"),
                    "Another race or ethnicity not listed": st.checkbox("Another race or ethnicity not listed"),
                    "Prefer not to answer": st.checkbox("Prefer not to answer")
                }
                
                # Textures
                st.subheader("Describe how the affected skin area feels")
                st.write("Select all that apply")
                textures = {
                    "Raised or bumpy": st.checkbox("Raised or bumpy", key="texture1"),
                    "Flat": st.checkbox("Flat", key="texture2"),
                    "Rough or flaky": st.checkbox("Rough or flaky", key="texture3"),
                    "Filled with fluid": st.checkbox("Filled with fluid", key="texture4")
                }
                
                # Body parts
                st.subheader("Where on your body is the issue?")
                st.write("Select all that apply")
                body_parts = {
                    "Head / Neck": st.checkbox("Head / Neck", key="body1"),
                    "Arm": st.checkbox("Arm", key="body2"),
                    "Palm of hand": st.checkbox("Palm of hand", key="body3"),
                    "Back of hand": st.checkbox("Back of hand", key="body4"),
                    "Front torso": st.checkbox("Front torso", key="body5"),
                    "Back torso": st.checkbox("Back torso", key="body6"),
                    "Genitalia / Groin": st.checkbox("Genitalia / Groin", key="body7"),
                    "Buttocks": st.checkbox("Buttocks", key="body8"),
                    "Leg": st.checkbox("Leg", key="body9"),
                    "Top / side of foot": st.checkbox("Top / side of foot", key="body10"),
                    "Sole of foot": st.checkbox("Sole of foot", key="body11"),
                    "Other": st.checkbox("Other", key="body12")
                }
                
                # Condition symptoms
                st.subheader("Are you experiencing any of the following with your skin issue?")
                st.write("Select all that apply")
                condition_symptoms = {
                    "Concerning in appearance": st.checkbox("Concerning in appearance", key="symptom1"),
                    "Bleeding": st.checkbox("Bleeding", key="symptom2"),
                    "Increasing in size": st.checkbox("Increasing in size", key="symptom3"),
                    "Darkening": st.checkbox("Darkening", key="symptom4"),
                    "Itching": st.checkbox("Itching", key="symptom5"),
                    "Burning": st.checkbox("Burning", key="symptom6"),
                    "Pain": st.checkbox("Pain", key="symptom7"),
                    "None of the above": st.checkbox("None of the above", key="symptom8")
                }
                
                # Other symptoms
                st.subheader("Do you have any of these symptoms?")
                st.write("Select all that apply")
                other_symptoms = {
                    "Fever": st.checkbox("Fever", key="other1"),
                    "Chills": st.checkbox("Chills", key="other2"),
                    "Fatigue": st.checkbox("Fatigue", key="other3"),
                    "Joint pain": st.checkbox("Joint pain", key="other4"),
                    "Mouth sores": st.checkbox("Mouth sores", key="other5"),
                    "Shortness of breath": st.checkbox("Shortness of breath", key="other6"),
                    "None of the above": st.checkbox("None of the above", key="other7")
                }
                
                # Condition duration
                st.subheader("For how long have you had this skin issue?")
                condition_duration = st.selectbox(
                    "Select option",
                    [
                        "1 day",
                        "Less than 1 week",
                        "1-4 weeks",
                        "1-3 months",
                        "More than 3 months",
                        "More than 1 year",
                        "More than 5 years",
                        "Since childhood",
                        "None of the above"
                    ]
                )
                
                # Submit button
                submit_button = st.form_submit_button("Analyze")
                
                if submit_button:
                    # Save form data to session state
                    st.session_state.form_data = {
                        "age": age,
                        "sex": sex,
                        "skin_type": skin_type,
                        "race_ethnicity": [k for k, v in race_ethnicity.items() if v],
                        "textures": [k for k, v in textures.items() if v],
                        "body_parts": [k for k, v in body_parts.items() if v],
                        "condition_symptoms": [k for k, v in condition_symptoms.items() if v 
                                              and k != "None of the above"],
                        "other_symptoms": [k for k, v in other_symptoms.items() if v 
                                          and k != "None of the above"],
                        "condition_duration": condition_duration
                    }
                    
                    # Create a progress bar for the analysis
                    progress_bar = st.progress(0)
                    
                    with st.spinner("Loading model and analyzing data..."):
                        try:
                            # Load model here, not at app initialization
                            model, metadata, model_loaded = load_model_and_metadata()
                            
                            if not model_loaded:
                                st.error("""
                                **Analysis failed!** The model could not be loaded.
                                
                                Please contact the administrator to ensure the required model files 
                                are properly installed.
                                """)
                                progress_bar.progress(100)
                            else:
                                progress_bar.progress(20)
                                
                                # Process all images and get predictions
                                st.session_state.predictions = []
                                
                                # Extract tabular features once
                                tabular_tensor = extract_tabular_features(st.session_state.form_data)
                                progress_bar.progress(40)
                                
                                # Process each image
                                for i, img_data in enumerate(st.session_state.images):
                                    # Update progress
                                    progress_percent = 40 + (i+1) * 50 / len(st.session_state.images)
                                    progress_bar.progress(int(progress_percent))
                                    
                                    # Preprocess the image
                                    img_tensor = preprocess_image(img_data['image'], metadata)
                                    
                                    # Convert tensor back to image for visualization
                                    preprocessed_img = tensor_to_image(img_tensor, metadata)
                                    
                                    # Get prediction for this image
                                    prediction = predict_skin_condition(model, img_tensor, tabular_tensor, metadata, model_loaded)
                                    
                                    if prediction is None:
                                        st.error(f"Prediction failed for image {i+1}.")
                                    else:
                                        # Store prediction with image type and preprocessed image
                                        st.session_state.predictions.append({
                                            'image_type': img_data['type'],
                                            'results': prediction,
                                            'preprocessed_image': preprocessed_img
                                        })
                                    
                                progress_bar.progress(100)
                        except Exception as e:
                            st.error(f"An error occurred during analysis: {e}")
                            
                            if st.session_state.show_debug:
                                import traceback
                                st.error("Detailed error information:")
                                st.code(traceback.format_exc())
                    
                    # Set analysis complete flag
                    st.session_state.analysis_complete = True
                    
                    # Move to results page only if we have predictions
                    if st.session_state.predictions:
                        st.session_state.page = 'results'
                        st.rerun()
                    else:
                        st.error("Analysis failed to produce any predictions. Cannot proceed to results.")
            
        # Page 3: Results
        elif st.session_state.page == 'results':
            st.subheader("Analysis Results")
            
            # Verify we have predictions before proceeding
            if not st.session_state.predictions:
                st.error("No predictions available. Please return to the previous steps.")
                if st.button("Return to Image Upload"):
                    st.session_state.page = 'upload'
                    st.rerun()
            else:
                # Debug info
                if st.session_state.get('show_debug', False):
                    st.info("Number of prediction sets: " + str(len(st.session_state.predictions)))
                    
                    with st.expander("Raw Prediction Data", expanded=False):
                        for i, pred_set in enumerate(st.session_state.predictions):
                            st.write(f"Image {i+1} ({pred_set['image_type']}):")
                            for j, pred in enumerate(pred_set['results'][:5]):
                                st.write(f"- {pred['condition']}: {pred['probability']*100:.1f}% (Above threshold: {pred['above_threshold']})")
                
                # Combine predictions across images
                # Aggregate predictions from all images
                all_conditions = {}
                for pred_set in st.session_state.predictions:
                    for pred in pred_set['results'][:5]:  # Consider top 5 from each image
                        condition = pred['condition']
                        prob = pred['probability']
                        above_threshold = pred.get('above_threshold', False)
                        
                        if condition in all_conditions:
                            all_conditions[condition]['probs'].append(prob)
                            # If any image shows it above threshold, consider it above threshold
                            all_conditions[condition]['above_threshold'] = (
                                all_conditions[condition]['above_threshold'] or above_threshold
                            )
                        else:
                            all_conditions[condition] = {
                                'probs': [prob],
                                'above_threshold': above_threshold
                            }
                
                # Average the probabilities
                sorted_conditions = [
                    {
                        "condition": cond, 
                        "probability": sum(data['probs'])/len(data['probs']),
                        "above_threshold": data['above_threshold']
                    } 
                    for cond, data in all_conditions.items()
                ]
                
                # Sort by probability
                sorted_conditions.sort(key=lambda x: x['probability'], reverse=True)
                
                # First separate into confirmed and differential diagnoses
                confirmed = [c for c in sorted_conditions if c.get("above_threshold", False)]
                differential = [c for c in sorted_conditions if not c.get("above_threshold", False)][:3]
                
                # Display confirmed conditions if any
                if confirmed:
                    st.write("### Likely Conditions")
                    confirmed_cols = st.columns(min(3, len(confirmed)))
                    for i, col in enumerate(confirmed_cols):
                        if i < len(confirmed):
                            with col:
                                prob = confirmed[i]['probability'] * 100
                                condition = confirmed[i]['condition']
                                st.markdown(f"""
                                <div style="padding: 20px; border-radius: 10px; border: 1px solid green;">
                                    <h3 style="color: green;">{condition}</h3>
                                    <h2 style="color: green;">{prob:.1f}%</h2>
                                    <p>Confidence</p>
                                </div>
                                """, unsafe_allow_html=True)
                
                # Always show differential diagnoses
                st.write("### Differential Diagnosis" if confirmed else "### Top Conditions Identified")
                diff_cols = st.columns(min(3, len(differential)))
                for i, col in enumerate(diff_cols):
                    if i < len(differential):
                        with col:
                            prob = differential[i]['probability'] * 100
                            condition = differential[i]['condition']
                            
                            # Color code by probability
                            if prob >= 40:
                                color = "orange"
                            else:
                                color = "gray"
                                
                            st.markdown(f"""
                            <div style="padding: 20px; border-radius: 10px; border: 1px solid {color};">
                                <h3 style="color: {color};">{condition}</h3>
                                <h2 style="color: {color};">{prob:.1f}%</h2>
                                <p>Considered</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Show detailed breakdown with original and preprocessed images
                with st.expander("See detailed analysis for each image"):
                    for i, pred_set in enumerate(st.session_state.predictions):
                        st.write(f"#### Image {i+1}: {pred_set['image_type'].replace('_', ' ').title()}")
                        
                        # Display original and preprocessed images side by side
                        img_col1, img_col2, pred_col = st.columns([1, 1, 2])
                        
                        with img_col1:
                            img_data = next((img for img in st.session_state.images if img['type'] == pred_set['image_type']), None)
                            if img_data:
                                st.write("Original:")
                                st.image(img_data['image'], width=150)
                        
                        with img_col2:
                            if 'preprocessed_image' in pred_set:
                                st.write("Preprocessed (Model Input):")
                                st.image(pred_set['preprocessed_image'], width=150)
                                
                                # Convert PIL Image to bytes for download
                                img_byte_arr = io.BytesIO()
                                pred_set['preprocessed_image'].save(img_byte_arr, format='PNG')
                                img_byte_arr = img_byte_arr.getvalue()
                                
                                # Download button for preprocessed image
                                st.download_button(
                                    label="Download preprocessed",
                                    data=img_byte_arr,
                                    file_name=f"preprocessed_{pred_set['image_type']}_{i}.png",
                                    mime="image/png",
                                    key=f"download_img_{i}"
                                )
                        
                        with pred_col:
                            # First show confirmed conditions for this image
                            confirmed_for_image = [p for p in pred_set['results'] if p.get('above_threshold', False)]
                            if confirmed_for_image:
                                st.write("**Likely conditions from this image:**")
                                for j, pred in enumerate(confirmed_for_image):
                                    st.write(f"• {pred['condition']}: {pred['probability']*100:.1f}%")
                            
                            # Then show top differential diagnoses
                            differential_for_image = [p for p in pred_set['results'] if not p.get('above_threshold', False)][:3]
                            if differential_for_image:
                                st.write("**Also considered from this image:**")
                                for j, pred in enumerate(differential_for_image):
                                    st.write(f"• {pred['condition']}: {pred['probability']*100:.1f}%")
            
                # Display patient information
                st.write("### Patient Information")
                info_col1, info_col2 = st.columns(2)
                
                with info_col1:
                    st.write(f"**Age:** {st.session_state.form_data['age']}")
                    st.write(f"**Sex at birth:** {st.session_state.form_data['sex']}")
                    st.write(f"**Skin type:** {st.session_state.form_data['skin_type']}")
                    st.write("**Ethnicity:** " + ", ".join(st.session_state.form_data['race_ethnicity']))
                    st.write("**Skin texture:** " + ", ".join(st.session_state.form_data['textures']))
                
                with info_col2:
                    st.write("**Body location:** " + ", ".join(st.session_state.form_data['body_parts']))
                    st.write("**Skin symptoms:** " + ", ".join(st.session_state.form_data['condition_symptoms']))
                    st.write("**Other symptoms:** " + ", ".join(st.session_state.form_data['other_symptoms']))
                    st.write(f"**Duration:** {st.session_state.form_data['condition_duration']}")
                
                # Recommendations section
                st.write("### Assessment")
                
                # Check for concerning symptoms
                concerning_symptoms = ["Bleeding", "Increasing in size", "Darkening"]
                has_concerning_symptoms = any(symptom in st.session_state.form_data['condition_symptoms'] 
                                            for symptom in concerning_symptoms)
                
                # Get confirmed conditions
                confirmed_conditions = [c["condition"] for c in sorted_conditions if c.get("above_threshold", False)]
                
                # Create assessment text for multiple potential conditions
                if confirmed_conditions:
                    if len(confirmed_conditions) == 1:
                        condition_text = f"**{confirmed_conditions[0]}**"
                        severity = "potentially concerning" if has_concerning_symptoms else "likely not immediately concerning"
                    elif len(confirmed_conditions) == 2:
                        condition_text = f"**{confirmed_conditions[0]}** and **{confirmed_conditions[1]}**"
                        severity = "potentially concerning" if has_concerning_symptoms else "of mixed concern"
                    else:
                        condition_list = [f"**{c}**" for c in confirmed_conditions[:-1]]
                        condition_text = f"{', '.join(condition_list)}, and **{confirmed_conditions[-1]}**"
                        severity = "potentially concerning" if has_concerning_symptoms else "of mixed concern"
                    
                    if has_concerning_symptoms:
                        recommendation = "We recommend consulting with a dermatologist promptly."
                    else:
                        recommendation = "Consider a routine dermatology check-up."
                    
                    st.info(f"""
                    Based on the analysis of your images and information, this condition appears most consistent with {condition_text} and is {severity}.
                    
                    **Recommendation:** {recommendation}
                    
                    **Important:** This is not a medical diagnosis. The analysis is provided for informational purposes only and should not replace professional medical advice.
                    """)
                else:
                    # Fallback to top condition when no conditions are above threshold
                    top_condition = sorted_conditions[0]['condition'] if sorted_conditions else "Unknown"
                    top_confidence = sorted_conditions[0]['probability'] if sorted_conditions else 0
                    
                    if top_confidence >= 0.4:
                        confidence_text = "moderate confidence"
                    else:
                        confidence_text = "low confidence"
                    
                    severity = "potentially concerning" if has_concerning_symptoms else "of uncertain severity"
                    
                    if has_concerning_symptoms:
                        recommendation = "We recommend consulting with a dermatologist promptly."
                    else:
                        recommendation = "Consider consulting with a healthcare provider for further evaluation."
                    
                    st.info(f"""
                    Based on the analysis of your images and information, no conditions were identified with high confidence, 
                    but the condition that appears most similar is **{top_condition}** (with {confidence_text}) and is {severity}.
                    
                    **Recommendation:** {recommendation}
                    
                    **Important:** This is not a medical diagnosis. The analysis is provided for informational purposes only and should not replace professional medical advice.
                    """)
                
                # Actions section
                st.write("### Actions")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Export data option
                    if st.button("Export Report (JSON)"):
                        try:
                            # Create a clean, serializable version of the form data
                            clean_form_data = {
                                "age": st.session_state.form_data['age'],
                                "sex": st.session_state.form_data['sex'],
                                "skin_type": st.session_state.form_data['skin_type'],
                                "race_ethnicity": list(st.session_state.form_data['race_ethnicity']),
                                "textures": list(st.session_state.form_data['textures']),
                                "body_parts": list(st.session_state.form_data['body_parts']),
                                "condition_symptoms": list(st.session_state.form_data['condition_symptoms']),
                                "other_symptoms": list(st.session_state.form_data['other_symptoms']),
                                "condition_duration": st.session_state.form_data['condition_duration']
                            }
                            
                            # Create serializable predictions
                            clean_predictions = []
                            for condition in sorted_conditions[:5]:
                                clean_predictions.append({
                                    "condition": condition["condition"],
                                    "probability": float(condition["probability"]),
                                    "above_threshold": bool(condition["above_threshold"])
                                })
                            
                            # Include multi-label information in export
                            export_data = {
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "patient_info": clean_form_data,
                                "likely_conditions": confirmed_conditions,
                                "all_predictions": clean_predictions,
                                "has_concerning_symptoms": has_concerning_symptoms,
                                "recommendation": recommendation
                            }
                            
                            # Convert to JSON and offer download
                            json_str = json.dumps(export_data, indent=4)
                            st.download_button(
                                label="Download JSON Report",
                                data=json_str,
                                file_name=f"skin_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                        except Exception as e:
                            st.error(f"Error creating export: {str(e)}")
                            
                            if st.session_state.get('show_debug', False):
                                import traceback
                                st.error("Detailed error information:")
                                st.code(traceback.format_exc())
                
                with col2:
                    # Reset button
                    if st.button("Start New Analysis"):
                        st.session_state.page = 'upload'
                        st.session_state.images = []
                        st.session_state.analysis_complete = False
                        st.session_state.form_data = {}
                        st.session_state.predictions = []
                        st.rerun()

if __name__ == "__main__":
    main()