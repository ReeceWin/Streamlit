import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix,
                           balanced_accuracy_score, roc_auc_score, 
                           precision_recall_fscore_support)
from PIL import Image
import random
import warnings
import joblib

# Ignore warnings - helpful for cleaner output
warnings.filterwarnings('ignore')

# Set seeds for reproducibility across all libraries
def set_seed(seed=42):
    """Set seeds for reproducibility in numpy, random, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make operations deterministic when possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Set up device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")

#################################################
# IMPROVED DATASET CLASS
#################################################

class ImprovedSkinDataset(Dataset):
    """
    Enhanced dataset class with better handling of:
    - Multiple images per case
    - Class imbalance
    - Tabular feature extraction
    - Image verification
    - Multi-label classification
    """
    def __init__(self, cases_df, labels_df, transform=None, base_img_dir="", 
                 verify_images=True, top_n_classes=None, mode='train'):
        """
        Initialize the dataset with cases and labels data.
        
        Args:
            cases_df: DataFrame with case information
            labels_df: DataFrame with diagnosis labels
            transform: Image transformations to apply
            base_img_dir: Base directory for image paths
            verify_images: Whether to check if image files exist
            top_n_classes: Limit to top N most common classes (optional)
            mode: 'train', 'val', or 'test' - affects augmentation and sampling
        """
        self.mode = mode
        self.transform = transform
        self.base_img_dir = base_img_dir
        
        # Merge datasets on case_id
        self.data = pd.merge(cases_df, labels_df, on='case_id', how='inner')
        print(f"Initial merged data size: {len(self.data)}")
        
        # Extract image paths and gradability for each image
        image_data = []
        
        print("Processing multiple images per case...")
        for _, row in tqdm(self.data.iterrows(), total=len(self.data)):
            case_id = row['case_id']
            diagnosis = None
            
            # Extract diagnosis - we still need a primary diagnosis for stratification
            if isinstance(row.get('dermatologist_skin_condition_on_label_name'), str):
                if row['dermatologist_skin_condition_on_label_name'].startswith('['):
                    try:
                        diagnosis_list = eval(row['dermatologist_skin_condition_on_label_name'])
                        if len(diagnosis_list) > 0:
                            diagnosis = diagnosis_list[0]
                    except:
                        pass
                else:
                    # Handle case where diagnosis is directly provided as string
                    diagnosis = row['dermatologist_skin_condition_on_label_name']
            
            # Skip if no diagnosis
            if not diagnosis:
                continue
                
            # Process each image for this case
            for img_idx in range(1, 4):  # Images 1, 2, 3
                img_path_col = f'image_{img_idx}_path'
                gradability_col = f'dermatologist_gradable_for_skin_condition_{img_idx}'
                
                # Check if image path exists and is gradable
                # More robust checking of image quality
                is_gradable = (
                    (gradability_col in row) and 
                    pd.notna(row[gradability_col]) and 
                    (row[gradability_col] == 'DEFAULT_YES_IMAGE_QUALITY_SUFFICIENT')
                )
                
                has_path = (
                    (img_path_col in row) and 
                    pd.notna(row[img_path_col]) and 
                    isinstance(row[img_path_col], str)
                )
                
                if has_path and is_gradable:
                    # Extract enhanced tabular features for this case
                    tabular_features = self.extract_enhanced_tabular_features(row)
                    
                    # Add to image data
                    image_data.append({
                        'case_id': case_id,
                        'image_path': row[img_path_col],
                        'diagnosis': diagnosis,  # Primary diagnosis for stratification
                        'tabular_features': tabular_features
                    })
        
        print(f"Total valid case-image pairs: {len(image_data)}")
        
        # Convert to DataFrame for easier handling
        self.image_data = pd.DataFrame(image_data)
        
        # Verify images exist if requested - critical for real-world deployment
        if verify_images and len(self.image_data) > 0:
            print("Verifying image files exist...")
            valid_indices = []
            for idx, row in tqdm(self.image_data.iterrows(), total=len(self.image_data)):
                img_path = os.path.join(base_img_dir, row['image_path'])
                if os.path.exists(img_path):
                    valid_indices.append(idx)
            
            print(f"Found {len(valid_indices)} valid images out of {len(self.image_data)}")
            self.image_data = self.image_data.iloc[valid_indices].reset_index(drop=True)
        
        # Keep only the top N most common skin conditions if specified
        if top_n_classes:
            top_classes = self.image_data['diagnosis'].value_counts().head(top_n_classes).index.tolist()
            self.image_data = self.image_data[self.image_data['diagnosis'].isin(top_classes)].reset_index(drop=True)
            print(f"Keeping only top {top_n_classes} conditions")
        
        # Create label encoder
        unique_diagnoses = sorted(self.image_data['diagnosis'].unique())
        self.diagnosis_to_idx = {diagnosis: idx for idx, diagnosis in enumerate(unique_diagnoses)}
        self.idx_to_diagnosis = {idx: diagnosis for diagnosis, idx in self.diagnosis_to_idx.items()}
        
        # Store class names for later
        self.classes = list(self.diagnosis_to_idx.keys())
        print(f"Final number of classes: {len(self.classes)}")
        
        # Print class distribution
        class_dist = self.image_data['diagnosis'].value_counts()
        print("Class distribution:")
        print(class_dist)
        
        # Calculate class weights for handling imbalance
        # Inverse frequency weighting: give higher weights to underrepresented classes
        total_samples = len(self.image_data)
        num_classes = len(class_dist)
        
        # Calculate weights inversely proportional to class frequency
        self.class_weights = {
            class_name: total_samples / (num_classes * count) 
            for class_name, count in class_dist.items()
        }
        
        print("Class weights (for handling class imbalance):")
        for class_name, weight in sorted(self.class_weights.items(), key=lambda x: x[1], reverse=True):
            print(f"{class_name}: {weight:.2f}")
        
        # Store the tabular feature dimension for consistency
        self.tabular_dim = len(self.extract_enhanced_tabular_features(self.data.iloc[0]))
        print(f"Tabular feature dimension: {self.tabular_dim}")
    
    def extract_enhanced_tabular_features(self, row):
        """
        Extract and normalize tabular features from a case.
        Includes demographics, condition duration, body location, symptoms, etc.
        
        This function is critical for incorporating non-image information that
        doctors use for diagnosis.
        """
        # Helper function to convert values to float (YES -> 1.0, others -> 0.0)
        def convert_value(val):
            if pd.isna(val):  # Check for NaN values
                return 0.0    # Convert NaN to 0.0
            elif isinstance(val, (int, float)):
                return float(val)
            elif isinstance(val, str) and val.upper() == 'YES':
                return 1.0
            else:
                return 0.0
        
        # Helper to convert categorical to numerical
        def convert_category(val, categories=None, default=0):
            if pd.isna(val):
                return default
            if categories is None:
                return float(val) if isinstance(val, (int, float)) else 0.0
            else:
                return float(categories.index(val)) if val in categories else default
        
        # 1. Demographics - critical as some conditions are age/sex/skin-type specific
        # Age group (convert to numerical)
        age_groups = ['AGE_0_TO_2', 'AGE_3_TO_11', 'AGE_12_TO_17', 'AGE_18_TO_29', 
                      'AGE_30_TO_39', 'AGE_40_TO_49', 'AGE_50_TO_64', 'AGE_65_PLUS']
        age_value = convert_category(row.get('age_group'), age_groups, 0)
        
        # Sex at birth (0 for unknown, 1 for male, 2 for female)
        sex_mapping = {'MALE': 1.0, 'FEMALE': 2.0}
        sex_value = sex_mapping.get(row.get('sex_at_birth'), 0.0)
        
        # Fitzpatrick skin type (1-6 scale) - extremely important for diagnosis
        fst_mapping = {'FST1': 1.0, 'FST2': 2.0, 'FST3': 3.0, 'FST4': 4.0, 'FST5': 5.0, 'FST6': 6.0}
        fst_value = fst_mapping.get(row.get('fitzpatrick_skin_type'), 0.0)
        
        demographics = [age_value, sex_value, fst_value]
        
        # 2. Race/ethnicity (binary features)
        ethnicity = [
            convert_value(row.get('race_ethnicity_white', 0)),
            convert_value(row.get('race_ethnicity_black_or_african_american', 0)),
            convert_value(row.get('race_ethnicity_asian', 0)),
            convert_value(row.get('race_ethnicity_hispanic_latino_or_spanish_origin', 0)),
            convert_value(row.get('race_ethnicity_middle_eastern_or_north_african', 0)),
        ]
        
        # 3. Duration of condition (critical for diagnosis)
        duration_mapping = {
            'ONE_DAY': 1.0,
            'LESS_THAN_ONE_WEEK': 2.0,
            'ONE_TO_FOUR_WEEKS': 3.0,
            'ONE_TO_THREE_MONTHS': 4.0,
            'MORE_THAN_THREE_MONTHS': 5.0
        }
        duration_value = duration_mapping.get(row.get('condition_duration'), 0.0)
        
        # 4. Body parts - where on the body the issue is
        # This is diagnostic as some conditions only affect certain body parts
        body_parts = [
            convert_value(row.get('body_parts_head_or_neck', 0)),
            convert_value(row.get('body_parts_arm', 0)),
            convert_value(row.get('body_parts_palm', 0)),
            convert_value(row.get('body_parts_back_of_hand', 0)),
            convert_value(row.get('body_parts_torso_front', 0)),
            convert_value(row.get('body_parts_torso_back', 0)),
            convert_value(row.get('body_parts_genitalia_or_groin', 0)),
            convert_value(row.get('body_parts_buttocks', 0)),
            convert_value(row.get('body_parts_leg', 0)),
            convert_value(row.get('body_parts_foot_top_or_side', 0)),
            convert_value(row.get('body_parts_foot_sole', 0))
        ]
        
        # 5. Symptoms - key for differential diagnosis
        symptoms = [
            convert_value(row.get('condition_symptoms_bothersome_appearance', 0)),
            convert_value(row.get('condition_symptoms_bleeding', 0)),
            convert_value(row.get('condition_symptoms_increasing_size', 0)),
            convert_value(row.get('condition_symptoms_darkening', 0)),
            convert_value(row.get('condition_symptoms_itching', 0)),
            convert_value(row.get('condition_symptoms_burning', 0)),
            convert_value(row.get('condition_symptoms_pain', 0)),
            convert_value(row.get('other_symptoms_fever', 0)),
            convert_value(row.get('other_symptoms_chills', 0)),
            convert_value(row.get('other_symptoms_fatigue', 0)),
            convert_value(row.get('other_symptoms_joint_pain', 0)),
            convert_value(row.get('other_symptoms_mouth_sores', 0)),
            convert_value(row.get('other_symptoms_shortness_of_breath', 0))
        ]
        
        # 6. Texture features - distinctive for many conditions
        textures = [
            convert_value(row.get('textures_raised_or_bumpy', 0)),
            convert_value(row.get('textures_flat', 0)),
            convert_value(row.get('textures_rough_or_flaky', 0)),
            convert_value(row.get('textures_fluid_filled', 0))
        ]
        
        # 7. Image metadata - shot type can be informative
        shot_type_mapping = {
            'CLOSE_UP': 1.0,
            'AT_AN_ANGLE': 2.0,
            'AT_DISTANCE': 3.0
        }
        shot_type = shot_type_mapping.get(row.get('image_1_shot_type'), 0.0)
        
        # Combine all features
        all_features = demographics + [duration_value] + ethnicity + body_parts + symptoms + textures + [shot_type]
        return all_features
    
    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset with multi-hot encoding for labels."""
        # Get row from dataframe
        row = self.image_data.iloc[idx]
        
        # Get image
        img_path = os.path.join(self.base_img_dir, row['image_path'])
        
        try:
            # Load and convert image to RGB
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Create a black image as fallback
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
        
        # Create multi-hot encoding for labels
        multi_hot_label = np.zeros(len(self.classes), dtype=np.float32)
        
        # Get case_id to retrieve original label information
        case_id = row['case_id']
        case_rows = self.data[self.data['case_id'] == case_id]
        if not case_rows.empty:
            original_row = case_rows.iloc[0]
            
            # Parse weighted diagnosis information
            try:
                weight_dict = {}
                if isinstance(original_row.get('weighted_skin_condition_label'), str):
                    if original_row['weighted_skin_condition_label'].startswith('{'):
                        weight_dict = eval(original_row['weighted_skin_condition_label'])
                elif isinstance(original_row.get('weighted_skin_condition_label'), dict):
                    weight_dict = original_row['weighted_skin_condition_label']
                    
                # For each diagnosis with weight above threshold, set to 1
                threshold = 0.2  # Consider diagnoses with at least 20% confidence
                for diagnosis, weight in weight_dict.items():
                    if diagnosis in self.diagnosis_to_idx and weight >= threshold:
                        multi_hot_label[self.diagnosis_to_idx[diagnosis]] = 1.0
            except Exception as e:
                pass
        
        # If no labels were assigned, use the primary diagnosis as fallback
        if np.sum(multi_hot_label) == 0 and row['diagnosis'] in self.diagnosis_to_idx:
            multi_hot_label[self.diagnosis_to_idx[row['diagnosis']]] = 1.0
        
        # Get tabular features
        tabular_features = np.array(row['tabular_features'], dtype=np.float32)
        
        # Calculate sample weight - use maximum class weight for any positive label
        weight = 1.0
        for i, is_present in enumerate(multi_hot_label):
            if is_present > 0:
                class_name = self.idx_to_diagnosis[i]
                weight = max(weight, self.class_weights.get(class_name, 1.0))
        
        return image, tabular_features, multi_hot_label, weight

#################################################
# MODEL ARCHITECTURE
#################################################

class CombinedModel(nn.Module):
    """
    Model that processes images and tabular data separately before combining them.
    This architecture allows the model to make better use of both data types.
    """
    def __init__(self, num_classes, tabular_dim):
        super(CombinedModel, self).__init__()
        
        # Image feature extraction using EfficientNet-B3
        # More modern than ResNet50 and better with limited data
        self.image_model = models.efficientnet_b3(weights='IMAGENET1K_V1')
        self.image_features = 1536  # Get feature dimension
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
            nn.BatchNorm1d(tabular_dim),  # Normalize inputs
            nn.Linear(tabular_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 + 64, 256),  # Combine image and tabular features
            nn.ReLU(),
            nn.Dropout(0.5),  # Higher dropout for final layers to prevent overfitting
            nn.Linear(256, num_classes)
        )
    
    def forward(self, images, tabular):
        """Forward pass combining both image and tabular data."""
        # Process image path
        img_features = self.image_model(images)
        img_features = self.image_fc(img_features)
        
        # Process tabular path
        tab_features = self.tabular_model(tabular)
        
        # Combine features from both paths
        combined = torch.cat((img_features, tab_features), dim=1)
        
        # Final classification
        return self.classifier(combined)

#################################################
# TRAINING AND EVALUATION FUNCTIONS
#################################################

def create_dataloaders(dataset, batch_size=16, val_split=0.15, test_split=0.15):
    """
    Create train, validation, and test splits with appropriate samplers.
    Returns appropriate data loaders for each split.
    
    For multi-label classification, we still stratify by the primary diagnosis
    to ensure a balanced distribution.
    """
    # Get dataset indices
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    # Get all primary labels for stratification
    # We need to use the primary diagnosis for stratification since we can't stratify by multi-hot vectors
    all_labels = [dataset.image_data.iloc[i]['diagnosis'] for i in range(dataset_size)]
    
    # Create stratified train, val, test splits
    # First split off test set
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_split, random_state=42, 
        stratify=[all_labels[i] for i in indices]
    )
    
    # Then split train/val
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_split/(1-test_split), random_state=42,
        stratify=[all_labels[i] for i in train_val_indices]
    )
    
    # Create weighted sampler for training set to handle class imbalance
    # Get class weights for each sample in training set
    train_samples_weights = [dataset[i][3] for i in train_indices]
    train_sampler = WeightedRandomSampler(
        weights=train_samples_weights,
        num_samples=len(train_indices),
        replacement=True
    )
    
    # Create data loaders
    # Train with weighted sampler
    train_loader = DataLoader(
        dataset, batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4, pin_memory=True
    )
    
    # Validation and test with sequential sampling
    val_loader = DataLoader(
        dataset, batch_size=batch_size,
        sampler=val_indices,
        num_workers=4, pin_memory=True
    )
    
    test_loader = DataLoader(
        dataset, batch_size=batch_size,
        sampler=test_indices,
        num_workers=4, pin_memory=True
    )
    
    # Create subset indices for reference
    split_indices = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }
    
    return train_loader, val_loader, test_loader, split_indices

def train_model(model, train_loader, val_loader, num_epochs=300, patience=15):
    """
    Train the model with early stopping based on validation performance.
    Modified for multi-label classification.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Maximum number of epochs to train
        patience: Number of epochs to wait for improvement before stopping
    
    Returns:
        Trained model and training history
    """
    # Move model to device
    model = model.to(device)
    
    # Use Binary Cross Entropy with Logits for multi-label classification
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    # Use AdamW optimizer which handles weight decay better than Adam
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Learning rate scheduler - reduce LR when performance plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6, verbose=True
    )
    
    # Initialize tracking variables
    best_val_f1 = 0.0  # Track F1 score instead of accuracy for multi-label
    patience_counter = 0
    history = {
        'train_loss': [], 'train_f1': [], 
        'val_loss': [], 'val_f1': [],
        'lr': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 10)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_true_pos = 0
        train_false_pos = 0
        train_false_neg = 0
        train_samples = 0
        
        # Use tqdm for a progress bar
        train_bar = tqdm(train_loader, desc='Training')
        for images, tabular, labels, weights in train_bar:
            # Move data to device
            images = images.to(device)
            tabular = tabular.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images, tabular)
            
            # For multi-label prediction
            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            # Calculate weighted loss
            loss = criterion(outputs, labels)
            loss = (loss.mean(dim=1) * weights).mean()  # Apply sample weights
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics for F1 score
            train_true_pos += torch.sum(preds * labels).item()
            train_false_pos += torch.sum(preds * (1 - labels)).item()
            train_false_neg += torch.sum((1 - preds) * labels).item()
            train_samples += images.size(0)
            train_loss += loss.item() * images.size(0)
            
            # Calculate F1 for progress bar
            precision = train_true_pos / (train_true_pos + train_false_pos + 1e-8)
            recall = train_true_pos / (train_true_pos + train_false_neg + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            # Update progress bar
            train_bar.set_postfix({
                'loss': loss.item(), 
                'f1': f1
            })
        
        # Calculate epoch statistics
        epoch_train_loss = train_loss / train_samples
        
        # Calculate F1 score
        precision = train_true_pos / (train_true_pos + train_false_pos + 1e-8)
        recall = train_true_pos / (train_true_pos + train_false_neg + 1e-8)
        epoch_train_f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_true_pos = 0
        val_false_pos = 0
        val_false_neg = 0
        val_samples = 0
        
        # No gradients needed for validation
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc='Validation')
            for images, tabular, labels, _ in val_bar:
                # Move data to device
                images = images.to(device)
                tabular = tabular.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(images, tabular)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                loss = criterion(outputs, labels).mean(dim=1).mean()  # No weighting for validation
                
                # Update statistics
                val_loss += loss.item() * images.size(0)
                val_true_pos += torch.sum(preds * labels).item()
                val_false_pos += torch.sum(preds * (1 - labels)).item()
                val_false_neg += torch.sum((1 - preds) * labels).item()
                val_samples += images.size(0)
                
                # Calculate metrics for progress bar
                precision = val_true_pos / (val_true_pos + val_false_pos + 1e-8)
                recall = val_true_pos / (val_true_pos + val_false_neg + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                
                # Update progress bar
                val_bar.set_postfix({
                    'loss': loss.item(), 
                    'f1': f1
                })
        
        # Calculate epoch statistics
        epoch_val_loss = val_loss / val_samples
        
        # Calculate validation F1 score
        val_precision = val_true_pos / (val_true_pos + val_false_pos + 1e-8)
        val_recall = val_true_pos / (val_true_pos + val_false_neg + 1e-8)
        epoch_val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall + 1e-8)
        
        # Print epoch results
        print(f'Train Loss: {epoch_train_loss:.4f} F1: {epoch_train_f1:.4f}')
        print(f'Val Loss: {epoch_val_loss:.4f} F1: {epoch_val_f1:.4f}')
        
        # Update learning rate based on validation performance
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(epoch_val_f1)  # Use F1 score for LR scheduling
        
        # Save history
        history['train_loss'].append(epoch_train_loss)
        history['train_f1'].append(epoch_train_f1)
        history['val_loss'].append(epoch_val_loss)
        history['val_f1'].append(epoch_val_f1)
        history['lr'].append(current_lr)
        
        # Check for improvement
        if epoch_val_f1 > best_val_f1:
            print(f"Validation F1 improved from {best_val_f1:.4f} to {epoch_val_f1:.4f}")
            best_val_f1 = epoch_val_f1
            # Save the best model
            torch.save(model.state_dict(), 'best_skin_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs (best F1: {best_val_f1:.4f})")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load the best model for return
    model.load_state_dict(torch.load('best_skin_model.pth'))
    return model, history

def evaluate_model(model, test_loader, class_names):
    """
    Comprehensive evaluation of model performance with metrics suitable for medical applications.
    Modified for multi-label classification.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        class_names: List of class names
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    # No gradients needed for evaluation
    with torch.no_grad():
        for images, tabular, labels, _ in tqdm(test_loader, desc='Evaluating'):
            # Move data to device
            images = images.to(device)
            tabular = tabular.to(device)
            
            # Forward pass
            outputs = model(images, tabular)
            probs = torch.sigmoid(outputs)  # Sigmoid for multi-label
            preds = (probs > 0.5).float()  # Binary prediction
            
            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    # Calculate exact match (all labels correct)
    exact_match = np.mean(np.all(y_true == y_pred, axis=1))
    
    # Calculate hamming score (proportion of correct labels)
    hamming_score = np.mean(y_true == y_pred)
    
    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Calculate overall metrics
    avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Create DataFrame for per-class metrics
    class_metrics = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    
    # Sort by support (number of samples)
    class_metrics_sorted = class_metrics.sort_values('Support', ascending=False)
    
    # Print summary metrics
    print("\nMulti-Label Classification Metrics:")
    print(f"Exact Match Ratio: {exact_match:.4f}")
    print(f"Hamming Score: {hamming_score:.4f}")
    print(f"Weighted Precision: {avg_precision:.4f}")
    print(f"Weighted Recall: {avg_recall:.4f}")
    print(f"Weighted F1-Score: {avg_f1:.4f}")
    
    return {
        'exact_match': exact_match,
        'hamming_score': hamming_score,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
        'class_metrics': class_metrics,
        'class_metrics_sorted': class_metrics_sorted,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

#################################################
# VISUALIZATION FUNCTIONS
#################################################

def plot_confusion_matrix(metrics, class_names, save_path='confusion_matrix.png'):
    """
    Plot multi-label confusion statistics instead of a traditional confusion matrix.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        class_names: List of class names
        save_path: Path to save the visualization
    """
    y_true = metrics['y_true']
    y_pred = metrics['y_pred']
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # For each class, calculate TP, FP, TN, FN
    stats = []
    for i, class_name in enumerate(class_names):
        tp = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
        fp = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1))
        tn = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 0))
        fn = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0))
        
        # Add to stats
        stats.append([class_name, tp, fp, tn, fn, tp+fn])
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(
        stats, 
        columns=['Class', 'True Positive', 'False Positive', 'True Negative', 'False Negative', 'Total']
    )
    stats_df['Precision'] = stats_df['True Positive'] / (stats_df['True Positive'] + stats_df['False Positive']).replace(0, np.nan)
    stats_df['Recall'] = stats_df['True Positive'] / stats_df['Total'].replace(0, np.nan)
    
    # Sort by total count
    stats_df = stats_df.sort_values('Total', ascending=False)
    
    # Plot as bar chart
    plt.subplot(1, 2, 1)
    plt.bar(range(len(stats_df)), stats_df['Precision'], label='Precision')
    plt.bar(range(len(stats_df)), stats_df['Recall'], alpha=0.7, label='Recall')
    plt.xticks(range(len(stats_df)), stats_df['Class'], rotation=90)
    plt.ylabel('Score')
    plt.title('Precision and Recall by Class')
    plt.legend()
    
    # Plot TP/FP/FN
    plt.subplot(1, 2, 2)
    plt.bar(range(len(stats_df)), stats_df['True Positive'], label='True Positive')
    plt.bar(range(len(stats_df)), stats_df['False Positive'], bottom=stats_df['True Positive'], label='False Positive')
    plt.bar(range(len(stats_df)), stats_df['False Negative'], bottom=stats_df['True Positive']+stats_df['False Positive'], label='False Negative')
    plt.xticks(range(len(stats_df)), stats_df['Class'], rotation=90)
    plt.ylabel('Count')
    plt.title('Classification Results by Class')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Classification statistics saved to {save_path}")

def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training history showing loss, F1 score, and learning rate.
    
    Args:
        history: Dictionary containing training metrics
        save_path: Path to save the visualization
    """
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot F1 score instead of accuracy
    ax2.plot(history['train_f1'], label='Train F1 Score')
    ax2.plot(history['val_f1'], label='Validation F1 Score')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Training and Validation F1 Score')
    ax2.legend()
    ax2.grid(True)
    
    # Plot learning rate
    ax3.plot(history['lr'], label='Learning Rate')
    ax3.set_ylabel('Learning Rate')
    ax3.set_xlabel('Epochs')
    ax3.set_title('Learning Rate')
    ax3.set_yscale('log')  # Log scale for learning rate
    ax3.grid(True)
    
    # Ensure layout is tight
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training history saved to {save_path}")

def plot_class_performance(metrics, save_path='class_performance.png'):
    """
    Plot per-class performance metrics.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        save_path: Path to save the visualization
    """
    # Extract metrics
    class_metrics = metrics['class_metrics_sorted']
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot precision, recall, F1 for each class
    x = np.arange(len(class_metrics))
    width = 0.25
    
    # Plot bars
    plt.bar(x - width, class_metrics['Precision'], width, label='Precision')
    plt.bar(x, class_metrics['Recall'], width, label='Recall')
    plt.bar(x + width, class_metrics['F1-Score'], width, label='F1-Score')
    
    # Add labels and title
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Performance Metrics by Class')
    plt.xticks(x, class_metrics['Class'], rotation=45, ha='right')
    plt.legend()
    
    # Add support values as text
    for i, support in enumerate(class_metrics['Support']):
        plt.text(i, 0.05, f'n={support}', ha='center', fontsize=8)
    
    # Ensure layout is tight
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Class performance metrics saved to {save_path}")

def plot_label_co_occurrence(metrics, class_names, save_path='label_co_occurrence.png'):
    """
    Plot a heatmap showing how often labels co-occur in the dataset.
    This is useful for understanding relationships between skin conditions.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        class_names: List of class names
        save_path: Path to save the visualization
    """
    y_true = metrics['y_true']
    
    # Create co-occurrence matrix
    co_occurrence = np.zeros((len(class_names), len(class_names)))
    
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            # Count cases where both labels are present
            if i == j:
                # Diagonal shows total instances of each class
                co_occurrence[i, j] = np.sum(y_true[:, i])
            else:
                co_occurrence[i, j] = np.sum((y_true[:, i] == 1) & (y_true[:, j] == 1))
    
    # Create figure
    plt.figure(figsize=(14, 12))
    
    # Plot heatmap
    sns.heatmap(
        co_occurrence, annot=True, fmt='g', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 8}
    )
    
    # Add labels and title
    plt.ylabel('Skin Condition', fontsize=12)
    plt.xlabel('Co-occurring Condition', fontsize=12)
    plt.title('Label Co-occurrence Matrix', fontsize=14)
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    
    # Ensure layout is tight
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Label co-occurrence matrix saved to {save_path}")

#################################################
# MAIN FUNCTION
#################################################

def main():
    """Main function to run the full pipeline."""
    # Define image transforms with data augmentation
    # Different transforms for training vs. validation/testing
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomRotation(20),  # Skin conditions can appear at any orientation
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # Lighting variations
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),  # Skin images often have no inherent up/down orientation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load data
    print("Loading data...")
    cases_df = pd.read_csv('dataset/scin_cases.csv')
    labels_df = pd.read_csv('dataset/scin_labels.csv')
    
    print(f"Cases DataFrame shape: {cases_df.shape}")
    print(f"Labels DataFrame shape: {labels_df.shape}")
    
    # Check if dataset directory exists
    if os.path.exists('dataset'):
        base_img_dir = ""  # No prefix needed
    elif os.path.exists('images'):
        base_img_dir = "images"
    else:
        base_img_dir = ""  # Default to no prefix
    
    # Create dataset with the training transforms
    print("\nCreating dataset with improved handling...")
    dataset = ImprovedSkinDataset(
        cases_df, labels_df, 
        transform=train_transforms,  # Start with training transforms
        base_img_dir=base_img_dir,
        verify_images=True,
        top_n_classes=20,  # Focus on top 20 conditions
        mode='train'
    )
    
    # Check if we have a valid dataset
    if len(dataset) == 0:
        print("Error: No valid samples in the dataset. Please check the data.")
        return
    
    print(f"Dataset created with {len(dataset)} valid samples")
    
    # Create data loaders with appropriate splits
    print("\nCreating data loaders with balanced sampling...")
    train_loader, val_loader, test_loader, split_indices = create_dataloaders(
        dataset, batch_size=32, val_split=0.15, test_split=0.15
    )
    
    print(f"Training set: {len(split_indices['train'])} samples")
    print(f"Validation set: {len(split_indices['val'])} samples")
    print(f"Test set: {len(split_indices['test'])} samples")
    
    # Create model
    print("\nCreating dual-input model (image + tabular features)...")
    num_classes = len(dataset.classes)
    tabular_dim = dataset.tabular_dim
    
    model = CombinedModel(num_classes=num_classes, tabular_dim=tabular_dim)
    print(f"Model created with {num_classes} output classes")
    
    # Train the model
    print("\nTraining model...")
    trained_model, history = train_model(
        model, train_loader, val_loader, 
        num_epochs=500,  # Maximum epochs
        patience=15     # Early stopping patience
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate the model on the test set
    print("\nEvaluating model on test set...")
    class_names = dataset.classes
    metrics = evaluate_model(trained_model, test_loader, class_names)
    
    # Plot multi-label classification statistics
    plot_confusion_matrix(metrics, class_names)
    
    # Plot class performance
    plot_class_performance(metrics)
    
    # Plot label co-occurrence
    plot_label_co_occurrence(metrics, class_names)
    
    # Save the final model
    print("\nSaving model...")
    torch.save(trained_model.state_dict(), 'test_final_skin_model.pth')
    
    # Save class names and other metadata
    model_metadata = {
        'class_names': class_names,
        'tabular_dim': tabular_dim,
        'input_size': (224, 224),
        'normalize_mean': [0.485, 0.456, 0.406],
        'normalize_std': [0.229, 0.224, 0.225]
    }
    
    joblib.dump(model_metadata, 'model_metadata.joblib')
    
    # Save class names to text file for easy reference
    with open('class_names.txt', 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    print("\nTraining and evaluation complete!")
    print("Model, metadata, and visualizations saved.")

#################################################
# PREDICTION FUNCTION FOR DEPLOYMENT
#################################################

def predict_skin_condition(image_path, tabular_data, model_path, metadata_path):
    """
    Function to make predictions with the trained model.
    Modified for multi-label prediction.
    
    Args:
        image_path: Path to skin image
        tabular_data: Dict with tabular features
        model_path: Path to saved model
        metadata_path: Path to model metadata
        
    Returns:
        Dict with prediction results including multiple potential conditions
    """
    # Load model metadata
    metadata = joblib.load(metadata_path)
    
    # Create the model architecture
    model = CombinedModel(
        num_classes=len(metadata['class_names']),
        tabular_dim=metadata['tabular_dim']
    )
    
    # Load model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Move model to device
    model = model.to(device)
    
    # Create transforms for inference
    inference_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            metadata['normalize_mean'],
            metadata['normalize_std']
        )
    ])
    
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = inference_transforms(image).unsqueeze(0).to(device)
    except Exception as e:
        return {'error': f"Image processing error: {str(e)}"}
    
    # Process tabular data
    try:
        # Convert to numpy array and ensure correct shape
        tabular_features = np.array(tabular_data, dtype=np.float32)
        tabular_tensor = torch.tensor(tabular_features).unsqueeze(0).to(device)
    except Exception as e:
        return {'error': f"Tabular data processing error: {str(e)}"}
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor, tabular_tensor)
        probabilities = torch.sigmoid(outputs)[0].cpu().numpy()  # Sigmoid for multi-label
    
    # Get all predictions above threshold
    threshold = 0.5
    positive_indices = np.where(probabilities >= threshold)[0]
    
    # If no conditions above threshold, get top 3
    if len(positive_indices) == 0:
        top_indices = np.argsort(probabilities)[::-1][:3]
    else:
        # Sort positive predictions by confidence
        top_indices = positive_indices[np.argsort(probabilities[positive_indices])[::-1]]
    
    # Format results
    predictions = []
    for i, idx in enumerate(top_indices):
        predictions.append({
            'condition': metadata['class_names'][idx],
            'probability': float(probabilities[idx]),
            'rank': i + 1,
            'above_threshold': bool(probabilities[idx] >= threshold)
        })
    
    # Group results by whether they're above threshold
    return {
        'predictions': predictions,
        'conditions_above_threshold': [p['condition'] for p in predictions if p['above_threshold']],
        'top_condition': metadata['class_names'][top_indices[0]],
        'confidence': float(probabilities[top_indices[0]])
    }

if __name__ == "__main__":
    main()