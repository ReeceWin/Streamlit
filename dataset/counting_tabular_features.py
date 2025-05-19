import pandas as pd
import numpy as np
import json
import os
from collections import defaultdict

def count_features_by_disease(data_path=None):
    """
    Simple script to count the frequency of each tabular feature by disease.
    
    Args:
        data_path: Path to the dataset CSV. If None, uses synthetic data.
    
    Returns:
        Dictionary with counts for each feature by disease
    """
    # Load dataset
    if data_path and os.path.exists(data_path):
        print(f"Loading dataset from {data_path}")
        df = pd.read_csv(data_path)
    else:
        print("No dataset found. Creating synthetic data for demonstration.")
        df = create_synthetic_dataset()
    
    print(f"Dataset contains {len(df)} records")
    
    # Get list of all diseases
    diseases = df['diagnosis'].unique()
    print(f"Found {len(diseases)} unique skin conditions")
    
    # Initialize results dictionary
    results = {}
    
    # Feature categories and their column prefixes
    feature_categories = {
        'demographics': ['age_group', 'sex_at_birth', 'fitzpatrick_skin_type'],
        'race_ethnicity': ['race_ethnicity_'],
        'duration': ['condition_duration'],
        'body_parts': ['body_parts_'],
        'symptoms': ['condition_symptoms_'],
        'other_symptoms': ['other_symptoms_'],
        'textures': ['textures_']
    }
    
    # Process each disease
    for disease in diseases:
        print(f"Processing {disease}...")
        
        # Filter data for this disease
        disease_df = df[df['diagnosis'] == disease]
        sample_count = len(disease_df)
        
        # Skip if no data
        if sample_count == 0:
            continue
        
        # Initialize disease results
        disease_results = {
            'sample_count': sample_count,
            'features': {}
        }
        
        # Process each feature category
        for category, prefixes in feature_categories.items():
            category_results = {}
            
            # Handle demographics (categorical variables)
            if category == 'demographics':
                for col in prefixes:
                    if col in disease_df.columns:
                        # Get value counts and percentages
                        counts = disease_df[col].value_counts()
                        percentages = (counts / sample_count * 100).round(1)
                        
                        # Convert to dictionary
                        feature_data = {
                            str(value): {
                                'count': int(count),
                                'percentage': float(pct)
                            }
                            for value, count, pct in zip(counts.index, counts, percentages)
                        }
                        
                        category_results[col] = feature_data
            
            # Handle binary features (columns with prefixes)
            else:
                # Get all columns that match the prefix
                for prefix in prefixes:
                    matching_cols = [col for col in disease_df.columns if col.startswith(prefix)]
                    
                    for col in matching_cols:
                        # Count YES/1/TRUE values
                        yes_count = disease_df[col].astype(str).str.upper().isin(['YES', '1', 'TRUE']).sum()
                        percentage = (yes_count / sample_count * 100).round(1)
                        
                        # Only include if at least one patient has this feature
                        if yes_count > 0:
                            # Extract feature name from column name
                            if '_' in col:
                                parts = col.split('_')
                                if len(parts) > 1:
                                    feature_name = '_'.join(parts[1:])  # Remove prefix
                                else:
                                    feature_name = col
                            else:
                                feature_name = col
                            
                            category_results[feature_name] = {
                                'count': int(yes_count),
                                'percentage': float(percentage)
                            }
            
            # Add category results to disease results
            if category_results:
                disease_results['features'][category] = category_results
        
        # Add disease results to overall results
        results[disease] = disease_results
    
    return results

def create_synthetic_dataset():
    """
    Creates a simple synthetic dataset for demonstration
    """
    # Define some example diseases
    diseases = ['Acne', 'Eczema', 'Psoriasis', 'Herpes Zoster', 'Tinea']
    
    # Define columns
    columns = [
        'case_id', 'diagnosis', 
        'age_group', 'sex_at_birth', 'fitzpatrick_skin_type',
        'condition_duration',
        'race_ethnicity_white', 'race_ethnicity_black', 'race_ethnicity_asian',
        'body_parts_head', 'body_parts_arm', 'body_parts_leg', 'body_parts_torso',
        'condition_symptoms_itching', 'condition_symptoms_pain', 'condition_symptoms_burning',
        'other_symptoms_fever', 'other_symptoms_fatigue',
        'textures_raised', 'textures_flat', 'textures_rough'
    ]
    
    # Create empty dataframe
    df = pd.DataFrame(columns=columns)
    
    # Generate 500 random records
    import random
    records = []
    
    for i in range(500):
        disease = random.choice(diseases)
        
        # Create record with default 'NO' values
        record = {col: 'NO' for col in columns if col not in ['case_id', 'diagnosis', 'age_group', 'sex_at_birth', 'fitzpatrick_skin_type', 'condition_duration']}
        record['case_id'] = i
        record['diagnosis'] = disease
        
        # Demographics
        record['age_group'] = random.choice(['AGE_0_TO_2', 'AGE_3_TO_11', 'AGE_12_TO_17', 'AGE_18_TO_29', 'AGE_30_TO_39', 'AGE_40_TO_49', 'AGE_50_TO_64', 'AGE_65_PLUS'])
        record['sex_at_birth'] = random.choice(['MALE', 'FEMALE'])
        record['fitzpatrick_skin_type'] = random.choice(['FST1', 'FST2', 'FST3', 'FST4', 'FST5', 'FST6'])
        record['condition_duration'] = random.choice(['ONE_DAY', 'LESS_THAN_ONE_WEEK', 'ONE_TO_FOUR_WEEKS', 'ONE_TO_THREE_MONTHS', 'MORE_THAN_THREE_MONTHS'])
        
        # Choose one ethnicity
        ethnicity_cols = [col for col in columns if col.startswith('race_ethnicity_')]
        record[random.choice(ethnicity_cols)] = 'YES'
        
        # Choose 1-2 body parts
        body_part_cols = [col for col in columns if col.startswith('body_parts_')]
        for col in random.sample(body_part_cols, random.randint(1, 2)):
            record[col] = 'YES'
            
        # Choose 0-2 symptoms
        symptom_cols = [col for col in columns if col.startswith('condition_symptoms_')]
        for col in random.sample(symptom_cols, random.randint(0, 2)):
            record[col] = 'YES'
            
        # Choose 0-1 other symptoms
        other_symptom_cols = [col for col in columns if col.startswith('other_symptoms_')]
        if random.random() < 0.3:  # 30% chance to have other symptoms
            record[random.choice(other_symptom_cols)] = 'YES'
            
        # Choose 1 texture
        texture_cols = [col for col in columns if col.startswith('textures_')]
        record[random.choice(texture_cols)] = 'YES'
        
        records.append(record)
    
    return pd.DataFrame(records)

def save_results(results, output_path="disease_feature_counts.json"):
    """Save results to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

def print_summary(results):
    """Print a simple summary of the results"""
    print("\n===== FEATURE COUNTS BY DISEASE =====\n")
    
    for disease, data in results.items():
        print(f"\n{disease.upper()} (n={data['sample_count']})")
        print("-" * 50)
        
        for category, features in data.get('features', {}).items():
            print(f"\n{category.title()}:")
            
            # Handle different data structures more carefully
            for feature, values in features.items():
                if isinstance(values, dict) and 'count' in values and 'percentage' in values:
                    # Simple feature structure with count and percentage
                    print(f"  {feature}: {values['count']} ({values['percentage']}%)")
                elif isinstance(values, dict):
                    # Nested structure (e.g., demographics)
                    print(f"  {feature}:")
                    for value, stats in values.items():
                        if isinstance(stats, dict) and 'count' in stats and 'percentage' in stats:
                            print(f"    {value}: {stats['count']} ({stats['percentage']}%)")
                        else:
                            print(f"    {value}: {stats}")
                else:
                    # Handle unexpected structure
                    print(f"  {feature}: {values}")

def main():
    """Main function"""
    # Try several possible file paths
    possible_paths = [
        "skin_disease_dataset.csv",
        "dataset/skin_disease_data.csv"
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    # Get feature counts
    results = count_features_by_disease(data_path)
    
    # Save to file
    save_results(results)
    
    # Print summary
    print_summary(results)

if __name__ == "__main__":
    main()