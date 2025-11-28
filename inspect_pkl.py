"""
This script inspects and displays the contents of PKL files
Use this to see what's inside best_rf_model.pkl and label_encoders.pkl

Usage: python inspect_pkl.py
"""

import pickle
import os

print("=" * 70)
print("  PKL FILE INSPECTOR")
print("=" * 70)

# ============================================================================
# INSPECT label_encoders.pkl
# ============================================================================

print("\n[1] Inspecting label_encoders.pkl")
print("-" * 70)

try:
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    
    print(f"✓ File loaded successfully!")
    print(f"  Type: {type(label_encoders)}")
    print(f"  Size: {os.path.getsize('label_encoders.pkl')} bytes")
    print(f"\n  Contents: Dictionary with {len(label_encoders)} encoders")
    print("-" * 70)
    
    for key, encoder in label_encoders.items():
        print(f"\n  📋 {key} Encoder:")
        print(f"     Type: {type(encoder).__name__}")
        print(f"     Classes: {list(encoder.classes_)}")
        print(f"     Number of classes: {len(encoder.classes_)}")
        
        # Show encoding examples
        print(f"     Encoding examples:")
        for i, cls in enumerate(encoder.classes_[:3]):  # Show first 3
            print(f"       '{cls}' → {i}")
        if len(encoder.classes_) > 3:
            print(f"       ... and {len(encoder.classes_) - 3} more")
    
except FileNotFoundError:
    print("❌ File not found! Run 'python create_model_pkl.py' first")
except Exception as e:
    print(f"❌ Error: {e}")

# ============================================================================
# INSPECT best_rf_model.pkl
# ============================================================================

print("\n\n[2] Inspecting best_rf_model.pkl")
print("-" * 70)

try:
    with open('best_rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    
    print(f"✓ File loaded successfully!")
    print(f"  Type: {type(rf_model).__name__}")
    print(f"  Size: {os.path.getsize('best_rf_model.pkl')} bytes")
    
    print(f"\n  Model Parameters:")
    print(f"     Number of trees: {rf_model.n_estimators}")
    print(f"     Max depth: {rf_model.max_depth}")
    print(f"     Min samples split: {rf_model.min_samples_split}")
    print(f"     Min samples leaf: {rf_model.min_samples_leaf}")
    print(f"     Random state: {rf_model.random_state}")
    print(f"     Number of features: {rf_model.n_features_in_}")
    print(f"     Number of classes: {rf_model.n_classes_}")
    
    print(f"\n  Feature Importance (Top 10):")
    feature_names = ['BMI', 'Smoking', 'Alcohol', 'Stroke', 'PhysicalHealth', 
                     'MentalHealth', 'DiffWalking', 'Sex', 'Age', 'Race',
                     'Diabetic', 'PhysicalActivity', 'GenHealth', 'Sleep',
                     'Asthma', 'KidneyDisease', 'SkinCancer']
    
    importances = sorted(
        zip(feature_names, rf_model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    
    for i, (feat, imp) in enumerate(importances[:10], 1):
        bar = '█' * int(imp * 100)
        print(f"     {i:2d}. {feat:20s} {imp:.4f} {bar}")
    
    print(f"\n  Model is trained and ready to predict!")
    
except FileNotFoundError:
    print("❌ File not found! Run 'python create_model_pkl.py' first")
except Exception as e:
    print(f"❌ Error: {e}")

# ============================================================================
# CREATE MANUAL ENCODERS (Alternative method)
# ============================================================================

print("\n\n[3] Alternative: Create Encoders Manually")
print("-" * 70)
print("""
If you want to create label_encoders.pkl manually without training:

from sklearn.preprocessing import LabelEncoder
import pickle

# Create encoders
encoders = {
    'Sex': LabelEncoder().fit(['Female', 'Male']),
    'AgeCategory': LabelEncoder().fit([
        '18-24', '25-29', '30-34', '35-39', '40-44', '45-49',
        '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'
    ]),
    'Race': LabelEncoder().fit([
        'White', 'Black', 'Asian', 'American Indian/Alaskan Native', 
        'Hispanic', 'Other'
    ]),
    'Diabetic': LabelEncoder().fit([
        'No', 'No, borderline diabetes', 'Yes', 'Yes (during pregnancy)'
    ]),
    'GenHealth': LabelEncoder().fit([
        'Excellent', 'Very good', 'Good', 'Fair', 'Poor'
    ])
}

# Save
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print('✅ Created!')
""")

print("\n" + "=" * 70)
print("  INSPECTION COMPLETE")
print("=" * 70)