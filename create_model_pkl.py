"""
Complete script to generate best_rf_model.pkl and label_encoders.pkl
Run this to create all required pickle files for the Heart Disease Prediction app.

Usage: python create_model_pkl.py
"""

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

print("=" * 70)
print("  HEART DISEASE PREDICTION - MODEL & ENCODER GENERATOR")
print("=" * 70)

# ============================================================================
# PART 1: CREATE AND TRAIN RANDOM FOREST MODEL
# ============================================================================

print("\n[STEP 1/4] Creating Random Forest Model...")
print("-" * 70)

# Initialize Random Forest with optimal parameters
rf_model = RandomForestClassifier(
    n_estimators=100,          # Number of trees
    max_depth=20,              # Maximum depth of trees
    min_samples_split=5,       # Minimum samples to split
    min_samples_leaf=2,        # Minimum samples in leaf
    max_features='sqrt',       # Features to consider for split
    random_state=42,           # Reproducibility
    n_jobs=-1,                 # Use all CPU cores
    class_weight='balanced'    # Handle class imbalance
)

print("✓ Model initialized with parameters:")
print(f"  - Trees: {rf_model.n_estimators}")
print(f"  - Max Depth: {rf_model.max_depth}")
print(f"  - Random State: {rf_model.random_state}")

# ============================================================================
# PART 2: GENERATE TRAINING DATA
# ============================================================================

print("\n[STEP 2/4] Generating Training Data...")
print("-" * 70)

np.random.seed(42)
n_samples = 15000  # Larger dataset for better training

print(f"Generating {n_samples} samples with 17 features...")

# Generate feature data
X_train = np.column_stack([
    np.random.uniform(15, 50, n_samples),      # 0: BMI (15-50)
    np.random.randint(0, 2, n_samples),        # 1: Smoking (0=No, 1=Yes)
    np.random.randint(0, 2, n_samples),        # 2: AlcoholDrinking (0=No, 1=Yes)
    np.random.randint(0, 2, n_samples),        # 3: Stroke (0=No, 1=Yes)
    np.random.randint(0, 31, n_samples),       # 4: PhysicalHealth (0-30 days)
    np.random.randint(0, 31, n_samples),       # 5: MentalHealth (0-30 days)
    np.random.randint(0, 2, n_samples),        # 6: DiffWalking (0=No, 1=Yes)
    np.random.randint(0, 2, n_samples),        # 7: Sex (0=Female, 1=Male)
    np.random.randint(0, 13, n_samples),       # 8: AgeCategory (0-12)
    np.random.randint(0, 6, n_samples),        # 9: Race (0-5)
    np.random.randint(0, 4, n_samples),        # 10: Diabetic (0-3)
    np.random.randint(0, 2, n_samples),        # 11: PhysicalActivity (0=No, 1=Yes)
    np.random.randint(0, 5, n_samples),        # 12: GenHealth (0-4)
    np.random.randint(4, 12, n_samples),       # 13: SleepTime (4-11 hours)
    np.random.randint(0, 2, n_samples),        # 14: Asthma (0=No, 1=Yes)
    np.random.randint(0, 2, n_samples),        # 15: KidneyDisease (0=No, 1=Yes)
    np.random.randint(0, 2, n_samples),        # 16: SkinCancer (0=No, 1=Yes)
])

print("✓ Feature matrix created: shape", X_train.shape)

# Generate target variable with realistic risk modeling
print("\nGenerating target labels with risk modeling...")
y_train = np.zeros(n_samples, dtype=int)

for i in range(n_samples):
    risk_score = 0.0
    
    # Major risk factors
    if X_train[i, 0] > 35: risk_score += 0.30      # Obesity (BMI > 35)
    if X_train[i, 0] > 30: risk_score += 0.15      # Overweight (BMI > 30)
    if X_train[i, 1] == 1: risk_score += 0.35      # Smoking
    if X_train[i, 3] == 1: risk_score += 0.40      # Previous Stroke
    if X_train[i, 4] > 20: risk_score += 0.25      # Very poor physical health
    if X_train[i, 4] > 15: risk_score += 0.15      # Poor physical health
    
    # Age risk (exponential with age)
    if X_train[i, 8] >= 10: risk_score += 0.35     # Age 70+
    elif X_train[i, 8] >= 8: risk_score += 0.25    # Age 60-69
    elif X_train[i, 8] >= 6: risk_score += 0.15    # Age 50-59
    
    # Chronic conditions
    if X_train[i, 10] >= 2: risk_score += 0.30     # Diabetic
    elif X_train[i, 10] == 1: risk_score += 0.15   # Borderline diabetic
    if X_train[i, 6] == 1: risk_score += 0.20      # Difficulty walking
    if X_train[i, 15] == 1: risk_score += 0.25     # Kidney disease
    
    # Lifestyle factors
    if X_train[i, 2] == 1: risk_score += 0.15      # Heavy alcohol
    if X_train[i, 11] == 0: risk_score += 0.10     # No physical activity
    
    # General health
    if X_train[i, 12] == 4: risk_score += 0.25     # Poor health
    elif X_train[i, 12] == 3: risk_score += 0.15   # Fair health
    
    # Sleep
    if X_train[i, 13] < 6 or X_train[i, 13] > 9:
        risk_score += 0.10  # Poor sleep
    
    # Mental health correlation
    if X_train[i, 5] > 20: risk_score += 0.15      # Poor mental health
    
    # Assign heart disease label
    threshold = 0.7 + np.random.uniform(-0.2, 0.2)  # Variable threshold
    y_train[i] = 1 if risk_score > threshold else 0

# Add some random noise for realism (5% random flips)
noise_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
y_train[noise_indices] = 1 - y_train[noise_indices]

# Statistics
positive_cases = int(y_train.sum())
negative_cases = len(y_train) - positive_cases

print(f"\n✓ Target labels generated:")
print(f"  - Total samples: {n_samples}")
print(f"  - Positive cases (Heart Disease): {positive_cases} ({positive_cases/n_samples*100:.1f}%)")
print(f"  - Negative cases (No Disease): {negative_cases} ({negative_cases/n_samples*100:.1f}%)")

# ============================================================================
# PART 3: TRAIN THE MODEL
# ============================================================================

print("\n[STEP 3/4] Training Random Forest Model...")
print("-" * 70)

print("Training in progress... (this may take a moment)")
rf_model.fit(X_train, y_train)

# Calculate training metrics
train_accuracy = rf_model.score(X_train, y_train)
train_predictions = rf_model.predict(X_train)
train_probabilities = rf_model.predict_proba(X_train)

print(f"\n✓ Model training completed!")
print(f"  - Training Accuracy: {train_accuracy*100:.2f}%")
print(f"  - Features: {X_train.shape[1]}")
print(f"  - Trees in forest: {len(rf_model.estimators_)}")

# Feature importance
feature_names = ['BMI', 'Smoking', 'Alcohol', 'Stroke', 'PhysicalHealth', 
                 'MentalHealth', 'DiffWalking', 'Sex', 'Age', 'Race',
                 'Diabetic', 'PhysicalActivity', 'GenHealth', 'Sleep',
                 'Asthma', 'KidneyDisease', 'SkinCancer']

importances = rf_model.feature_importances_
top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:5]

print(f"\n  Top 5 Important Features:")
for feat, imp in top_features:
    print(f"    - {feat}: {imp:.4f}")

# ============================================================================
# SAVE MODEL
# ============================================================================

print("\n💾 Saving model to 'best_rf_model.pkl'...")
with open('best_rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

file_size = os.path.getsize('best_rf_model.pkl') / 1024
print(f"✓ Model saved successfully! (Size: {file_size:.2f} KB)")

# ============================================================================
# PART 4: CREATE LABEL ENCODERS
# ============================================================================

print("\n[STEP 4/4] Creating Label Encoders...")
print("-" * 70)

label_encoders = {}

# 1. Sex Encoder
sex_encoder = LabelEncoder()
sex_encoder.fit(['Female', 'Male'])
label_encoders['Sex'] = sex_encoder
print("✓ Sex encoder: ['Female', 'Male']")

# 2. Age Category Encoder
age_encoder = LabelEncoder()
age_categories = [
    '18-24', '25-29', '30-34', '35-39', '40-44', '45-49',
    '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'
]
age_encoder.fit(age_categories)
label_encoders['AgeCategory'] = age_encoder
print(f"✓ Age encoder: {len(age_categories)} categories")

# 3. Race Encoder
race_encoder = LabelEncoder()
races = ['White', 'Black', 'Asian', 'American Indian/Alaskan Native', 'Hispanic', 'Other']
race_encoder.fit(races)
label_encoders['Race'] = race_encoder
print(f"✓ Race encoder: {len(races)} categories")

# 4. Diabetic Encoder
diabetic_encoder = LabelEncoder()
diabetic_categories = ['No', 'No, borderline diabetes', 'Yes', 'Yes (during pregnancy)']
diabetic_encoder.fit(diabetic_categories)
label_encoders['Diabetic'] = diabetic_encoder
print(f"✓ Diabetic encoder: {len(diabetic_categories)} categories")

# 5. General Health Encoder
health_encoder = LabelEncoder()
health_categories = ['Excellent', 'Very good', 'Good', 'Fair', 'Poor']
health_encoder.fit(health_categories)
label_encoders['GenHealth'] = health_encoder
print(f"✓ General Health encoder: {len(health_categories)} categories")

# Save encoders
print("\n💾 Saving encoders to 'label_encoders.pkl'...")
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

encoder_size = os.path.getsize('label_encoders.pkl') / 1024
print(f"✓ Encoders saved successfully! (Size: {encoder_size:.2f} KB)")

# ============================================================================
# VERIFICATION & SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("  ✅ SUCCESS! ALL FILES GENERATED")
print("=" * 70)

print("\n📁 Generated Files:")
print(f"  1. best_rf_model.pkl       → {file_size:.2f} KB")
print(f"  2. label_encoders.pkl      → {encoder_size:.2f} KB")

print("\n📊 Model Summary:")
print(f"  - Algorithm: Random Forest Classifier")
print(f"  - Training Samples: {n_samples}")
print(f"  - Features: 17")
print(f"  - Accuracy: {train_accuracy*100:.2f}%")
print(f"  - Positive Class: {positive_cases} ({positive_cases/n_samples*100:.1f}%)")

print("\n🚀 Next Steps:")
print("  1. Ensure you have 'app.py' in the same directory")
print("  2. Run: streamlit run app.py")
print("  3. Access the app in your browser")

print("\n" + "=" * 70)
print("  Ready to predict heart disease! 🫀")
print("=" * 70)