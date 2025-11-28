"""
শুধুমাত্র label_encoders.pkl file create করার জন্য
যদি আপনি শুধু encoder file টা চান (model ছাড়া)

Usage: python create_encoders_only.py
"""

import pickle
from sklearn.preprocessing import LabelEncoder

print("=" * 70)
print("  LABEL ENCODERS GENERATOR")
print("=" * 70)

print("\n📝 Creating Label Encoders...")
print("-" * 70)

# Dictionary to store all encoders
label_encoders = {}

# ============================================================================
# 1. SEX ENCODER
# ============================================================================
print("\n[1/5] Creating Sex Encoder...")
sex_encoder = LabelEncoder()
sex_categories = ['Female', 'Male']
sex_encoder.fit(sex_categories)
label_encoders['Sex'] = sex_encoder

print(f"  ✓ Sex Encoder created")
print(f"     Categories: {sex_categories}")
print(f"     Encoding: Female=0, Male=1")

# ============================================================================
# 2. AGE CATEGORY ENCODER
# ============================================================================
print("\n[2/5] Creating Age Category Encoder...")
age_encoder = LabelEncoder()
age_categories = [
    '18-24',    # 0
    '25-29',    # 1
    '30-34',    # 2
    '35-39',    # 3
    '40-44',    # 4
    '45-49',    # 5
    '50-54',    # 6
    '55-59',    # 7
    '60-64',    # 8
    '65-69',    # 9
    '70-74',    # 10
    '75-79',    # 11
    '80 or older'  # 12
]
age_encoder.fit(age_categories)
label_encoders['AgeCategory'] = age_encoder

print(f"  ✓ Age Category Encoder created")
print(f"     Total categories: {len(age_categories)}")
print(f"     Range: 18-24 (0) to 80+ (12)")

# ============================================================================
# 3. RACE ENCODER
# ============================================================================
print("\n[3/5] Creating Race Encoder...")
race_encoder = LabelEncoder()
race_categories = [
    'American Indian/Alaskan Native',  # 0
    'Asian',                           # 1
    'Black',                           # 2
    'Hispanic',                        # 3
    'Other',                           # 4
    'White'                            # 5
]
race_encoder.fit(race_categories)
label_encoders['Race'] = race_encoder

print(f"  ✓ Race Encoder created")
print(f"     Categories: {len(race_categories)}")
for i, cat in enumerate(race_categories):
    print(f"       {i}: {cat}")

# ============================================================================
# 4. DIABETIC STATUS ENCODER
# ============================================================================
print("\n[4/5] Creating Diabetic Status Encoder...")
diabetic_encoder = LabelEncoder()
diabetic_categories = [
    'No',                        # 0
    'No, borderline diabetes',   # 1
    'Yes',                       # 2
    'Yes (during pregnancy)'     # 3
]
diabetic_encoder.fit(diabetic_categories)
label_encoders['Diabetic'] = diabetic_encoder

print(f"  ✓ Diabetic Status Encoder created")
print(f"     Categories:")
for i, cat in enumerate(diabetic_categories):
    print(f"       {i}: {cat}")

# ============================================================================
# 5. GENERAL HEALTH ENCODER
# ============================================================================
print("\n[5/5] Creating General Health Encoder...")
health_encoder = LabelEncoder()
health_categories = [
    'Excellent',    # 0
    'Fair',         # 1
    'Good',         # 2
    'Poor',         # 3
    'Very good'     # 4
]
health_encoder.fit(health_categories)
label_encoders['GenHealth'] = health_encoder

print(f"  ✓ General Health Encoder created")
print(f"     Categories:")
for i, cat in enumerate(health_categories):
    print(f"       {i}: {cat}")

# ============================================================================
# SAVE ENCODERS TO PKL FILE
# ============================================================================

print("\n" + "-" * 70)
print("💾 Saving encoders to 'label_encoders.pkl'...")

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

import os
file_size = os.path.getsize('label_encoders.pkl')

print(f"✓ Encoders saved successfully!")
print(f"  File: label_encoders.pkl")
print(f"  Size: {file_size:,} bytes ({file_size/1024:.2f} KB)")

# ============================================================================
# VERIFICATION
# ============================================================================

print("\n" + "-" * 70)
print("🔍 Verifying saved encoders...")

# Load and verify
with open('label_encoders.pkl', 'rb') as f:
    loaded_encoders = pickle.load(f)

print(f"✓ Loaded {len(loaded_encoders)} encoders:")
for key in loaded_encoders.keys():
    print(f"  - {key}")

# Test encoding
print("\n📊 Test Encoding Examples:")
print(f"  Sex['Male'] = {loaded_encoders['Sex'].transform(['Male'])[0]}")
print(f"  AgeCategory['60-64'] = {loaded_encoders['AgeCategory'].transform(['60-64'])[0]}")
print(f"  Race['Asian'] = {loaded_encoders['Race'].transform(['Asian'])[0]}")
print(f"  Diabetic['Yes'] = {loaded_encoders['Diabetic'].transform(['Yes'])[0]}")
print(f"  GenHealth['Good'] = {loaded_encoders['GenHealth'].transform(['Good'])[0]}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("  ✅ SUCCESS! LABEL ENCODERS CREATED")
print("=" * 70)

print(f"""
📦 File Created:
   label_encoders.pkl ({file_size/1024:.2f} KB)

📋 Contents:
   5 LabelEncoder objects for categorical features:
   1. Sex (2 categories)
   2. AgeCategory (13 categories)
   3. Race (6 categories)
   4. Diabetic (4 categories)
   5. GenHealth (5 categories)

💡 Usage in Code:
   import pickle
   
   with open('label_encoders.pkl', 'rb') as f:
       encoders = pickle.load(f)
   
   # Encode
   encoded_sex = encoders['Sex'].transform(['Male'])[0]
   
   # Decode
   decoded_sex = encoders['Sex'].inverse_transform([1])[0]

🚀 Next Step:
   Now create the model file:
   python create_model_pkl.py
   
   OR run the app directly if you have best_rf_model.pkl:
   streamlit run app.py
""")

print("=" * 70)