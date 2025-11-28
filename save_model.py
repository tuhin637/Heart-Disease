import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

print("=" * 60)
print("Heart Disease Prediction - Model & Encoder Generator")
print("=" * 60)

# Create and train a Random Forest model
print("\n[1/3] Creating Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# Create dummy training data (17 features)
print("[2/3] Generating training data...")
np.random.seed(42)
n_samples = 5000

# Generate realistic-looking dummy data
X_train = np.column_stack([
    np.random.uniform(15, 50, n_samples),  # BMI
    np.random.randint(0, 2, n_samples),    # Smoking
    np.random.randint(0, 2, n_samples),    # AlcoholDrinking
    np.random.randint(0, 2, n_samples),    # Stroke
    np.random.randint(0, 31, n_samples),   # PhysicalHealth
    np.random.randint(0, 31, n_samples),   # MentalHealth
    np.random.randint(0, 2, n_samples),    # DiffWalking
    np.random.randint(0, 2, n_samples),    # Sex
    np.random.randint(0, 13, n_samples),   # AgeCategory
    np.random.randint(0, 6, n_samples),    # Race
    np.random.randint(0, 4, n_samples),    # Diabetic
    np.random.randint(0, 2, n_samples),    # PhysicalActivity
    np.random.randint(0, 5, n_samples),    # GenHealth
    np.random.randint(4, 12, n_samples),   # SleepTime
    np.random.randint(0, 2, n_samples),    # Asthma
    np.random.randint(0, 2, n_samples),    # KidneyDisease
    np.random.randint(0, 2, n_samples),    # SkinCancer
])

# Generate target with some logic
y_train = np.zeros(n_samples)
for i in range(n_samples):
    risk_score = 0
    if X_train[i, 0] > 30: risk_score += 0.3  # High BMI
    if X_train[i, 1] == 1: risk_score += 0.25  # Smoking
    if X_train[i, 3] == 1: risk_score += 0.3   # Stroke
    if X_train[i, 4] > 15: risk_score += 0.15  # Poor physical health
    if X_train[i, 8] > 8: risk_score += 0.2    # Older age
    
    y_train[i] = 1 if risk_score > 0.5 or np.random.random() < 0.3 else 0

print(f"   Training samples: {n_samples}")
print(f"   Features: {X_train.shape[1]}")
print(f"   Positive cases: {int(y_train.sum())} ({y_train.mean()*100:.1f}%)")
print(f"   Negative cases: {int((1-y_train).sum())} ({(1-y_train.mean())*100:.1f}%)")

# Train the model
print("\n[3/3] Training model...")
rf_model.fit(X_train, y_train)

# Calculate training accuracy
train_accuracy = rf_model.score(X_train, y_train)
print(f"   Training accuracy: {train_accuracy*100:.2f}%")

# Save the model
print("\n💾 Saving model...")
with open('best_rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("   ✅ Model saved as 'best_rf_model.pkl'")

# Create and save label encoders
print("\n🔤 Creating label encoders...")
label_encoders = {}

# Sex encoder
sex_encoder = LabelEncoder()
sex_encoder.fit(['Female', 'Male'])
label_encoders['Sex'] = sex_encoder
print("   ✓ Sex encoder created")

# AgeCategory encoder
age_encoder = LabelEncoder()
age_categories = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', 
                  '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older']
age_encoder.fit(age_categories)
label_encoders['AgeCategory'] = age_encoder
print("   ✓ Age category encoder created")

# Race encoder
race_encoder = LabelEncoder()
races = ['White', 'Black', 'Asian', 'American Indian/Alaskan Native', 'Hispanic', 'Other']
race_encoder.fit(races)
label_encoders['Race'] = race_encoder
print("   ✓ Race encoder created")

# Diabetic encoder
diabetic_encoder = LabelEncoder()
diabetic_categories = ['No', 'No, borderline diabetes', 'Yes', 'Yes (during pregnancy)']
diabetic_encoder.fit(diabetic_categories)
label_encoders['Diabetic'] = diabetic_encoder
print("   ✓ Diabetic encoder created")

# GenHealth encoder
health_encoder = LabelEncoder()
health_categories = ['Excellent', 'Very good', 'Good', 'Fair', 'Poor']
health_encoder.fit(health_categories)
label_encoders['GenHealth'] = health_encoder
print("   ✓ General health encoder created")

# Save encoders
print("\n💾 Saving label encoders...")
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("   ✅ Label encoders saved as 'label_encoders.pkl'")

# Verify files
print("\n" + "=" * 60)
print("✅ SUCCESS! All files created successfully!")
print("=" * 60)
print("\n📁 Files created:")
print("   1. best_rf_model.pkl          ({:.2f} KB)".format(
    __import__('os').path.getsize('best_rf_model.pkl') / 1024))
print("   2. label_encoders.pkl         ({:.2f} KB)".format(
    __import__('os').path.getsize('label_encoders.pkl') / 1024))

print("\n🚀 Next steps:")
print("   Run: streamlit run app.py")
print("\n" + "=" * 60)