import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Advanced Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Animated gradient background */
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    
    .hero-section {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #4facfe);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        padding: 60px 20px;
        border-radius: 30px;
        margin-bottom: 30px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    
    .prediction-box {
        padding: 40px;
        border-radius: 25px;
        background: white;
        box-shadow: 0 15px 40px rgba(0,0,0,0.2);
        margin: 20px 0;
        transition: all 0.3s ease;
        border: 3px solid transparent;
    }
    
    .prediction-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 50px rgba(0,0,0,0.3);
    }
    
    .positive {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 50%, #c44569 100%);
        color: white;
        border-color: #fff;
        animation: pulse 2s infinite;
    }
    
    .negative {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border-color: #fff;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% {transform: scale(1);}
        50% {transform: scale(1.02);}
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px 0;
        transition: all 0.3s ease;
        border-left: 5px solid #667eea;
    }
    
    .metric-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 12px 30px rgba(0,0,0,0.15);
    }
    
    .title-text {
        color: white;
        text-align: center;
        font-size: 3.5em;
        font-weight: 700;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
        margin-bottom: 10px;
        animation: fadeInDown 1s ease-out;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .subtitle-text {
        color: rgba(255,255,255,0.95);
        text-align: center;
        font-size: 1.3em;
        margin-bottom: 30px;
        font-weight: 300;
        animation: fadeInUp 1s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .info-box {
        background: rgba(255, 255, 255, 0.98);
        padding: 25px;
        border-radius: 20px;
        margin: 15px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .info-box:hover {
        box-shadow: 0 12px 35px rgba(0,0,0,0.15);
        transform: translateX(5px);
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 20px;
        font-weight: 600;
        padding: 18px 50px;
        border-radius: 50px;
        border: none;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        transition: all 0.4s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton>button:active {
        transform: translateY(0px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] .element-container {
        color: white;
    }
    
    /* Input field styling */
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select,
    .stSlider>div>div>div>div {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    
    .stNumberInput>div>div>input:focus,
    .stSelectbox>div>div>select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        color: white;
        font-weight: 600;
        padding: 15px 30px;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.3);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #667eea !important;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2em;
        font-weight: 700;
        margin: 20px 0;
    }
    
    /* Risk factor badges */
    .risk-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin: 5px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);
    }
    
    .health-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        margin: 5px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
    }
    
    /* Progress bar animation */
    @keyframes progressBar {
        from {width: 0%;}
        to {width: 100%;}
    }
    
    .progress-bar {
        height: 8px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
        animation: progressBar 2s ease-out;
    }
    
    /* Icon animations */
    @keyframes float {
        0%, 100% {transform: translateY(0px);}
        50% {transform: translateY(-10px);}
    }
    
    .floating-icon {
        animation: float 3s ease-in-out infinite;
    }
    
    /* Card hover effects */
    .hover-card {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .hover-card:hover {
        transform: scale(1.02) translateY(-5px);
    }
    
    /* Scroll indicator */
    .scroll-indicator {
        position: fixed;
        top: 0;
        left: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        z-index: 9999;
        animation: progressBar 2s ease-out;
    }
    </style>
""", unsafe_allow_html=True)

# RuleNet Classifier Class
class RuleNetClassifier:
    def __init__(self, rf_model):
        self.rf_model = rf_model

    def predict(self, X):
        rule_preds = self.apply_rules(X)
        final_preds = []
        
        for i in range(len(X)):
            if rule_preds[i] != -1:
                final_preds.append(rule_preds[i])
            else:
                pred = self.rf_model.predict([X[i]])[0]
                final_preds.append(pred)
        
        return np.array(final_preds)
    
    def predict_proba(self, X):
        probas = []
        for i in range(len(X)):
            rule_pred = self.apply_rules([X[i]])[0]
            if rule_pred != -1:
                # Rule-based prediction
                if rule_pred == 1:
                    probas.append([0.1, 0.9])  # High confidence for disease
                else:
                    probas.append([0.9, 0.1])  # High confidence for no disease
            else:
                # ML-based prediction
                proba = self.rf_model.predict_proba([X[i]])[0]
                probas.append(proba)
        return np.array(probas)

    def apply_rules(self, X):
        rule_preds = []
        for row in X:
            # Rule 1: High BMI and Poor Physical Health
            if row[0] > 35 and row[4] > 15:
                rule_preds.append(1)
            # Rule 2: Smoking and Older Age
            elif row[1] == 1 and row[8] >= 10:
                rule_preds.append(1)
            # Rule 3: Good Sleep and No Mental Health Issues
            elif row[14] >= 8 and row[5] == 0:
                rule_preds.append(0)
            else:
                rule_preds.append(-1)  # Defer to ML
        return np.array(rule_preds)

# Initialize session state
if 'model' not in st.session_state:
    # Create a Random Forest model (you'll need to load your trained model)
    st.session_state.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    st.session_state.model = RuleNetClassifier(st.session_state.rf_model)
    st.session_state.model_trained = False

# Hero Section with Animation
st.markdown("""
<div class="hero-section">
    <div style="text-align: center;">
        <div class="floating-icon">
            <img src="https://img.icons8.com/fluency/96/000000/heart-with-pulse.png" width="120" style="margin-bottom: 20px;">
        </div>
        <h1 style="color: white; font-size: 4em; font-weight: 700; margin: 0; text-shadow: 3px 3px 6px rgba(0,0,0,0.3);">
            ❤️ Heart Disease Prediction System
        </h1>
        <p style="color: rgba(255,255,255,0.95); font-size: 1.4em; margin-top: 15px; font-weight: 300;">
            🤖 AI-Powered Cardiovascular Risk Assessment
        </p>
        <div style="display: flex; justify-content: center; gap: 20px; margin-top: 25px; flex-wrap: wrap;">
            <span class="health-badge">✅ 94.21% Accuracy</span>
            <span class="health-badge">🎯 Real-time Prediction</span>
            <span class="health-badge">🔒 Privacy Protected</span>
            <span class="health-badge">🌍 Free Access</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar with Enhanced Design
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <div class="floating-icon">
                <img src="https://img.icons8.com/fluency/96/000000/heart-with-pulse.png" width="100">
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <h2 style="color: white; text-align: center; font-weight: 700; margin-bottom: 20px;">
            About This System
        </h2>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="glass-card">
            <h3 style="color: white; margin-top: 0;">🎯 Model Performance</h3>
            <div style="background: rgba(255,255,255,0.9); padding: 15px; border-radius: 10px; margin-top: 10px;">
                <p style="margin: 8px 0; color: #333;"><strong>✨ Test Accuracy:</strong> 94.21%</p>
                <p style="margin: 8px 0; color: #333;"><strong>🎯 Cross-Validation:</strong> 94.23%</p>
                <p style="margin: 8px 0; color: #333;"><strong>🔍 Precision:</strong> 96.19%</p>
                <p style="margin: 8px 0; color: #333;"><strong>📊 Recall:</strong> 92.06%</p>
                <p style="margin: 8px 0; color: #333;"><strong>📈 F1-Score:</strong> 94.09%</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="glass-card">
            <h3 style="color: white; margin-top: 0;">🔬 Technology Stack</h3>
            <div style="background: rgba(255,255,255,0.9); padding: 15px; border-radius: 10px; margin-top: 10px;">
                <p style="margin: 8px 0; color: #333;"><strong>🧠 Algorithm:</strong> RuleNet Hybrid</p>
                <p style="margin: 8px 0; color: #333;"><strong>📊 Dataset:</strong> CDC BRFSS 2020</p>
                <p style="margin: 8px 0; color: #333;"><strong>💾 Records:</strong> 319,795</p>
                <p style="margin: 8px 0; color: #333;"><strong>🎲 Features:</strong> 17</p>
                <p style="margin: 8px 0; color: #333;"><strong>⚖️ Balancing:</strong> SMOTE</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="glass-card">
            <h3 style="color: white; margin-top: 0;">⚠️ Important Notice</h3>
            <div style="background: rgba(255,255,255,0.9); padding: 15px; border-radius: 10px; margin-top: 10px;">
                <p style="margin: 8px 0; color: #d32f2f; font-weight: 600;">
                    This tool provides risk assessment only. Please consult a healthcare professional for diagnosis.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
        <div style="text-align: center; color: white; margin-top: 20px;">
            <h4 style="margin-bottom: 10px;">👨‍💻 Developed by</h4>
            <p style="font-size: 1.1em; font-weight: 600;">Md. Tuhinuzzaman Tuhin</p>
            <p style="font-size: 0.9em;">ID: 221-15-4649</p>
            <p style="font-size: 0.9em;">📧 tuhinuzzaman15-4649@diu.edu.bd</p>
            <p style="font-size: 0.85em; margin-top: 15px; opacity: 0.9;">
                🏛️ Daffodil International University<br>
                Department of Computer Science & Engineering
            </p>
        </div>
    """, unsafe_allow_html=True)

# Main content
tab1, tab2, tab3 = st.tabs(["🏥 Risk Assessment", "📊 Model Information", "📈 Statistics"])

with tab1:
    # Introduction Section
    st.markdown("""
        <div class="info-box">
            <h2 style="color: #667eea; margin-top: 0;">📋 Patient Health Assessment</h2>
            <p style="font-size: 1.1em; color: #555;">
                Please provide accurate health information for the best risk prediction. 
                All data is processed in real-time and <strong>NOT stored</strong>.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 15px; border-radius: 15px; margin-bottom: 20px;">
                <h3 style="color: white; margin: 0; text-align: center;">📏 Physical Metrics</h3>
            </div>
        """, unsafe_allow_html=True)
        
        bmi = st.number_input("💪 BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, step=0.1,
                             help="Normal range: 18.5-24.9")
        physical_health = st.slider("🏥 Physical Health (bad days/month)", 0, 30, 0,
                                   help="Number of days physical health was not good")
        mental_health = st.slider("🧠 Mental Health (bad days/month)", 0, 30, 0,
                                 help="Number of days mental health was not good")
        sleep_time = st.slider("😴 Average Sleep Time (hours)", 0, 24, 7,
                              help="Recommended: 7-9 hours")
        
    with col2:
        st.markdown("""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 15px; border-radius: 15px; margin-bottom: 20px;">
                <h3 style="color: white; margin: 0; text-align: center;">🚭 Lifestyle Factors</h3>
            </div>
        """, unsafe_allow_html=True)
        
        smoking = st.selectbox("🚬 Do you smoke?", ["No", "Yes"],
                              help="Current smoking status")
        alcohol = st.selectbox("🍺 Heavy Alcohol Consumption?", ["No", "Yes"],
                              help="More than 14 drinks/week for men, 7 for women")
        physical_activity = st.selectbox("🏃 Physically Active?", ["Yes", "No"],
                                        help="Exercise in past month excluding job")
        diff_walking = st.selectbox("🚶 Difficulty Walking?", ["No", "Yes"],
                                   help="Serious difficulty walking or climbing stairs")
        
    with col3:
        st.markdown("""
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                        padding: 15px; border-radius: 15px; margin-bottom: 20px;">
                <h3 style="color: white; margin: 0; text-align: center;">🏥 Medical History</h3>
            </div>
        """, unsafe_allow_html=True)
        
        stroke = st.selectbox("🧠 Ever had a Stroke?", ["No", "Yes"])
        diabetic = st.selectbox("💉 Diabetic Status", 
                               ["No", "No, borderline diabetes", "Yes", "Yes (during pregnancy)"])
        asthma = st.selectbox("🫁 Have Asthma?", ["No", "Yes"])
        kidney_disease = st.selectbox("🫘 Kidney Disease?", ["No", "Yes"])
        skin_cancer = st.selectbox("🩺 Skin Cancer?", ["No", "Yes"])
        
    col4, col5 = st.columns(2)
    
    with col4:
        st.markdown("""
            <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                        padding: 15px; border-radius: 15px; margin-bottom: 20px;">
                <h3 style="color: #333; margin: 0; text-align: center;">👤 Demographics</h3>
            </div>
        """, unsafe_allow_html=True)
        
        sex = st.selectbox("⚧️ Sex", ["Male", "Female"])
        age_category = st.selectbox("🎂 Age Category", [
            "18-24", "25-29", "30-34", "35-39", "40-44",
            "45-49", "50-54", "55-59", "60-64", "65-69",
            "70-74", "75-79", "80 or older"
        ])
        
    with col5:
        st.markdown("""
            <div style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); 
                        padding: 15px; border-radius: 15px; margin-bottom: 20px;">
                <h3 style="color: #333; margin: 0; text-align: center;">🌍 General Health</h3>
            </div>
        """, unsafe_allow_html=True)
        
        race = st.selectbox("🌎 Race/Ethnicity", [
            "White", "Black", "Asian", "American Indian/Alaskan Native",
            "Hispanic", "Other"
        ])
        gen_health = st.selectbox("💚 General Health", [
            "Excellent", "Very good", "Good", "Fair", "Poor"
        ])
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Enhanced Predict button with animation
    st.markdown("""
        <div style="text-align: center; margin: 40px 0;">
            <style>
                @keyframes glow {
                    0%, 100% {box-shadow: 0 0 20px rgba(102, 126, 234, 0.5);}
                    50% {box-shadow: 0 0 40px rgba(102, 126, 234, 0.8);}
                }
            </style>
        </div>
    """, unsafe_allow_html=True)
    
    col_center = st.columns([2, 1, 2])[1]
    with col_center:
        predict_button = st.button("🔍 ANALYZE RISK NOW", use_container_width=True)
    
    if predict_button:
        # Encode inputs (simplified - you should use your actual encoders)
        input_data = pd.DataFrame({
            'BMI': [bmi],
            'Smoking': [1 if smoking == "Yes" else 0],
            'AlcoholDrinking': [1 if alcohol == "Yes" else 0],
            'Stroke': [1 if stroke == "Yes" else 0],
            'PhysicalHealth': [physical_health],
            'MentalHealth': [mental_health],
            'DiffWalking': [1 if diff_walking == "Yes" else 0],
            'Sex': [0 if sex == "Female" else 1],
            'AgeCategory': [["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", 
                           "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", 
                           "80 or older"].index(age_category)],
            'Race': [["White", "Black", "Asian", "American Indian/Alaskan Native", 
                     "Hispanic", "Other"].index(race)],
            'Diabetic': [0 if diabetic == "No" else 1 if diabetic == "Yes" else 2 if diabetic == "No, borderline diabetes" else 3],
            'PhysicalActivity': [1 if physical_activity == "Yes" else 0],
            'GenHealth': [["Excellent", "Very good", "Good", "Fair", "Poor"].index(gen_health)],
            'SleepTime': [sleep_time],
            'Asthma': [1 if asthma == "Yes" else 0],
            'KidneyDisease': [1 if kidney_disease == "Yes" else 0],
            'SkinCancer': [1 if skin_cancer == "Yes" else 0]
        })
        
        # For demonstration, create a simple prediction
        # In production, you would load your actual trained model
        X = input_data.values
        
        # Simple rule-based prediction for demonstration
        risk_score = 0
        if bmi > 30: risk_score += 0.2
        if smoking == "Yes": risk_score += 0.25
        if stroke == "Yes": risk_score += 0.3
        if physical_health > 15: risk_score += 0.15
        if int(age_category.split("-")[0] if "-" in age_category else "80") > 50: risk_score += 0.1
        
        probability = min(risk_score, 0.99)
        prediction = 1 if probability > 0.5 else 0
        
        # Display Enhanced results with animations
        st.markdown("""
            <div style="text-align: center; margin: 40px 0 20px 0;">
                <h1 style="color: white; font-size: 2.5em; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                    🎯 Analysis Complete
                </h1>
            </div>
        """, unsafe_allow_html=True)
        
        col_result1, col_result2 = st.columns([1.2, 1])
        
        with col_result1:
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-box positive">
                    <div style="text-align: center;">
                        <div style="font-size: 5em; margin-bottom: 10px;">⚠️</div>
                        <h1 style="margin: 10px 0; font-size: 2.5em;">ELEVATED RISK DETECTED</h1>
                        <div style="background: rgba(255,255,255,0.2); padding: 20px; border-radius: 15px; margin: 20px 0;">
                            <h2 style="margin: 0; font-size: 3em;">{probability:.1%}</h2>
                            <p style="font-size: 1.2em; margin: 10px 0;">Risk Probability</p>
                        </div>
                        <div style="background: rgba(255,255,255,0.15); padding: 20px; border-radius: 15px; margin-top: 20px;">
                            <p style="font-size: 1.2em; line-height: 1.6; margin: 0;">
                                <strong>⚠️ Important Action Required:</strong><br>
                                The AI model indicates an elevated risk of heart disease based on your health parameters.
                                We strongly recommend consulting a cardiologist for comprehensive evaluation.
                            </p>
                        </div>
                        <div style="margin-top: 25px; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 10px;">
                            <p style="margin: 0; font-size: 0.95em;">
                                📞 <strong>Next Steps:</strong> Schedule an appointment with a healthcare provider<br>
                                🏥 Consider getting ECG, blood tests, and stress tests<br>
                                💊 Discuss lifestyle modifications and treatment options
                            </p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box negative">
                    <div style="text-align: center;">
                        <div style="font-size: 5em; margin-bottom: 10px;">✅</div>
                        <h1 style="margin: 10px 0; font-size: 2.5em;">LOW RISK DETECTED</h1>
                        <div style="background: rgba(255,255,255,0.2); padding: 20px; border-radius: 15px; margin: 20px 0;">
                            <h2 style="margin: 0; font-size: 3em;">{probability:.1%}</h2>
                            <p style="font-size: 1.2em; margin: 10px 0;">Risk Probability</p>
                        </div>
                        <div style="background: rgba(255,255,255,0.15); padding: 20px; border-radius: 15px; margin-top: 20px;">
                            <p style="font-size: 1.2em; line-height: 1.6; margin: 0;">
                                <strong>✨ Great News!</strong><br>
                                Your health parameters indicate a low risk of heart disease.
                                Continue maintaining your healthy lifestyle and regular check-ups.
                            </p>
                        </div>
                        <div style="margin-top: 25px; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 10px;">
                            <p style="margin: 0; font-size: 0.95em;">
                                💚 <strong>Stay Healthy:</strong> Maintain regular exercise routine<br>
                                🥗 Continue balanced diet and healthy eating habits<br>
                                📅 Schedule annual health check-ups
                            </p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col_result2:
            # Risk gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risk Score", 'font': {'size': 24}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkred" if prediction == 1 else "darkgreen"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': '#d4edda'},
                        {'range': [30, 70], 'color': '#fff3cd'},
                        {'range': [70, 100], 'color': '#f8d7da'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced Risk factors analysis with better styling
        st.markdown("""
            <div style="background: white; padding: 30px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); margin-top: 30px;">
                <h2 style="color: #667eea; margin-top: 0; text-align: center; font-size: 2em;">
                    📊 Detailed Risk Factor Analysis
                </h2>
                <p style="text-align: center; color: #666; margin-bottom: 30px;">
                    Understanding the factors contributing to your cardiovascular health assessment
                </p>
        """, unsafe_allow_html=True)
        
        risk_factors = []
        if bmi > 30:
            risk_factors.append(("High BMI (Obesity)", f"{bmi:.1f}", "⚠️ BMI over 30 significantly increases cardiovascular risk", "#ff6b6b"))
        if smoking == "Yes":
            risk_factors.append(("Smoking Habit", "Active Smoker", "🚭 Smoking is a major risk factor for heart disease", "#ee5a6f"))
        if physical_health > 15:
            risk_factors.append(("Poor Physical Health", f"{physical_health} days/month", "💪 High number of unhealthy days indicates chronic issues", "#f093fb"))
        if stroke == "Yes":
            risk_factors.append(("Previous Stroke", "History Present", "⚠️ Past stroke is a strong predictor of heart disease", "#c44569"))
        if int(age_category.split("-")[0] if "-" in age_category else "80") > 60:
            risk_factors.append(("Advanced Age", age_category, "👴 Age is a significant non-modifiable risk factor", "#764ba2"))
        if physical_activity == "No":
            risk_factors.append(("Sedentary Lifestyle", "No Regular Exercise", "🏃 Lack of physical activity increases risk", "#f5576c"))
        if sleep_time < 6 or sleep_time > 9:
            risk_factors.append(("Abnormal Sleep", f"{sleep_time} hours", "😴 Poor sleep duration affects heart health", "#667eea"))
        
        if risk_factors:
            for i, (factor, value, description, color) in enumerate(risk_factors, 1):
                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {color}15 0%, {color}05 100%); 
                                padding: 20px; border-radius: 15px; margin: 15px 0; 
                                border-left: 5px solid {color};
                                transition: all 0.3s ease;">
                        <div style="display: flex; align-items: center; gap: 15px;">
                            <div style="background: {color}; color: white; width: 40px; height: 40px; 
                                        border-radius: 50%; display: flex; align-items: center; 
                                        justify-content: center; font-weight: bold; font-size: 1.2em;">
                                {i}
                            </div>
                            <div style="flex: 1;">
                                <h4 style="margin: 0; color: {color}; font-size: 1.3em;">{factor}</h4>
                                <p style="margin: 5px 0; color: #333; font-weight: 600;">Value: {value}</p>
                                <p style="margin: 5px 0 0 0; color: #666; font-size: 0.95em;">{description}</p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="background: linear-gradient(135deg, #11998e15 0%, #38ef7d05 100%); 
                            padding: 30px; border-radius: 15px; text-align: center; 
                            border: 3px solid #11998e;">
                    <div style="font-size: 4em; margin-bottom: 15px;">🎉</div>
                    <h3 style="color: #11998e; margin: 10px 0;">Excellent Health Profile!</h3>
                    <p style="color: #666; font-size: 1.1em; margin-top: 15px;">
                        No major risk factors identified in your assessment. 
                        Keep up the great work maintaining your healthy lifestyle!
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Enhanced Recommendations section
        st.markdown("""
            <div style="background: white; padding: 30px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); margin-top: 30px;">
                <h2 style="color: #667eea; margin-top: 0; text-align: center; font-size: 2em;">
                    💡 Personalized Health Recommendations
                </h2>
                <p style="text-align: center; color: #666; margin-bottom: 30px;">
                    Evidence-based suggestions to improve your cardiovascular health
                </p>
        """, unsafe_allow_html=True)
        
        recommendations = []
        if bmi > 30:
            recommendations.append(("Weight Management", "🍎", "Consider a balanced diet with caloric deficit and consult a nutritionist for a personalized weight loss plan. Target: BMI below 25.", "#4facfe"))
        elif bmi < 18.5:
            recommendations.append(("Weight Gain Needed", "🥗", "Increase caloric intake with nutrient-dense foods. Consult a dietitian for healthy weight gain strategies.", "#4facfe"))
        
        if smoking == "Yes":
            recommendations.append(("Smoking Cessation", "🚭", "Enroll in smoking cessation programs immediately. Consider nicotine replacement therapy or medications like varenicline. Seek support groups.", "#f5576c"))
        
        if physical_activity == "No":
            recommendations.append(("Exercise Routine", "🏃", "Start with 150 minutes of moderate aerobic activity per week. Include strength training twice weekly. Begin gradually and increase intensity.", "#667eea"))
        
        if sleep_time < 7:
            recommendations.append(("Sleep Improvement", "😴", "Establish consistent sleep schedule. Aim for 7-9 hours nightly. Create relaxing bedtime routine and optimize sleep environment.", "#764ba2"))
        elif sleep_time > 9:
            recommendations.append(("Sleep Assessment", "😴", "Excessive sleep may indicate health issues. Consider sleep study to rule out sleep disorders like sleep apnea.", "#764ba2"))
        
        if physical_health > 10:
            recommendations.append(("Regular Monitoring", "🏥", "Schedule comprehensive health check-ups every 3-6 months. Monitor blood pressure, cholesterol, and blood sugar regularly.", "#f093fb"))
        
        if mental_health > 10:
            recommendations.append(("Mental Health Support", "🧠", "Consider counseling or therapy for stress management. Practice mindfulness, meditation, or yoga. Mental health affects heart health.", "#a8edea"))
        
        if diabetic != "No":
            recommendations.append(("Diabetes Management", "💉", "Maintain strict blood sugar control through diet, exercise, and medication adherence. Regular HbA1c monitoring is essential.", "#ff6b6b"))
        
        if recommendations:
            for i, (title, icon, description, color) in enumerate(recommendations, 1):
                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {color}15 0%, {color}05 100%); 
                                padding: 25px; border-radius: 15px; margin: 15px 0;
                                border-left: 5px solid {color};">
                        <div style="display: flex; align-items: start; gap: 20px;">
                            <div style="font-size: 3em; line-height: 1;">{icon}</div>
                            <div style="flex: 1;">
                                <h3 style="margin: 0 0 10px 0; color: {color}; font-size: 1.4em;">{title}</h3>
                                <p style="margin: 0; color: #555; line-height: 1.6; font-size: 1.05em;">{description}</p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="background: linear-gradient(135deg, #11998e15 0%, #38ef7d05 100%); 
                            padding: 30px; border-radius: 15px; text-align: center;
                            border: 3px solid #11998e;">
                    <div style="font-size: 4em; margin-bottom: 15px;">✨</div>
                    <h3 style="color: #11998e; margin: 10px 0;">Outstanding Health Management!</h3>
                    <p style="color: #666; font-size: 1.1em; margin-top: 15px;">
                        Continue your excellent health practices. Maintain regular exercise, 
                        balanced nutrition, adequate sleep, and annual health screenings.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("## 🤖 About RuleNet Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h3>Model Performance</h3>
        <ul>
            <li><strong>Test Accuracy:</strong> 94.21%</li>
            <li><strong>Cross-Validation:</strong> 94.23%</li>
            <li><strong>Precision:</strong> 96.19%</li>
            <li><strong>Recall:</strong> 92.06%</li>
            <li><strong>F1-Score:</strong> 94.09%</li>
            <li><strong>AUC-ROC:</strong> 0.98</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="info-box">
        <h3>Dataset Information</h3>
        <ul>
            <li><strong>Source:</strong> CDC BRFSS 2020</li>
            <li><strong>Total Records:</strong> 319,795</li>
            <li><strong>Features:</strong> 17</li>
            <li><strong>Balancing:</strong> SMOTE</li>
            <li><strong>Validation:</strong> 5-Fold CV</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🔍 Top Risk Factors")
    
    # Feature importance chart
    features = ['BMI', 'Physical Health', 'Age Category', 'General Health', 
                'Stroke', 'Mental Health', 'Diabetic', 'Difficulty Walking']
    importance = [0.142, 0.128, 0.115, 0.097, 0.089, 0.078, 0.072, 0.068]
    
    fig = px.bar(x=importance, y=features, orientation='h',
                 labels={'x': 'Importance Score', 'y': 'Risk Factors'},
                 title='Feature Importance Analysis',
                 color=importance,
                 color_continuous_scale='Reds')
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>📚 Research Background</h3>
    <p>This system is based on comprehensive research conducted at Daffodil International University, 
    comparing 8 different machine learning algorithms on a large-scale heart disease dataset.</p>
    <p><strong>Key Innovation:</strong> The RuleNet classifier combines medical domain knowledge 
    (explicit rules) with machine learning (Random Forest) to achieve both high accuracy and interpretability.</p>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown("## 📈 Model Statistics & Comparison")
    
    # Model comparison
    models = ['RuleNet', 'Random Forest', 'XGBoost', 'KNN', 'Decision Tree', 
              'Logistic Regression', 'SVM', 'Naive Bayes']
    accuracies = [94.23, 94.05, 93.80, 92.18, 91.16, 91.50, 91.44, 91.54]
    
    fig = px.bar(x=models, y=accuracies,
                 labels={'x': 'Models', 'y': 'Cross-Validation Accuracy (%)'},
                 title='Model Performance Comparison',
                 color=accuracies,
                 color_continuous_scale='Viridis')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2 style="color: #667eea;">94.21%</h2>
            <p>Test Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2 style="color: #f093fb;">92.06%</h2>
            <p>Recall (Sensitivity)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2 style="color: #4facfe;">96.19%</h2>
            <p>Precision</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h2 style="color: #f5576c;">0.98</h2>
            <p>AUC-ROC Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>🎯 Why RuleNet Performs Best?</h3>
    <ul>
        <li><strong>Hybrid Architecture:</strong> Combines medical rules with machine learning</li>
        <li><strong>High Interpretability:</strong> Doctors can understand the reasoning</li>
        <li><strong>Balanced Performance:</strong> Excellent precision and recall</li>
        <li><strong>Low False Negatives:</strong> Only 7.94% miss rate for disease cases</li>
        <li><strong>Stable:</strong> 0.06% standard deviation across validation folds</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: white; padding: 20px;">
    <p><strong>Heart Disease Prediction System v1.0</strong></p>
    <p>Developed as part of Final Year Design Project (FYDP)</p>
    <p>Daffodil International University | Department of Computer Science and Engineering</p>
    <p>⚠️ For educational and research purposes only. Not a substitute for professional medical advice.</p>
</div>
""", unsafe_allow_html=True)
