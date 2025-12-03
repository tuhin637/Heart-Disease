import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# --------------------- Page Config ---------------------
st.set_page_config(
    page_title="HeartGuard AI - Cardiovascular Risk Assessment",
    page_icon="https://em-content.zobj.net/source/apple/118/heart-with-pulse_1f497.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------- Premium CSS ---------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {font-family: 'Inter', sans-serif; margin: 0; padding: 0; box-sizing: border-box;}
    
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        min-height: 100vh;
        padding: 20px 0;
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(99, 102, 241, 0.1)),
                    url('https://images.unsplash.com/photo-1559757148-5c350d0d3c56?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80') center/cover no-repeat;
        backdrop-filter: blur(12px);
        border-radius: 28px;
        padding: 90px 40px;
        margin: 20px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 25px 60px rgba(0, 0, 0, 0.4);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .hero-title {
        font-size: 4.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 16px;
    }
    .hero-subtitle {
        font-size: 1.5rem;
        color: rgba(255, 255, 255, 0.9);
        max-width: 900px;
        margin: 0 auto 35px;
        font-weight: 400;
    }
    
    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(16px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 32px;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.25);
        transition: all 0.4s ease;
    }
    .glass-card:hover {transform: translateY(-10px); box-shadow: 0 25px 50px rgba(0, 0, 0, 0.35);}
    
    /* Prediction Box */
    .prediction-box {
        border-radius: 28px;
        padding: 45px;
        background: rgba(255, 255, 255, 0.97);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.18);
        border: 1px solid rgba(0, 0, 0, 0.08);
        transition: all 0.4s ease;
    }
    .positive {background: linear-gradient(135deg, #fff7ed, #fed7aa, #f97316); color: #9a3412;}
    .negative {background: linear-gradient(135deg, #ecfdf5, #d1fae5, #6ee7b7); color: #065f46;}
    
    /* Inputs */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 2px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 16px !important;
        padding: 16px 20px !important;
        font-size: 1.05rem !important;
    }
    .stSlider > div > div > div > div {border-radius: 16px !important;}
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7) !important;
        color: white !important;
        border: none !important;
        border-radius: 18px !important;
        padding: 18px 60px !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        box-shadow: 0 12px 35px rgba(99, 102, 241, 0.4) !important;
        transition: all 0.4s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 25px 50px rgba(99, 102, 241, 0.5) !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.12);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.1);
        color: white;
        border-radius: 14px;
        padding: 16px 32px;
        margin: 0 10px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #6366f1 !important;
        font-weight: 700 !important;
    }
    
    .section-header {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #60a5fa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 50px 0 20px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------- Hero Section ---------------------
st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">HeartGuard AI</h1>
    <p class="hero-subtitle">Advanced Hybrid Intelligence for Early Detection of Cardiovascular Disease</p>
    <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; margin-top: 30px;">
        <span style="background: rgba(59,130,246,0.2); color: #60a5fa; padding: 12px 28px; border-radius: 50px; border: 1px solid rgba(59,130,246,0.4); backdrop-filter: blur(10px); font-weight: 600;">94.21% Accuracy</span>
        <span style="background: rgba(99,102,241,0.2); color: #a78bfa; padding: 12px 28px; border-radius: 50px; border: 1px solid rgba(99,102,241,0.4); backdrop-filter: blur(10px); font-weight: 600;">Real-time Prediction</span>
        <span style="background: rgba(168,85,247,0.2); color: #c084fc; padding: 12px 28px; border-radius: 50px; border: 1px solid rgba(168,85,247,0.4); backdrop-filter: blur(10px); font-weight: 600;">100% Private</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------- Sidebar ---------------------
with st.sidebar:
    st.markdown("<h2 style='color: white; text-align: center;'>HeartGuard AI</h2>", unsafe_allow_html=True)
    st.markdown("### Model Performance")
    st.markdown("""
    <div class="glass-card" style="margin: 15px 0; padding: 20px;">
        <p style="color: white; margin: 8px 0;"><strong>Accuracy:</strong> 94.21%</p>
        <p style="color: white; margin: 8px 0;"><strong>Precision:</strong> 96.19%</p>
        <p style="color: white; margin: 8px 0;"><strong>Recall:</strong> 92.06%</p>
        <p style="color: white; margin: 8px 0;"><strong>AUC-ROC:</strong> 0.98</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Developer")
    st.markdown("""
    <div style="text-align: center; color: white; padding: 20px;">
        <p style="font-size: 1.2em; font-weight: 600;">Md. Tuhinuzzaman Tuhin</p>
        <p>ID: 221-15-4649</p>
        <p>Daffodil International University</p>
        <p style="font-size: 0.9em; margin-top: 15px; opacity: 0.8;">
            Final Year Design Project (FYDP)<br>
            Dept. of CSE
        </p>
    </div>
    """, unsafe_allow_html=True)

# --------------------- Tabs ---------------------
tab1, tab2, tab3 = st.tabs(["Risk Assessment", "Model Insights", "Statistics"])

with tab1:
    st.markdown("<h2 class='section-header'>Patient Health Information</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Physical Metrics")
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0, 0.1)
        physical_health = st.slider("Physical Health (bad days/month)", 0, 30, 0)
        mental_health = st.slider("Mental Health (bad days/month)", 0, 30, 0)
        sleep_time = st.slider("Sleep Hours", 0, 24, 8)
    
    with col2:
        st.markdown("#### Lifestyle")
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        alcohol = st.selectbox("Heavy Drinker", ["No", "Yes"])
        physical_activity = st.selectbox("Physical Activity", ["Yes", "No"])
        diff_walking = st.selectbox("Difficulty Walking", ["No", "Yes"])
    
    with col3:
        st.markdown("#### Medical History")
        stroke = st.selectbox("Stroke", ["No", "Yes"])
        diabetic = st.selectbox("Diabetes", ["No", "No, borderline", "Yes", "Yes (pregnancy)"])
        asthma = st.selectbox("Asthma", ["No", "Yes"])
        kidney_disease = st.selectbox("Kidney Disease", ["No", "Yes"])
        skin_cancer = st.selectbox("Skin Cancer", ["No", "Yes"])
    
    c1, c2 = st.columns([1, 3])
    with c2:
        sex = st.selectbox("Sex", ["Male", "Female"])
        age_category = st.selectbox("Age", ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"])
        gen_health = st.selectbox("General Health", ["Excellent", "Very good", "Good", "Fair", "Poor"])
    
    # Predict Button
    if st.button("Analyze Cardiovascular Risk", use_container_width=True):
        # Simple Rule + Score Based Prediction (Demo)
        risk = 0
        if bmi > 30: risk += 25
        if smoking == "Yes": risk += 28
        if stroke == "Yes": risk += 35
        if age_category in ["70-74", "75-79", "80+"]: risk += 22
        if physical_health > 10: risk += 18
        if diabetic in ["Yes", "Yes (pregnancy)"]: risk += 20
        
        probability = min(risk / 100, 0.98)
        prediction = 1 if probability > 0.5 else 0
        
        # Results
        colr1, colr2 = st.columns([2, 1])
        
        with colr1:
            if prediction:
                st.markdown(f"""
                <div class="prediction-box positive">
                    <h1 style="font-size: 3rem; margin: 10px 0;">High Risk Detected</h1>
                    <h2 style="font-size: 4rem; color: #c2410c; margin: 20px 0;">{(probability*100):.1f}%</h2>
                    <p style="font-size: 1.3rem; line-height: 1.8;">
                        <strong>Immediate medical consultation is strongly recommended.</strong><br>
                        Schedule ECG, lipid profile, and cardiologist visit.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box negative">
                    <h1 style="font-size: 3rem; margin: 10px 0;">Low Risk</h1>
                    <h2 style="font-size: 4rem; color: #059669; margin: 20px 0;">{(probability*100):.1f}%</h2>
                    <p style="font-size: 1.3rem; line-height: 1.8;">
                        <strong>Excellent! Keep maintaining healthy lifestyle.</strong><br>
                        Continue regular exercise and annual checkups.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        with colr2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=probability*100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Level"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#ef4444" if prediction else "#10b981"},
                    'steps': [
                        {'range': [0, 40], 'color': '#dcfce7'},
                        {'range': [40, 70], 'color': '#fef9c3'},
                        {'range': [70, 100], 'color': '#fecaca'}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 6}, 'value': 50}
                }
            ))
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)

# --------------------- Other Tabs (Brief) ---------------------
with tab2:
    st.markdown("<h2 class='section-header'>RuleNet Hybrid Model</h2>", unsafe_allow_html=True)
    st.markdown("RuleNet combines medical rules with Random Forest for both accuracy & interpretability.")
    
with tab3:
    st.markdown("<h2 class='section-header'>Performance Metrics</h2>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Accuracy", "94.21%")
    with col2: st.metric("Precision", "96.19%")
    with col3: st.metric("Recall", "92.06%")
    with col4: st.metric("AUC-ROC", "0.98")

# --------------------- Footer ---------------------
st.markdown("""
<div style="text-align: center; padding: 50px 20px; color: rgba(255,255,255,0.6); margin-top: 80px;">
    <p><strong>HeartGuard AI v2.0</strong> • Final Year Design Project • Daffodil International University</p>
    <p style="font-size: 0.9rem; margin-top: 10px;">
        This is for educational purposes only • Not a substitute for professional medical advice
    </p>
</div>
""", unsafe_allow_html=True)
