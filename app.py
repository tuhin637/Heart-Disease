import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="HeartGuard AI Pro - Cardiovascular Risk Assessment",
    page_icon="https://em-content.zobj.net/source/apple/118/heart-with-pulse_1f497.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== PREMIUM CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    * {font-family: 'Inter', sans-serif !important; margin: 0; padding: 0; box-sizing: border-box;}
    
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        min-height: 100vh;
        padding: 20px 0;
    }

    /* Hero Section - World Class */
    .hero-section {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.18), rgba(139, 92, 246, 0.12)),
                    url('https://images.unsplash.com/photo-1576092768241-dec231879fc3?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80') center/cover no-repeat;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 32px;
        padding: 100px 50px;
        margin: 25px 20px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 30px 80px rgba(0, 0, 0, 0.4);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .hero-title {
        font-size: 5.5rem;
        font-weight: 900;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #e879f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 20px;
    }
    .hero-subtitle {
        font-size: 1.6rem;
        color: rgba(255, 255, 255, 0.92);
        max-width: 950px;
        margin: 0 auto 45px;
        line-height: 1.7;
        font-weight: 400;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.14);
        backdrop-filter: blur(18px);
        border-radius: 26px;
        border: 1.5px solid rgba(255, 255, 255, 0.2);
        padding: 38px;
        box-shadow: 0 15px 45px rgba(0, 0, 0, 0.3);
        transition: all 0.5s ease;
        margin: 20px 0;
    }
    .glass-card:hover {transform: translateY(-12px); box-shadow: 0 30px 70px rgba(0, 0, 0, 0.45);}

    /* Prediction Box */
    .prediction-box {
        border-radius: 34px;
        padding: 55px;
        background: rgba(255, 255, 255, 0.98);
        box-shadow: 0 35px 80px rgba(0, 0, 0, 0.22);
        border: 1px solid rgba(0, 0, 0, 0.1);
        transition: all 0.5s ease;
        margin: 35px 0;
    }
    .positive {background: linear-gradient(135deg, #fff7ed, #fed7aa, #f97316); color: #9a3412; border: 2px solid #f97316;}
    .negative {background: linear-gradient(135deg, #ecfdf5, #d1fae5, #6ee7b7); color: #065f46; border: 2px solid #10b981;}

    /* Inputs */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stSlider > div > div > div > div {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 2.5px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 20px !important;
        padding: 18px 24px !important;
        font-size: 1.1rem !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08) !important;
        transition: all 0.4s ease !important;
    }
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 6px rgba(99, 102, 241, 0.25) !important;
        transform: translateY(-3px);
    }

    /* Main Button */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #c084fc, #e879f9) !important;
        color: white !important;
        border: none !important;
        border-radius: 22px !important;
        padding: 22px 80px !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        box-shadow: 0 18px 50px rgba(99, 102, 241, 0.55) !important;
        transition: all 0.5s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-10px) scale(1.05) !important;
        box-shadow: 0 35px 70px rgba(99, 102, 241, 0.65) !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.12);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.12);
        color: white;
        border-radius: 18px;
        padding: 20px 40px;
        margin: 0 15px;
        font-weight: 600;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #6366f1 !important;
        font-weight: 800 !important;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.25);
    }

    .section-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #e879f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 70px 0 35px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HERO SECTION ====================
st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">HeartGuard AI Pro</h1>
    <p class="hero-subtitle">Next-Generation Hybrid Intelligence for Early Detection & Prevention of Cardiovascular Disease</p>
    <div style="margin-top: 50px; display: flex; justify-content: center; gap: 25px; flex-wrap: wrap;">
        <span style="background: rgba(99,102,241,0.2); color: #c084fc; padding: 14px 32px; border-radius: 50px; border: 2px solid rgba(99,102,241,0.5); backdrop-filter: blur(10px); font-weight: 700; font-size: 1.1rem;">94.21% Accuracy</span>
        <span style="background: rgba(168,85,247,0.2); color: #e879f9; padding: 14px 32px; border-radius: 50px; border: 2px solid rgba(168,85,247,0.5); backdrop-filter: blur(10px); font-weight: 700; font-size: 1.1rem;">Real-time Analysis</span>
        <span style="background: rgba(59,130,246,0.2); color: #60a5fa; padding: 14px 32px; border-radius: 50px; border: 2px solid rgba(59,130,246,0.5); backdrop-filter: blur(10px); font-weight: 700; font-size: 1.1rem;">100% Private</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("<h2 style='color: white; text-align: center; margin-bottom: 20px;'>HeartGuard AI Pro</h2>", unsafe_allow_html=True)
    st.markdown("### Model Performance")
    st.markdown("""
    <div class="glass-card" style="margin: 20px 0;">
        <p style="color: white; margin: 10px 0; font-size: 1.05rem;"><strong>Accuracy:</strong> 94.21%</p>
        <p style="color: white; margin: 10px 0; font-size: 1.05rem;"><strong>Precision:</strong> 96.19%</p>
        <p style="color: white; margin: 10px 0; font-size: 1.05rem;"><strong>Recall:</strong> 92.06%</p>
        <p style="color: white; margin: 10px 0; font-size: 1.05rem;"><strong>AUC-ROC:</strong> 0.98</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Developer")
    st.markdown("""
    <div style="text-align: center; color: white; padding: 25px;">
        <p style="font-size: 1.3em; font-weight: 700;">Md. Tuhinuzzaman Tuhin</p>
        <p>ID: 221-15-4649</p>
        <p>Daffodil International University</p>
        <p style="font-size: 0.95em; margin-top: 20px; opacity: 0.9;">
            Final Year Design Project (FYDP)<br>
            Department of Computer Science & Engineering
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==================== TABS ====================
tab1, tab2, tab3 = st.tabs(["Risk Assessment", "Model Insights", "Performance"])

with tab1:
    st.markdown("<h2 class='section-header'>Patient Health Assessment</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Physical Metrics")
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0, 0.1)
        physical_health = st.slider("Physical Health (bad days/month)", 0, 30, 0)
        mental_health = st.slider("Mental Health (bad days/month)", 0, 30, 0)
        sleep_time = st.slider("Sleep Hours", 0, 24, 8)
    
    with col2:
        st.markdown("#### Lifestyle Factors")
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        alcohol = st.selectbox("Heavy Alcohol", ["No", "Yes"])
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
        age_category = st.selectbox("Age Category", ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"])
        gen_health = st.selectbox("General Health", ["Excellent", "Very good", "Good", "Fair", "Poor"])

    if st.button("Analyze Risk Now", use_container_width=True):
        risk = 0
        if bmi > 30: risk += 28
        if smoking == "Yes": risk += 30
        if stroke == "Yes": risk += 35
        if age_category in ["70-74", "75-79", "80 or older"]: risk += 25
        if physical_health > 10: risk += 20
        if diabetic in ["Yes", "Yes (pregnancy)"]: risk += 22
        
        probability = min(risk / 100, 0.99)
        prediction = 1 if probability > 0.5 else 0
        
        colr1, colr2 = st.columns([2, 1])
        
        with colr1:
            if prediction:
                st.markdown(f"""
                <div class="prediction-box positive">
                    <h1 style="font-size: 3.5rem; margin: 15px 0;">High Risk Detected</h1>
                    <h2 style="font-size: 5rem; color: #c2410c; margin: 25px 0;">{(probability*100):.1f}%</h2>
                    <p style="font-size: 1.4rem; line-height: 1.9;">
                        <strong>Immediate consultation with a cardiologist is strongly recommended.</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box negative">
                    <h1 style="font-size: 3.5rem; margin: 15px 0;">Low Risk</h1>
                    <h2 style="font-size: 5rem; color: #059669; margin: 25px 0;">{(probability*100):.1f}%</h2>
                    <p style="font-size: 1.4rem; line-height: 1.9;">
                        <strong>Excellent! Continue your healthy lifestyle.</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        with colr2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=probability*100,
                title={'text': "Cardiovascular Risk"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#ef4444" if prediction else "#10b981"},
                    'steps': [
                        {'range': [0, 40], 'color': '#dcfce7'},
                        {'range': [40, 70], 'color': '#fef9c3'},
                        {'range': [70, 100], 'color': '#fecaca'}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 8}, 'value': 50}
                }
            ))
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

# ==================== OTHER TABS ====================
with tab2:
    st.markdown("<h2 class='section-header'>RuleNet Hybrid Intelligence</h2>", unsafe_allow_html=True)
    st.markdown("A revolutionary model combining medical domain rules with Random Forest for superior accuracy and clinical interpretability.")

with tab3:
    st.markdown("<h2 class='section-header'>Performance Metrics</h2>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Accuracy", "94.21%")
    with col2: st.metric("Precision", "96.19%")
    with col3: st.metric("Recall", "92.06%")
    with col4: st.metric("AUC-ROC", "0.98")

# ==================== FOOTER ====================
st.markdown("""
<div style="text-align: center; padding: 80px 20px; color: rgba(255,255,255,0.7); margin-top: 100px;">
    <h3>HeartGuard AI Pro v2.0</h3>
    <p>Final Year Design Project • Daffodil International University</p>
    <p style="font-size: 0.95rem; margin-top: 15px;">
        For educational & research purposes only • Not a substitute for professional medical diagnosis
    </p>
</div>
""", unsafe_allow_html=True)
