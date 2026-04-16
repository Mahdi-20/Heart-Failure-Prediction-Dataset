"""
Heart Failure Prediction Web App - PRODUCTION VERSION
Interactive Streamlit application with patient history tracking, model explainability (SHAP), and deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import shap
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Heart Failure Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
    <style>
    .main-header {
        font-size: 3em;
        color: #065A82;
        text-align: center;
        margin-bottom: 0.5em;
    }
    .metric-box {
        background-color: #EBF4FA;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #028090;
        text-align: center;
    }
    .risk-high {
        color: #C0392B;
        font-weight: bold;
    }
    .risk-low {
        color: #1E7E34;
        font-weight: bold;
    }
    [role="tablist"] {
        background: linear-gradient(135deg, #028090 0%, #00A896 100%);
        padding: 10px;
        border-radius: 10px;
        gap: 10px;
    }
    [role="tab"] {
        font-size: 1000px !important;
        font-weight: 1000 !important;
        padding: 12px 24px !important;
        background-color: rgba(255, 255, 0, 0.2) !important;
        color: white !important;
        border-radius: 8px !important;
        border: 2px solid transparent !important;
        transition: all 0.3s ease;
    }
    [role="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.3) !important;
        transform: translateY(-2px);
    }
    [role="tab"][aria-selected="true"] {
        background-color: white !important;
        color: #028090 !important;
        border: 2px solid #028090 !important;
        font-weight: 800 !important;
    }
    button[kind="secondary"] {
        background: linear-gradient(135deg, #27AE60 0%, #2ECC71 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        padding: 12px 24px !important;
        border: 2px solid #27AE60 !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    button[kind="secondary"]:hover {
        background: linear-gradient(135deg, #1E8449 0%, #27AE60 100%) !important;
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(39, 174, 96, 0.4) !important;
    }
    .sidebar-section {
        background-color: #E8F4F8;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0px;
        border-left: 5px solid #028090;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND PREPROCESSING OBJECTS
# ============================================================================

@st.cache_resource
def load_model():
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, scaler, label_encoders, feature_names

model, scaler, label_encoders, feature_names = load_model()

# ============================================================================
# LOAD OR CREATE PATIENT HISTORY
# ============================================================================

def load_patient_history():
    if os.path.exists('patient_history.json'):
        with open('patient_history.json', 'r') as f:
            return json.load(f)
    return []

def save_patient_history(history):
    with open('patient_history.json', 'w') as f:
        json.dump(history, f, indent=2)

patient_history = load_patient_history()

# ============================================================================
# HEADER
# ============================================================================

st.markdown("<h1 class='main-header'>❤️ Heart Failure Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1em; color: #666;'>Advanced ML Model for Heart Disease Risk Assessment</p>", unsafe_allow_html=True)
st.markdown("---")

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Make Prediction", "📊 Patient History & Trends", "📈 Data Insights", "ℹ️ About Project"])

# ============================================================================
# TAB 1: MAKE PREDICTION
# ============================================================================

# ============================================================================
# SIDEBAR INPUT SECTION
# ============================================================================

st.sidebar.markdown("### 👤 Patient Information")
patient_id = st.sidebar.text_input("Patient ID (e.g., P001)", value="P001", placeholder="Unique ID")
patient_name = st.sidebar.text_input("Patient Name", value="", placeholder="Patient name")

# Demographics section
st.sidebar.markdown("""
<div style='background-color: #E8F4F8; padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #028090;'>
    <h3 style='color: #065A82; margin-top: 0;'>👥 Demographics</h3>
</div>
""", unsafe_allow_html=True)
age = st.sidebar.slider("Age (years)", min_value=25, max_value=85, value=50, step=1)
sex = st.sidebar.radio("Sex", options=["Male", "Female"], horizontal=True)
sex_encoded = 1 if sex == "Male" else 0

# Chest Symptoms section
st.sidebar.markdown("""
<div style='background-color: #FFF3E0; padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #FF9800;'>
    <h3 style='color: #E65100; margin-top: 0;'>💓 Chest Symptoms</h3>
</div>
""", unsafe_allow_html=True)
chest_pain_type = st.sidebar.selectbox(
    "Chest Pain Type",
    options=["Asymptomatic (ASY)", "Atypical Angina (ATA)", "Non-anginal Pain (NAP)", "Typical Angina (TA)"]
)
cp_map = {"Asymptomatic (ASY)": 0, "Atypical Angina (ATA)": 1, "Non-anginal Pain (NAP)": 2, "Typical Angina (TA)": 3}
cp_encoded = cp_map[chest_pain_type]

exercise_angina = st.sidebar.radio("Exercise Induced Angina", options=["No", "Yes"], horizontal=True)
exercise_angina_encoded = 1 if exercise_angina == "Yes" else 0

# Blood Measurements section
st.sidebar.markdown("""
<div style='background-color: #F3E5F5; padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #9C27B0;'>
    <h3 style='color: #6A1B9A; margin-top: 0;'>🩸 Blood Measurements</h3>
</div>
""", unsafe_allow_html=True)
resting_bp = st.sidebar.slider("Resting Blood Pressure (mmHg)", min_value=80, max_value=200, value=120, step=1)
cholesterol = st.sidebar.slider("Serum Cholesterol (mg/dL)", min_value=100, max_value=600, value=200, step=1)
fasting_bs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dL", options=["No", "Yes"], horizontal=True)
fasting_bs_encoded = 1 if fasting_bs == "Yes" else 0

# Exercise & ECG section
st.sidebar.markdown("""
<div style='background-color: #E8F5E9; padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #4CAF50;'>
    <h3 style='color: #1B5E20; margin-top: 0;'>🏃 Exercise & ECG</h3>
</div>
""", unsafe_allow_html=True)
max_hr = st.sidebar.slider("Max Heart Rate Achieved", min_value=60, max_value=220, value=150, step=1)
oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", min_value=-5.0, max_value=7.0, value=0.0, step=0.1)
resting_ecg = st.sidebar.selectbox(
    "Resting ECG",
    options=["Normal", "ST-T Abnormality", "LV Hypertrophy"]
)
ecg_map = {"Normal": 1, "ST-T Abnormality": 2, "LV Hypertrophy": 0}
ecg_encoded = ecg_map[resting_ecg]

st_slope = st.sidebar.selectbox(
    "ST Slope",
    options=["Up", "Flat", "Down"]
)
st_slope_map = {"Up": 2, "Flat": 1, "Down": 0}
st_slope_encoded = st_slope_map[st_slope]

# ============================================================================
# AUTOMATIC PREDICTION (No button needed)
# ============================================================================

# Prepare input data
input_data = pd.DataFrame({
    'Age': [age],
    'Sex': [sex_encoded],
    'ChestPainType': [cp_encoded],
    'RestingBP': [resting_bp],
    'Cholesterol': [cholesterol],
    'FastingBS': [fasting_bs_encoded],
    'RestingECG': [ecg_encoded],
    'MaxHR': [max_hr],
    'ExerciseAngina': [exercise_angina_encoded],
    'Oldpeak': [oldpeak],
    'ST_Slope': [st_slope_encoded]
})

# Define numeric columns (same as training)
numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']

# Scale numeric features
numeric_values = input_data[numeric_cols].values
numeric_scaled = scaler.transform(numeric_values)
input_data[numeric_cols] = numeric_scaled
input_scaled = input_data

# Make prediction
prediction = model.predict(input_scaled)[0]
prediction_proba = model.predict_proba(input_scaled)[0]
risk_percentage = prediction_proba[1] * 100
confidence = prediction_proba[prediction] * 100

# ============================================================================
# TAB 1: MAKE PREDICTION
# ============================================================================

with tab1:
    st.markdown("## 🔮 Heart Failure Risk Prediction")

    # Display top 3 metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class='metric-box'>
            <h3 style='color: #065A82; margin: 0;'>Risk Score</h3>
            <h2 style='color: #C0392B; margin: 0;'>{risk_percentage:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        prediction_text = "⚠️ HIGH RISK" if prediction == 1 else "✅ LOW RISK"
        color = "#C0392B" if prediction == 1 else "#1E7E34"
        st.markdown(f"""
        <div class='metric-box'>
            <h3 style='color: #065A82; margin: 0;'>Prediction</h3>
            <h2 style='color: {color}; margin: 0;'>{prediction_text}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='metric-box'>
            <h3 style='color: #065A82; margin: 0;'>Confidence</h3>
            <h2 style='color: #028090; margin: 0;'>{confidence:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Display results section
    st.markdown("### 📊 Risk Assessment")

    col1, col2 = st.columns(2)

    with col1:
        # Risk gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_percentage,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Heart Failure Risk (%)"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#C0392B" if risk_percentage > 50 else "#1E7E34"},
                'steps': [
                    {'range': [0, 25], 'color': "#E8F5E9"},
                    {'range': [25, 50], 'color': "#C8E6C9"},
                    {'range': [50, 75], 'color': "#FFE0B2"},
                    {'range': [75, 100], 'color': "#FFCDD2"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # SHAP Explainability
        st.markdown("#### Feature Importance (SHAP Analysis)")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_scaled)

            # Handle SHAP values for binary classification
            if isinstance(shap_values, list):
                shap_values_pred = np.abs(shap_values[1][0])  # For disease class, first sample
            else:
                # For single output, take first sample
                if shap_values.ndim > 1:
                    shap_values_pred = np.abs(shap_values[0])
                else:
                    shap_values_pred = np.abs(shap_values)

            # Ensure 1D array and fix length if needed
            shap_values_pred = np.asarray(shap_values_pred).flatten()
            feature_list = list(input_data.columns)

            # Match lengths if there's a mismatch
            if len(shap_values_pred) != len(feature_list):
                # Take the minimum length
                min_len = min(len(shap_values_pred), len(feature_list))
                shap_values_pred = shap_values_pred[:min_len]
                feature_list = feature_list[:min_len]

            # Create simple feature importance visualization
            if len(shap_values_pred) > 0 and len(feature_list) > 0:
                importance_df = pd.DataFrame({
                    'Feature': feature_list,
                    'Impact': shap_values_pred
                }).sort_values('Impact', ascending=False)

                fig_shap = go.Figure(go.Bar(
                    y=importance_df['Feature'],
                    x=importance_df['Impact'],
                    orientation='h',
                    marker=dict(color='#028090')
                ))
                fig_shap.update_layout(
                    title="Feature Importance",
                    xaxis_title="Impact Magnitude",
                    yaxis_title="Feature",
                    height=400
                )
                st.plotly_chart(fig_shap, use_container_width=True)
            else:
                st.info("Feature importance data unavailable")
        except Exception as e:
            st.info("SHAP analysis unavailable for this prediction")

    st.markdown("---")

    # Current Prediction Input Summary
    st.markdown("### Current Prediction Input Summary")
    summary_data = {
        'Age': [age],
        'Sex': [sex],
        'Chest Pain Type': [chest_pain_type],
        'Resting BP (mmHg)': [resting_bp],
        'Cholesterol (mg/dL)': [cholesterol],
        'Fasting BS > 120': [fasting_bs],
        'Resting ECG': [resting_ecg],
        'Max HR': [max_hr],
        'Exercise Angina': [exercise_angina],
        'ST Depression': [oldpeak],
        'ST Slope': [st_slope]
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

    # Save to History button
    if st.button("Save This Prediction to History", type="secondary", use_container_width=True):
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'patient_id': patient_id,
            'patient_name': patient_name,
            'age': age,
            'sex': sex,
            'chest_pain_type': chest_pain_type,
            'resting_bp': resting_bp,
            'cholesterol': cholesterol,
            'fasting_bs': fasting_bs,
            'resting_ecg': resting_ecg,
            'max_hr': max_hr,
            'exercise_angina': exercise_angina,
            'oldpeak': oldpeak,
            'st_slope': st_slope,
            'prediction': int(prediction),
            'risk_percentage': float(risk_percentage),
            'confidence': float(confidence)
        }
        patient_history.append(history_entry)
        save_patient_history(patient_history)
        st.success("✓ Prediction saved to patient history!")

    st.markdown("---")

    # Important disclaimers
    st.markdown("### ⚠️ Important Disclaimers")
    st.warning("""
    This application is for **educational purposes only** and should NOT be used for clinical diagnosis.
    Always consult with qualified healthcare professionals for medical advice and diagnosis.
    The model predictions are estimates based on training data and should not replace professional medical evaluation.
    """)

# ============================================================================
# TAB 2: PATIENT HISTORY & TRENDS
# ============================================================================

with tab2:
    st.markdown("## 📊 Patient History & Risk Trend Analysis")

    if len(patient_history) > 0:
        # Patient selection
        patient_list = list(set([f"{h['patient_id']} - {h['patient_name']}" for h in patient_history]))
        patient_list.sort()
        patient_selection_label = st.selectbox("Select Patient", options=patient_list, key="patient_select")

        # Filter history for selected patient
        selected_patient_id = patient_selection_label.split(' - ')[0]
        patient_data = [h for h in patient_history if h['patient_id'] == selected_patient_id]

        # Convert to DataFrame
        patient_df = pd.DataFrame(patient_data)
        # Handle different timestamp formats flexibly
        patient_df['timestamp'] = pd.to_datetime(patient_df['timestamp'], format='mixed', errors='coerce')
        patient_df = patient_df.dropna(subset=['timestamp'])  # Remove rows with invalid timestamps
        patient_df = patient_df.sort_values('timestamp')

        # Display prediction history table
        st.markdown("### Prediction History Table")
        display_cols = ['timestamp', 'patient_id', 'patient_name', 'age', 'cholesterol', 'resting_bp', 'max_hr', 'prediction', 'risk_percentage', 'confidence']
        available_cols = [col for col in display_cols if col in patient_df.columns]
        st.dataframe(patient_df[available_cols], use_container_width=True)

        # Trend Analysis
        if len(patient_data) >= 2:
            st.markdown("### Risk Trend Analysis")

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=patient_df['timestamp'],
                y=patient_df['risk_percentage'],
                mode='lines+markers',
                name='Risk %',
                line=dict(color='#C0392B', width=3),
                marker=dict(size=10)
            ))

            fig_trend.update_layout(
                title="Heart Failure Risk Over Time",
                xaxis_title="Date/Time",
                yaxis_title="Risk Percentage (%)",
                hovermode='x unified',
                height=400,
                template='plotly_white'
            )
            st.plotly_chart(fig_trend, use_container_width=True)

            # Statistics
            st.markdown("### Patient Statistics")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Average Risk", f"{patient_df['risk_percentage'].mean():.1f}%")

            with col2:
                st.metric("Max Risk", f"{patient_df['risk_percentage'].max():.1f}%")

            with col3:
                st.metric("Min Risk", f"{patient_df['risk_percentage'].min():.1f}%")

        else:
            st.info("Patient has only one prediction. Need at least 2 predictions for trend analysis.")
            st.markdown("**Tip:** Add more predictions for this patient to see trends over time.")

        # Patient Information
        st.markdown("### Patient Information")
        latest_entry = patient_data[-1]
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Age", latest_entry.get('age', 'N/A'))

        with col2:
            st.metric("Sex", latest_entry.get('sex', 'N/A'))

        with col3:
            st.metric("Resting BP", f"{latest_entry.get('resting_bp', 'N/A')} mmHg")

        with col4:
            st.metric("Cholesterol", f"{latest_entry.get('cholesterol', 'N/A')} mg/dL")

    else:
        st.info("No patient history yet. Make some predictions in the 'Make Prediction' tab to see them here!")

# ============================================================================
# TAB 3: DATA INSIGHTS & VISUALIZATIONS
# ============================================================================

with tab3:
    st.markdown("## 📈 Data Insights & Visualizations")

    # Load original data for insights
    df_viz = pd.read_csv('heart.csv')

    st.markdown("### Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patients", len(df_viz))
    with col2:
        st.metric("Heart Failure Cases", (df_viz['HeartDisease'] == 1).sum())
    with col3:
        st.metric("Healthy Cases", (df_viz['HeartDisease'] == 0).sum())
    with col4:
        st.metric("Features", len(df_viz.columns) - 1)

    st.markdown("---")

    # Disease Distribution
    st.markdown("### Disease Distribution")
    disease_counts = df_viz['HeartDisease'].value_counts()
    fig_disease = go.Figure(data=[
        go.Bar(x=['No Heart Failure', 'Heart Failure'],
               y=[disease_counts[0], disease_counts[1]],
               marker=dict(color=['#2ecc71', '#e74c3c']))
    ])
    fig_disease.update_layout(
        title="Heart Failure Distribution",
        xaxis_title="Status",
        yaxis_title="Number of Patients",
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig_disease, use_container_width=True)

    st.markdown("---")

    # Age and Sex Distribution
    st.markdown("### Demographics")
    col1, col2 = st.columns(2)

    with col1:
        fig_age = go.Figure(data=[go.Histogram(x=df_viz['Age'], nbinsx=20, marker=dict(color='steelblue'))])
        fig_age.update_layout(title="Age Distribution", xaxis_title="Age (years)", yaxis_title="Count", height=400)
        st.plotly_chart(fig_age, use_container_width=True)

    with col2:
        sex_counts = df_viz['Sex'].value_counts()
        fig_sex = go.Figure(data=[
            go.Pie(labels=['Male', 'Female'], values=[sex_counts['M'], sex_counts['F']],
                   marker=dict(colors=['#3498db', '#e91e63']))
        ])
        fig_sex.update_layout(title="Gender Distribution", height=400)
        st.plotly_chart(fig_sex, use_container_width=True)

    st.markdown("---")

    # Clinical Measurements Distribution
    st.markdown("### Clinical Measurements")

    # Resting BP Distribution by Disease
    col1, col2 = st.columns(2)

    with col1:
        fig_bp = go.Figure()
        fig_bp.add_trace(go.Histogram(x=df_viz[df_viz['HeartDisease']==0]['RestingBP'],
                                       name='No Disease', opacity=0.7, nbinsx=20, marker=dict(color='green')))
        fig_bp.add_trace(go.Histogram(x=df_viz[df_viz['HeartDisease']==1]['RestingBP'],
                                       name='Heart Failure', opacity=0.7, nbinsx=20, marker=dict(color='red')))
        fig_bp.update_layout(title="Resting Blood Pressure by Disease Status", xaxis_title="Resting BP (mmHg)",
                            yaxis_title="Count", barmode='overlay', height=400)
        st.plotly_chart(fig_bp, use_container_width=True)

    with col2:
        fig_chol = go.Figure()
        fig_chol.add_trace(go.Histogram(x=df_viz[df_viz['HeartDisease']==0]['Cholesterol'],
                                        name='No Disease', opacity=0.7, nbinsx=20, marker=dict(color='green')))
        fig_chol.add_trace(go.Histogram(x=df_viz[df_viz['HeartDisease']==1]['Cholesterol'],
                                        name='Heart Failure', opacity=0.7, nbinsx=20, marker=dict(color='red')))
        fig_chol.update_layout(title="Cholesterol Level by Disease Status", xaxis_title="Cholesterol (mg/dL)",
                              yaxis_title="Count", barmode='overlay', height=400)
        st.plotly_chart(fig_chol, use_container_width=True)

    st.markdown("---")

    # Max Heart Rate
    col1, col2 = st.columns(2)

    with col1:
        fig_hr = go.Figure()
        fig_hr.add_trace(go.Histogram(x=df_viz[df_viz['HeartDisease']==0]['MaxHR'],
                                      name='No Disease', opacity=0.7, nbinsx=20, marker=dict(color='green')))
        fig_hr.add_trace(go.Histogram(x=df_viz[df_viz['HeartDisease']==1]['MaxHR'],
                                      name='Heart Failure', opacity=0.7, nbinsx=20, marker=dict(color='red')))
        fig_hr.update_layout(title="Max Heart Rate by Disease Status", xaxis_title="Max HR (bpm)",
                            yaxis_title="Count", barmode='overlay', height=400)
        st.plotly_chart(fig_hr, use_container_width=True)

    with col2:
        fig_oldpeak = go.Figure()
        fig_oldpeak.add_trace(go.Histogram(x=df_viz[df_viz['HeartDisease']==0]['Oldpeak'],
                                          name='No Disease', opacity=0.7, nbinsx=20, marker=dict(color='green')))
        fig_oldpeak.add_trace(go.Histogram(x=df_viz[df_viz['HeartDisease']==1]['Oldpeak'],
                                          name='Heart Failure', opacity=0.7, nbinsx=20, marker=dict(color='red')))
        fig_oldpeak.update_layout(title="ST Depression by Disease Status", xaxis_title="Oldpeak (ST depression)",
                                 yaxis_title="Count", barmode='overlay', height=400)
        st.plotly_chart(fig_oldpeak, use_container_width=True)

    st.markdown("---")

    # Categorical Features
    st.markdown("### Categorical Features Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        cp_counts = df_viz['ChestPainType'].value_counts()
        fig_cp = go.Figure(data=[go.Bar(x=cp_counts.index, y=cp_counts.values, marker=dict(color='steelblue'))])
        fig_cp.update_layout(title="Chest Pain Types", xaxis_title="Type", yaxis_title="Count", height=350)
        st.plotly_chart(fig_cp, use_container_width=True)

    with col2:
        ecg_counts = df_viz['RestingECG'].value_counts()
        fig_ecg = go.Figure(data=[go.Bar(x=ecg_counts.index, y=ecg_counts.values, marker=dict(color='orange'))])
        fig_ecg.update_layout(title="Resting ECG Results", xaxis_title="Type", yaxis_title="Count", height=350)
        st.plotly_chart(fig_ecg, use_container_width=True)

    with col3:
        slope_counts = df_viz['ST_Slope'].value_counts()
        fig_slope = go.Figure(data=[go.Bar(x=slope_counts.index, y=slope_counts.values, marker=dict(color='green'))])
        fig_slope.update_layout(title="ST Slope Distribution", xaxis_title="Slope", yaxis_title="Count", height=350)
        st.plotly_chart(fig_slope, use_container_width=True)

# ============================================================================
# TAB 4: ABOUT PROJECT
# ============================================================================

with tab4:
    st.markdown("<h2 style='text-align: center; color: #065A82;'>📚 About This Project</h2>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("## 📋 Project Overview")
    st.markdown("""
    **Objective:** Predict heart failure risk using supervised machine learning classification with clinical features.

    **Machine Learning Problem:**
    - **Type:** Binary Classification (Heart Failure Present/Absent)
    - **Dataset:** Heart Failure Prediction Dataset (918 patients, 11 clinical features)
    - **Features:** Age, Sex, Chest Pain Type, Resting BP, Cholesterol, Fasting BS, Resting ECG, Max HR, Exercise Angina, ST Depression, ST Slope

    **Machine Learning Approaches Evaluated:**
    - **SVM (Linear & RBF)** - Support Vector Machine classifiers
    - **Random Forest** - Ensemble decision tree method
    - **Logistic Regression** - Baseline statistical model
    - **Linear Discriminant Analysis (LDA)** - Dimensionality reduction classifier

    **Model Evaluation & Selection:**
    - **Methodology:** Stratified 5-fold Cross-Validation
    - **Metrics:** Accuracy, ROC-AUC, Recall, Precision, F1-Score
    - **Key Finding:** Random Forest achieved 92.39% ROC-AUC with 86.41% test accuracy
    - **Data Quality:** 100% complete, no missing values, no duplicates, well-balanced classes

    **Model Deployment & Productionization:**
    - **Framework:** Streamlit for interactive web application
    - **Deployment Platform:** Streamlit Cloud for live, shareable predictions
    - **Real-time Inference:** Instant risk predictions with clinical feature inputs
    - **Model Interpretability:** SHAP explainability layer shows which features influence predictions
    - **Data Persistence:** Patient history tracking with trend analysis capabilities

    **Deep Learning Consideration:**
    - Foundation laid for future neural network implementations
    - Current ML models provide strong baseline for comparison with deep learning approaches

    **Key Insights:**
    - Random Forest shows excellent generalization with clean data
    - Clinical features provide strong discriminative power for disease prediction
    - Model explainability (SHAP) reveals which features most influence predictions
    - Successful transition from research model to production-ready web application
    """)

    st.markdown("---")

    st.markdown("## 🎓 Course Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("""
        ### Course
        **Advanced ML & Data Analytics**
        """)

    with col2:
        st.info("""
        ### Institution
        **Nexa-land**
        """)

    with col3:
        st.info("""
        ### Professor
        **[Prof. Hamed Mamani](https://foster.uw.edu/faculty-research/directory/hamed-mamani/)**

        University of Washington
        """)

    st.markdown("---")

    st.markdown("## 👨‍💻 Author")
    st.markdown("""
    **Mahdi Bakhtiari** (@mahdi-20)

    GitHub: [github.com/mahdi-20](https://github.com/mahdi-20)
    """)

    st.markdown("---")

    st.markdown("## 🛠️ Technologies & Libraries")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Python ML Stack
        - **NumPy** - Numerical computing & arrays
        - **Pandas** - Data manipulation & analysis
        - **Scikit-Learn** - ML algorithms & preprocessing
        - **Matplotlib/Seaborn** - Data visualization
        - **Plotly** - Interactive visualizations

        ### ML Model Development
        - **SVM** - Support Vector Machine
        - **Random Forest** - Ensemble classification (Selected)
        - **Logistic Regression** - Baseline model
        - **Model Training & Evaluation**
        - **Cross-Validation** - 5-fold CV
        """)

    with col2:
        st.markdown("""
        ### ML Techniques & Concepts
        - **Preprocessing** - Feature scaling & normalization
        - **Feature Engineering** - Clinical feature selection
        - **Exploratory Data Analysis (EDA)** - Statistical analysis
        - **Classification Models** - Binary disease prediction
        - **Regression Concepts** - Model relationships
        - **Evaluation Metrics** - Accuracy, ROC-AUC, Recall, Precision

        ### Model Interpretability & Deployment
        - **SHAP** - Feature importance analysis
        - **Streamlit** - Interactive web framework
        - **Model Explainability** - Understanding predictions
        - **Interactive Visualizations** - Real-time insights
        """)

    st.markdown("---")

    st.markdown("## 📊 Dataset")
    st.markdown("""
    - **Source:** [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
    - **Samples:** 918 patients
    - **Features:** 11 clinical measurements
    - **Target:** Binary classification (presence/absence of heart failure)
    - **Data Quality:** 100% complete, no missing values, no duplicates
    - **Class Balance:** 410 (No Disease) vs 508 (Disease) = 0.81 ratio
    """)

    st.markdown("---")

    st.markdown("## 📈 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Algorithm", "Random Forest", "100 trees")

    with col2:
        st.metric("Test Accuracy", "86.41%", "✓ Excellent")

    with col3:
        st.metric("ROC-AUC", "0.9239", "✓ High")

    with col4:
        st.metric("Recall", "87.25%", "Disease Detection")

    st.markdown("---")

    st.markdown("## ✨ Application Features")
    st.markdown("""
    ✅ **Interactive Web Application Deployment** - Real-time predictions with Streamlit framework

    ✅ **Data Visualization and Analysis** - Interactive charts, gauges, and trend analysis

    ✅ **Model Explainability (SHAP)** - Understand which features most influence predictions

    ✅ **Patient History Tracking and Trend Analysis** - Save, compare, and analyze multiple predictions over time

    **Additional Features:**
    - 🔮 Real-time Predictions with clinical feature inputs
    - 👥 Patient ID Support for distinguishing similar patients
    - 🎯 Confidence Scores for each prediction
    - 📱 Responsive Design for desktop and mobile devices
    """)

    st.markdown("---")

    st.markdown("## 📂 Source Code")
    st.markdown("""
    Full project source code available on GitHub:

    [![GitHub](https://img.shields.io/badge/GitHub-View%20Repository-black?logo=github&style=for-the-badge)](https://github.com/mahdi-20/heart-disease-predictor)

    **Repository includes:**
    - `app.py` - Standard version
    - `app_with_retraining.py` - Advanced version with model retraining
    - `heart.csv` - Heart Failure dataset
    - `ml_pipeline.py` - Complete ML pipeline
    - `requirements.txt` - Dependencies
    - Complete documentation
    """)

    st.markdown("---")

    st.warning("""
    ### ⚠️ Important Disclaimer

    This application is for **educational purposes only** and should not be used for clinical diagnosis.
    Always consult with qualified healthcare professionals for medical advice and diagnosis.

    The model is trained on historical data and predictions are estimates only.
    """)

    st.markdown("---")

    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em; margin-top: 30px;'>
        <p>❤️ Built with Python, Machine Learning, and Streamlit</p>
        <p>Advanced ML & Data Analytics Course | Nexa-land | Prof. Hamed Mamani, University of Washington</p>
    </div>
    """, unsafe_allow_html=True)
