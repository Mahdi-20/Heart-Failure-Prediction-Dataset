"""
Heart Failure Prediction Web App - OPTIMIZED VERSION
Streamlit application for Heart Disease Risk Assessment using GridSearchCV-tuned Random Forest
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import warnings
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

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
        font-size: 1.2em;
    }
    .risk-low {
        color: #1E7E34;
        font-weight: bold;
        font-size: 1.2em;
    }
    .info-box {
        background-color: #E8F4F8;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #028090;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS AND PREPROCESSING OBJECTS
# ============================================================================

def load_models():
    """Load trained models and preprocessing objects from pickle files"""
    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        return model, scaler, label_encoders, feature_names
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# Load models (NOT cached - reload fresh each time)
model, scaler, label_encoders, feature_names = load_models()

# ============================================================================
# MODEL INFORMATION
# ============================================================================

MODEL_INFO = {
    'name': 'Random Forest (GridSearchCV Optimized)',
    'test_accuracy': 0.8750,
    'test_roc_auc': 0.9250,
    'test_recall': 0.9118,
    'test_precision': 0.8692,
    'test_f1': 0.8900,
    'cv_roc_auc': 0.9321,
    'hyperparameters': {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 10,
        'min_samples_leaf': 2,
        'max_features': 'sqrt'
    }
}

# ============================================================================
# PATIENT HISTORY MANAGEMENT
# ============================================================================

def load_patient_history():
    """Load patient history from JSON file"""
    if os.path.exists('patient_history.json'):
        with open('patient_history.json', 'r') as f:
            return json.load(f)
    return []

def save_patient_history(history):
    """Save patient history to JSON file"""
    with open('patient_history.json', 'w') as f:
        json.dump(history, f, indent=2)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def prepare_features(age, sex, chest_pain, resting_bp, cholesterol, fasting_bs,
                     resting_ecg, max_hr, exercise_angina, oldpeak, st_slope):
    """Prepare input features for model prediction"""

    # Create a dictionary with feature values
    input_data = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': 1 if fasting_bs == 'Yes' else 0,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': 1 if exercise_angina == 'Yes' else 0,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }

    # Create DataFrame
    df = pd.DataFrame([input_data])

    # Encode categorical variables
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']

    for col in categorical_cols:
        if col in label_encoders:
            df[col] = label_encoders[col].transform([df[col].values[0]])

    # Scale numeric features
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Ensure correct feature order
    df = df[feature_names]

    return df

def make_prediction(input_features):
    """Make prediction using the trained model"""
    try:
        prediction = model.predict(input_features)[0]
        probability = model.predict_proba(input_features)[0]

        risk_score = probability[1]  # Probability of heart disease
        risk_level = "HIGH RISK" if risk_score > 0.5 else "LOW RISK"

        return prediction, risk_score, risk_level
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None, None

# ============================================================================
# PAGE HEADER
# ============================================================================

st.markdown("<h1 class='main-header'>❤️ Heart Failure Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1em; color: #666;'>Advanced ML Model for Heart Disease Risk Assessment</p>", unsafe_allow_html=True)
st.markdown("---")

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Make Prediction", "📊 Patient History", "📈 Model Performance", "ℹ️ About"])

# ============================================================================
# TAB 1: MAKE PREDICTION
# ============================================================================

with tab1:
    st.markdown("## 🔮 Heart Failure Risk Prediction")

    # Sidebar input section
    st.sidebar.markdown("### 👤 Patient Information")
    patient_id = st.sidebar.text_input("Patient ID", value="P001", placeholder="e.g., P001")
    patient_name = st.sidebar.text_input("Patient Name", value="", placeholder="Patient name (optional)")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Clinical Parameters")

    # Basic info
    age = st.sidebar.slider("Age (years)", min_value=25, max_value=85, value=50, step=1)
    sex = st.sidebar.radio("Sex", options=["Male", "Female"], horizontal=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Chest & Heart**")

    chest_pain_type = st.sidebar.selectbox(
        "Chest Pain Type",
        options=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
    )

    exercise_angina = st.sidebar.radio("Exercise Induced Angina", options=["No", "Yes"], horizontal=True)
    st_slope = st.sidebar.selectbox(
        "ST Slope (Exercise)",
        options=["Upsloping", "Flat", "Downsloping"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Blood Pressure & Cholesterol**")

    resting_bp = st.sidebar.slider("Resting BP (mmHg)", min_value=80, max_value=200, value=120, step=1)
    cholesterol = st.sidebar.slider("Cholesterol (mg/dL)", min_value=100, max_value=600, value=200, step=1)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Heart Rate & Other**")

    max_hr = st.sidebar.slider("Max Heart Rate Achieved", min_value=60, max_value=220, value=150, step=1)
    oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", min_value=-5.0, max_value=7.0, value=0.0, step=0.1)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Lab Results**")

    fasting_bs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dL", options=["No", "Yes"], horizontal=True)
    resting_ecg = st.sidebar.selectbox(
        "Resting ECG",
        options=["Normal", "ST-T Abnormality", "LV Hypertrophy"]
    )

    # Make prediction
    input_features = prepare_features(
        age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs,
        resting_ecg, max_hr, exercise_angina, oldpeak, st_slope
    )

    prediction, risk_score, risk_level = make_prediction(input_features)

    # Display results
    if prediction is not None:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class='metric-box'>
                <h3>Risk Level</h3>
                <p class='{'risk-high' if risk_level == 'HIGH RISK' else 'risk-low'}'>{risk_level}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class='metric-box'>
                <h3>Risk Score</h3>
                <p style='font-size: 1.8em; color: #028090; font-weight: bold;'>{risk_score:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class='metric-box'>
                <h3>Prediction</h3>
                <p style='font-size: 1.8em; color: #028090; font-weight: bold;'>{'Positive' if prediction == 1 else 'Negative'}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Risk assessment gauge
        st.markdown("### 📊 Risk Assessment Gauge")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_score * 100,
            title={'text': "Heart Disease Risk (%)"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#C0392B" if risk_score > 0.5 else "#1E7E34"},
                'steps': [
                    {'range': [0, 25], 'color': "#90EE90"},
                    {'range': [25, 50], 'color': "#FFD700"},
                    {'range': [50, 75], 'color': "#FFA500"},
                    {'range': [75, 100], 'color': "#FF6347"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))

        fig_gauge.update_layout(
            font={'size': 12},
            height=400,
            margin=dict(l=20, r=20, t=70, b=20)
        )

        st.plotly_chart(fig_gauge, use_container_width=True)

        # Save to history
        if st.button("💾 Save to Patient History"):
            history = load_patient_history()
            record = {
                'timestamp': datetime.now().isoformat(),
                'patient_id': patient_id,
                'patient_name': patient_name,
                'age': age,
                'sex': sex,
                'risk_score': float(risk_score),
                'risk_level': risk_level,
                'prediction': int(prediction)
            }
            history.append(record)
            save_patient_history(history)
            st.success("✓ Prediction saved to patient history!")

# ============================================================================
# TAB 2: PATIENT HISTORY
# ============================================================================

with tab2:
    st.markdown("## 📊 Patient History & Trends")

    history = load_patient_history()

    if history:
        # Display history table
        st.markdown("### 📋 Patient Records")

        history_df = pd.DataFrame(history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        history_df['risk_score'] = history_df['risk_score'].apply(lambda x: f"{x:.1%}")

        st.dataframe(
            history_df[['timestamp', 'patient_id', 'patient_name', 'age', 'risk_level', 'risk_score']],
            use_container_width=True
        )

        # Risk trend chart
        st.markdown("### 📈 Risk Score Trends")

        if len(history) > 1:
            history_plot = pd.DataFrame(history)
            history_plot['timestamp'] = pd.to_datetime(history_plot['timestamp'])

            fig = px.line(
                history_plot,
                x='timestamp',
                y='risk_score',
                color='patient_id',
                markers=True,
                title='Patient Risk Score Over Time'
            )

            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Risk Score',
                hovermode='x unified',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 records to show trend chart.")

        # Clear history button
        if st.button("🗑️ Clear All History"):
            save_patient_history([])
            st.success("History cleared!")
            st.rerun()
    else:
        st.info("No patient history yet. Make a prediction and save it to see records here.")

# ============================================================================
# TAB 3: MODEL PERFORMANCE
# ============================================================================

with tab3:
    st.markdown("## 📈 Model Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class='metric-box'>
            <h4>Test Accuracy</h4>
            <p style='font-size: 1.6em; color: #028090; font-weight: bold;'>{MODEL_INFO['test_accuracy']:.2%}</p>
            <p style='font-size: 0.9em; color: #666;'>Excellent</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='metric-box'>
            <h4>ROC-AUC Score</h4>
            <p style='font-size: 1.6em; color: #028090; font-weight: bold;'>{MODEL_INFO['test_roc_auc']:.4f}</p>
            <p style='font-size: 0.9em; color: #666;'>Very High</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='metric-box'>
            <h4>Recall (Sensitivity)</h4>
            <p style='font-size: 1.6em; color: #028090; font-weight: bold;'>{MODEL_INFO['test_recall']:.2%}</p>
            <p style='font-size: 0.9em; color: #666;'>Disease Detection</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class='metric-box'>
            <h4>Precision</h4>
            <p style='font-size: 1.6em; color: #028090; font-weight: bold;'>{MODEL_INFO['test_precision']:.2%}</p>
            <p style='font-size: 0.9em; color: #666;'>Accuracy</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Model details
    st.markdown("### 🤖 Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class='info-box'>
            <h4>Algorithm</h4>
            <p><strong>{MODEL_INFO['name']}</strong></p>
            <p>GridSearchCV optimized with StratifiedKFold (5-fold) cross-validation</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='info-box'>
            <h4>Cross-Validation Performance</h4>
            <p>Best CV ROC-AUC: <strong>{MODEL_INFO['cv_roc_auc']:.4f}</strong></p>
            <p>From 216 parameter combinations × 5 folds = 1,080 models trained</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='info-box'>
            <h4>Optimal Hyperparameters</h4>
        """, unsafe_allow_html=True)

        for param, value in MODEL_INFO['hyperparameters'].items():
            st.markdown(f"- **{param}**: {value}")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class='info-box'>
            <h4>Additional Metrics</h4>
            <p>F1-Score: <strong>{MODEL_INFO['test_f1']:.4f}</strong></p>
            <p>Test Set Size: 184 samples (20% of 918)</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# TAB 4: ABOUT
# ============================================================================

with tab4:
    st.markdown("## ℹ️ About This Application")

    st.markdown("""
    ### Overview
    This Heart Failure Prediction System uses a **GridSearchCV-optimized Random Forest classifier**
    to assess the risk of heart disease based on clinical parameters.

    ### Features
    - **Real-time Prediction**: Get instant risk assessment based on patient data
    - **Patient History**: Track multiple patient records and view trends
    - **Model Performance**: View detailed metrics and model information
    - **Optimized Model**: Uses hyperparameter tuning with 1,080 trained models

    ### Dataset
    - **Total Samples**: 918 patient records
    - **Features**: 11 clinical parameters
    - **Target**: Binary classification (Heart Disease: Yes/No)

    ### Model Training
    - **Algorithm**: Random Forest Classifier
    - **Hyperparameter Tuning**: GridSearchCV
    - **Cross-Validation**: StratifiedKFold (5-fold)
    - **Best Parameters Found**:
      - n_estimators: 200 trees
      - max_depth: 10
      - min_samples_split: 10
      - min_samples_leaf: 2
      - max_features: sqrt

    ### Performance
    - **Test Accuracy**: 87.50%
    - **Test ROC-AUC**: 0.9250 (Excellent)
    - **Test Recall**: 91.18% (Great disease detection)
    - **Test Precision**: 86.92% (Low false positives)
    - **Test F1-Score**: 0.8900

    ### How to Use
    1. **Enter Patient Data**: Use the sidebar to input clinical parameters
    2. **Get Prediction**: The model automatically predicts heart disease risk
    3. **Save Record**: Click "Save to Patient History" to store the result
    4. **View Trends**: Check the "Patient History" tab to see patient trends

    ### Clinical Parameters
    - **Age**: Patient age in years
    - **Sex**: Biological sex (Male/Female)
    - **Chest Pain Type**: Type of chest pain experienced
    - **Resting BP**: Blood pressure at rest
    - **Cholesterol**: Serum cholesterol level
    - **Fasting BS**: Fasting blood sugar > 120 mg/dL
    - **Resting ECG**: Electrocardiogram results
    - **Max HR**: Maximum heart rate achieved
    - **Exercise Angina**: Angina induced by exercise
    - **Oldpeak**: ST depression induced by exercise
    - **ST Slope**: Slope of the ST segment

    ### Disclaimer
    This tool is for educational and research purposes only.
    It should not be used as a substitute for professional medical advice.
    Always consult with qualified healthcare professionals for medical decisions.

    ### Contact & Support
    For questions or issues, please contact the development team.
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    <p>Heart Failure Prediction System | GridSearchCV-Optimized Model | Updated: April 19, 2026</p>
</div>
""", unsafe_allow_html=True)
