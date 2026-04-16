# ❤️ Heart Failure Prediction - Advanced ML & Deployment

A comprehensive machine learning project for heart failure prediction with interactive web application deployment using Streamlit.

## 🎯 Project Overview

**Objective:** Predict heart failure risk using supervised machine learning classification with clinical features.

### ML Problem Definition
- **Type:** Binary Classification (Heart Failure Present/Absent)
- **Dataset:** Heart Failure Prediction (918 patients, 11 clinical features)
- **Data Quality:** 100% complete, no missing values, no duplicates
- **Class Balance:** 410:508 (0.81 ratio - well balanced)

### Models Evaluated
- **SVM (Linear & RBF)** - Support Vector Machines
- **Random Forest** - Ensemble decision trees (✓ SELECTED)
- **Logistic Regression** - Baseline statistical model
- **Linear Discriminant Analysis (LDA)** - Dimensionality reduction

### Best Model Performance
| Metric | Value |
|--------|-------|
| Algorithm | Random Forest (100 trees) |
| Test Accuracy | **86.41%** |
| ROC-AUC | **0.9239** |
| Test Recall | **87.25%** |
| Test Precision | **88.12%** |
| Cross-Validation (5-fold) | 92.75% ROC-AUC |

## 📊 Dataset

**Heart Failure Prediction Dataset**
- **Source:** [Kaggle Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Samples:** 918 patients
- **Features:** 11 clinical measurements
- **Target:** Binary (0=No disease, 1=Heart failure)

### Features
| Feature | Type | Description |
|---------|------|-------------|
| Age | Numeric | Patient age (28-77 years) |
| Sex | Categorical | M (Male) or F (Female) |
| ChestPainType | Categorical | ASY, ATA, NAP, TA |
| RestingBP | Numeric | Resting blood pressure (mmHg) |
| Cholesterol | Numeric | Serum cholesterol (mg/dL) |
| FastingBS | Binary | Fasting blood sugar > 120 mg/dL |
| RestingECG | Categorical | Normal, ST, LVH |
| MaxHR | Numeric | Max heart rate achieved |
| ExerciseAngina | Binary | Exercise-induced angina (Y/N) |
| Oldpeak | Numeric | ST depression induced by exercise |
| ST_Slope | Categorical | Up, Flat, Down |

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip or conda
```

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/mahdi-20/heart-failure-predictor.git
cd heart-failure-predictor
```

2. **Create virtual environment (optional):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Running Locally

**Option 1: Standard App**
```bash
streamlit run app.py
```

**Option 2: Advanced App (with model retraining)**
```bash
streamlit run app_with_retraining.py
```

The app will be available at `http://localhost:8501`

## 📁 Project Structure

```
heart-failure-predictor/
├── app.py                          # Standard Streamlit app
├── app_with_retraining.py         # Advanced app with retraining
├── ml_pipeline.py                  # Complete ML training pipeline
├── train_model.py                  # Model training & saving script
├── heart.csv                       # Dataset (918 samples)
├── best_model.pkl                 # Trained Random Forest model
├── scaler.pkl                     # Feature scaler
├── label_encoders.pkl             # Categorical encoders
├── feature_names.pkl              # Feature names
├── requirements.txt               # Python dependencies
├── DATASET_ANALYSIS.md           # Dataset quality report
├── README.md                      # This file
└── .gitignore                    # Git ignore rules
```

## 🛠️ Technologies Used

### Python ML Stack
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation & analysis
- **Scikit-Learn** - Machine learning algorithms
- **Matplotlib/Seaborn** - Static visualizations
- **Plotly** - Interactive visualizations

### ML Development
- **SVM** - Support Vector Machine classifier
- **Random Forest** - Ensemble learning
- **Logistic Regression** - Baseline model
- **LDA** - Linear Discriminant Analysis
- **Cross-Validation** - 5-fold stratified CV

### Model Interpretability & Deployment
- **SHAP** - Feature importance analysis
- **Streamlit** - Interactive web framework
- **Streamlit Cloud** - Cloud deployment

## 📊 ML Pipeline

### Step 1: Data Preprocessing
- Categorical feature encoding (LabelEncoder)
- Numeric feature scaling (StandardScaler)
- Zero value handling (replaced with median)
- Train-test split (80/20, stratified)

### Step 2: Model Training
- 5 models trained with stratified 5-fold CV
- Metrics: Accuracy, ROC-AUC, Recall, Precision, F1-Score
- Best model: Random Forest (ROC-AUC: 0.9275)

### Step 3: Model Evaluation
- Cross-validation scores computed
- Test set evaluation
- Feature importance analysis via SHAP
- Detailed confusion matrix & classification report

### Step 4: Deployment
- Model saved as pickle (best_model.pkl)
- Preprocessing objects saved (scaler, encoders)
- Interactive Streamlit application
- Streamlit Cloud deployment

## 🎯 Application Features

### Standard App (app.py)
✅ Real-time predictions with clinical feature inputs  
✅ Interactive risk visualization (gauge charts)  
✅ SHAP feature importance analysis  
✅ Patient history tracking (JSON storage)  
✅ Risk trend analysis with multiple predictions  
✅ Patient ID support for distinguishing patients  
✅ Responsive design for desktop/mobile  

### Advanced App (app_with_retraining.py)
✅ All standard features PLUS:  
✅ Model retraining with new patient data  
✅ Continuous learning capability  
✅ Cross-validation scores after retraining  
✅ Session-based model management  

## 📈 How to Use

### 1. Make a Prediction
1. Go to "Make Prediction" tab
2. Enter patient clinical data
3. Click "Get Heart Failure Risk Assessment"
4. View risk percentage, confidence, and feature importance
5. Save to patient history

### 2. Analyze Patient Trends
1. Go to "Patient History & Trends" tab
2. Select patient from dropdown
3. View prediction history table
4. See risk trends over time (if 2+ predictions)
5. View patient statistics

### 3. Retrain Model (Advanced App)
1. Save 5+ patient predictions
2. Scroll to "Model Management & Retraining"
3. Click "Retrain Model with New Data"
4. See updated cross-validation accuracy
5. All predictions now use improved model

## 🌐 Deployment Options

### Streamlit Cloud (Recommended)
1. Push to GitHub repository
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect GitHub account
4. Deploy new app from repository
5. Share app URL

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
```

## 📚 Course Information

**Course:** Advanced ML & Data Analytics  
**Institution:** Nexa-land  
**Professor:** [Prof. Hamed Mamani](https://foster.uw.edu/faculty-research/directory/hamed-mamani/)  
University of Washington  

**Author:** Mahdi Bakhtiari  
**GitHub:** [github.com/mahdi-20](https://github.com/mahdi-20)

## ⚠️ Important Disclaimer

**This application is for EDUCATIONAL PURPOSES ONLY** and should not be used for clinical diagnosis.

Always consult with qualified healthcare professionals for medical advice and diagnosis. The model is trained on historical data and predictions are estimates only. This tool does not replace professional medical evaluation.

## 📝 Results Summary

### Data Quality
- Total Samples: 918
- Missing Values: 0% ✓
- Duplicates: 0% ✓
- Class Balance: 0.81 ratio (excellent) ✓

### Model Selection
- **5 Models Tested**: SVM, Random Forest, Logistic Regression, LDA, SVM RBF
- **Best Model**: Random Forest
  - **Why RF?** Best ROC-AUC (0.9275), excellent generalization
  - **Metrics**: 86.41% accuracy, 92.39% test AUC, 87.25% recall

### Preprocessing Applied
- Categorical encoding: 5 features (Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope)
- Numeric scaling: StandardScaler on 6 features
- Zero handling: Median imputation for Cholesterol & RestingBP
- Train-test split: 80/20 stratified

### Production Ready
✅ Trained model saved (best_model.pkl)  
✅ All preprocessing objects saved  
✅ Interactive Streamlit app  
✅ Patient history tracking  
✅ SHAP explainability  
✅ Ready for cloud deployment  

## 🔗 Links

- **Dataset:** https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
- **Professor:** https://foster.uw.edu/faculty-research/directory/hamed-mamani/
- **Streamlit Docs:** https://docs.streamlit.io/
- **SHAP Documentation:** https://shap.readthedocs.io/
- **Scikit-Learn:** https://scikit-learn.org/

## 📧 Questions & Support

For questions about this project, please reach out via GitHub Issues or GitHub Discussions.

---

**Built with Python, Machine Learning, and Streamlit** ❤️  
Advanced ML & Data Analytics Course | Nexa-land | Prof. Hamed Mamani, University of Washington
