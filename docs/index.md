# 🫀 Heart Failure Prediction System

**Advanced Machine Learning Pipeline with Interactive Web Application**

[![GitHub](https://img.shields.io/badge/GitHub-Mahdi--20-blue?logo=github)](https://github.com/Mahdi-20/Heart-Failure-Prediction-Dataset)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red?logo=streamlit)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📊 Quick Overview

A **complete machine learning project** for predicting heart failure risk using clinical data. Features a production-ready Random Forest classifier achieving **92.75% cross-validation ROC-AUC**, comprehensive data analysis notebooks, and an interactive Streamlit web application for real-time predictions.

### ✨ Key Highlights

- **Best Model**: Random Forest
- **Performance**: 92.75% CV ROC-AUC | 92.39% Test AUC | 86% Recall
- **Dataset**: 918 patient records | 11 clinical features | 0 duplicates
- **Deployment**: Live Streamlit Web App with SHAP explainability
- **Status**: Production Ready ✓

---

## 🚀 Live Demo

**[Open Interactive Web App →](https://heart-disease-predictor-7crsjwjiyk7su62cji42fz.streamlit.app/)**

The web application features:
- 🔮 **Real-time Risk Prediction** - Input patient data and get instant heart failure risk assessment
- 📊 **Data Visualizations** - Comprehensive disease distribution and clinical feature analysis
- 📈 **Patient History Tracking** - Save and monitor multiple patient predictions over time
- 🧠 **SHAP Explainability** - Understand which features influence each prediction
- 📋 **About & Information** - Course details and model performance metrics

---

## 📁 Project Structure

```
Heart-Failure-Prediction-Dataset/
│
├── app.py                          # Main Streamlit application
├── ml_pipeline.py                  # ML training pipeline
├── train_model.py                  # Model training & artifact saving
├── requirements.txt                # Python dependencies
├── heart.csv                       # Dataset (918 samples, 11 features)
│
├── notebooks/
│   ├── ML_Pipeline.ipynb           # Complete pipeline with visualizations
│   └── ML_Analysis.ipynb           # Comprehensive analysis (8 sections)
│
├── models/
│   ├── best_model.pkl              # Trained Random Forest
│   ├── scaler.pkl                  # StandardScaler for preprocessing
│   ├── label_encoders.pkl          # Categorical encoders
│   └── feature_names.pkl           # Feature names mapping
│
├── docs/
│   └── index.md                    # GitHub Pages documentation
│
├── README.md                       # Project README
└── DATASET_ANALYSIS.md             # Dataset quality report
```

---

## 🎯 Features & Capabilities

### Machine Learning Pipeline
- **5 Models Evaluated**: Logistic Regression, SVM (Linear & RBF), Random Forest, LDA
- **5-Fold Stratified Cross-Validation** for robust performance evaluation
- **Comprehensive Metrics**: Accuracy, ROC-AUC, Recall, Precision, F1-Score
- **Hyperparameter Tuning**: GridSearchCV optimization for Random Forest
- **Feature Importance Analysis**: SHAP & Random Forest importance rankings

### Data Preprocessing
- ✅ Categorical Encoding (LabelEncoder)
- ✅ Numeric Scaling (StandardScaler)
- ✅ Zero Value Handling (Median Imputation)
- ✅ Train-Test Stratified Split (80/20)

### Interactive Web Application
- 📱 **Make Prediction Tab** - Real-time clinical assessment
- 📊 **Patient History Tab** - Prediction tracking & trend analysis
- 📈 **Data Insights Tab** - Dataset statistics & visualizations
- ℹ️ **About Project Tab** - Course & model information

---

## 📈 Model Performance

| Model | CV ROC-AUC | Test ROC-AUC | Accuracy | Recall | Precision |
|-------|-----------|-------------|----------|--------|-----------|
| **Random Forest** | **0.9275** | **0.9239** | 0.8641 | **0.86** | 0.8641 |
| SVM (RBF) | 0.9171 | 0.9065 | 0.8534 | 0.7869 | 0.8545 |
| SVM (Linear) | 0.9158 | 0.8967 | 0.8426 | 0.7582 | 0.8605 |
| Logistic Regression | 0.8988 | 0.8791 | 0.8317 | 0.7275 | 0.8586 |
| LDA | 0.8897 | 0.8736 | 0.8207 | 0.7059 | 0.8509 |

**Why Random Forest?**
- Highest ROC-AUC across CV and test sets
- Excellent generalization (minimal CV-Test gap)
- Balanced performance across all metrics
- Provides feature importance insights

---

## 🔬 Dataset Information

### Source
**Heart Failure Prediction Dataset**
- 918 patient records
- 11 clinical features
- Binary classification (Heart Failure vs No Disease)
- 100% complete data | 0 missing values | 0 duplicates
- Balanced classes (55.3% Heart Failure, 44.7% Healthy)

### Clinical Features
| Category | Features |
|----------|----------|
| Demographics | Age, Sex |
| Symptoms | ChestPainType, ExerciseAngina |
| Measurements | RestingBP, Cholesterol, MaxHR, Oldpeak |
| ECG Data | RestingECG, ST_Slope |
| Blood Tests | FastingBS |

---

## 🛠️ Technologies Used

### ML & Data Science Stack
- **pandas, NumPy** - Data manipulation & analysis
- **scikit-learn** - Machine learning models & preprocessing
- **SHAP** - Model explainability
- **Plotly, Seaborn, Matplotlib** - Data visualization

### Web Framework & Deployment
- **Streamlit** - Interactive web application
- **Streamlit Cloud** - Cloud hosting & deployment
- **Git & GitHub** - Version control & repository management

### Development Environment
- **Python 3.9+**
- **Jupyter Notebook** - Data exploration & analysis
- **VS Code** - Code editor

---

## 📖 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/Mahdi-20/Heart-Failure-Prediction-Dataset.git
cd Heart-Failure-Prediction-Dataset
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Streamlit App Locally
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`

### 4. Explore Jupyter Notebooks
```bash
jupyter notebook
```
Open `ML_Pipeline.ipynb` or `ML_Analysis.ipynb` to see detailed analysis

### 5. Train Model (Optional)
```bash
python train_model.py
```

---

## 📚 Jupyter Notebooks

### ML_Pipeline.ipynb
Complete ML pipeline with:
- Data loading & exploration
- Preprocessing & feature engineering
- Model training (5 models)
- Cross-validation & evaluation
- Confusion matrices & ROC curves
- Performance comparison visualizations

### ML_Analysis.ipynb
Comprehensive analysis with 8 sections:
1. Load & Explore Data
2. Exploratory Data Analysis (EDA)
3. Model Comparison - 5-Fold CV
4. Best Model Evaluation
5. ROC Curves
6. Feature Importance
7. Hyperparameter Tuning
8. Conclusion & Summary

---

## 🎓 Course Information

**Course**: Advanced ML & Data Analytics  
**Institution**: Nexa-land  
**Instructor**: Prof. Hamed Mamani, University of Washington  
**Author**: Mahdi Bakhtiari

---

## 📝 Important Disclaimer

⚠️ **This application is for educational purposes only.**

- **NOT a substitute for professional medical diagnosis**
- Results should only be used in consultation with qualified healthcare professionals
- Always consult a doctor for medical advice
- Model predictions are based on training data patterns and may not reflect all clinical scenarios

---

## 📖 Documentation

- **[README.md](https://github.com/Mahdi-20/Heart-Failure-Prediction-Dataset/blob/master/README.md)** - Detailed project documentation
- **[DATASET_ANALYSIS.md](https://github.com/Mahdi-20/Heart-Failure-Prediction-Dataset/blob/master/DATASET_ANALYSIS.md)** - Dataset quality & specifications
- **[GitHub Repository](https://github.com/Mahdi-20/Heart-Failure-Prediction-Dataset)** - Source code & version control

---

## 🤝 Contributing

This is an educational project. For improvements or suggestions, feel free to:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📞 Contact

- **Author**: Mahdi Bakhtiari
- **GitHub**: [@Mahdi-20](https://github.com/Mahdi-20)
- **Repository**: [Heart-Failure-Prediction-Dataset](https://github.com/Mahdi-20/Heart-Failure-Prediction-Dataset)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**[🫀 Try the Live App →](https://heart-disease-predictor-7crsjwjiyk7su62cji42fz.streamlit.app/)**

Built with ❤️ for machine learning education

</div>
