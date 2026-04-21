# ❤️ Heart Failure Prediction System

Advanced ML Model for Heart Disease Risk Assessment using supervised machine learning classification.

## 📋 Project Overview

**Objective:** Predict heart failure risk using clinical features and machine learning.

- **Type:** Binary Classification (Heart Failure Present/Absent)
- **Dataset:** 918 patients with 11 clinical features
- **Best Model:** Random Forest with GridSearchCV optimization
- **Performance:** 87.5% accuracy, 92.5% ROC-AUC
- **Deployment:** Interactive Streamlit web application

## 🚀 Live Application

**Try it now:** [Heart Failure Prediction System](https://heart-failure-prediction-mlcourse2025-2026.streamlit.app/)

## 📁 Project Structure

```
Heart-Failure-Prediction-Dataset/
│
├── 📁 Final/                                (DELIVERABLES - Ready to Share)
│   ├── heart.csv                          (Dataset)
│   ├── ML_Analysis_Final_ver_2_0.ipynb    (Complete analysis notebook)
│   ├── ML_Analysis_Final_ver_2_0.pdf      (Detailed report)
│   ├── ML_Analysis_Final_ver_2_0.tex      (LaTeX source)
│   ├── app.py                             (Streamlit webapp)
│   ├── best_model.pkl                     (Trained Random Forest model)
│   ├── scaler.pkl                         (StandardScaler)
│   ├── label_encoders.pkl                 (Categorical encoders)
│   ├── feature_names.pkl                  (Feature reference)
│   ├── model_metrics.json                 (Performance metrics)
│   ├── patient_history.json               (Sample patient data)
│   ├── ML_Course_Project_Presentatoin_Group5_ver1(20260419).pptx  (Group presentation)
│   ├── RF_Deployment_Pipeline.png         (Deployment diagram)
│   ├── Data_Preprocessing_Flowchart.png   (Data pipeline diagram)
│   ├── requirements.txt                   (Python dependencies)
│   └── README.md                          (Project documentation)
│
├── 📁 Scripts/                            (Utility & Visualization Scripts)
│   ├── create_deployment_flowchart.py
│   ├── create_preprocessing_image.py
│   ├── create_feature_table_ppt.py
│   ├── create_flowchart_image.py
│   ├── ml_pipeline.py
│   ├── train_model.py
│   ├── export_models.py
│   └── ... (15+ utility scripts)
│
├── 📁 BuildArtifacts/                     (LaTeX/PDF Build Temporary Files)
│   ├── *.log, *.aux, *.out, *.toc
│   └── *_files/ (HTML build artifacts)
│
├── 📁 Images/                             (Generated Visualizations)
│   ├── ML_Pipeline_Flowchart.png
│   ├── folowchart.png
│   ├── EKG*.png
│   └── preprocessing_pipline.png
│
├── 📁 Development/                        (Experimental/Demo Files)
│   └── webapp_demo_60sec.gif
│
├── 📁 Archive/                            (Old Versions & Backups)
│   ├── ML_Analysis.ipynb (old)
│   ├── ML_Pipeline.ipynb (old)
│   ├── LLM_presentation.* (old)
│   └── ML_Project_Summary.* (old)
│
├── .git/                                  (Version control)
├── .gitignore                             (Git ignore rules)
└── README.md                              (This file)
```

## 🔧 Technologies Used

### Machine Learning & Data Science
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Scikit-Learn** - ML algorithms & preprocessing
- **GridSearchCV** - Hyperparameter optimization
- **StratifiedKFold** - 5-fold cross-validation

### Visualization & Deployment
- **Matplotlib/Seaborn** - Static visualizations
- **Plotly** - Interactive charts
- **SHAP** - Model explainability
- **Streamlit** - Web application framework

### Tools & Environments
- **Python 3.x** - Primary language
- **Jupyter Notebook** - Analysis & documentation
- **Streamlit Cloud** - Live deployment
- **Git/GitHub** - Version control

## 📊 Model Performance

| Metric | Score | Interpretation |
|--------|-------|-----------------|
| Test Accuracy | 87.5% | Excellent overall performance |
| ROC-AUC | 92.5% | Excellent discrimination |
| Recall | 91.18% | High disease detection rate |
| Precision | 86.92% | Low false positive rate |
| F1-Score | 89.0% | Balanced precision-recall |

## 🎯 Key Features

✅ **Real-time Predictions** - Instant risk assessment with clinical inputs

✅ **Model Explainability** - SHAP analysis shows feature importance

✅ **Patient History Tracking** - Save and analyze multiple predictions over time

✅ **Data Visualizations** - Interactive charts and trend analysis

✅ **Production-Ready Deployment** - Streamlit Cloud integration

## 📈 Dataset Information

- **Source:** [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Samples:** 918 patients
- **Features:** 11 clinical measurements
- **Target:** Binary (0 = No disease, 1 = Heart failure)
- **Data Quality:** 100% complete, no missing values, balanced classes

## 🔬 Clinical Features

| Feature | Description |
|---------|-------------|
| Age | Patient age in years |
| Sex | Gender (M=Male, F=Female) |
| ChestPainType | Type of chest pain |
| RestingBP | Resting blood pressure (mmHg) |
| Cholesterol | Serum cholesterol level (mg/dL) |
| FastingBS | Fasting blood sugar > 120 mg/dL |
| RestingECG | Resting electrocardiogram results |
| MaxHR | Maximum heart rate achieved (bpm) |
| ExerciseAngina | Exercise-induced angina (Yes/No) |
| Oldpeak | ST depression induced by exercise |
| ST_Slope | Slope of ST segment |

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r Final/requirements.txt
```

### Run the Webapp Locally
```bash
streamlit run Final/app.py
```

### View Analysis
Open `Final/ML_Analysis_Final_ver_2_0.ipynb` in Jupyter Notebook

## 📚 Course Information

- **Course:** Advanced ML & Data Analytics
- **Institution:** Nexa-land
- **Instructor:** Prof. Hamed Mamani, University of Washington
- **Semester:** 2026 Spring

## 👨‍💻 Author

**Mahdi Bakhtiari** (@mahdi-20)

- GitHub: [github.com/mahdi-20](https://github.com/mahdi-20)
- Email: mahdi6563@gmail.com

## ⚠️ Important Disclaimer

This application is for **educational purposes only** and should NOT be used for clinical diagnosis. Always consult with qualified healthcare professionals for medical advice and diagnosis.

The model predictions are estimates based on training data and should not replace professional medical evaluation.

## 📄 License

This project is part of an educational course. Use for learning purposes only.

---

**Last Updated:** April 21, 2026

Built with ❤️ using Python, Machine Learning, and Streamlit
