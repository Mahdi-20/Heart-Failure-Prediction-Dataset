#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Heart Failure Prediction - Complete ML Pipeline
Comprehensive machine learning pipeline with multiple model comparison and evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (accuracy_score, roc_auc_score, recall_score,
                              precision_score, f1_score, confusion_matrix,
                              classification_report)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA LOADING AND EXPLORATION
# ============================================================================

print("=" * 80)
print("HEART FAILURE PREDICTION - ML PIPELINE")
print("=" * 80)

# Load dataset
df = pd.read_csv('heart.csv')
print(f"\n[OK] Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

# Display dataset info
print(f"\nDataset Info:")
print(f"  - Target: HeartDisease (0=No disease, 1=Disease)")
print(f"  - Features: {', '.join(df.columns[:-1])}")
print(f"\nTarget Distribution:")
print(df['HeartDisease'].value_counts())

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================

print("\n" + "=" * 80)
print("DATA PREPROCESSING")
print("=" * 80)

# Separate features and target
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"\nFeature Types:")
print(f"  Categorical: {categorical_cols}")
print(f"  Numeric: {numeric_cols}")

# Encode categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    mappings = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"  [ENCODED] {col}: {mappings}")

# Handle zero values in Cholesterol and RestingBP
print(f"\nHandling Zero Values:")
print(f"  - Cholesterol zeros: {(X['Cholesterol'] == 0).sum()}")
print(f"  - RestingBP zeros: {(X['RestingBP'] == 0).sum()}")

# Replace zero values with median
for col in ['Cholesterol', 'RestingBP']:
    median_val = X[X[col] != 0][col].median()
    X[col] = X[col].replace(0, median_val)
    print(f"  [FIXED] Replaced {col} zeros with median: {median_val}")

# Standardize numeric features
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
print(f"\n[OK] Standardized {len(numeric_cols)} numeric features")

print(f"\nPreprocessed Data Shape: {X.shape}")

# ============================================================================
# 3. TRAIN-TEST SPLIT
# ============================================================================

print("\n" + "=" * 80)
print("TRAIN-TEST SPLIT")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain Set: {X_train.shape[0]} samples")
print(f"  - Class 0: {(y_train == 0).sum()} ({(y_train == 0).sum() / len(y_train) * 100:.1f}%)")
print(f"  - Class 1: {(y_train == 1).sum()} ({(y_train == 1).sum() / len(y_train) * 100:.1f}%)")

print(f"\nTest Set: {X_test.shape[0]} samples")
print(f"  - Class 0: {(y_test == 0).sum()} ({(y_test == 0).sum() / len(y_test) * 100:.1f}%)")
print(f"  - Class 1: {(y_test == 1).sum()} ({(y_test == 1).sum() / len(y_test) * 100:.1f}%)")

# ============================================================================
# 4. MODEL TRAINING AND CROSS-VALIDATION
# ============================================================================

print("\n" + "=" * 80)
print("MODEL TRAINING AND EVALUATION")
print("=" * 80)

# Define models
models = {
    'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'LDA': LinearDiscriminantAnalysis()
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store results
cv_results = {}
test_results = {}

print("\n5-FOLD CROSS-VALIDATION RESULTS:")
print("-" * 80)

for model_name, model in models.items():
    print(f"\n{model_name}:")

    # Cross-validation
    scoring = ['accuracy', 'roc_auc', 'recall', 'precision', 'f1']
    cv_scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)

    cv_results[model_name] = {
        'Accuracy': cv_scores['test_accuracy'].mean(),
        'Accuracy_std': cv_scores['test_accuracy'].std(),
        'ROC-AUC': cv_scores['test_roc_auc'].mean(),
        'ROC-AUC_std': cv_scores['test_roc_auc'].std(),
        'Recall': cv_scores['test_recall'].mean(),
        'Precision': cv_scores['test_precision'].mean(),
        'F1': cv_scores['test_f1'].mean(),
    }

    print(f"  Accuracy:   {cv_results[model_name]['Accuracy']:.4f} +/- {cv_results[model_name]['Accuracy_std']:.4f}")
    print(f"  ROC-AUC:    {cv_results[model_name]['ROC-AUC']:.4f} +/- {cv_results[model_name]['ROC-AUC_std']:.4f}")
    print(f"  Recall:     {cv_results[model_name]['Recall']:.4f}")
    print(f"  Precision:  {cv_results[model_name]['Precision']:.4f}")
    print(f"  F1-Score:   {cv_results[model_name]['F1']:.4f}")

    # Train on full training set and evaluate on test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    test_results[model_name] = {
        'Test_Accuracy': accuracy_score(y_test, y_pred),
        'Test_ROC_AUC': roc_auc_score(y_test, y_pred_proba),
        'Test_Recall': recall_score(y_test, y_pred),
        'Test_Precision': precision_score(y_test, y_pred),
        'Test_F1': f1_score(y_test, y_pred),
    }

# ============================================================================
# 5. MODEL COMPARISON AND SELECTION
# ============================================================================

print("\n" + "=" * 80)
print("TEST SET RESULTS")
print("=" * 80)

# Create comparison dataframe
cv_df = pd.DataFrame(cv_results).T
test_df = pd.DataFrame(test_results).T

print("\nCROSS-VALIDATION RESULTS:")
print(cv_df.to_string())
print("\nTEST SET RESULTS:")
print(test_df.to_string())

# Select best model based on ROC-AUC
best_model_name = cv_df['ROC-AUC'].idxmax()
best_cv_auc = cv_df.loc[best_model_name, 'ROC-AUC']
best_test_auc = test_df.loc[best_model_name, 'Test_ROC_AUC']

print("\n" + "=" * 80)
print("BEST MODEL SELECTED")
print("=" * 80)
print(f"\nModel: {best_model_name}")
print(f"CV ROC-AUC: {best_cv_auc:.4f}")
print(f"Test ROC-AUC: {best_test_auc:.4f}")
print(f"Test Accuracy: {test_df.loc[best_model_name, 'Test_Accuracy']:.4f}")
print(f"Test Recall: {test_df.loc[best_model_name, 'Test_Recall']:.4f}")
print(f"Test Precision: {test_df.loc[best_model_name, 'Test_Precision']:.4f}")

# Train final best model
best_model = models[best_model_name]
best_model.fit(X_train, y_train)

print(f"\n[OK] Best model trained and ready for deployment")

# ============================================================================
# 6. DETAILED ANALYSIS OF BEST MODEL
# ============================================================================

print("\n" + "=" * 80)
print("DETAILED ANALYSIS - BEST MODEL")
print("=" * 80)

y_pred_best = best_model.predict(X_test)
y_pred_proba_best = best_model.predict_proba(X_test)[:, 1]

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=['No Disease', 'Disease']))

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
print(cm)
print(f"  True Negatives:  {cm[0, 0]}")
print(f"  False Positives: {cm[0, 1]}")
print(f"  False Negatives: {cm[1, 0]}")
print(f"  True Positives:  {cm[1, 1]}")

# ============================================================================
# 7. SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("PIPELINE SUMMARY")
print("=" * 80)

print(f"""
Dataset Information:
  - Total Samples: {len(df)}
  - Features: {X.shape[1]}
  - Classes: 2 (No Disease, Disease)
  - Class Balance: {(y == 0).sum()}:{(y == 1).sum()}

Preprocessing:
  - Categorical Encoding: [OK]
  - Feature Scaling: [OK]
  - Missing Value Handling: [OK]
  - Train/Test Split: 80/20 (Stratified)

Model Comparison:
  - Models Tested: 5
  - Best Model: {best_model_name}
  - CV ROC-AUC: {best_cv_auc:.4f}
  - Test ROC-AUC: {best_test_auc:.4f}
  - Test Accuracy: {test_df.loc[best_model_name, 'Test_Accuracy']:.4f}

Status: Ready for Deployment [OK]
""")

print("=" * 80)
