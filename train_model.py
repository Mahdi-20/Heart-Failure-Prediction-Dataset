#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train and save the best model (Random Forest)
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

print("Training and saving Random Forest model...")

# Load data
df = pd.read_csv('heart.csv')
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Identify feature types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Encode categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Handle zero values
for col in ['Cholesterol', 'RestingBP']:
    median_val = X[X[col] != 0][col].median()
    X[col] = X[col].replace(0, median_val)

# Standardize numeric features
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Save model and preprocessing objects
joblib.dump(model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

# Save feature names for later use
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'feature_names.pkl')

print("[OK] Model saved: best_model.pkl")
print("[OK] Scaler saved: scaler.pkl")
print("[OK] Label encoders saved: label_encoders.pkl")
print("[OK] Feature names saved: feature_names.pkl")

# Print model info
from sklearn.metrics import accuracy_score, roc_auc_score
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print(f"\nModel Performance:")
print(f"  Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"  Test ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
