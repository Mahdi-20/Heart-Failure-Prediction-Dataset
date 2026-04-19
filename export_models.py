"""
Export trained models from ML_Analysis_Final_ver_2_0.ipynb
Saves models as pickle files for use with app.py
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score

print("\n" + "="*80)
print("EXPORTING TRAINED MODELS")
print("="*80)

# Load dataset
print("\n[1/6] Loading dataset...")
df = pd.read_csv('heart.csv')
print(f"  Dataset shape: {df.shape}")

# Prepare features and target
print("\n[2/6] Preprocessing data...")
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"  Categorical columns: {categorical_cols}")
print(f"  Numeric columns: {numeric_cols}")

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Handle missing values (zeros in Cholesterol and RestingBP)
for col in ['Cholesterol', 'RestingBP']:
    median_val = X[X[col] != 0][col].median()
    X[col] = X[col].replace(0, median_val)

# Scale numeric features
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

print(f"  Features scaled and encoded")

# Train-test split with stratification
print("\n[3/6] Splitting data (80-20 train-test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Training set: {X_train.shape}")
print(f"  Test set: {X_test.shape}")

# Set up StratifiedKFold for GridSearchCV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define Random Forest parameter grid
print("\n[4/6] Setting up GridSearchCV for Random Forest...")
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

total_combinations = 1
for v in rf_param_grid.values():
    total_combinations *= len(v)
print(f"  Total parameter combinations: {total_combinations}")
print(f"  With 5-fold CV: {total_combinations * 5} models to train")

# Perform GridSearchCV
print("\n[5/6] Training Random Forest with GridSearchCV...")
print("  (This may take 2-3 minutes...)")

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

rf_grid.fit(X_train, y_train)

print(f"\n  Best CV ROC-AUC Score: {rf_grid.best_score_:.4f}")
print(f"  Optimal Hyperparameters:")
for param, value in rf_grid.best_params_.items():
    print(f"    - {param}: {value}")

# Get the best estimator
best_model = rf_grid.best_estimator_

# Evaluate on test set
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

test_accuracy = accuracy_score(y_test, y_pred)
test_roc_auc = roc_auc_score(y_test, y_proba)
test_recall = recall_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)

print(f"\n  Test Set Performance:")
print(f"    - Accuracy:  {test_accuracy:.4f}")
print(f"    - ROC-AUC:   {test_roc_auc:.4f}")
print(f"    - Recall:    {test_recall:.4f}")
print(f"    - Precision: {test_precision:.4f}")
print(f"    - F1-Score:  {test_f1:.4f}")

# Save models and preprocessing objects as pickle
print("\n[6/6] Saving models as pickle files...")

with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("  [OK] Saved: best_model.pkl")

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("  [OK] Saved: scaler.pkl")

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("  [OK] Saved: label_encoders.pkl")

with open('feature_names.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)
print("  [OK] Saved: feature_names.pkl")

print("\n" + "="*80)
print("[OK] MODEL EXPORT COMPLETED SUCCESSFULLY")
print("="*80)
print("\nThe following files have been created:")
print("  1. best_model.pkl       - Trained Random Forest classifier")
print("  2. scaler.pkl           - StandardScaler for numeric features")
print("  3. label_encoders.pkl   - LabelEncoders for categorical features")
print("  4. feature_names.pkl    - List of feature names")
print("\nThese files are ready to be used with app.py")
print("="*80 + "\n")
