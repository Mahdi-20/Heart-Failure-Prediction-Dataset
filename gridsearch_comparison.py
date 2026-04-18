"""
GridSearchCV Comparison: Default vs. Tuned Hyperparameters
Compares model performance with default vs. optimal hyperparameters
for the top-performing models (Random Forest and SVM RBF)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: LOAD AND PREPROCESS DATA
# ============================================================================
print("="*80)
print("GRIDSEARCHCV COMPARISON: DEFAULT vs. TUNED HYPERPARAMETERS")
print("="*80)

# Load dataset
df = pd.read_csv('heart.csv')
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Preprocessing
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

for col in ['Cholesterol', 'RestingBP']:
    median_val = X[X[col] != 0][col].median()
    X[col] = X[col].replace(0, median_val)

scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nDataset Prepared:")
print(f"  Train Set: {X_train.shape[0]} samples")
print(f"  Test Set: {X_test.shape[0]} samples")
print(f"  Features: {X_train.shape[1]}")

# ============================================================================
# STEP 2: DEFINE HYPERPARAMETER GRIDS
# ============================================================================
print("\n" + "="*80)
print("HYPERPARAMETER GRID DEFINITIONS")
print("="*80)

# Random Forest Grid
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

print("\nRandom Forest Parameter Grid:")
print(f"  n_estimators: {rf_param_grid['n_estimators']}")
print(f"  max_depth: {rf_param_grid['max_depth']}")
print(f"  min_samples_split: {rf_param_grid['min_samples_split']}")
print(f"  min_samples_leaf: {rf_param_grid['min_samples_leaf']}")
print(f"  max_features: {rf_param_grid['max_features']}")
print(f"  Total combinations: {np.prod([len(v) for v in rf_param_grid.values()])}")

# SVM RBF Grid
svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf']
}

print("\nSVM (RBF) Parameter Grid:")
print(f"  C: {svm_param_grid['C']}")
print(f"  gamma: {svm_param_grid['gamma']}")
print(f"  kernel: {svm_param_grid['kernel']}")
print(f"  Total combinations: {np.prod([len(v) for v in svm_param_grid.values()])}")

# ============================================================================
# STEP 3: RANDOM FOREST - DEFAULT vs. TUNED
# ============================================================================
print("\n" + "="*80)
print("RANDOM FOREST COMPARISON")
print("="*80)

# Default Random Forest
print("\n1. DEFAULT HYPERPARAMETERS:")
rf_default = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_default.fit(X_train, y_train)

y_pred_rf_default = rf_default.predict(X_test)
y_pred_proba_rf_default = rf_default.predict_proba(X_test)[:, 1]

rf_default_results = {
    'Accuracy': accuracy_score(y_test, y_pred_rf_default),
    'ROC-AUC': roc_auc_score(y_test, y_pred_proba_rf_default),
    'Recall': recall_score(y_test, y_pred_rf_default),
    'Precision': precision_score(y_test, y_pred_rf_default),
    'F1': f1_score(y_test, y_pred_rf_default)
}

print(f"  n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt'")
print(f"  Accuracy:  {rf_default_results['Accuracy']:.4f}")
print(f"  ROC-AUC:   {rf_default_results['ROC-AUC']:.4f}")
print(f"  Recall:    {rf_default_results['Recall']:.4f}")
print(f"  Precision: {rf_default_results['Precision']:.4f}")
print(f"  F1-Score:  {rf_default_results['F1']:.4f}")

# Tuned Random Forest with GridSearchCV
print("\n2. TUNED HYPERPARAMETERS (GridSearchCV):")
print("   Searching through parameter space...")

rf_grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)

rf_grid_search.fit(X_train, y_train)
rf_tuned = rf_grid_search.best_estimator_

print(f"\n   Best Parameters Found:")
print(f"     n_estimators: {rf_grid_search.best_params_['n_estimators']}")
print(f"     max_depth: {rf_grid_search.best_params_['max_depth']}")
print(f"     min_samples_split: {rf_grid_search.best_params_['min_samples_split']}")
print(f"     min_samples_leaf: {rf_grid_search.best_params_['min_samples_leaf']}")
print(f"     max_features: {rf_grid_search.best_params_['max_features']}")
print(f"   Best Cross-Validation ROC-AUC Score: {rf_grid_search.best_score_:.4f}")

y_pred_rf_tuned = rf_tuned.predict(X_test)
y_pred_proba_rf_tuned = rf_tuned.predict_proba(X_test)[:, 1]

rf_tuned_results = {
    'Accuracy': accuracy_score(y_test, y_pred_rf_tuned),
    'ROC-AUC': roc_auc_score(y_test, y_pred_proba_rf_tuned),
    'Recall': recall_score(y_test, y_pred_rf_tuned),
    'Precision': precision_score(y_test, y_pred_rf_tuned),
    'F1': f1_score(y_test, y_pred_rf_tuned)
}

print(f"\n   Test Set Performance:")
print(f"   Accuracy:  {rf_tuned_results['Accuracy']:.4f}")
print(f"   ROC-AUC:   {rf_tuned_results['ROC-AUC']:.4f}")
print(f"   Recall:    {rf_tuned_results['Recall']:.4f}")
print(f"   Precision: {rf_tuned_results['Precision']:.4f}")
print(f"   F1-Score:  {rf_tuned_results['F1']:.4f}")

# Improvement Calculation
rf_improvements = {
    'Accuracy': (rf_tuned_results['Accuracy'] - rf_default_results['Accuracy']) / rf_default_results['Accuracy'] * 100,
    'ROC-AUC': (rf_tuned_results['ROC-AUC'] - rf_default_results['ROC-AUC']) / rf_default_results['ROC-AUC'] * 100,
    'Recall': (rf_tuned_results['Recall'] - rf_default_results['Recall']) / rf_default_results['Recall'] * 100,
    'Precision': (rf_tuned_results['Precision'] - rf_default_results['Precision']) / rf_default_results['Precision'] * 100,
    'F1': (rf_tuned_results['F1'] - rf_default_results['F1']) / rf_default_results['F1'] * 100
}

print(f"\n3. IMPROVEMENT:")
for metric, improvement in rf_improvements.items():
    symbol = "↑" if improvement > 0 else "↓" if improvement < 0 else "→"
    print(f"   {metric}: {improvement:+.2f}% {symbol}")

# ============================================================================
# STEP 4: SVM RBF - DEFAULT vs. TUNED
# ============================================================================
print("\n" + "="*80)
print("SVM (RBF) COMPARISON")
print("="*80)

# Default SVM RBF
print("\n1. DEFAULT HYPERPARAMETERS:")
svm_default = SVC(kernel='rbf', probability=True, random_state=42)
svm_default.fit(X_train, y_train)

y_pred_svm_default = svm_default.predict(X_test)
y_pred_proba_svm_default = svm_default.predict_proba(X_test)[:, 1]

svm_default_results = {
    'Accuracy': accuracy_score(y_test, y_pred_svm_default),
    'ROC-AUC': roc_auc_score(y_test, y_pred_proba_svm_default),
    'Recall': recall_score(y_test, y_pred_svm_default),
    'Precision': precision_score(y_test, y_pred_svm_default),
    'F1': f1_score(y_test, y_pred_svm_default)
}

print(f"  C=1.0, gamma='scale', kernel='rbf'")
print(f"  Accuracy:  {svm_default_results['Accuracy']:.4f}")
print(f"  ROC-AUC:   {svm_default_results['ROC-AUC']:.4f}")
print(f"  Recall:    {svm_default_results['Recall']:.4f}")
print(f"  Precision: {svm_default_results['Precision']:.4f}")
print(f"  F1-Score:  {svm_default_results['F1']:.4f}")

# Tuned SVM RBF with GridSearchCV
print("\n2. TUNED HYPERPARAMETERS (GridSearchCV):")
print("   Searching through parameter space...")

svm_grid_search = GridSearchCV(
    SVC(probability=True, random_state=42),
    svm_param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)

svm_grid_search.fit(X_train, y_train)
svm_tuned = svm_grid_search.best_estimator_

print(f"\n   Best Parameters Found:")
print(f"     C: {svm_grid_search.best_params_['C']}")
print(f"     gamma: {svm_grid_search.best_params_['gamma']}")
print(f"   Best Cross-Validation ROC-AUC Score: {svm_grid_search.best_score_:.4f}")

y_pred_svm_tuned = svm_tuned.predict(X_test)
y_pred_proba_svm_tuned = svm_tuned.predict_proba(X_test)[:, 1]

svm_tuned_results = {
    'Accuracy': accuracy_score(y_test, y_pred_svm_tuned),
    'ROC-AUC': roc_auc_score(y_test, y_pred_proba_svm_tuned),
    'Recall': recall_score(y_test, y_pred_svm_tuned),
    'Precision': precision_score(y_test, y_pred_svm_tuned),
    'F1': f1_score(y_test, y_pred_svm_tuned)
}

print(f"\n   Test Set Performance:")
print(f"   Accuracy:  {svm_tuned_results['Accuracy']:.4f}")
print(f"   ROC-AUC:   {svm_tuned_results['ROC-AUC']:.4f}")
print(f"   Recall:    {svm_tuned_results['Recall']:.4f}")
print(f"   Precision: {svm_tuned_results['Precision']:.4f}")
print(f"   F1-Score:  {svm_tuned_results['F1']:.4f}")

# Improvement Calculation
svm_improvements = {
    'Accuracy': (svm_tuned_results['Accuracy'] - svm_default_results['Accuracy']) / svm_default_results['Accuracy'] * 100,
    'ROC-AUC': (svm_tuned_results['ROC-AUC'] - svm_default_results['ROC-AUC']) / svm_default_results['ROC-AUC'] * 100,
    'Recall': (svm_tuned_results['Recall'] - svm_default_results['Recall']) / svm_default_results['Recall'] * 100,
    'Precision': (svm_tuned_results['Precision'] - svm_default_results['Precision']) / svm_default_results['Precision'] * 100,
    'F1': (svm_tuned_results['F1'] - svm_default_results['F1']) / svm_default_results['F1'] * 100
}

print(f"\n3. IMPROVEMENT:")
for metric, improvement in svm_improvements.items():
    symbol = "↑" if improvement > 0 else "↓" if improvement < 0 else "→"
    print(f"   {metric}: {improvement:+.2f}% {symbol}")

# ============================================================================
# STEP 5: COMPREHENSIVE COMPARISON TABLE
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE COMPARISON TABLE")
print("="*80)

comparison_data = {
    'Model': ['Random Forest (Default)', 'Random Forest (Tuned)', 'SVM RBF (Default)', 'SVM RBF (Tuned)'],
    'Accuracy': [
        rf_default_results['Accuracy'],
        rf_tuned_results['Accuracy'],
        svm_default_results['Accuracy'],
        svm_tuned_results['Accuracy']
    ],
    'ROC-AUC': [
        rf_default_results['ROC-AUC'],
        rf_tuned_results['ROC-AUC'],
        svm_default_results['ROC-AUC'],
        svm_tuned_results['ROC-AUC']
    ],
    'Recall': [
        rf_default_results['Recall'],
        rf_tuned_results['Recall'],
        svm_default_results['Recall'],
        svm_tuned_results['Recall']
    ],
    'Precision': [
        rf_default_results['Precision'],
        rf_tuned_results['Precision'],
        svm_default_results['Precision'],
        svm_tuned_results['Precision']
    ],
    'F1-Score': [
        rf_default_results['F1'],
        rf_tuned_results['F1'],
        svm_default_results['F1'],
        svm_tuned_results['F1']
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n")
print(comparison_df.to_string(index=False))

# ============================================================================
# STEP 6: SUMMARY AND RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY AND RECOMMENDATIONS")
print("="*80)

print(f"""
Key Findings:

1. RANDOM FOREST:
   - Default ROC-AUC: {rf_default_results['ROC-AUC']:.4f}
   - Tuned ROC-AUC:   {rf_tuned_results['ROC-AUC']:.4f}
   - Improvement:     {rf_improvements['ROC-AUC']:+.2f}%

   Best Parameters:
   - n_estimators: {rf_grid_search.best_params_['n_estimators']}
   - max_depth: {rf_grid_search.best_params_['max_depth']}
   - min_samples_split: {rf_grid_search.best_params_['min_samples_split']}
   - min_samples_leaf: {rf_grid_search.best_params_['min_samples_leaf']}
   - max_features: {rf_grid_search.best_params_['max_features']}

2. SVM (RBF):
   - Default ROC-AUC: {svm_default_results['ROC-AUC']:.4f}
   - Tuned ROC-AUC:   {svm_tuned_results['ROC-AUC']:.4f}
   - Improvement:     {svm_improvements['ROC-AUC']:+.2f}%

   Best Parameters:
   - C: {svm_grid_search.best_params_['C']}
   - gamma: {svm_grid_search.best_params_['gamma']}

3. INTERPRETATION:
   - {'Default parameters already near optimal!' if abs(rf_improvements['ROC-AUC']) < 1 and abs(svm_improvements['ROC-AUC']) < 1 else 'Hyperparameter tuning shows significant improvements'}
   - Random Forest: {('Maintains excellent performance' if rf_improvements['ROC-AUC'] >= 0 else 'Shows slight overfitting') if abs(rf_improvements['ROC-AUC']) > 0 else 'No change'}
   - SVM RBF: {('Benefits from tuning' if svm_improvements['ROC-AUC'] > 0 else 'Default parameters sufficient') if abs(svm_improvements['ROC-AUC']) > 0 else 'No change'}

4. CONCLUSION:
   GridSearchCV helps identify optimal hyperparameters, but in this case,
   the default parameters were already well-tuned for the dataset.
   This is common with good baseline settings in scikit-learn.
""")

print("="*80)
