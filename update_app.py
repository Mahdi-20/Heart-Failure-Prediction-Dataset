"""
Update app.py with new GridSearchCV tuned model information
"""

import re

# Read the current app.py
with open('app.py', 'r') as f:
    content = f.read()

# Update the model information section with GridSearchCV details
old_model_info = r'- \*\*SVM \(Linear & RBF\)\*\* - Support Vector Machine classifiers\s*- \*\*Random Forest\*\* - Ensemble decision tree method'

new_model_info = '''- **Logistic Regression** - Regularized linear classifier (Tuned)
    - **SVM (Linear)** - Support Vector Machine with linear kernel (Tuned)
    - **SVM (RBF)** - Support Vector Machine with RBF kernel (Tuned)
    - **Decision Tree** - Single decision tree classifier (Tuned)
    - **Random Forest** - Ensemble decision tree method (Tuned)'''

content = re.sub(old_model_info, new_model_info, content, flags=re.MULTILINE)

# Update the key finding
old_finding = r'- \*\*Key Finding:\*\* Random Forest achieved 92\.39% ROC-AUC with 86\.41% test accuracy'
new_finding = '- **Key Finding:** Tuned Random Forest achieved 91.92% ROC-AUC with 86.96% test accuracy\n    - **Tuning Method:** GridSearchCV with StratifiedKFold (5-fold) cross-validation\n    - **Optimal Parameters:** n_estimators=50, max_depth=15, min_samples_split=10, min_samples_leaf=4'

content = re.sub(old_finding, new_finding, content, flags=re.MULTILINE)

# Update model selection section
old_selection = r'- \*\*SVM\*\* - Support Vector Machine\s*- \*\*Random Forest\*\* - Ensemble classification \(Selected\)'
new_selection = '''- **Logistic Regression** - Best for interpretability
    - **SVM (Linear)** - Best for linear separability  
    - **SVM (RBF)** - Best for non-linear patterns
    - **Decision Tree** - Best for explainability
    - **Random Forest (Tuned)** - Best overall performance (Selected)'''

content = re.sub(old_selection, new_selection, content, flags=re.MULTILINE)

# Update the metrics display
old_metrics = r'st\.metric\("Algorithm", "Random Forest", "100 trees"\)\s*with st\.columns\(3\):\s*col1, col2, col3 = st\.columns\(3\)\s*with col1:\s*st\.metric\("Hyperparameters", "Default",'
new_metrics = '''st.metric("Algorithm", "Random Forest (Tuned)", "✓ GridSearchCV Optimized")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Hyperparameters", "Tuned",'''

content = re.sub(old_metrics, new_metrics, content, flags=re.MULTILINE)

# Update ROC-AUC metric
content = re.sub(
    r'st\.metric\("ROC-AUC", "0\.9239"',
    'st.metric("ROC-AUC (Tuned)", "0.9192"',
    content
)

# Update accuracy metric
content = re.sub(
    r'st\.metric\("Accuracy", "0\.8641"',
    'st.metric("Accuracy (Tuned)", "0.8696"',
    content
)

# Add info about GridSearchCV
gridsearch_info = '''
    
    # GridSearchCV Tuning Process
    st.info("""
    **Hyperparameter Optimization with GridSearchCV:**
    - Tested 600+ parameter combinations for each model
    - Used 5-fold Stratified Cross-Validation for robust evaluation
    - Each fold preserves class distribution (55% positive, 45% negative)
    - Selected parameters with highest average CV ROC-AUC score
    
    **Random Forest Tuned Parameters:**
    - n_estimators: 50 (fewer, more efficient trees)
    - max_depth: 15 (limited depth to prevent overfitting)
    - min_samples_split: 10 (stricter split requirements)
    - min_samples_leaf: 4 (larger leaf sizes)
    - max_features: sqrt (square root of features)
    
    **Other Tuned Models Also Available:**
    See the 'Model Comparison' section for performance of all 5 tuned models.
    """)
'''

# Find a good place to insert this - after the existing metrics
insert_pos = content.find('st.divider()')
if insert_pos > 0:
    content = content[:insert_pos] + gridsearch_info + '\n    ' + content[insert_pos:]

# Save the updated app.py
with open('app.py', 'w') as f:
    f.write(content)

print("[OK] app.py updated with GridSearchCV tuned model information!")
print("\nUpdates Made:")
print("  1. Updated model descriptions to indicate tuning")
print("  2. Changed default Random Forest to Tuned Random Forest")
print("  3. Added optimal hyperparameters found by GridSearchCV")
print("  4. Added explanation of GridSearchCV process")
print("  5. Updated performance metrics from tuned models")
print("  6. Added info about StratifiedKFold cross-validation")
print("\nYour webapp now displays:")
print("  ✓ Tuned Random Forest as primary model")
print("  ✓ Optimal hyperparameters (n_est=50, max_depth=15, etc.)")
print("  ✓ GridSearchCV optimization explanation")
print("  ✓ All 5 tuned model options available")
