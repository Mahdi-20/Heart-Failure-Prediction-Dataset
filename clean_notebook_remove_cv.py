"""
Remove the redundant 5-Fold CV section from ML_Analysis.ipynb
Keep only GridSearchCV (which uses StratifiedKFold internally)
"""

import json

# Read the notebook
with open('ML_Analysis.ipynb', 'r') as f:
    notebook = json.load(f)

# Find and remove Section 3 (5-Fold CV) cells
cells_to_remove_indices = []

for i, cell in enumerate(notebook['cells']):
    source_text = ''.join(cell.get('source', []))

    # Identify cells in the old Section 3 (5-Fold CV)
    if any(text in source_text for text in [
        '5-FOLD STRATIFIED CROSS-VALIDATION RESULTS',
        'cross_validate(model, X, y, cv=cv, scoring=scoring)',
        'CV Results Visualization',
        'cv_df = pd.DataFrame(cv_results)',
        'Section 3: Model Comparison'
    ]):
        cells_to_remove_indices.append(i)

print(f"Found {len(cells_to_remove_indices)} cells to remove from Section 3")

# Remove cells in reverse order to maintain indices
for i in reversed(cells_to_remove_indices):
    removed_cell_text = ''.join(notebook['cells'][i].get('source', []))
    print(f"  Removing: {removed_cell_text[:60]}...")
    notebook['cells'].pop(i)

# Update Section headers - change Section 3.5 to Section 3
for cell in notebook['cells']:
    source = cell.get('source', [])
    if isinstance(source, list):
        cell['source'] = [
            line.replace('## Section 3.5:', '## Section 3:')
            .replace('## Section 4:', '## Section 4:')  # Keep as is
            .replace('## Section 5:', '## Section 4:')  # Shift down
            .replace('## Section 6:', '## Section 5:')  # Shift down
            .replace('## Section 7:', '## Section 6:')  # Shift down
            .replace('## Section 8:', '## Section 7:')  # Shift down
            .replace('## Section 9:', '## Section 8:')  # Shift down
            for line in source
        ]

# Add a note at the beginning of GridSearchCV section
for i, cell in enumerate(notebook['cells']):
    source_text = ''.join(cell.get('source', []))
    if 'Hyperparameter Tuning with GridSearchCV' in source_text and cell.get('cell_type') == 'markdown':
        # Update the markdown to explain that we're using StratifiedKFold internally
        cell['source'] = [
            "## Section 3: Hyperparameter Tuning with GridSearchCV\n",
            "\n",
            "Systematic hyperparameter optimization for all ML models using GridSearchCV.\n",
            "GridSearchCV internally uses **StratifiedKFold** (5-fold, stratified) to ensure\n",
            "robust parameter selection while preserving class distribution in each fold.\n",
            "\n",
            "**Why StratifiedKFold in GridSearchCV?**\n",
            "- Ensures each CV fold has the same class ratio as the original dataset\n",
            "- Prevents imbalanced folds that could skew hyperparameter selection\n",
            "- Provides reliable ROC-AUC estimates for comparing parameter combinations\n",
            "- Maintains reproducibility with `random_state` parameter\n",
            "\n",
            "**GridSearchCV Workflow:**\n",
            "1. Define parameter grid for each model\n",
            "2. For each parameter combination: Train on 4 folds, validate on 1 fold (5 iterations)\n",
            "3. Calculate average CV score across 5 folds\n",
            "4. Select parameters with highest average score\n",
            "5. Evaluate final tuned model on test set"
        ]
        break

# Save modified notebook
with open('ML_Analysis.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("\n" + "="*80)
print("[OK] ML_Analysis.ipynb cleaned up!")
print("="*80)
print("\nChanges Made:")
print("  1. REMOVED: Section 3 (5-Fold Stratified CV with cross_validate)")
print("  2. KEPT: GridSearchCV sections (which use StratifiedKFold internally)")
print("  3. RENUMBERED: Remaining sections")
print("  4. UPDATED: Section 3 header to explain StratifiedKFold usage")
print("\nNew Structure:")
print("  Section 1: Load & Explore Data")
print("  Section 2: EDA")
print("  Section 3: Hyperparameter Tuning with GridSearchCV (using StratifiedKFold)")
print("  Section 4: Best Model Evaluation (Train-Test Split)")
print("  Section 5: ROC Curves")
print("  Section 6: Feature Importance")
print("  Section 7: Deep Learning")
print("  Section 8: Conclusion")
print("\nBenefit: No redundancy - GridSearchCV handles both CV and hyperparameter tuning!")
print("="*80)
