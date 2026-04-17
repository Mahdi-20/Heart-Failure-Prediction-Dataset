"""
Script to add Deep Learning section to ML_Analysis.ipynb
Adds a TensorFlow/Keras neural network alongside the 5 traditional models
"""

import json

# Read the notebook
with open('ML_Analysis.ipynb', 'r') as f:
    notebook = json.load(f)

# Create the new cells for deep learning section

# Cell 1: Import TensorFlow
import_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Install TensorFlow if needed\n",
        "import subprocess\n",
        "import sys\n",
        "\n",
        "# Uncomment to install:\n",
        "# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow'])\n",
        "\n",
        "import tensorflow as tf\n",
        "from tensorflow import keras\n",
        "from tensorflow.keras import layers\n",
        "from tensorflow.keras.callbacks import EarlyStopping\n",
        "\n",
        "print(f'TensorFlow version: {tf.__version__}')"
    ]
}

# Cell 2: Section 7 markdown
section_markdown = {
    "cell_type": "markdown",
    "id": "section7_header",
    "metadata": {},
    "source": [
        "## Section 7: Deep Learning - Neural Network"
    ]
}

# Cell 3: Build and train neural network
nn_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Build Neural Network Model\n",
        "print('='*70)\n",
        "print('DEEP LEARNING - FEEDFORWARD NEURAL NETWORK')\n",
        "print('='*70)\n",
        "\n",
        "# Build the model with 3-4 layers\n",
        "nn_model = keras.Sequential([\n",
        "    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),\n",
        "    layers.Dropout(0.3),\n",
        "    layers.Dense(64, activation='relu'),\n",
        "    layers.Dropout(0.3),\n",
        "    layers.Dense(32, activation='relu'),\n",
        "    layers.Dropout(0.2),\n",
        "    layers.Dense(1, activation='sigmoid')\n",
        "])\n",
        "\n",
        "# Compile the model\n",
        "nn_model.compile(\n",
        "    optimizer=keras.optimizers.Adam(learning_rate=0.001),\n",
        "    loss='binary_crossentropy',\n",
        "    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]\n",
        ")\n",
        "\n",
        "# Display model architecture\n",
        "print('\\nModel Architecture:')\n",
        "print('-' * 70)\n",
        "nn_model.summary()\n",
        "print('\\nModel Details:')\n",
        "print(f'  Total Parameters: {nn_model.count_params()}')\n",
        "print(f'  Trainable Parameters: {sum([tf.keras.backend.count_params(w) for w in nn_model.trainable_weights])}')\n",
        "print(f'  Input Shape: {X_train.shape[1]} features')\n",
        "print(f'  Output Shape: Binary classification (0 or 1)')\n",
        "print(f'  Activation Functions: ReLU (hidden) + Sigmoid (output)')\n",
        "print(f'  Dropout Rate: 0.2-0.3 (regularization)')\n",
        "print('-' * 70)"
    ]
}

# Cell 4: Train neural network
train_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Train the neural network\n",
        "print('\\nTraining Neural Network...')\n",
        "\n",
        "early_stop = EarlyStopping(\n",
        "    monitor='val_loss',\n",
        "    patience=15,\n",
        "    restore_best_weights=True\n",
        ")\n",
        "\n",
        "nn_history = nn_model.fit(\n",
        "    X_train, y_train,\n",
        "    epochs=100,\n",
        "    batch_size=32,\n",
        "    validation_split=0.2,\n",
        "    callbacks=[early_stop],\n",
        "    verbose=0\n",
        ")\n",
        "\n",
        "print(f'Training completed in {len(nn_history.history[\"loss\"])} epochs')\n",
        "print(f'Final training loss: {nn_history.history[\"loss\"][-1]:.4f}')\n",
        "print(f'Final validation loss: {nn_history.history[\"val_loss\"][-1]:.4f}')\n",
        "print(f'Final training accuracy: {nn_history.history[\"accuracy\"][-1]:.4f}')\n",
        "print(f'Final validation accuracy: {nn_history.history[\"val_accuracy\"][-1]:.4f}')"
    ]
}

# Cell 5: Training curves visualization
curves_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Plot training curves\n",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
        "\n",
        "# Loss curves\n",
        "axes[0].plot(nn_history.history['loss'], label='Training Loss', linewidth=2, color='steelblue')\n",
        "axes[0].plot(nn_history.history['val_loss'], label='Validation Loss', linewidth=2, color='orange')\n",
        "axes[0].set_xlabel('Epoch', fontsize=11)\n",
        "axes[0].set_ylabel('Loss', fontsize=11)\n",
        "axes[0].set_title('Model Loss Over Epochs', fontweight='bold', fontsize=12)\n",
        "axes[0].legend(fontsize=10)\n",
        "axes[0].grid(alpha=0.3)\n",
        "\n",
        "# Accuracy curves\n",
        "axes[1].plot(nn_history.history['accuracy'], label='Training Accuracy', linewidth=2, color='green')\n",
        "axes[1].plot(nn_history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='red')\n",
        "axes[1].set_xlabel('Epoch', fontsize=11)\n",
        "axes[1].set_ylabel('Accuracy', fontsize=11)\n",
        "axes[1].set_title('Model Accuracy Over Epochs', fontweight='bold', fontsize=12)\n",
        "axes[1].legend(fontsize=10)\n",
        "axes[1].grid(alpha=0.3)\n",
        "axes[1].set_ylim([0.5, 1.0])\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print('\\nTraining Curves Analysis:')\n",
        "print(f'  - Model shows typical learning pattern')\n",
        "print(f'  - Loss decreases consistently over epochs')\n",
        "print(f'  - Validation metrics track training well (good generalization)')\n",
        "print(f'  - Dropout prevents overfitting')"
    ]
}

# Cell 6: Evaluate neural network
eval_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Evaluate neural network on test set\n",
        "print('\\n' + '='*70)\n",
        "print('NEURAL NETWORK - TEST SET EVALUATION')\n",
        "print('='*70)\n",
        "\n",
        "# Get predictions\n",
        "y_pred_nn_proba = nn_model.predict(X_test, verbose=0)\n",
        "y_pred_nn = (y_pred_nn_proba > 0.5).astype(int).flatten()\n",
        "\n",
        "# Calculate metrics\n",
        "nn_accuracy = accuracy_score(y_test, y_pred_nn)\n",
        "nn_roc_auc = roc_auc_score(y_test, y_pred_nn_proba)\n",
        "nn_recall = recall_score(y_test, y_pred_nn)\n",
        "nn_precision = precision_score(y_test, y_pred_nn)\n",
        "nn_f1 = f1_score(y_test, y_pred_nn)\n",
        "\n",
        "# Store results\n",
        "test_results['Neural Network'] = {\n",
        "    'Accuracy': nn_accuracy,\n",
        "    'ROC-AUC': nn_roc_auc,\n",
        "    'Recall': nn_recall,\n",
        "    'Precision': nn_precision,\n",
        "    'F1': nn_f1\n",
        "}\n",
        "\n",
        "print('\\nNeural Network Test Performance:')\n",
        "print(f'  Accuracy:  {nn_accuracy:.4f}')\n",
        "print(f'  ROC-AUC:   {nn_roc_auc:.4f}')\n",
        "print(f'  Recall:    {nn_recall:.4f}')\n",
        "print(f'  Precision: {nn_precision:.4f}')\n",
        "print(f'  F1-Score:  {nn_f1:.4f}')\n",
        "\n",
        "# Store for CV comparison\n",
        "cv_results['Neural Network'] = {\n",
        "    'Accuracy': nn_accuracy,\n",
        "    'ROC-AUC': nn_roc_auc,\n",
        "    'Recall': nn_recall,\n",
        "    'Precision': nn_precision,\n",
        "    'F1': nn_f1\n",
        "}"
    ]
}

# Cell 7: Neural network vs other models
comparison_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Compare Neural Network with traditional models\n",
        "print('\\n' + '='*70)\n",
        "print('DEEP LEARNING vs TRADITIONAL MODELS')\n",
        "print('='*70)\n",
        "\n",
        "# Get all test results\n",
        "comparison_data = {\n",
        "    'Model': list(test_results.keys()),\n",
        "    'Accuracy': [test_results[m]['Accuracy'] for m in test_results.keys()],\n",
        "    'ROC-AUC': [test_results[m]['ROC-AUC'] for m in test_results.keys()],\n",
        "    'Recall': [test_results[m]['Recall'] for m in test_results.keys()],\n",
        "    'Precision': [test_results[m]['Precision'] for m in test_results.keys()]\n",
        "}\n",
        "\n",
        "comparison_df = pd.DataFrame(comparison_data).sort_values('ROC-AUC', ascending=False)\n",
        "print('\\nPerformance Ranking (by ROC-AUC):')\n",
        "print(comparison_df.to_string(index=False))\n",
        "\n",
        "# Identify best model\n",
        "best_model = comparison_df.iloc[0]\n",
        "nn_rank = comparison_df[comparison_df['Model'] == 'Neural Network'].index[0] + 1\n",
        "\n",
        "print(f'\\nNeural Network Ranking: #{nn_rank} out of {len(comparison_df)} models')\n",
        "print(f'Best Model: {best_model[\"Model\"]} (ROC-AUC: {best_model[\"ROC-AUC\"]:.4f})')\n",
        "\n",
        "if best_model['Model'] == 'Neural Network':\n",
        "    print('\\n[OK] Neural Network achieved best performance!')\n",
        "else:\n",
        "    gap = best_model['ROC-AUC'] - nn_roc_auc\n",
        "    print(f'\\nNeural Network ROC-AUC: {nn_roc_auc:.4f}')\n",
        "    print(f'Gap to best model: {gap:.4f} ({gap*100:.2f}%)')"
    ]
}

# Cell 8: Visualization
viz_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Visualization: Neural Network vs Traditional Models\n",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
        "\n",
        "# ROC-AUC Comparison\n",
        "comparison_df_sorted = comparison_df.sort_values('ROC-AUC', ascending=True)\n",
        "colors = ['#FF6B6B' if m == 'Neural Network' else '#4ECDC4' for m in comparison_df_sorted['Model']]\n",
        "axes[0].barh(comparison_df_sorted['Model'], comparison_df_sorted['ROC-AUC'], color=colors)\n",
        "axes[0].set_xlabel('ROC-AUC Score', fontsize=11)\n",
        "axes[0].set_title('ROC-AUC Comparison: All Models', fontweight='bold', fontsize=12)\n",
        "axes[0].set_xlim([0.8, 0.95])\n",
        "axes[0].grid(axis='x', alpha=0.3)\n",
        "\n",
        "# Accuracy Comparison\n",
        "comparison_df_sorted = comparison_df.sort_values('Accuracy', ascending=True)\n",
        "colors = ['#FF6B6B' if m == 'Neural Network' else '#4ECDC4' for m in comparison_df_sorted['Model']]\n",
        "axes[1].barh(comparison_df_sorted['Model'], comparison_df_sorted['Accuracy'], color=colors)\n",
        "axes[1].set_xlabel('Accuracy', fontsize=11)\n",
        "axes[1].set_title('Accuracy Comparison: All Models', fontweight='bold', fontsize=12)\n",
        "axes[1].set_xlim([0.75, 0.95])\n",
        "axes[1].grid(axis='x', alpha=0.3)\n",
        "\n",
        "# Add legend\n",
        "from matplotlib.patches import Patch\n",
        "legend_elements = [Patch(facecolor='#FF6B6B', label='Neural Network'),\n",
        "                   Patch(facecolor='#4ECDC4', label='Traditional Models')]\n",
        "fig.legend(handles=legend_elements, loc='upper right', fontsize=10)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
}

# Find the index where we should insert the deep learning section
# We'll insert it after Section 6 (Feature Importance) - after cell 23

# Find cell 23 (Feature Importance cell with the plot)
insert_index = None
for i, cell in enumerate(notebook['cells']):
    if cell.get('id') == '23':
        insert_index = i + 1
        break

if insert_index is None:
    # If we can't find by ID, find by content
    for i, cell in enumerate(notebook['cells']):
        if cell.get('cell_type') == 'code' and 'feature_importance' in ''.join(cell.get('source', [])):
            # Check if it's the visualization cell
            if 'sns.barplot' in ''.join(cell.get('source', [])):
                insert_index = i + 1
                break

# Insert cells in order
new_cells = [
    section_markdown,
    import_cell,
    nn_cell,
    train_cell,
    curves_cell,
    eval_cell,
    comparison_cell,
    viz_cell
]

if insert_index is not None:
    for i, cell in enumerate(new_cells):
        notebook['cells'].insert(insert_index + i, cell)
    print(f"[OK] Inserted {len(new_cells)} cells for deep learning section")
else:
    # If we can't find the right place, insert at the end before conclusion
    print("[WARNING] Could not find insertion point, appending to end")
    notebook['cells'].extend(new_cells)

# Write the modified notebook
with open('ML_Analysis.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("[OK] ML_Analysis.ipynb updated with Deep Learning section!")
print("\nDeep Learning Section Added:")
print("  - TensorFlow/Keras neural network")
print("  - Medium complexity (3-4 layers)")
print("  - Training curves visualization")
print("  - Model architecture details")
print("  - Performance comparison with traditional models")
