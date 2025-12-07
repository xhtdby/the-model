"""
Generate ROC-AUC curve for the trained stroke prediction model.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score
from src.data_ingestion import load_stroke_data
from src.preprocessing_pipeline import split_data

# Load data
print("Loading data...")
df = load_stroke_data('healthcare-dataset-stroke-data.csv')
X_train, X_test, y_train, y_test = split_data(df)

# Load trained model
print("Loading trained model...")
model = joblib.load('models/stroke_xgboost_model.joblib')

# Get predictions
print("Generating predictions...")
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Plot ROC curve
ax.plot(fpr, tpr, color='#2E86AB', linewidth=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')

# Plot diagonal (random classifier)
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, label='Random Classifier (AUC = 0.5000)')

# Styling
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=13, fontweight='bold')
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=13, fontweight='bold')
ax.set_title('ROC Curve - Stroke Prediction Model', fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])

# Add text box with performance metrics
textstr = f'Test Set Performance\n' \
          f'─────────────────\n' \
          f'ROC-AUC: {roc_auc:.4f}\n' \
          f'Samples: {len(y_test)}\n' \
          f'Stroke Cases: {y_test.sum()}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.55, 0.25, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, family='monospace')

plt.tight_layout()

# Save figure
output_path = Path('figures/evaluation/roc_curve.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ ROC curve saved to: {output_path}")

# Also save to main figures directory
main_output = Path('figures/roc_curve.png')
plt.savefig(main_output, dpi=300, bbox_inches='tight')
print(f"✓ ROC curve also saved to: {main_output}")

plt.show()

print(f"\nROC-AUC Score: {roc_auc:.4f}")
