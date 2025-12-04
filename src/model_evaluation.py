"""
Model Evaluation, Fairness Audit, and Explainability Module
============================================================

This module provides comprehensive evaluation for the stroke prediction model:

1. Performance Metrics
   - Classification Report
   - Confusion Matrix
   - Precision-Recall Curve
   - Recall (Stroke Detection Rate)

2. Fairness Audit
   - Equalized Odds: TPR comparison (Male vs Female)
   - Demographic Parity: Positive Selection Rate (Urban vs Rural)
   - Disparity warnings when > 10%

3. Explainability (SHAP)
   - Global feature importance (Summary Plot)
   - Dependence Plot (BMI vs Age interaction)
   - Analysis of smoking_status_Unknown impact

Author: Healthcare AI Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay
)
from sklearn.pipeline import Pipeline


# =============================================================================
# Configuration
# =============================================================================

# Fairness disparity threshold (10%)
FAIRNESS_DISPARITY_THRESHOLD = 0.10

# Output directories
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# =============================================================================
# Performance Metrics
# =============================================================================

def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dict: bool = False
) -> str | Dict[str, Any]:
    """
    Generate detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_dict: If True, return dict instead of string
        
    Returns:
        Classification report as string or dict
    """
    target_names = ['No Stroke', 'Stroke']
    
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=target_names,
        output_dict=output_dict,
        zero_division=0
    )
    
    return report


def generate_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Generate and visualize confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Optional path to save figure
        
    Returns:
        Confusion matrix array
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['No Stroke', 'Stroke']
    )
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    
    ax.set_title('Confusion Matrix - Stroke Prediction Model', fontsize=14, fontweight='bold')
    
    # Add annotations
    tn, fp, fn, tp = cm.ravel()
    annotation_text = (
        f"TN={tn} (Correct No-Stroke) | FP={fp} (False Alarm)\n"
        f"FN={fn} (Missed Stroke)     | TP={tp} (Detected Stroke)"
    )
    plt.figtext(0.5, 0.02, annotation_text, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()
    
    return cm


def generate_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Precision-Recall curve for stroke prediction.
    
    Critical for Imbalanced Data:
    =============================
    PR curves are more informative than ROC curves when:
    - Class imbalance is severe (4.87% positive in our case)
    - The positive class (stroke) is the minority class of interest
    - We want to understand the precision-recall trade-off
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        save_path: Optional path to save figure
        
    Returns:
        Tuple of (precision, recall, thresholds)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Main PR curve
    disp = PrecisionRecallDisplay(
        precision=precision,
        recall=recall,
        average_precision=avg_precision,
        estimator_name='XGBoost Stroke Predictor'
    )
    disp.plot(ax=ax, color='darkorange', lw=2)
    
    # Add baseline (random classifier)
    positive_rate = y_true.mean()
    ax.axhline(y=positive_rate, color='navy', linestyle='--', lw=2, 
               label=f'Random Classifier (AP={positive_rate:.3f})')
    
    # Highlight key thresholds
    key_thresholds = [0.1, 0.3, 0.5, 0.7]
    for thresh in key_thresholds:
        idx = np.argmin(np.abs(thresholds - thresh))
        if idx < len(precision) - 1:
            ax.scatter(recall[idx], precision[idx], s=100, zorder=5)
            ax.annotate(f't={thresh}', (recall[idx], precision[idx]), 
                       textcoords="offset points", xytext=(5, 5), fontsize=9)
    
    ax.set_title('Precision-Recall Curve - Stroke Detection', fontsize=14, fontweight='bold')
    ax.set_xlabel('Recall (Stroke Detection Rate)', fontsize=12)
    ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add interpretation box
    interpretation = (
        f"Average Precision: {avg_precision:.3f}\n"
        f"At 80% Recall: Precision ≈ {precision[np.argmin(np.abs(recall - 0.8))]:.3f}\n"
        "Higher curve = Better model"
    )
    ax.text(0.02, 0.02, interpretation, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"PR curve saved to: {save_path}")
    
    plt.show()
    
    return precision, recall, thresholds


def calculate_recall_positive_class(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate recall (sensitivity) for the positive class (stroke).
    
    This is the STROKE DETECTION RATE - the most critical metric
    for a clinical screening tool.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Recall for positive class
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return recall


# =============================================================================
# Fairness Audit
# =============================================================================

def calculate_equalized_odds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_feature: np.ndarray,
    group_names: Tuple[str, str] = ('Group A', 'Group B')
) -> Dict[str, Any]:
    """
    Calculate Equalized Odds by comparing True Positive Rates across groups.
    
    Equalized Odds Definition:
    ==========================
    A classifier satisfies Equalized Odds if the True Positive Rate (TPR)
    and False Positive Rate (FPR) are equal across different groups.
    
    For stroke prediction, we focus on TPR (Recall) because:
    - Missing a stroke case (FN) is more dangerous than a false alarm (FP)
    - We want equal detection rates regardless of demographics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_feature: Array indicating group membership (e.g., gender)
        group_names: Names for the two groups
        
    Returns:
        Dict with TPR for each group and disparity metrics
    """
    unique_groups = np.unique(sensitive_feature)
    
    if len(unique_groups) != 2:
        raise ValueError(f"Expected 2 groups, got {len(unique_groups)}: {unique_groups}")
    
    results = {
        'group_names': group_names,
        'tpr': {},
        'fpr': {},
        'support': {}
    }
    
    for i, group in enumerate(unique_groups):
        mask = sensitive_feature == group
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]
        
        # Calculate TPR (Recall) for this group
        cm = confusion_matrix(y_true_group, y_pred_group, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        group_name = group_names[i] if i < len(group_names) else str(group)
        results['tpr'][group_name] = tpr
        results['fpr'][group_name] = fpr
        results['support'][group_name] = {
            'total': int(mask.sum()),
            'positive': int(y_true_group.sum())
        }
    
    # Calculate disparity
    tpr_values = list(results['tpr'].values())
    results['tpr_disparity'] = abs(tpr_values[0] - tpr_values[1])
    results['tpr_ratio'] = min(tpr_values) / max(tpr_values) if max(tpr_values) > 0 else 0
    
    fpr_values = list(results['fpr'].values())
    results['fpr_disparity'] = abs(fpr_values[0] - fpr_values[1])
    
    return results


def calculate_demographic_parity(
    y_pred: np.ndarray,
    sensitive_feature: np.ndarray,
    group_names: Tuple[str, str] = ('Group A', 'Group B')
) -> Dict[str, Any]:
    """
    Calculate Demographic Parity by comparing Positive Selection Rates across groups.
    
    Demographic Parity Definition:
    ==============================
    A classifier satisfies Demographic Parity if the probability of being
    classified as positive is equal across different groups:
    
    P(Y_pred = 1 | Group = A) = P(Y_pred = 1 | Group = B)
    
    Note: Demographic Parity may conflict with accuracy if base rates differ
    between groups. It's one of several fairness metrics to consider.
    
    Args:
        y_pred: Predicted labels
        sensitive_feature: Array indicating group membership
        group_names: Names for the two groups
        
    Returns:
        Dict with selection rates for each group and disparity metrics
    """
    unique_groups = np.unique(sensitive_feature)
    
    if len(unique_groups) != 2:
        raise ValueError(f"Expected 2 groups, got {len(unique_groups)}: {unique_groups}")
    
    results = {
        'group_names': group_names,
        'selection_rate': {},
        'support': {}
    }
    
    for i, group in enumerate(unique_groups):
        mask = sensitive_feature == group
        y_pred_group = y_pred[mask]
        
        selection_rate = y_pred_group.mean()
        
        group_name = group_names[i] if i < len(group_names) else str(group)
        results['selection_rate'][group_name] = selection_rate
        results['support'][group_name] = int(mask.sum())
    
    # Calculate disparity
    rates = list(results['selection_rate'].values())
    results['disparity'] = abs(rates[0] - rates[1])
    results['ratio'] = min(rates) / max(rates) if max(rates) > 0 else 0
    
    return results


def run_fairness_audit(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Run comprehensive fairness audit on the model.
    
    Checks:
    1. Equalized Odds: Male vs Female TPR
    2. Demographic Parity: Urban vs Rural selection rates
    
    Args:
        model: Trained sklearn Pipeline
        X_test: Test features
        y_test: True test labels
        threshold: Classification threshold
        
    Returns:
        Dict with all fairness metrics and warnings
    """
    print("\n" + "=" * 70)
    print("FAIRNESS AUDIT")
    print("=" * 70)
    
    # Get predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    results = {
        'threshold': threshold,
        'warnings': []
    }
    
    # -------------------------------------------------------------------------
    # 1. Equalized Odds: Male vs Female
    # -------------------------------------------------------------------------
    print("\n[1] Equalized Odds Analysis: Gender")
    print("-" * 50)
    
    # Extract gender from test data
    gender_feature = X_test['gender'].values
    gender_map = {'Male': 'Male', 'Female': 'Female', 'Other': 'Other'}
    
    # Filter to Male vs Female only (exclude 'Other' for binary comparison)
    binary_mask = np.isin(gender_feature, ['Male', 'Female'])
    
    if binary_mask.sum() > 0:
        eo_results = calculate_equalized_odds(
            y_true=y_test[binary_mask],
            y_pred=y_pred[binary_mask],
            sensitive_feature=gender_feature[binary_mask],
            group_names=('Female', 'Male')
        )
        results['equalized_odds_gender'] = eo_results
        
        print(f"\nTrue Positive Rate (Stroke Detection) by Gender:")
        for group, tpr in eo_results['tpr'].items():
            support = eo_results['support'][group]
            print(f"  {group:10s}: TPR = {tpr:.4f} "
                  f"(n={support['total']}, positive={support['positive']})")
        
        print(f"\nTPR Disparity: {eo_results['tpr_disparity']:.4f}")
        print(f"TPR Ratio (min/max): {eo_results['tpr_ratio']:.4f}")
        
        if eo_results['tpr_disparity'] > FAIRNESS_DISPARITY_THRESHOLD:
            warning = (
                f"⚠️  WARNING: Gender TPR disparity ({eo_results['tpr_disparity']:.2%}) "
                f"exceeds {FAIRNESS_DISPARITY_THRESHOLD:.0%} threshold!"
            )
            print(f"\n{warning}")
            results['warnings'].append(warning)
        else:
            print(f"\n✓ Gender TPR disparity within acceptable range (<{FAIRNESS_DISPARITY_THRESHOLD:.0%})")
    
    # -------------------------------------------------------------------------
    # 2. Demographic Parity: Urban vs Rural
    # -------------------------------------------------------------------------
    print("\n[2] Demographic Parity Analysis: Residence Type")
    print("-" * 50)
    
    residence_feature = X_test['Residence_type'].values
    
    dp_results = calculate_demographic_parity(
        y_pred=y_pred,
        sensitive_feature=residence_feature,
        group_names=('Rural', 'Urban')
    )
    results['demographic_parity_residence'] = dp_results
    
    print(f"\nPositive Selection Rate by Residence Type:")
    for group, rate in dp_results['selection_rate'].items():
        support = dp_results['support'][group]
        print(f"  {group:10s}: Selection Rate = {rate:.4f} (n={support})")
    
    print(f"\nSelection Rate Disparity: {dp_results['disparity']:.4f}")
    print(f"Selection Rate Ratio (min/max): {dp_results['ratio']:.4f}")
    
    if dp_results['disparity'] > FAIRNESS_DISPARITY_THRESHOLD:
        warning = (
            f"⚠️  WARNING: Residence type selection rate disparity ({dp_results['disparity']:.2%}) "
            f"exceeds {FAIRNESS_DISPARITY_THRESHOLD:.0%} threshold!"
        )
        print(f"\n{warning}")
        results['warnings'].append(warning)
    else:
        print(f"\n✓ Residence type disparity within acceptable range (<{FAIRNESS_DISPARITY_THRESHOLD:.0%})")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("FAIRNESS AUDIT SUMMARY")
    print("-" * 50)
    
    if results['warnings']:
        print(f"\n⚠️  Found {len(results['warnings'])} fairness concern(s):")
        for i, warning in enumerate(results['warnings'], 1):
            print(f"   {i}. {warning}")
    else:
        print("\n✓ Model passes all fairness checks (disparity < 10%)")
    
    return results


# =============================================================================
# SHAP Explainability
# =============================================================================

def generate_shap_analysis(
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    sample_size: int = 500,
    save_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Generate SHAP-based explainability analysis.
    
    SHAP (SHapley Additive exPlanations):
    =====================================
    SHAP values provide a unified measure of feature importance that:
    - Shows how each feature contributes to individual predictions
    - Is based on game theory (Shapley values)
    - Provides both local and global interpretability
    - Handles feature interactions properly
    
    Generates:
    1. Summary Plot (global importance)
    2. Dependence Plot (BMI vs Age interaction)
    3. Analysis of smoking_status_Unknown
    
    Args:
        model: Trained sklearn Pipeline
        X_train: Training features (for background)
        X_test: Test features (for explanation)
        sample_size: Number of samples for SHAP analysis
        save_dir: Directory to save plots
        
    Returns:
        Dict with SHAP values and analysis results
    """
    import shap
    
    print("\n" + "=" * 70)
    print("SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 70)
    
    if save_dir is None:
        save_dir = FIGURES_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    results = {}
    
    # -------------------------------------------------------------------------
    # Prepare data
    # -------------------------------------------------------------------------
    print("\n[1] Preparing data for SHAP analysis...")
    
    # Get preprocessor and model from pipeline
    preprocessor = model.named_steps['preprocessor']
    xgb_model = model.named_steps['model']
    
    # Transform data
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Get feature names after transformation
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        # Fallback for older sklearn versions
        from .preprocessing_pipeline import get_feature_names
        feature_names = get_feature_names(preprocessor)
    
    # Convert to DataFrame for better visualization
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)
    
    # Sample if dataset is large
    if len(X_test_df) > sample_size:
        sample_idx = np.random.choice(len(X_test_df), sample_size, replace=False)
        X_test_sample = X_test_df.iloc[sample_idx]
    else:
        X_test_sample = X_test_df
    
    print(f"  Using {len(X_test_sample)} samples for SHAP analysis")
    print(f"  Number of features: {len(feature_names)}")
    
    # -------------------------------------------------------------------------
    # Create SHAP explainer
    # -------------------------------------------------------------------------
    print("\n[2] Creating SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(xgb_model)
    
    print("\n[3] Computing SHAP values...")
    shap_values = explainer.shap_values(X_test_sample)
    
    results['shap_values'] = shap_values
    results['feature_names'] = list(feature_names)
    results['expected_value'] = explainer.expected_value
    
    # -------------------------------------------------------------------------
    # 1. Summary Plot (Global Feature Importance)
    # -------------------------------------------------------------------------
    print("\n[4] Generating Summary Plot (Global Feature Importance)...")
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, 
        X_test_sample,
        feature_names=feature_names,
        show=False,
        max_display=15
    )
    plt.title('SHAP Summary Plot - Global Feature Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    summary_path = save_dir / 'shap_summary_plot.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"  Summary plot saved to: {summary_path}")
    plt.show()
    
    # -------------------------------------------------------------------------
    # 2. Dependence Plot (BMI vs Age interaction)
    # -------------------------------------------------------------------------
    print("\n[5] Generating Dependence Plot (BMI vs Age)...")
    
    # Find indices for BMI and Age
    bmi_idx = None
    age_idx = None
    for i, name in enumerate(feature_names):
        if 'bmi' in name.lower():
            bmi_idx = i
        if 'age' in name.lower():
            age_idx = i
    
    if bmi_idx is not None and age_idx is not None:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            bmi_idx,
            shap_values,
            X_test_sample,
            feature_names=feature_names,
            interaction_index=age_idx,
            show=False
        )
        plt.title('SHAP Dependence Plot: BMI (colored by Age)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        dependence_path = save_dir / 'shap_dependence_bmi_age.png'
        plt.savefig(dependence_path, dpi=150, bbox_inches='tight')
        print(f"  Dependence plot saved to: {dependence_path}")
        plt.show()
        
        results['bmi_age_interaction'] = {
            'bmi_feature_idx': bmi_idx,
            'age_feature_idx': age_idx
        }
    else:
        print("  WARNING: Could not find BMI or Age features for dependence plot")
    
    # -------------------------------------------------------------------------
    # 3. Analysis of smoking_status_Unknown
    # -------------------------------------------------------------------------
    print("\n[6] Analyzing smoking_status_Unknown SHAP values...")
    
    # Find the smoking_status features
    # Note: With OneHotEncoder(drop='first'), 'Unknown' is the reference category
    # It's implicitly encoded when all other smoking columns are 0
    smoking_feature_indices = []
    smoking_feature_names_found = []
    for i, name in enumerate(feature_names):
        if 'smoking_status' in name.lower():
            smoking_feature_indices.append(i)
            smoking_feature_names_found.append(name)
    
    if smoking_feature_indices:
        print(f"\n  Found smoking features: {smoking_feature_names_found}")
        print("  Note: 'Unknown' is the reference category (all smoking columns = 0)")
        
        # Get SHAP values for all smoking features
        smoking_shap_matrix = shap_values[:, smoking_feature_indices]
        smoking_features_df = X_test_sample.iloc[:, smoking_feature_indices]
        
        # Identify samples where smoking_status = 'Unknown' 
        # (i.e., all smoking indicator columns are 0)
        unknown_mask = (smoking_features_df.sum(axis=1) == 0).values
        known_mask = ~unknown_mask
        
        # Total SHAP contribution from smoking features
        # For Unknown: all smoking features are 0, so their SHAP contribution is minimal
        # For Known: the active smoking feature contributes its SHAP value
        total_smoking_shap = smoking_shap_matrix.sum(axis=1)
        
        shap_when_unknown = total_smoking_shap[unknown_mask]
        shap_when_known = total_smoking_shap[known_mask]
        
        analysis = {
            'smoking_features': smoking_feature_names_found,
            'reference_category': 'Unknown (all smoking indicators = 0)',
            'mean_total_smoking_shap_unknown': float(np.mean(shap_when_unknown)) if len(shap_when_unknown) > 0 else None,
            'mean_total_smoking_shap_known': float(np.mean(shap_when_known)) if len(shap_when_known) > 0 else None,
            'n_unknown': int(unknown_mask.sum()),
            'n_known': int(known_mask.sum())
        }
        
        # Also analyze each smoking category separately
        analysis['per_category'] = {}
        for idx, name in zip(smoking_feature_indices, smoking_feature_names_found):
            feature_values = X_test_sample.iloc[:, idx].values
            active_mask = feature_values == 1
            if active_mask.sum() > 0:
                analysis['per_category'][name] = {
                    'mean_shap': float(np.mean(shap_values[active_mask, idx])),
                    'count': int(active_mask.sum())
                }
        
        results['smoking_unknown_analysis'] = analysis
        
        print(f"\n  Samples with Unknown smoking status: {analysis['n_unknown']}")
        print(f"  Samples with Known smoking status: {analysis['n_known']}")
        
        # The key insight: compare smoking SHAP contribution for Unknown vs Known
        print(f"\n  Mean total smoking SHAP (Unknown status): {analysis['mean_total_smoking_shap_unknown']:.4f}")
        print(f"  Mean total smoking SHAP (Known status):   {analysis['mean_total_smoking_shap_known']:.4f}")
        
        # For Unknown status: all smoking features are 0
        # The SHAP baseline assumes average feature values
        # If mean SHAP is positive when Unknown, it means having Unknown raises risk vs baseline
        # If mean SHAP is negative when Unknown, it means having Unknown lowers risk vs baseline
        
        if analysis['mean_total_smoking_shap_unknown'] is not None:
            # Compare to the overall mean SHAP for smoking
            overall_mean = np.mean(total_smoking_shap)
            unknown_vs_overall = analysis['mean_total_smoking_shap_unknown'] - overall_mean
            
            print(f"\n  Per-category analysis:")
            for cat_name, cat_data in analysis['per_category'].items():
                short_name = cat_name.replace('categorical__smoking_status_', '')
                direction = "↑ raises" if cat_data['mean_shap'] > 0 else "↓ lowers"
                print(f"    - {short_name}: {direction} risk (SHAP = {cat_data['mean_shap']:.4f}, n={cat_data['count']})")
            
            # Interpretation based on relative contribution
            relative_unknown_effect = analysis['mean_total_smoking_shap_unknown'] - analysis['mean_total_smoking_shap_known']
            
            if relative_unknown_effect > 0.01:
                direction_text = "RAISES"
                interpretation = (
                    "Having 'Unknown' smoking status is associated with HIGHER predicted stroke risk\n"
                    "compared to patients with known smoking status. This could indicate:\n"
                    "1. Patients who don't disclose smoking may have other correlated risk factors\n"
                    "2. Missing smoking data may be correlated with healthcare access issues\n"
                    "3. The model learned patterns associating incomplete records with higher risk"
                )
            elif relative_unknown_effect < -0.01:
                direction_text = "LOWERS"
                interpretation = (
                    "Having 'Unknown' smoking status is associated with LOWER predicted stroke risk\n"
                    "compared to patients with known smoking status (especially 'smokes'). This suggests:\n"
                    "1. Active smoking has a strong positive effect on stroke risk\n"
                    "2. Unknown status acts as a neutral baseline between smokers and non-smokers\n"
                    "3. The model treats missing data conservatively, not assuming worst case"
                )
            else:
                direction_text = "has MINIMAL EFFECT on"
                interpretation = (
                    "Having 'Unknown' smoking status has minimal differential effect on stroke risk.\n"
                    "The model treats unknown smoking status similarly to the population average."
                )
            
            print(f"\n  📊 FINDING: 'Unknown' smoking status {direction_text} predicted stroke risk")
            print(f"     (Relative effect vs known: {relative_unknown_effect:.4f})")
            print(f"\n  Interpretation:\n  {interpretation}")
        
        # Visualize distribution
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        if len(shap_when_unknown) > 0:
            plt.hist(shap_when_unknown, bins=30, alpha=0.7, color='gray', label='Unknown', edgecolor='black')
        if len(shap_when_known) > 0:
            plt.hist(shap_when_known, bins=30, alpha=0.7, color='steelblue', label='Known', edgecolor='black')
        plt.xlabel('Total Smoking SHAP Value')
        plt.ylabel('Frequency')
        plt.title('SHAP Distribution: Unknown vs Known Smoking Status')
        plt.legend()
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Baseline')
        
        plt.subplot(1, 2, 2)
        # Bar plot for each category
        categories = ['Unknown'] + [name.replace('categorical__smoking_status_', '') 
                                    for name in smoking_feature_names_found]
        values = [analysis['mean_total_smoking_shap_unknown']]
        for name in smoking_feature_names_found:
            if name in analysis['per_category']:
                values.append(analysis['per_category'][name]['mean_shap'])
            else:
                values.append(0)
        
        colors = ['gray'] + ['steelblue'] * len(smoking_feature_names_found)
        bars = plt.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        plt.ylabel('Mean SHAP Value')
        plt.title('Mean SHAP by Smoking Status Category')
        plt.xticks(rotation=15, ha='right')
        
        for bar, val in zip(bars, values):
            ypos = bar.get_height() + 0.002 if val >= 0 else bar.get_height() - 0.015
            plt.text(bar.get_x() + bar.get_width()/2, ypos,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        smoking_analysis_path = save_dir / 'shap_smoking_unknown_analysis.png'
        plt.savefig(smoking_analysis_path, dpi=150, bbox_inches='tight')
        print(f"\n  Analysis plot saved to: {smoking_analysis_path}")
        plt.show()
        
    else:
        print("  WARNING: Could not find any smoking_status features")
    
    return results


# =============================================================================
# Comprehensive Evaluation
# =============================================================================

def run_full_evaluation(
    model_path: str,
    data_path: str,
    threshold: float = 0.5,
    shap_sample_size: int = 500
) -> Dict[str, Any]:
    """
    Run complete model evaluation including performance, fairness, and explainability.
    
    Args:
        model_path: Path to saved model
        data_path: Path to dataset
        threshold: Classification threshold
        shap_sample_size: Sample size for SHAP analysis
        
    Returns:
        Dict with all evaluation results
    """
    print("=" * 70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 70)
    
    # Load model and data
    print("\n[Loading] Model and data...")
    model = joblib.load(model_path)
    
    from .data_ingestion import load_stroke_data
    from .preprocessing_pipeline import split_data
    
    df = load_stroke_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
    
    print(f"  Test set size: {len(X_test)}")
    print(f"  Positive class rate: {y_test.mean():.2%}")
    
    results = {}
    
    # -------------------------------------------------------------------------
    # 1. Performance Metrics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)
    
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Classification Report
    print("\n[1] Classification Report:")
    print("-" * 50)
    report = generate_classification_report(y_test.values, y_pred)
    print(report)
    results['classification_report'] = generate_classification_report(y_test.values, y_pred, output_dict=True)
    
    # Recall for positive class (Stroke Detection Rate)
    recall_positive = calculate_recall_positive_class(y_test.values, y_pred)
    print(f"\n📊 STROKE DETECTION RATE (Recall): {recall_positive:.4f} ({recall_positive:.1%})")
    results['stroke_detection_rate'] = recall_positive
    
    # Confusion Matrix
    print("\n[2] Confusion Matrix:")
    cm = generate_confusion_matrix(
        y_test.values, y_pred,
        save_path=str(FIGURES_DIR / 'confusion_matrix.png')
    )
    results['confusion_matrix'] = cm
    
    # Precision-Recall Curve
    print("\n[3] Precision-Recall Curve:")
    precision, recall, thresholds = generate_precision_recall_curve(
        y_test.values, y_proba,
        save_path=str(FIGURES_DIR / 'precision_recall_curve.png')
    )
    results['pr_curve'] = {
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds,
        'average_precision': average_precision_score(y_test.values, y_proba)
    }
    
    # -------------------------------------------------------------------------
    # 2. Fairness Audit
    # -------------------------------------------------------------------------
    fairness_results = run_fairness_audit(model, X_test, y_test.values, threshold)
    results['fairness'] = fairness_results
    
    # -------------------------------------------------------------------------
    # 3. SHAP Explainability
    # -------------------------------------------------------------------------
    shap_results = generate_shap_analysis(
        model, X_train, X_test,
        sample_size=shap_sample_size
    )
    results['shap'] = shap_results
    
    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    print(f"\n📊 Performance:")
    print(f"   - ROC-AUC: {roc_auc_score(y_test.values, y_proba):.4f}")
    print(f"   - Average Precision: {results['pr_curve']['average_precision']:.4f}")
    print(f"   - Stroke Detection Rate: {recall_positive:.1%}")
    
    print(f"\n⚖️  Fairness:")
    if fairness_results['warnings']:
        for warning in fairness_results['warnings']:
            print(f"   {warning}")
    else:
        print("   ✓ All fairness checks passed")
    
    print(f"\n🔍 Explainability:")
    if 'smoking_unknown_analysis' in shap_results:
        analysis = shap_results['smoking_unknown_analysis']
        if 'mean_total_smoking_shap_unknown' in analysis and analysis['mean_total_smoking_shap_unknown'] is not None:
            unknown_val = analysis['mean_total_smoking_shap_unknown']
            known_val = analysis.get('mean_total_smoking_shap_known', 0)
            diff = unknown_val - known_val if known_val else unknown_val
            if abs(diff) < 0.01:
                print(f"   - smoking_status_Unknown has minimal effect on predicted risk")
            else:
                direction = "raises" if diff > 0 else "lowers"
                print(f"   - smoking_status_Unknown {direction} predicted risk (vs known)")
    
    print(f"\n📁 Figures saved to: {FIGURES_DIR}")
    
    return results


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Paths
    project_root = Path(__file__).parent.parent
    model_path = project_root / "models" / "stroke_xgboost_model.joblib"
    data_path = project_root / "healthcare-dataset-stroke-data.csv"
    
    # Run evaluation
    results = run_full_evaluation(
        model_path=str(model_path),
        data_path=str(data_path),
        threshold=0.5,
        shap_sample_size=500
    )
