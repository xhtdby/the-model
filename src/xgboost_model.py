"""
Stroke Prediction - XGBoost Model Module
=========================================
XGBoost classifier with cost-sensitive learning and clinical constraints.

This module implements:
- Cost-sensitive learning via scale_pos_weight (no SMOTE)
- Monotonic constraints for clinical interpretability
- RandomizedSearchCV for hyperparameter tuning
- Full pipeline integration (preprocessing + model)

Author: Senior ML Engineer
Project: Healthcare AI - Stroke Prediction
"""

import warnings
from typing import Dict, Tuple, List, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import uniform, randint, loguniform
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_curve
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
import xgboost as xgb


# =============================================================================
# Cost-Sensitive Learning
# =============================================================================

def calculate_scale_pos_weight(y: pd.Series) -> float:
    """
    Calculate scale_pos_weight for cost-sensitive learning.
    
    For imbalanced binary classification, XGBoost's scale_pos_weight parameter
    adjusts the gradient for the positive class, effectively giving more weight
    to positive samples during training.
    
    Formula: scale_pos_weight = sum(negative) / sum(positive)
    
    This is equivalent to adjusting the loss function to penalize
    misclassification of the minority class (stroke=1) more heavily.
    
    WHY NOT SMOTE:
    ==============
    1. SMOTE creates synthetic samples, which can introduce artifacts
    2. Cost-sensitive learning works on the actual data distribution
    3. XGBoost's built-in handling is computationally efficient
    4. No risk of data leakage from synthetic sample generation
    5. Better for small minority classes where SMOTE may overfit
    
    Args:
        y: Target labels (binary: 0 or 1)
        
    Returns:
        scale_pos_weight value for XGBoost
        
    Example:
        >>> y = pd.Series([0, 0, 0, 0, 1])  # 4 negative, 1 positive
        >>> calculate_scale_pos_weight(y)
        4.0
    """
    n_negative = (y == 0).sum()
    n_positive = (y == 1).sum()
    
    if n_positive == 0:
        raise ValueError("No positive samples found in training data!")
    
    scale_pos_weight = n_negative / n_positive
    
    print(f"Class Distribution:")
    print(f"  Negative (no stroke): {n_negative} ({n_negative/len(y)*100:.2f}%)")
    print(f"  Positive (stroke):    {n_positive} ({n_positive/len(y)*100:.2f}%)")
    print(f"  Imbalance Ratio:      {scale_pos_weight:.2f}:1")
    print(f"  scale_pos_weight:     {scale_pos_weight:.4f}")
    
    return scale_pos_weight


# =============================================================================
# Monotonic Constraints
# =============================================================================

def get_monotonic_constraints(feature_names: List[str]) -> str:
    """
    Define monotonic constraints for clinical interpretability.
    
    CLINICAL RATIONALE:
    ===================
    In healthcare AI, model predictions should align with established medical
    knowledge. For stroke prediction:
    
    1. AGE (+1): Stroke risk increases with age. This is well-established in
       medical literature. The model should never predict lower stroke risk
       for an older patient when all other factors are equal.
    
    2. AVG_GLUCOSE_LEVEL (+1): Higher glucose levels (especially persistent
       hyperglycemia) are associated with increased stroke risk. This is
       linked to diabetes, vascular damage, and atherosclerosis.
    
    Constraint Values:
    - +1: Feature must have positive monotonic relationship (increasing)
    - -1: Feature must have negative monotonic relationship (decreasing)
    -  0: No constraint (non-monotonic relationship allowed)
    
    NOTE: We return a string representation for XGBoost's monotone_constraints
    parameter which handles dynamic feature counts better than tuples.
    
    Args:
        feature_names: List of feature names from preprocessing pipeline
        
    Returns:
        String of constraint values matching feature order (e.g., "(1,1,0,0,...)")
    """
    # Initialize all constraints to 0 (no constraint)
    constraints = [0] * len(feature_names)
    
    # Apply clinical constraints
    clinical_constraints = {
        'numerical__age': +1,              # Stroke risk increases with age
        'numerical__avg_glucose_level': +1  # Stroke risk increases with glucose
    }
    
    # Map constraints to feature indices
    for i, feature in enumerate(feature_names):
        if feature in clinical_constraints:
            constraints[i] = clinical_constraints[feature]
    
    # Log applied constraints
    print("\nMonotonic Constraints Applied:")
    for feature, constraint in clinical_constraints.items():
        direction = "↑ (increasing)" if constraint == +1 else "↓ (decreasing)"
        print(f"  {feature}: {direction}")
    
    # Return as string format for XGBoost (handles dynamic feature counts better)
    return "(" + ",".join(str(c) for c in constraints) + ")"


# =============================================================================
# XGBoost Model Creation
# =============================================================================

def create_xgboost_model(
    scale_pos_weight: float,
    monotonic_constraints: Optional[str] = None,
    random_state: int = 42,
    **kwargs
) -> xgb.XGBClassifier:
    """
    Create XGBoost classifier with cost-sensitive learning and constraints.
    
    Args:
        scale_pos_weight: Weight for positive class (handles imbalance)
        monotonic_constraints: String of constraint values per feature (optional)
        random_state: Random seed for reproducibility
        **kwargs: Additional hyperparameters to override defaults
        
    Returns:
        Configured XGBClassifier
    """
    # Default hyperparameters (will be tuned)
    default_params = {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 4,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 1.0,  # L1 regularization
        'reg_lambda': 1.0,  # L2 regularization
        'gamma': 0.1,  # Minimum loss reduction for split
        
        # Cost-sensitive learning
        'scale_pos_weight': scale_pos_weight,
        
        # Technical settings
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'use_label_encoder': False,
        'random_state': random_state,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    # Add monotonic constraints only if provided
    if monotonic_constraints is not None:
        default_params['monotone_constraints'] = monotonic_constraints
    
    # Override with any provided kwargs
    default_params.update(kwargs)
    
    return xgb.XGBClassifier(**default_params)


# =============================================================================
# Hyperparameter Search Space
# =============================================================================

def get_hyperparameter_search_space() -> Dict[str, Any]:
    """
    Define hyperparameter search space for RandomizedSearchCV.
    
    Search Space:
    =============
    - learning_rate: [0.01, 0.1] - Controls step size during gradient descent
      Lower = more robust but slower, Higher = faster but may overshoot
    
    - max_depth: [3, 6] - Maximum tree depth
      Shallow = less overfitting, Deep = captures complex patterns
      For healthcare with ~5k samples, 3-6 is appropriate
    
    - reg_alpha: [0, 10] - L1 regularization (Lasso)
      Promotes sparsity, helps feature selection
    
    Additional parameters included for robustness:
    - n_estimators: Number of boosting rounds
    - min_child_weight: Minimum sum of instance weight in child
    - subsample: Row sampling ratio
    - colsample_bytree: Column sampling ratio
    - gamma: Minimum loss reduction for split
    - reg_lambda: L2 regularization (Ridge)
    
    Returns:
        Dictionary of parameter distributions for RandomizedSearchCV
    """
    search_space = {
        # PRIMARY SEARCH PARAMETERS (as specified)
        'model__learning_rate': loguniform(0.01, 0.1),  # Log-uniform for learning rate
        'model__max_depth': randint(3, 7),  # Discrete: 3, 4, 5, 6
        'model__reg_alpha': uniform(0, 10),  # Uniform: [0, 10]
        
        # SECONDARY PARAMETERS for better optimization
        'model__n_estimators': randint(100, 500),  # Number of trees
        'model__min_child_weight': randint(1, 10),  # Min samples in leaf
        'model__subsample': uniform(0.6, 0.4),  # Row sampling [0.6, 1.0]
        'model__colsample_bytree': uniform(0.6, 0.4),  # Column sampling [0.6, 1.0]
        'model__gamma': uniform(0, 0.5),  # Min split loss [0, 0.5]
        'model__reg_lambda': uniform(0, 10),  # L2 regularization [0, 10]
    }
    
    return search_space


# =============================================================================
# Full Pipeline Creation
# =============================================================================

def create_full_pipeline(
    preprocessor: ColumnTransformer,
    scale_pos_weight: float,
    monotonic_constraints: Optional[str] = None,
    random_state: int = 42
) -> Pipeline:
    """
    Create complete pipeline: Preprocessing + XGBoost.
    
    Architecture:
    =============
    ┌────────────────────────────────────────────────────────────┐
    │                    sklearn.Pipeline                        │
    │  ┌─────────────────────────────────────────────────────┐  │
    │  │           Step 1: Preprocessor                      │  │
    │  │  ┌─────────────────┐  ┌─────────────────────────┐  │  │
    │  │  │ Numerical       │  │ Categorical             │  │  │
    │  │  │ RobustScaler    │  │ OneHotEncoder           │  │  │
    │  │  │ KNNImputer      │  │                         │  │  │
    │  │  └─────────────────┘  └─────────────────────────┘  │  │
    │  └─────────────────────────────────────────────────────┘  │
    │                          │                                 │
    │                          ▼                                 │
    │  ┌─────────────────────────────────────────────────────┐  │
    │  │           Step 2: XGBoost Classifier               │  │
    │  │  - scale_pos_weight (cost-sensitive)               │  │
    │  │  - monotone_constraints (clinical)                 │  │
    │  │  - Tunable hyperparameters                         │  │
    │  └─────────────────────────────────────────────────────┘  │
    └────────────────────────────────────────────────────────────┘
    
    Args:
        preprocessor: Fitted or unfitted ColumnTransformer
        scale_pos_weight: Weight for positive class
        monotonic_constraints: Clinical constraints tuple
        random_state: Random seed
        
    Returns:
        sklearn Pipeline ready for fitting or hyperparameter tuning
    """
    model = create_xgboost_model(
        scale_pos_weight=scale_pos_weight,
        monotonic_constraints=monotonic_constraints,
        random_state=random_state
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return pipeline


# =============================================================================
# Hyperparameter Tuning
# =============================================================================

def tune_hyperparameters(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 50,
    cv: int = 5,
    random_state: int = 42,
    verbose: int = 1
) -> Tuple[RandomizedSearchCV, Dict[str, Any]]:
    """
    Perform hyperparameter tuning using RandomizedSearchCV.
    
    Uses stratified K-fold cross-validation to ensure class balance
    is maintained in each fold, which is critical for imbalanced data.
    
    Optimization Metric: ROC-AUC
    ============================
    ROC-AUC is chosen over accuracy because:
    1. Insensitive to class imbalance
    2. Evaluates ranking ability across all thresholds
    3. Appropriate for probability-based predictions
    4. Standard metric for clinical decision support systems
    
    Args:
        pipeline: sklearn Pipeline with preprocessor and model
        X_train: Training features
        y_train: Training labels
        n_iter: Number of random combinations to try
        cv: Number of cross-validation folds
        random_state: Random seed
        verbose: Verbosity level
        
    Returns:
        Tuple of (fitted RandomizedSearchCV, best parameters dict)
    """
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING")
    print("=" * 70)
    print(f"Search iterations: {n_iter}")
    print(f"Cross-validation folds: {cv}")
    print(f"Optimization metric: ROC-AUC")
    
    # Get search space
    search_space = get_hyperparameter_search_space()
    
    print("\nSearch Space:")
    for param, dist in search_space.items():
        print(f"  {param}: {dist}")
    
    # Stratified K-Fold for imbalanced data
    cv_strategy = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state
    )
    
    # RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=search_space,
        n_iter=n_iter,
        scoring='roc_auc',  # Optimize for ROC-AUC
        cv=cv_strategy,
        random_state=random_state,
        verbose=verbose,
        n_jobs=-1,
        return_train_score=True,
        error_score='raise'
    )
    
    print("\nStarting hyperparameter search...")
    
    # Fit with verbose progress
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        random_search.fit(X_train, y_train)
    
    # Report results
    print("\n" + "-" * 70)
    print("TUNING RESULTS")
    print("-" * 70)
    print(f"Best ROC-AUC Score (CV): {random_search.best_score_:.4f}")
    
    print("\nBest Hyperparameters:")
    best_params = random_search.best_params_
    for param, value in sorted(best_params.items()):
        print(f"  {param}: {value}")
    
    return random_search, best_params


# =============================================================================
# Model Evaluation
# =============================================================================

def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation on test set.
    
    Args:
        model: Fitted pipeline
        X_test: Test features
        y_test: Test labels
        threshold: Classification threshold (default: 0.5)
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "=" * 70)
    print("MODEL EVALUATION ON TEST SET")
    print("=" * 70)
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    print(f"\nProbability Metrics:")
    print(f"  ROC-AUC Score:         {roc_auc:.4f}")
    print(f"  Average Precision:     {avg_precision:.4f}")
    
    print(f"\nClassification Report (threshold={threshold}):")
    print(classification_report(y_test, y_pred, target_names=['No Stroke', 'Stroke']))
    
    print(f"Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    
    # Calculate clinical metrics
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    ppv = cm[1, 1] / (cm[0, 1] + cm[1, 1]) if (cm[0, 1] + cm[1, 1]) > 0 else 0
    npv = cm[0, 0] / (cm[0, 0] + cm[1, 0]) if (cm[0, 0] + cm[1, 0]) > 0 else 0
    
    print(f"\nClinical Metrics:")
    print(f"  Sensitivity (Recall):  {sensitivity:.4f}")
    print(f"  Specificity:           {specificity:.4f}")
    print(f"  PPV (Precision):       {ppv:.4f}")
    print(f"  NPV:                   {npv:.4f}")
    
    return {
        'roc_auc': roc_auc,
        'average_precision': avg_precision,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'confusion_matrix': cm,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred
    }


def get_feature_importance(
    model: Pipeline,
    feature_names: List[str],
    top_n: int = 10
) -> pd.DataFrame:
    """
    Extract and display feature importance from trained model.
    
    Args:
        model: Fitted pipeline with XGBoost model
        feature_names: List of feature names after preprocessing
        top_n: Number of top features to display
        
    Returns:
        DataFrame with feature importances
    """
    # Get XGBoost model from pipeline
    xgb_model = model.named_steps['model']
    
    # Get importance scores
    importance_scores = xgb_model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop {top_n} Important Features:")
    print("-" * 50)
    for i, row in importance_df.head(top_n).iterrows():
        bar = "█" * int(row['importance'] * 50)
        print(f"  {row['feature']:40s} {row['importance']:.4f} {bar}")
    
    return importance_df


# =============================================================================
# Main Training Script
# =============================================================================

def train_stroke_model(
    df: pd.DataFrame,
    n_iter: int = 50,
    cv: int = 5,
    random_state: int = 42,
    verbose: int = 1
) -> Tuple[Pipeline, Dict[str, Any], Dict[str, Any]]:
    """
    Complete training pipeline for stroke prediction model.
    
    This function:
    1. Splits data with stratification
    2. Creates preprocessing pipeline
    3. Calculates scale_pos_weight for cost-sensitive learning
    4. Applies monotonic constraints for clinical interpretability
    5. Tunes hyperparameters using RandomizedSearchCV
    6. Evaluates on test set
    
    Args:
        df: Input DataFrame with features and target
        n_iter: Number of hyperparameter search iterations
        cv: Number of cross-validation folds
        random_state: Random seed
        verbose: Verbosity level
        
    Returns:
        Tuple of (best_model, best_params, evaluation_results)
    """
    print("=" * 70)
    print("STROKE PREDICTION MODEL TRAINING")
    print("=" * 70)
    
    # Import preprocessing functions
    from .preprocessing_pipeline import (
        split_data,
        create_preprocessing_pipeline,
        get_feature_names,
        AggressiveFeatureEngineer,
        BASE_NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES
    )
    
    # Step 1: Split data
    print("\n[Step 1] Splitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = split_data(
        df, test_size=0.2, random_state=random_state
    )
    
    # Step 2: Calculate scale_pos_weight
    print("\n[Step 2] Calculating cost-sensitive weight...")
    scale_pos_weight = calculate_scale_pos_weight(y_train)
    
    # Step 3: Create preprocessing pipeline with feature engineering
    print("\n[Step 3] Creating preprocessing pipeline...")
    feature_engineer = AggressiveFeatureEngineer()
    preprocessor = create_preprocessing_pipeline()
    
    # Apply feature engineering to training data
    X_train_eng = feature_engineer.fit_transform(X_train)
    
    # Fit preprocessor to get feature names (needed for monotonic constraints)
    preprocessor.fit(X_train_eng)
    feature_names = get_feature_names(preprocessor)
    print(f"  Total features after engineering + preprocessing: {len(feature_names)}")
    
    # Step 4: Get monotonic constraints (for final model, not during CV)
    print("\n[Step 4] Defining clinical monotonic constraints...")
    monotonic_constraints = get_monotonic_constraints(feature_names)
    print(f"  Constraints string: {monotonic_constraints[:50]}...")
    
    # Step 5: Create fresh pipeline (feature engineering + preprocessing + model)
    fresh_feature_engineer = AggressiveFeatureEngineer()
    preprocessor_for_pipeline = create_preprocessing_pipeline()
    
    # Step 6: Create full pipeline WITH feature engineering
    print("\n[Step 5] Creating pipeline for hyperparameter tuning...")
    print("  Note: Monotonic constraints applied after tuning to avoid CV issues")
    
    # Build manual pipeline to include feature engineering
    pipeline = Pipeline([
        ('feature_engineer', fresh_feature_engineer),
        ('preprocessor', preprocessor_for_pipeline),
        ('model', create_xgboost_model(
            scale_pos_weight=scale_pos_weight,
            monotonic_constraints=None,
            random_state=random_state
        ))
    ])
    
    # Step 7: Hyperparameter tuning
    print("\n[Step 6] Tuning hyperparameters...")
    random_search, best_params = tune_hyperparameters(
        pipeline=pipeline,
        X_train=X_train,
        y_train=y_train,
        n_iter=n_iter,
        cv=cv,
        random_state=random_state,
        verbose=verbose
    )
    
    # Step 8: Create final model with best parameters AND monotonic constraints
    print("\n[Step 7] Creating final model with monotonic constraints...")
    
    # Create final feature engineer and preprocessor
    final_feature_engineer = AggressiveFeatureEngineer()
    final_preprocessor = create_preprocessing_pipeline()
    
    # Extract best model parameters (remove 'model__' prefix)
    best_model_params = {
        k.replace('model__', ''): v 
        for k, v in best_params.items()
    }
    
    # Create final XGBoost with constraints
    final_xgb = create_xgboost_model(
        scale_pos_weight=scale_pos_weight,
        monotonic_constraints=monotonic_constraints,
        random_state=random_state,
        **best_model_params
    )
    
    # Create final pipeline with feature engineering
    final_pipeline = Pipeline([
        ('feature_engineer', final_feature_engineer),
        ('preprocessor', final_preprocessor),
        ('model', final_xgb)
    ])
    
    # Fit final model on full training data
    final_pipeline.fit(X_train, y_train)
    
    # Step 9: Evaluate on test set
    print("\n[Step 8] Evaluating on test set...")
    evaluation_results = evaluate_model(final_pipeline, X_test, y_test)
    
    # Step 10: Feature importance
    print("\n[Step 9] Analyzing feature importance...")
    importance_df = get_feature_importance(final_pipeline, feature_names)
    evaluation_results['feature_importance'] = importance_df
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best CV ROC-AUC: {random_search.best_score_:.4f}")
    print(f"Test ROC-AUC:    {evaluation_results['roc_auc']:.4f}")
    
    return final_pipeline, best_params, evaluation_results


# =============================================================================
# Model Persistence
# =============================================================================

def save_model(model: Pipeline, filepath: str) -> None:
    """Save trained model to disk."""
    import joblib
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")


def load_model(filepath: str) -> Pipeline:
    """Load trained model from disk."""
    import joblib
    model = joblib.load(filepath)
    print(f"Model loaded from: {filepath}")
    return model


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent))
    from data_ingestion import load_stroke_data
    
    # Load data
    print("Loading stroke dataset...")
    default_path = Path(__file__).parent.parent / "healthcare-dataset-stroke-data.csv"
    df = load_stroke_data(default_path)
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features\n")
    
    # Train model
    best_model, best_params, evaluation_results = train_stroke_model(
        df,
        n_iter=50,  # Number of hyperparameter combinations to try
        cv=5,       # 5-fold cross-validation
        random_state=42,
        verbose=1
    )
    
    # Save model
    model_path = Path(__file__).parent.parent / "models"
    model_path.mkdir(exist_ok=True)
    save_model(best_model, str(model_path / "stroke_xgboost_model.joblib"))
    
    # Save best parameters
    import json
    params_path = model_path / "best_hyperparameters.json"
    with open(params_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        serializable_params = {
            k: float(v) if hasattr(v, 'item') else v 
            for k, v in best_params.items()
        }
        json.dump(serializable_params, f, indent=2)
    print(f"Best parameters saved to: {params_path}")
