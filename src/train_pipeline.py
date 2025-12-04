"""
Training Pipeline Orchestrator
==============================

This module serves as the "One-Click Reproducibility" button for the
stroke prediction model. It orchestrates the entire training workflow:

1. Data Loading & Validation
2. Preprocessing Pipeline Creation
3. Model Training with Hyperparameter Tuning
4. Model Evaluation
5. Model Persistence

Usage:
    from src.train_pipeline import run_training_pipeline
    results = run_training_pipeline()

Or from command line:
    python -m src.train_pipeline

Author: Healthcare AI Team
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_training_pipeline(
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    n_iter: int = 50,
    cv: int = 5,
    random_state: int = 42,
    verbose: int = 1
) -> Dict[str, Any]:
    """
    Execute the complete training pipeline for stroke prediction.
    
    This is the "One-Click Reproducibility" function that:
    1. Loads and validates stroke data
    2. Creates the preprocessing pipeline
    3. Trains the XGBoost model with hyperparameter tuning
    4. Evaluates the model on test data
    5. Saves the trained model and results
    
    Pipeline Architecture:
    ======================
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                    TRAINING PIPELINE                            │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
    │  │ Data Loading │ -> │ Validation   │ -> │ Splitting    │     │
    │  │ (CSV)        │    │ (Pandera)    │    │ (Stratified) │     │
    │  └──────────────┘    └──────────────┘    └──────────────┘     │
    │                                               │                 │
    │                                               ▼                 │
    │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
    │  │ Evaluation   │ <- │ Tuning       │ <- │ Preprocessing│     │
    │  │ (Test Set)   │    │ (RandomCV)   │    │ (Pipeline)   │     │
    │  └──────────────┘    └──────────────┘    └──────────────┘     │
    │         │                                                       │
    │         ▼                                                       │
    │  ┌──────────────┐                                              │
    │  │ Save Model   │                                              │
    │  │ (.joblib)    │                                              │
    │  └──────────────┘                                              │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Args:
        data_path: Path to the stroke dataset CSV. If None, uses default.
        output_dir: Directory to save model and results. If None, uses default.
        n_iter: Number of hyperparameter search iterations.
        cv: Number of cross-validation folds.
        random_state: Random seed for reproducibility.
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed).
        
    Returns:
        Dict containing:
            - 'model': Trained sklearn Pipeline
            - 'best_params': Best hyperparameters found
            - 'evaluation': Evaluation metrics on test set
            - 'model_path': Path to saved model
            - 'training_time': Total training duration
            - 'timestamp': Training timestamp
    """
    from .data_ingestion import load_stroke_data, validate_stroke_data
    from .preprocessing_pipeline import create_preprocessing_pipeline
    from .xgboost_model import train_stroke_model, save_model
    from .model_evaluation import run_fairness_audit, generate_shap_analysis
    
    # Set default paths
    if data_path is None:
        data_path = str(PROJECT_ROOT / "healthcare-dataset-stroke-data.csv")
    
    if output_dir is None:
        output_dir = PROJECT_ROOT / "models"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    # Start timing
    start_time = datetime.now()
    
    print("=" * 70)
    print("STROKE PREDICTION - TRAINING PIPELINE")
    print("=" * 70)
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data path: {data_path}")
    print(f"Output dir: {output_dir}")
    print("=" * 70)
    
    results = {
        'timestamp': start_time.isoformat(),
        'config': {
            'data_path': data_path,
            'output_dir': str(output_dir),
            'n_iter': n_iter,
            'cv': cv,
            'random_state': random_state
        }
    }
    
    # =========================================================================
    # Step 1: Load and Validate Data
    # =========================================================================
    print("\n" + "─" * 70)
    print("STEP 1: DATA LOADING & VALIDATION")
    print("─" * 70)
    
    df = load_stroke_data(data_path)
    
    print(f"\n✓ Loaded {len(df):,} samples with {len(df.columns)} features")
    print(f"✓ Stroke rate: {df['stroke'].mean():.2%}")
    
    # Validate schema
    try:
        validate_stroke_data(df)
        print("✓ Schema validation passed")
    except Exception as e:
        print(f"⚠ Schema validation warning: {e}")
    
    results['data_info'] = {
        'n_samples': len(df),
        'n_features': len(df.columns),
        'stroke_rate': float(df['stroke'].mean())
    }
    
    # =========================================================================
    # Step 2: Create Preprocessing Pipeline
    # =========================================================================
    print("\n" + "─" * 70)
    print("STEP 2: PREPROCESSING PIPELINE")
    print("─" * 70)
    
    preprocessor = create_preprocessing_pipeline()
    print("✓ Preprocessing pipeline created")
    print("  - Numerical: RobustScaler → KNNImputer")
    print("  - Categorical: OneHotEncoder")
    
    # =========================================================================
    # Step 3: Train Model
    # =========================================================================
    print("\n" + "─" * 70)
    print("STEP 3: MODEL TRAINING")
    print("─" * 70)
    
    model, best_params, evaluation = train_stroke_model(
        df=df,
        n_iter=n_iter,
        cv=cv,
        random_state=random_state,
        verbose=verbose
    )
    
    results['model'] = model
    results['best_params'] = best_params
    results['evaluation'] = evaluation
    
    # =========================================================================
    # Step 4: Save Model
    # =========================================================================
    print("\n" + "─" * 70)
    print("STEP 4: MODEL PERSISTENCE")
    print("─" * 70)
    
    model_path = output_dir / "stroke_xgboost_model.joblib"
    save_model(model, str(model_path))
    results['model_path'] = str(model_path)
    
    # Save training metadata
    import json
    metadata_path = output_dir / "training_metadata.json"
    metadata = {
        'timestamp': results['timestamp'],
        'config': results['config'],
        'data_info': results['data_info'],
        'best_params': {k: float(v) if isinstance(v, (int, float)) else str(v) 
                        for k, v in best_params.items()},
        'metrics': {
            'roc_auc': float(evaluation['roc_auc']),
            'average_precision': float(evaluation['average_precision']),
            'sensitivity': float(evaluation['sensitivity']),
            'specificity': float(evaluation['specificity']),
            'ppv': float(evaluation['ppv']),
            'npv': float(evaluation['npv'])
        }
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Training metadata saved to: {metadata_path}")
    
    # =========================================================================
    # Step 5: Summary
    # =========================================================================
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    results['training_time'] = training_time
    
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n📊 Performance Summary:")
    print(f"   • ROC-AUC: {evaluation['roc_auc']:.4f}")
    print(f"   • Average Precision: {evaluation['average_precision']:.4f}")
    print(f"   • Stroke Detection Rate (Sensitivity): {evaluation['sensitivity']:.1%}")
    
    print(f"\n📁 Outputs:")
    print(f"   • Model: {model_path}")
    print(f"   • Metadata: {metadata_path}")
    
    print(f"\n⏱ Total training time: {training_time:.1f} seconds")
    print(f"   Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results


def quick_train(random_state: int = 42) -> Dict[str, Any]:
    """
    Quick training with minimal hyperparameter search.
    
    Useful for testing and development. Uses only 10 iterations
    instead of the full 50.
    
    Args:
        random_state: Random seed
        
    Returns:
        Training results dict
    """
    return run_training_pipeline(
        n_iter=10,
        cv=3,
        random_state=random_state,
        verbose=0
    )


if __name__ == "__main__":
    # Run the full training pipeline
    results = run_training_pipeline()
    
    print("\n" + "=" * 70)
    print("Pipeline execution completed successfully!")
    print("=" * 70)
