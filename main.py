#!/usr/bin/env python
"""
Stroke Prediction Model - Main Entry Point
===========================================

Command-line interface for the stroke prediction system.

Usage:
    python main.py --train          # Train the model
    python main.py --predict        # Run sample predictions
    python main.py --evaluate       # Evaluate model performance
    python main.py --help           # Show help

Examples:
    # Train with default settings
    python main.py --train
    
    # Train with custom iterations
    python main.py --train --n-iter 100
    
    # Quick training for testing
    python main.py --train --quick
    
    # Run sample predictions
    python main.py --predict
    
    # Predict for custom patient (JSON string)
    python main.py --predict --patient '{"age": 55, "gender": "Male", ...}'

Author: Healthcare AI Team
"""

import argparse
import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def train_model(args):
    """Execute model training pipeline."""
    from src.train_pipeline import run_training_pipeline, quick_train
    
    print("\n" + "=" * 70)
    print("🏥 STROKE PREDICTION MODEL - TRAINING MODE")
    print("=" * 70)
    
    if args.quick:
        print("\n⚡ Running quick training (reduced iterations)...")
        results = quick_train(random_state=args.seed)
    else:
        results = run_training_pipeline(
            data_path=args.data,
            output_dir=args.output,
            n_iter=args.n_iter,
            cv=args.cv,
            random_state=args.seed,
            verbose=args.verbose
        )
    
    print("\n✅ Training completed successfully!")
    print(f"   Model saved to: {results['model_path']}")
    
    return 0


def run_predictions(args):
    """Execute sample predictions."""
    from src.inference import StrokePredictor, SAMPLE_PATIENTS, run_demo
    
    print("\n" + "=" * 70)
    print("🏥 STROKE PREDICTION MODEL - INFERENCE MODE")
    print("=" * 70)
    
    if args.demo:
        run_demo()
        return 0
    
    # Initialize predictor
    predictor = StrokePredictor(model_path=args.model, verbose=True)
    
    if args.patient:
        # Custom patient data from command line
        try:
            patient_data = json.loads(args.patient)
            print(f"\n📋 Custom Patient Data:")
            for key, value in patient_data.items():
                print(f"   {key}: {value}")
            
            result = predictor.predict(patient_data)
            
            print(f"\n🎯 Prediction Result:")
            print(f"   {result}")
            print(f"   Probability: {result.probability:.4f}")
            print(f"   Risk Level: {result.risk_level.value}")
            print(f"   Confidence: {result.confidence}")
            
            if result.alert:
                print(f"\n   🚨 {result.alert}")
            
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {e}")
            return 1
        except ValueError as e:
            print(f"❌ Validation error: {e}")
            return 1
    else:
        # Run sample predictions
        print("\n📊 Running sample predictions...")
        print("-" * 70)
        
        for risk_category, patient_data in SAMPLE_PATIENTS.items():
            print(f"\n📋 {risk_category.replace('_', ' ').title()} Patient:")
            print(f"   Age: {patient_data['age']}, Gender: {patient_data['gender']}")
            print(f"   Hypertension: {patient_data['hypertension']}, "
                  f"Heart Disease: {patient_data['heart_disease']}")
            print(f"   Glucose: {patient_data['avg_glucose_level']}, BMI: {patient_data['bmi']}")
            print(f"   Smoking: {patient_data['smoking_status']}")
            
            result = predictor.predict(patient_data)
            
            print(f"\n   🎯 Result: {result}")
            if result.alert:
                print(f"   🚨 {result.alert}")
    
    print("\n" + "-" * 70)
    print("✅ Predictions completed successfully!")
    
    return 0


def evaluate_model(args):
    """Run model evaluation."""
    from src.model_evaluation import run_full_evaluation
    
    print("\n" + "=" * 70)
    print("🏥 STROKE PREDICTION MODEL - EVALUATION MODE")
    print("=" * 70)
    
    # Paths
    model_path = args.model or str(PROJECT_ROOT / "models" / "stroke_xgboost_model.joblib")
    data_path = args.data or str(PROJECT_ROOT / "healthcare-dataset-stroke-data.csv")
    
    results = run_full_evaluation(
        model_path=model_path,
        data_path=data_path,
        threshold=args.threshold,
        shap_sample_size=args.shap_samples
    )
    
    print("\n✅ Evaluation completed successfully!")
    
    return 0


def show_info(args):
    """Show project information."""
    print("\n" + "=" * 70)
    print("🏥 STROKE PREDICTION MODEL - PROJECT INFO")
    print("=" * 70)
    
    print("""
    A machine learning system for predicting stroke risk based on
    patient health indicators.
    
    📁 Project Structure:
        src/
        ├── data_ingestion.py      - Data loading and validation
        ├── statistical_analysis.py - Statistical analysis
        ├── preprocessing_pipeline.py - Feature preprocessing
        ├── xgboost_model.py       - Model training
        ├── model_evaluation.py    - Performance evaluation
        ├── train_pipeline.py      - Training orchestration
        └── inference.py           - Production inference
        
        tests/
        ├── test_data_ingestion.py
        └── test_preprocessing_pipeline.py
        
        models/
        └── stroke_xgboost_model.joblib
        
        figures/
        └── [evaluation plots]
    
    🚀 Quick Start:
        1. Train model:     python main.py --train
        2. Run predictions: python main.py --predict
        3. Evaluate:        python main.py --evaluate
    
    📊 Model Features:
        • XGBoost with cost-sensitive learning
        • Monotonic constraints for clinical interpretability
        • Hyperparameter tuning via RandomizedSearchCV
        • Fairness auditing (Equalized Odds, Demographic Parity)
        • SHAP explainability
    
    📧 Contact: Healthcare AI Team
    """)
    
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Stroke Prediction Model - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --train                Train the model
  python main.py --train --quick        Quick training (fewer iterations)
  python main.py --predict              Run sample predictions
  python main.py --predict --demo       Run full inference demo
  python main.py --evaluate             Evaluate model performance
  python main.py --info                 Show project information
        """
    )
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--train',
        action='store_true',
        help='Train the stroke prediction model'
    )
    mode_group.add_argument(
        '--predict',
        action='store_true',
        help='Run predictions using the trained model'
    )
    mode_group.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate model performance and fairness'
    )
    mode_group.add_argument(
        '--info',
        action='store_true',
        help='Show project information'
    )
    
    # Training arguments
    train_group = parser.add_argument_group('Training Options')
    train_group.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to training data CSV'
    )
    train_group.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for model and results'
    )
    train_group.add_argument(
        '--n-iter',
        type=int,
        default=50,
        help='Number of hyperparameter search iterations (default: 50)'
    )
    train_group.add_argument(
        '--cv',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    train_group.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    train_group.add_argument(
        '--quick',
        action='store_true',
        help='Quick training with reduced iterations'
    )
    train_group.add_argument(
        '--verbose',
        type=int,
        default=1,
        choices=[0, 1, 2],
        help='Verbosity level (0=silent, 1=progress, 2=detailed)'
    )
    
    # Prediction arguments
    predict_group = parser.add_argument_group('Prediction Options')
    predict_group.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to trained model file'
    )
    predict_group.add_argument(
        '--patient',
        type=str,
        default=None,
        help='Patient data as JSON string'
    )
    predict_group.add_argument(
        '--demo',
        action='store_true',
        help='Run full inference demonstration'
    )
    
    # Evaluation arguments
    eval_group = parser.add_argument_group('Evaluation Options')
    eval_group.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold (default: 0.5)'
    )
    eval_group.add_argument(
        '--shap-samples',
        type=int,
        default=500,
        help='Number of samples for SHAP analysis (default: 500)'
    )
    
    args = parser.parse_args()
    
    # Route to appropriate function
    try:
        if args.train:
            return train_model(args)
        elif args.predict:
            return run_predictions(args)
        elif args.evaluate:
            return evaluate_model(args)
        elif args.info:
            return show_info(args)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("   Make sure to train the model first: python main.py --train")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
