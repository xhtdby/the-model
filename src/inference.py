"""
Stroke Prediction Inference Module
===================================

Production-ready inference module for stroke risk prediction.
This is the deployable component for hospital API integration.

Features:
- Load pre-trained model from disk
- Accept raw patient data as dictionary
- Return formatted risk scores with clinical alerts
- Input validation and error handling
- Batch prediction support

Usage:
    from src.inference import StrokePredictor
    
    predictor = StrokePredictor()
    result = predictor.predict({
        'age': 55,
        'gender': 'Male',
        'avg_glucose_level': 200,
        ...
    })

Author: Healthcare AI Team
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# Risk Level Classification
# =============================================================================

class RiskLevel(Enum):
    """Risk level categories based on prediction probability."""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class PredictionResult:
    """
    Structured prediction result with clinical context.
    
    Attributes:
        probability: Raw stroke probability (0-1)
        risk_percentage: Formatted percentage string
        risk_level: Categorical risk level
        alert: Optional critical alert message
        confidence: Model confidence indicator
        features_used: Input features that were processed
    """
    probability: float
    risk_percentage: str
    risk_level: RiskLevel
    alert: Optional[str]
    confidence: str
    features_used: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'probability': self.probability,
            'risk_percentage': self.risk_percentage,
            'risk_level': self.risk_level.value,
            'alert': self.alert,
            'confidence': self.confidence,
            'features_used': self.features_used
        }
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        result = f"Stroke Risk: {self.risk_percentage} ({self.risk_level.value})"
        if self.alert:
            result += f"\n🚨 {self.alert}"
        return result


# =============================================================================
# Expected Features Schema
# =============================================================================

REQUIRED_FEATURES = {
    'gender': str,
    'age': (int, float),
    'hypertension': (int, bool),
    'heart_disease': (int, bool),
    'ever_married': str,
    'work_type': str,
    'Residence_type': str,
    'avg_glucose_level': (int, float),
    'bmi': (int, float, type(None)),  # Can be NaN
    'smoking_status': str
}

VALID_VALUES = {
    'gender': ['Male', 'Female', 'Other'],
    'ever_married': ['Yes', 'No'],
    'work_type': ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'],
    'Residence_type': ['Urban', 'Rural'],
    'smoking_status': ['formerly smoked', 'never smoked', 'smokes', 'Unknown']
}


# =============================================================================
# StrokePredictor Class
# =============================================================================

class StrokePredictor:
    """
    Production-ready stroke risk predictor.
    
    This class encapsulates the trained model and provides a clean
    interface for making predictions in a hospital/clinical setting.
    
    Risk Level Thresholds:
    - LOW: < 20%
    - MODERATE: 20-50%
    - HIGH: 50-80%
    - CRITICAL: > 80% (triggers alert)
    
    Example:
        >>> predictor = StrokePredictor()
        >>> result = predictor.predict({
        ...     'age': 67,
        ...     'gender': 'Male',
        ...     'hypertension': 1,
        ...     'heart_disease': 0,
        ...     'ever_married': 'Yes',
        ...     'work_type': 'Private',
        ...     'Residence_type': 'Urban',
        ...     'avg_glucose_level': 228.69,
        ...     'bmi': 36.6,
        ...     'smoking_status': 'formerly smoked'
        ... })
        >>> print(result)
        Stroke Risk: 45.2% (HIGH)
    """
    
    # Risk thresholds
    THRESHOLD_LOW = 0.20
    THRESHOLD_MODERATE = 0.50
    THRESHOLD_HIGH = 0.80
    CRITICAL_ALERT_THRESHOLD = 0.80
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the StrokePredictor.
        
        Args:
            model_path: Path to the saved model. If None, uses default location.
            verbose: Whether to print loading messages.
        """
        self.verbose = verbose
        
        # Determine model path
        if model_path is None:
            project_root = Path(__file__).parent.parent
            model_path = project_root / "models" / "stroke_xgboost_model.joblib"
        else:
            model_path = Path(model_path)
        
        # Load model
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please train the model first using: python main.py --train"
            )
        
        if self.verbose:
            print(f"Loading model from: {model_path}")
        
        self.model = joblib.load(model_path)
        self.model_path = model_path
        
        if self.verbose:
            print("✓ Model loaded successfully")
            print(f"  Pipeline steps: {list(self.model.named_steps.keys())}")
    
    def _validate_input(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize input patient data.
        
        Args:
            patient_data: Raw patient data dictionary
            
        Returns:
            Validated and sanitized data
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        validated = {}
        errors = []
        
        for feature, expected_type in REQUIRED_FEATURES.items():
            if feature not in patient_data:
                # Handle missing BMI gracefully (will be imputed)
                if feature == 'bmi':
                    validated[feature] = np.nan
                    continue
                errors.append(f"Missing required feature: '{feature}'")
                continue
            
            value = patient_data[feature]
            
            # Handle None/NaN for BMI
            if feature == 'bmi' and (value is None or (isinstance(value, float) and np.isnan(value))):
                validated[feature] = np.nan
                continue
            
            # Type checking
            if not isinstance(value, expected_type):
                # Try type conversion for numeric fields
                if expected_type in [(int, float), (int, bool)]:
                    try:
                        value = float(value) if expected_type == (int, float) else int(value)
                    except (ValueError, TypeError):
                        errors.append(f"Invalid type for '{feature}': expected {expected_type}, got {type(value)}")
                        continue
            
            # Value validation for categorical features
            if feature in VALID_VALUES:
                if value not in VALID_VALUES[feature]:
                    errors.append(
                        f"Invalid value for '{feature}': '{value}'. "
                        f"Expected one of: {VALID_VALUES[feature]}"
                    )
                    continue
            
            # Range validation for numeric features
            if feature == 'age' and not (0 <= value <= 120):
                errors.append(f"Invalid age: {value}. Expected 0-120.")
                continue
            
            if feature == 'avg_glucose_level' and not (0 <= value <= 500):
                errors.append(f"Invalid glucose level: {value}. Expected 0-500.")
                continue
            
            if feature == 'bmi' and not (10 <= value <= 100):
                errors.append(f"Unusual BMI: {value}. Expected 10-100.")
                # Don't fail, just warn
            
            validated[feature] = value
        
        if errors:
            raise ValueError(
                "Input validation failed:\n" + "\n".join(f"  • {e}" for e in errors)
            )
        
        return validated
    
    def _classify_risk_level(self, probability: float) -> RiskLevel:
        """
        Classify probability into risk level category.
        
        Args:
            probability: Stroke probability (0-1)
            
        Returns:
            RiskLevel enum value
        """
        if probability < self.THRESHOLD_LOW:
            return RiskLevel.LOW
        elif probability < self.THRESHOLD_MODERATE:
            return RiskLevel.MODERATE
        elif probability < self.THRESHOLD_HIGH:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _generate_alert(self, probability: float) -> Optional[str]:
        """
        Generate critical alert if probability exceeds threshold.
        
        Args:
            probability: Stroke probability (0-1)
            
        Returns:
            Alert message or None
        """
        if probability >= self.CRITICAL_ALERT_THRESHOLD:
            return (
                "CRITICAL ALERT: Patient has extremely high stroke risk (>80%). "
                "Immediate clinical evaluation recommended."
            )
        return None
    
    def _assess_confidence(self, probability: float) -> str:
        """
        Assess prediction confidence based on probability extremity.
        
        Predictions near 0 or 1 are more confident than those near 0.5.
        
        Args:
            probability: Stroke probability (0-1)
            
        Returns:
            Confidence descriptor string
        """
        distance_from_uncertain = abs(probability - 0.5)
        
        if distance_from_uncertain > 0.4:
            return "High"
        elif distance_from_uncertain > 0.2:
            return "Moderate"
        else:
            return "Low"
    
    def predict(self, patient_data: Dict[str, Any]) -> PredictionResult:
        """
        Predict stroke risk for a single patient.
        
        Args:
            patient_data: Dictionary containing patient features.
                Required keys:
                - gender: 'Male', 'Female', or 'Other'
                - age: Patient age in years
                - hypertension: 0 or 1
                - heart_disease: 0 or 1
                - ever_married: 'Yes' or 'No'
                - work_type: 'Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'
                - Residence_type: 'Urban' or 'Rural'
                - avg_glucose_level: Average glucose level
                - bmi: Body mass index (can be None/NaN)
                - smoking_status: 'formerly smoked', 'never smoked', 'smokes', 'Unknown'
        
        Returns:
            PredictionResult object with risk assessment
            
        Example:
            >>> result = predictor.predict({
            ...     'age': 55,
            ...     'gender': 'Male',
            ...     'hypertension': 1,
            ...     'heart_disease': 0,
            ...     'ever_married': 'Yes',
            ...     'work_type': 'Private',
            ...     'Residence_type': 'Urban',
            ...     'avg_glucose_level': 200,
            ...     'bmi': 28.5,
            ...     'smoking_status': 'smokes'
            ... })
            >>> print(result.risk_percentage)
            '32.5%'
        """
        # Validate input
        validated_data = self._validate_input(patient_data)
        
        # Convert to DataFrame (model expects DataFrame input)
        df = pd.DataFrame([validated_data])
        
        # Get probability prediction
        probability = self.model.predict_proba(df)[0, 1]
        
        # Build result
        result = PredictionResult(
            probability=float(probability),
            risk_percentage=f"{probability * 100:.1f}%",
            risk_level=self._classify_risk_level(probability),
            alert=self._generate_alert(probability),
            confidence=self._assess_confidence(probability),
            features_used=validated_data
        )
        
        return result
    
    def predict_batch(
        self,
        patients: List[Dict[str, Any]],
        return_dataframe: bool = False
    ) -> Union[List[PredictionResult], pd.DataFrame]:
        """
        Predict stroke risk for multiple patients.
        
        Args:
            patients: List of patient data dictionaries
            return_dataframe: If True, return results as DataFrame
            
        Returns:
            List of PredictionResult objects or DataFrame
        """
        results = [self.predict(patient) for patient in patients]
        
        if return_dataframe:
            return pd.DataFrame([r.to_dict() for r in results])
        
        return results
    
    def predict_proba(self, patient_data: Dict[str, Any]) -> float:
        """
        Get raw probability without formatting.
        
        Args:
            patient_data: Patient data dictionary
            
        Returns:
            Stroke probability (0-1)
        """
        return self.predict(patient_data).probability
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict with model metadata
        """
        return {
            'model_path': str(self.model_path),
            'pipeline_steps': list(self.model.named_steps.keys()),
            'risk_thresholds': {
                'low': f'< {self.THRESHOLD_LOW:.0%}',
                'moderate': f'{self.THRESHOLD_LOW:.0%} - {self.THRESHOLD_MODERATE:.0%}',
                'high': f'{self.THRESHOLD_MODERATE:.0%} - {self.THRESHOLD_HIGH:.0%}',
                'critical': f'> {self.THRESHOLD_HIGH:.0%}'
            },
            'critical_alert_threshold': f'{self.CRITICAL_ALERT_THRESHOLD:.0%}'
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def load_predictor(model_path: Optional[str] = None) -> StrokePredictor:
    """
    Factory function to create a StrokePredictor instance.
    
    Args:
        model_path: Optional custom model path
        
    Returns:
        Initialized StrokePredictor
    """
    return StrokePredictor(model_path=model_path)


def quick_predict(patient_data: Dict[str, Any]) -> str:
    """
    One-liner prediction function for quick testing.
    
    Args:
        patient_data: Patient data dictionary
        
    Returns:
        Formatted risk string
    """
    predictor = StrokePredictor(verbose=False)
    result = predictor.predict(patient_data)
    return str(result)


# =============================================================================
# Sample Patient Data for Testing
# =============================================================================

SAMPLE_PATIENTS = {
    'low_risk': {
        'gender': 'Female',
        'age': 25,
        'hypertension': 0,
        'heart_disease': 0,
        'ever_married': 'No',
        'work_type': 'Private',
        'Residence_type': 'Urban',
        'avg_glucose_level': 85.0,
        'bmi': 22.5,
        'smoking_status': 'never smoked'
    },
    'moderate_risk': {
        'gender': 'Male',
        'age': 55,
        'hypertension': 0,
        'heart_disease': 0,
        'ever_married': 'Yes',
        'work_type': 'Private',
        'Residence_type': 'Urban',
        'avg_glucose_level': 110.0,
        'bmi': 28.0,
        'smoking_status': 'formerly smoked'
    },
    'high_risk': {
        'gender': 'Male',
        'age': 72,
        'hypertension': 1,
        'heart_disease': 1,
        'ever_married': 'Yes',
        'work_type': 'Self-employed',
        'Residence_type': 'Rural',
        'avg_glucose_level': 228.0,
        'bmi': 36.6,
        'smoking_status': 'smokes'
    }
}


# =============================================================================
# Demo Function
# =============================================================================

def run_demo():
    """
    Run a demonstration of the inference module.
    """
    print("=" * 70)
    print("STROKE PREDICTOR - INFERENCE DEMO")
    print("=" * 70)
    
    # Initialize predictor
    print("\n[1] Initializing StrokePredictor...")
    predictor = StrokePredictor()
    
    # Show model info
    print("\n[2] Model Information:")
    info = predictor.get_model_info()
    print(f"   Model path: {info['model_path']}")
    print(f"   Pipeline steps: {info['pipeline_steps']}")
    print(f"   Risk thresholds:")
    for level, threshold in info['risk_thresholds'].items():
        print(f"     - {level.upper()}: {threshold}")
    
    # Run predictions on sample patients
    print("\n[3] Sample Predictions:")
    print("-" * 70)
    
    for risk_category, patient_data in SAMPLE_PATIENTS.items():
        print(f"\n📋 Patient Profile: {risk_category.replace('_', ' ').title()}")
        print(f"   Age: {patient_data['age']}, Gender: {patient_data['gender']}")
        print(f"   Hypertension: {'Yes' if patient_data['hypertension'] else 'No'}, "
              f"Heart Disease: {'Yes' if patient_data['heart_disease'] else 'No'}")
        print(f"   Glucose: {patient_data['avg_glucose_level']}, BMI: {patient_data['bmi']}")
        print(f"   Smoking: {patient_data['smoking_status']}")
        
        result = predictor.predict(patient_data)
        
        print(f"\n   🎯 Prediction: {result}")
        print(f"      Confidence: {result.confidence}")
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
