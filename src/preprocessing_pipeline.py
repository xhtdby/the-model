"""
Stroke Prediction - Preprocessing Pipeline Module
===================================================
Scikit-learn pipeline for data preprocessing without data leakage.

This module implements:
- Train/test split with stratification for class balance
- ColumnTransformer with numerical and categorical branches
- RobustScaler + KNNImputer for numerical features
- OneHotEncoder for categorical features
- Proper leakage prevention through sklearn Pipeline architecture

Author: Senior ML Engineer
Project: Healthcare AI - Stroke Prediction
"""

from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


# =============================================================================
# Feature Definitions
# =============================================================================

# Numerical features for scaling and imputation
NUMERICAL_FEATURES = [
    'age',
    'avg_glucose_level',
    'bmi'
]

# Categorical features for one-hot encoding
CATEGORICAL_FEATURES = [
    'gender',
    'hypertension',
    'heart_disease',
    'ever_married',
    'work_type',
    'Residence_type',
    'smoking_status'  # 'Unknown' is a valid category, NOT missing data
]

# Target variable
TARGET = 'stroke'


# =============================================================================
# Data Splitting
# =============================================================================

def split_data(
    df: pd.DataFrame,
    target: str = TARGET,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets with stratification.
    
    Uses stratify=y to maintain class balance in both splits, which is
    critical for imbalanced datasets like stroke prediction where the
    positive class (stroke=1) is much rarer than the negative class.
    
    Args:
        df: Input DataFrame with features and target
        target: Name of target column (default: 'stroke')
        test_size: Proportion for test set (default: 0.2 = 80/20 split)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Separate features and target
    X = df.drop(columns=[target])
    y = df[target]
    
    # Stratified split to maintain class distribution
    # This ensures both train and test sets have similar stroke/no-stroke ratios
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # CRITICAL: Maintains class balance in both splits
    )
    
    print(f"Data split complete:")
    print(f"  Training set: {len(X_train)} samples ({y_train.mean()*100:.2f}% stroke)")
    print(f"  Test set:     {len(X_test)} samples ({y_test.mean()*100:.2f}% stroke)")
    
    return X_train, X_test, y_train, y_test


# =============================================================================
# Pipeline Construction
# =============================================================================

def create_numerical_pipeline() -> Pipeline:
    """
    Create the numerical feature preprocessing pipeline.
    
    Pipeline: RobustScaler -> KNNImputer
    
    WHY SCALING BEFORE KNN IMPUTATION:
    ==================================
    KNNImputer uses distance-based calculations (typically Euclidean distance)
    to find the k nearest neighbors for imputing missing values. Without scaling:
    
    1. FEATURE MAGNITUDE DOMINANCE: Features with larger scales (e.g., 
       avg_glucose_level: 50-300) would dominate distance calculations over
       features with smaller scales (e.g., age: 0-100, bmi: 10-60).
       
    2. BIASED NEIGHBOR SELECTION: A difference of 10 in glucose_level would
       be treated the same as a difference of 10 in BMI, even though 10 points
       represents very different clinical significance for each.
       
    3. ACCURATE IMPUTATION: By scaling first, all features contribute equally
       to the distance calculation, ensuring neighbors are selected based on
       overall similarity across ALL features, not just the high-magnitude ones.
    
    WHY RobustScaler INSTEAD OF StandardScaler:
    ===========================================
    RobustScaler uses median and IQR instead of mean and std, making it:
    - Robust to outliers (glucose levels >300, extreme BMI values)
    - Better for healthcare data where outliers often represent real patients
    - Prevents outliers from skewing the scaling transformation
    
    Returns:
        sklearn Pipeline for numerical feature preprocessing
    """
    return Pipeline(
        steps=[
            # Step 1: Scale features using median/IQR (robust to outliers)
            # This normalizes all features to comparable scales BEFORE KNN
            ('scaler', RobustScaler()),
            
            # Step 2: Impute missing BMI values using K-Nearest Neighbors
            # Now that features are scaled, distance calculations are fair
            # n_neighbors=10 provides stable estimates while avoiding overfitting
            ('imputer', KNNImputer(
                n_neighbors=10,
                weights='distance',  # Closer neighbors have more influence
                metric='nan_euclidean'  # Handles NaN values in distance calc
            ))
        ]
    )


def create_categorical_pipeline() -> Pipeline:
    """
    Create the categorical feature preprocessing pipeline.
    
    Uses OneHotEncoder with:
    - handle_unknown='ignore': Safely handles unseen categories at inference
    - drop='first': Avoids multicollinearity (dummy variable trap)
    - sparse_output=False: Returns dense array for compatibility
    
    IMPORTANT: smoking_status 'Unknown' is encoded as a valid category,
    NOT treated as missing data. This is justified by our Chi-Square analysis
    which showed 'Unknown' has statistically distinct stroke risk (p < 0.05).
    
    Returns:
        sklearn Pipeline for categorical feature preprocessing
    """
    return Pipeline(
        steps=[
            ('encoder', OneHotEncoder(
                handle_unknown='ignore',  # Ignore unseen categories at inference
                drop='first',  # Drop first category to avoid multicollinearity
                sparse_output=False  # Return dense array
            ))
        ]
    )


def create_preprocessing_pipeline(
    numerical_features: List[str] = NUMERICAL_FEATURES,
    categorical_features: List[str] = CATEGORICAL_FEATURES
) -> ColumnTransformer:
    """
    Create the complete preprocessing pipeline with ColumnTransformer.
    
    Architecture:
    =============
    ColumnTransformer applies different transformations to different columns:
    
    ┌─────────────────────────────────────────────────────────────┐
    │                    ColumnTransformer                        │
    │  ┌─────────────────────────┐  ┌─────────────────────────┐  │
    │  │   Numerical Branch      │  │   Categorical Branch    │  │
    │  │   ─────────────────     │  │   ──────────────────    │  │
    │  │   age                   │  │   gender                │  │
    │  │   avg_glucose_level ───►│  │   hypertension      ───►│  │
    │  │   bmi                   │  │   heart_disease         │  │
    │  │         │               │  │   ever_married          │  │
    │  │         ▼               │  │   work_type             │  │
    │  │   RobustScaler          │  │   Residence_type        │  │
    │  │         │               │  │   smoking_status        │  │
    │  │         ▼               │  │         │               │  │
    │  │   KNNImputer            │  │         ▼               │  │
    │  │   (n_neighbors=10)      │  │   OneHotEncoder         │  │
    │  └─────────────────────────┘  └─────────────────────────┘  │
    │                    │                      │                 │
    │                    └──────────┬───────────┘                 │
    │                               ▼                             │
    │                    Concatenated Output                      │
    └─────────────────────────────────────────────────────────────┘
    
    DATA LEAKAGE PREVENTION:
    ========================
    sklearn.pipeline.Pipeline prevents data leakage by design:
    
    1. FIT vs TRANSFORM SEPARATION:
       - fit(): Learns statistics (scaling params, encoder categories) from data
       - transform(): Applies learned statistics to data
       - fit_transform(): Convenience method that does both
    
    2. TRAINING PHASE (pipeline.fit(X_train)):
       - RobustScaler learns median/IQR from X_train ONLY
       - KNNImputer learns neighbor structure from X_train ONLY
       - OneHotEncoder learns categories from X_train ONLY
    
    3. INFERENCE PHASE (pipeline.transform(X_test)):
       - Uses statistics learned from X_train
       - Test data NEVER influences the learned parameters
       - Even if test set has extreme outliers, they don't affect scaling
    
    4. CROSS-VALIDATION SAFETY:
       - When used with cross_val_score or GridSearchCV, each fold:
         * Fits on training fold only
         * Transforms validation fold using training fold statistics
       - No information from validation fold leaks into the model
    
    This is why we NEVER do:
        scaler.fit(X)  # BAD: Fits on all data including test
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
    Instead, the pipeline ensures:
        pipeline.fit(X_train)  # GOOD: Fits only on training data
        X_train_transformed = pipeline.transform(X_train)
        X_test_transformed = pipeline.transform(X_test)  # Uses train statistics
    
    Args:
        numerical_features: List of numerical column names
        categorical_features: List of categorical column names
        
    Returns:
        ColumnTransformer with configured preprocessing pipelines
    """
    preprocessor = ColumnTransformer(
        transformers=[
            # Numerical branch: RobustScaler -> KNNImputer
            ('numerical', create_numerical_pipeline(), numerical_features),
            
            # Categorical branch: OneHotEncoder
            ('categorical', create_categorical_pipeline(), categorical_features)
        ],
        # Keep column order consistent
        remainder='drop',  # Drop any columns not specified
        verbose_feature_names_out=True  # Prefix output names with transformer name
    )
    
    return preprocessor


def get_feature_names(
    preprocessor: ColumnTransformer,
    numerical_features: List[str] = NUMERICAL_FEATURES,
    categorical_features: List[str] = CATEGORICAL_FEATURES
) -> List[str]:
    """
    Get feature names after transformation.
    
    Args:
        preprocessor: Fitted ColumnTransformer
        numerical_features: Original numerical feature names
        categorical_features: Original categorical feature names
        
    Returns:
        List of feature names in transformed output
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError:
        # Fallback for older sklearn versions
        # Numerical features keep their names
        num_names = [f'numerical__{f}' for f in numerical_features]
        
        # Categorical features get one-hot encoded names
        cat_encoder = preprocessor.named_transformers_['categorical'].named_steps['encoder']
        cat_names = [f'categorical__{name}' for name in cat_encoder.get_feature_names_out(categorical_features)]
        
        return num_names + cat_names


# =============================================================================
# Complete Preprocessing Function
# =============================================================================

def preprocess_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, ColumnTransformer, List[str]]:
    """
    Complete preprocessing pipeline execution.
    
    This function:
    1. Splits data with stratification
    2. Fits preprocessor on training data only (no leakage)
    3. Transforms both train and test sets
    4. Returns transformed arrays and fitted preprocessor
    
    Args:
        df: Input DataFrame with all features and target
        test_size: Proportion for test set
        random_state: Random seed
        
    Returns:
        Tuple of:
        - X_train_transformed: Transformed training features (np.ndarray)
        - X_test_transformed: Transformed test features (np.ndarray)
        - y_train: Training labels
        - y_test: Test labels
        - preprocessor: Fitted ColumnTransformer (for inference)
        - feature_names: List of feature names after transformation
    """
    # Step 1: Split data with stratification
    X_train, X_test, y_train, y_test = split_data(
        df, test_size=test_size, random_state=random_state
    )
    
    # Step 2: Create preprocessor
    preprocessor = create_preprocessing_pipeline()
    
    # Step 3: Fit on training data ONLY, then transform
    # This is where leakage prevention happens - fit() only sees X_train
    X_train_transformed = preprocessor.fit_transform(X_train)
    
    # Step 4: Transform test data using statistics from training data
    # The test data does NOT influence the learned parameters
    X_test_transformed = preprocessor.transform(X_test)
    
    # Step 5: Get feature names for interpretability
    feature_names = get_feature_names(preprocessor)
    
    print(f"\nPreprocessing complete:")
    print(f"  Original features: {X_train.shape[1]}")
    print(f"  Transformed features: {X_train_transformed.shape[1]}")
    print(f"  Feature names: {feature_names[:5]}... (showing first 5)")
    
    return (
        X_train_transformed,
        X_test_transformed,
        y_train,
        y_test,
        preprocessor,
        feature_names
    )


# =============================================================================
# Utility Functions
# =============================================================================

def transform_new_data(
    preprocessor: ColumnTransformer,
    new_data: pd.DataFrame
) -> np.ndarray:
    """
    Transform new data using a fitted preprocessor.
    
    IMPORTANT: This uses the statistics learned during training.
    New data does not influence the transformation parameters.
    
    Args:
        preprocessor: Fitted ColumnTransformer from training
        new_data: New DataFrame to transform
        
    Returns:
        Transformed features as numpy array
    """
    return preprocessor.transform(new_data)


def verify_no_missing_values(X_transformed: np.ndarray) -> bool:
    """
    Verify that transformation eliminated all missing values.
    
    Args:
        X_transformed: Transformed feature array
        
    Returns:
        True if no missing values, raises AssertionError otherwise
    """
    n_missing = np.isnan(X_transformed).sum()
    assert n_missing == 0, f"Found {n_missing} missing values after transformation!"
    return True


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
    default_path = Path(__file__).parent.parent / "healthcare-dataset-stroke-data.csv"
    df = load_stroke_data(default_path)
    
    # Run preprocessing
    (X_train, X_test, y_train, y_test, 
     preprocessor, feature_names) = preprocess_data(df)
    
    # Verify no missing values
    verify_no_missing_values(X_train)
    verify_no_missing_values(X_test)
    
    print("\n✓ All preprocessing completed successfully!")
    print(f"✓ No missing values in transformed data")
    print(f"✓ Training set shape: {X_train.shape}")
    print(f"✓ Test set shape: {X_test.shape}")
