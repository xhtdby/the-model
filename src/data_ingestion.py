"""
Stroke Prediction - Data Ingestion Module
==========================================
This module handles data loading, schema validation, and sanitization
for the healthcare stroke prediction dataset.

Author: Senior ML Engineer
Project: Healthcare AI - Stroke Prediction
"""

import warnings
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema
from pandera.errors import SchemaError


# =============================================================================
# Schema Definition
# =============================================================================

# Define valid categories for smoking_status (including 'Unknown')
VALID_SMOKING_STATUS = ["formerly smoked", "never smoked", "smokes", "Unknown"]

# Define valid categories for other categorical columns
VALID_GENDER = ["Male", "Female", "Other"]
VALID_WORK_TYPE = ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
VALID_RESIDENCE_TYPE = ["Urban", "Rural"]
VALID_EVER_MARRIED = ["Yes", "No"]


def _check_glucose_outlier(glucose_series: pd.Series) -> pd.Series:
    """
    Check for potential glucose outliers and raise warnings.
    
    Args:
        glucose_series: Series of avg_glucose_level values
        
    Returns:
        Boolean series (all True for validation pass)
    """
    outlier_mask = glucose_series > 300
    if outlier_mask.any():
        outlier_count = outlier_mask.sum()
        outlier_values = glucose_series[outlier_mask].tolist()
        warnings.warn(
            f"Potential outliers detected: {outlier_count} glucose values > 300. "
            f"Values: {outlier_values[:5]}{'...' if len(outlier_values) > 5 else ''}",
            UserWarning
        )
    # Return all True - this is a warning check, not a failure condition
    return pd.Series([True] * len(glucose_series), index=glucose_series.index)


def _check_unknown_in_smoking_status(smoking_series: pd.Series) -> bool:
    """
    Validate that 'Unknown' is present as a level in smoking_status.
    
    Args:
        smoking_series: Series of smoking_status values
        
    Returns:
        True if 'Unknown' is present in the unique values
    """
    unique_values = smoking_series.unique()
    return "Unknown" in unique_values


# Define the strict schema for stroke prediction data
STROKE_DATA_SCHEMA = DataFrameSchema(
    columns={
        "gender": Column(
            str,
            Check.isin(VALID_GENDER),
            nullable=False,
            description="Patient gender"
        ),
        "age": Column(
            float,
            Check.greater_than(0, error="Age must be greater than 0"),
            nullable=False,
            coerce=True,
            description="Patient age in years"
        ),
        "hypertension": Column(
            int,
            Check.isin([0, 1]),
            nullable=False,
            coerce=True,
            description="Hypertension indicator (0=No, 1=Yes)"
        ),
        "heart_disease": Column(
            int,
            Check.isin([0, 1]),
            nullable=False,
            coerce=True,
            description="Heart disease indicator (0=No, 1=Yes)"
        ),
        "ever_married": Column(
            str,
            Check.isin(VALID_EVER_MARRIED),
            nullable=False,
            description="Marriage status"
        ),
        "work_type": Column(
            str,
            Check.isin(VALID_WORK_TYPE),
            nullable=False,
            description="Type of work"
        ),
        "Residence_type": Column(
            str,
            Check.isin(VALID_RESIDENCE_TYPE),
            nullable=False,
            description="Residence type (Urban/Rural)"
        ),
        "avg_glucose_level": Column(
            float,
            [
                Check.greater_than(0, error="Glucose level must be greater than 0"),
                Check(
                    _check_glucose_outlier,
                    element_wise=False,
                    error="Glucose outlier check failed"
                )
            ],
            nullable=False,
            coerce=True,
            description="Average glucose level"
        ),
        "bmi": Column(
            float,
            Check.greater_than(0, error="BMI must be greater than 0"),
            nullable=True,  # Allow NaNs for later imputation
            coerce=True,
            description="Body Mass Index (nullable for imputation)"
        ),
        "smoking_status": Column(
            str,
            Check.isin(VALID_SMOKING_STATUS),
            nullable=False,
            description="Smoking status category"
        ),
        "stroke": Column(
            int,
            Check.isin([0, 1]),
            nullable=False,
            coerce=True,
            description="Stroke outcome (0=No, 1=Yes)"
        ),
    },
    strict=True,  # Only allow columns defined in schema
    coerce=True,  # Coerce data types where possible
)


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_stroke_data(
    filepath: Union[str, Path],
    validate: bool = True,
    drop_id: bool = True
) -> pd.DataFrame:
    """
    Load and optionally validate the stroke prediction dataset.
    
    This function performs the following operations:
    1. Loads CSV data from the specified filepath
    2. Sanitizes data by dropping the 'id' column (prevents spurious correlations)
    3. Handles 'N/A' strings in BMI column as proper NaN values
    4. Validates data against the defined schema (if validate=True)
    
    Args:
        filepath: Path to the CSV file containing stroke data
        validate: Whether to validate against schema (default: True)
        drop_id: Whether to drop the 'id' column (default: True)
        
    Returns:
        pd.DataFrame: Validated and sanitized stroke dataset
        
    Raises:
        FileNotFoundError: If the specified file does not exist
        SchemaError: If validation fails (when validate=True)
        AssertionError: If dataframe shape is incorrect after dropping 'id'
        
    Example:
        >>> df = load_stroke_data("healthcare-dataset-stroke-data.csv")
        >>> print(df.shape)
        (5110, 11)
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    # Load data with 'N/A' treated as NaN
    df = pd.read_csv(filepath, na_values=["N/A", "N/a", "n/a", "NA", ""])
    
    # Store original shape for assertion
    original_columns = df.shape[1]
    original_rows = df.shape[0]
    
    # Sanitization: Drop 'id' column to prevent spurious correlations
    if drop_id and "id" in df.columns:
        df = df.drop(columns=["id"])
        
        # Assert correct shape after dropping
        expected_columns = original_columns - 1
        assert df.shape[1] == expected_columns, (
            f"Column count mismatch after dropping 'id': "
            f"expected {expected_columns}, got {df.shape[1]}"
        )
        assert df.shape[0] == original_rows, (
            f"Row count changed after dropping 'id': "
            f"expected {original_rows}, got {df.shape[0]}"
        )
    
    # Validate against schema if requested
    if validate:
        df = validate_stroke_data(df)
    
    return df


def validate_stroke_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate stroke data against the defined schema.
    
    Performs strict schema validation including:
    - Type checking and coercion
    - Range validation (age > 0, glucose > 0, etc.)
    - Categorical validation (smoking_status, gender, etc.)
    - Outlier warnings (glucose > 300)
    - Ensures 'Unknown' is present in smoking_status
    
    Args:
        df: DataFrame to validate
        
    Returns:
        pd.DataFrame: Validated DataFrame with coerced types
        
    Raises:
        SchemaError: If validation fails
        ValueError: If 'Unknown' not present in smoking_status
    """
    # Validate against schema
    validated_df = STROKE_DATA_SCHEMA.validate(df, lazy=True)
    
    # Additional check: Ensure 'Unknown' is present in smoking_status
    if not _check_unknown_in_smoking_status(validated_df["smoking_status"]):
        raise ValueError(
            "smoking_status column must contain 'Unknown' as a valid level. "
            f"Found levels: {validated_df['smoking_status'].unique().tolist()}"
        )
    
    return validated_df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate a summary of the loaded stroke dataset.
    
    Args:
        df: Validated stroke DataFrame
        
    Returns:
        dict: Summary statistics and data quality metrics
    """
    summary = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "stroke_distribution": df["stroke"].value_counts().to_dict(),
        "age_stats": {
            "min": df["age"].min(),
            "max": df["age"].max(),
            "mean": df["age"].mean(),
            "median": df["age"].median()
        },
        "glucose_stats": {
            "min": df["avg_glucose_level"].min(),
            "max": df["avg_glucose_level"].max(),
            "mean": df["avg_glucose_level"].mean(),
            "outliers_above_300": (df["avg_glucose_level"] > 300).sum()
        },
        "smoking_status_levels": df["smoking_status"].unique().tolist(),
        "bmi_missing_count": df["bmi"].isnull().sum()
    }
    return summary


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Example usage
    import sys
    
    # Default path
    default_path = Path(__file__).parent.parent / "healthcare-dataset-stroke-data.csv"
    
    filepath = sys.argv[1] if len(sys.argv) > 1 else default_path
    
    try:
        print(f"Loading stroke data from: {filepath}")
        df = load_stroke_data(filepath)
        print(f"\nData loaded successfully!")
        print(f"Shape: {df.shape}")
        
        summary = get_data_summary(df)
        print(f"\nData Summary:")
        print(f"  - Stroke cases: {summary['stroke_distribution']}")
        print(f"  - Age range: {summary['age_stats']['min']:.1f} - {summary['age_stats']['max']:.1f}")
        print(f"  - Missing BMI values: {summary['bmi_missing_count']}")
        print(f"  - Smoking status levels: {summary['smoking_status_levels']}")
        print(f"  - Glucose outliers (>300): {summary['glucose_stats']['outliers_above_300']}")
        
    except SchemaError as e:
        print(f"Validation Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
