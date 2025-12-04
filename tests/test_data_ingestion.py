"""
Unit Tests for Stroke Prediction Data Ingestion
================================================
Tests for data loading, validation, and sanitization functions.

Author: Senior ML Engineer
Project: Healthcare AI - Stroke Prediction
"""

import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

# Import module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_ingestion import (
    load_stroke_data,
    validate_stroke_data,
    get_data_summary,
    STROKE_DATA_SCHEMA,
    VALID_SMOKING_STATUS,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def valid_sample_data():
    """Create a valid sample DataFrame for testing."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "gender": ["Male", "Female", "Male", "Female", "Other"],
        "age": [67.0, 61.0, 80.0, 49.0, 35.0],
        "hypertension": [0, 0, 1, 0, 0],
        "heart_disease": [1, 0, 1, 0, 0],
        "ever_married": ["Yes", "Yes", "Yes", "No", "Yes"],
        "work_type": ["Private", "Self-employed", "Private", "Govt_job", "Private"],
        "Residence_type": ["Urban", "Rural", "Urban", "Rural", "Urban"],
        "avg_glucose_level": [228.69, 202.21, 105.92, 171.23, 85.5],
        "bmi": [36.6, np.nan, 32.5, 34.4, 25.0],  # Include NaN for BMI
        "smoking_status": ["formerly smoked", "never smoked", "never smoked", "smokes", "Unknown"],
        "stroke": [1, 1, 1, 0, 0]
    })


@pytest.fixture
def valid_csv_file(valid_sample_data, tmp_path):
    """Create a temporary valid CSV file."""
    csv_path = tmp_path / "valid_data.csv"
    valid_sample_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def invalid_negative_age_data():
    """Create data with invalid negative age."""
    return pd.DataFrame({
        "id": [1, 2],
        "gender": ["Male", "Female"],
        "age": [-5.0, 30.0],  # INVALID: Negative age
        "hypertension": [0, 0],
        "heart_disease": [0, 0],
        "ever_married": ["Yes", "No"],
        "work_type": ["Private", "Private"],
        "Residence_type": ["Urban", "Rural"],
        "avg_glucose_level": [100.0, 90.0],
        "bmi": [25.0, 28.0],
        "smoking_status": ["never smoked", "Unknown"],
        "stroke": [0, 0]
    })


# =============================================================================
# Test Cases: Data Loading
# =============================================================================

class TestDataLoading:
    """Tests for the load_stroke_data function."""
    
    def test_load_valid_csv(self, valid_csv_file):
        """Test loading a valid CSV file."""
        df = load_stroke_data(valid_csv_file)
        
        # Check that data was loaded
        assert df is not None
        assert len(df) == 5
        
        # Check that 'id' column was dropped
        assert "id" not in df.columns
        assert df.shape[1] == 11  # 12 columns - 1 (id) = 11
    
    def test_load_nonexistent_file(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            load_stroke_data("nonexistent_file.csv")
    
    def test_id_column_dropped(self, valid_csv_file):
        """Test that id column is properly dropped."""
        df = load_stroke_data(valid_csv_file, drop_id=True)
        assert "id" not in df.columns
    
    def test_id_column_preserved_when_requested(self, valid_csv_file):
        """Test that id column can be preserved."""
        df = load_stroke_data(valid_csv_file, drop_id=False, validate=False)
        assert "id" in df.columns
    
    def test_na_values_handled(self, tmp_path):
        """Test that 'N/A' strings are converted to NaN."""
        # Create CSV with N/A string values
        csv_content = """id,gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status,stroke
1,Male,67,0,1,Yes,Private,Urban,228.69,N/A,Unknown,1
2,Female,61,0,0,Yes,Self-employed,Rural,202.21,25.5,never smoked,0"""
        
        csv_path = tmp_path / "na_test.csv"
        csv_path.write_text(csv_content)
        
        df = load_stroke_data(csv_path)
        
        # Check that N/A was converted to NaN
        assert pd.isna(df["bmi"].iloc[0])
        assert df["bmi"].iloc[1] == 25.5


# =============================================================================
# Test Cases: Schema Validation
# =============================================================================

class TestSchemaValidation:
    """Tests for schema validation."""
    
    def test_valid_data_passes_validation(self, valid_sample_data):
        """Test that valid data passes schema validation."""
        # Remove id column as it would be dropped in normal flow
        df = valid_sample_data.drop(columns=["id"])
        validated_df = validate_stroke_data(df)
        assert validated_df is not None
        assert len(validated_df) == 5
    
    def test_negative_age_raises_error(self, invalid_negative_age_data, tmp_path):
        """Test that negative age raises SchemaError/SchemaErrors."""
        csv_path = tmp_path / "invalid_age.csv"
        invalid_negative_age_data.to_csv(csv_path, index=False)
        
        with pytest.raises((SchemaError, SchemaErrors)) as exc_info:
            load_stroke_data(csv_path)
        
        # Check that the error message mentions age validation
        assert "age" in str(exc_info.value).lower() or "greater_than" in str(exc_info.value).lower()
    
    def test_negative_glucose_raises_error(self, tmp_path):
        """Test that negative glucose level raises SchemaError."""
        invalid_data = pd.DataFrame({
            "id": [1],
            "gender": ["Male"],
            "age": [50.0],
            "hypertension": [0],
            "heart_disease": [0],
            "ever_married": ["Yes"],
            "work_type": ["Private"],
            "Residence_type": ["Urban"],
            "avg_glucose_level": [-50.0],  # INVALID: Negative glucose
            "bmi": [25.0],
            "smoking_status": ["Unknown"],
            "stroke": [0]
        })
        
        csv_path = tmp_path / "invalid_glucose.csv"
        invalid_data.to_csv(csv_path, index=False)
        
        with pytest.raises((SchemaError, SchemaErrors)):
            load_stroke_data(csv_path)
    
    def test_invalid_stroke_value_raises_error(self, tmp_path):
        """Test that stroke values outside {0, 1} raise SchemaError."""
        invalid_data = pd.DataFrame({
            "id": [1],
            "gender": ["Male"],
            "age": [50.0],
            "hypertension": [0],
            "heart_disease": [0],
            "ever_married": ["Yes"],
            "work_type": ["Private"],
            "Residence_type": ["Urban"],
            "avg_glucose_level": [100.0],
            "bmi": [25.0],
            "smoking_status": ["Unknown"],
            "stroke": [2]  # INVALID: Not in {0, 1}
        })
        
        csv_path = tmp_path / "invalid_stroke.csv"
        invalid_data.to_csv(csv_path, index=False)
        
        with pytest.raises((SchemaError, SchemaErrors)):
            load_stroke_data(csv_path)
    
    def test_invalid_smoking_status_raises_error(self, tmp_path):
        """Test that invalid smoking status raises SchemaError."""
        invalid_data = pd.DataFrame({
            "id": [1, 2],
            "gender": ["Male", "Female"],
            "age": [50.0, 45.0],
            "hypertension": [0, 0],
            "heart_disease": [0, 0],
            "ever_married": ["Yes", "Yes"],
            "work_type": ["Private", "Private"],
            "Residence_type": ["Urban", "Rural"],
            "avg_glucose_level": [100.0, 95.0],
            "bmi": [25.0, 27.0],
            "smoking_status": ["invalid_status", "Unknown"],  # INVALID status
            "stroke": [0, 0]
        })
        
        csv_path = tmp_path / "invalid_smoking.csv"
        invalid_data.to_csv(csv_path, index=False)
        
        with pytest.raises((SchemaError, SchemaErrors)):
            load_stroke_data(csv_path)
    
    def test_bmi_allows_nan(self, valid_sample_data):
        """Test that BMI column properly allows NaN values."""
        df = valid_sample_data.drop(columns=["id"])
        
        # Ensure there's a NaN in BMI
        assert df["bmi"].isna().any()
        
        # Validation should pass with NaN BMI
        validated_df = validate_stroke_data(df)
        assert validated_df["bmi"].isna().any()
    
    def test_unknown_required_in_smoking_status(self, tmp_path):
        """Test that 'Unknown' must be present in smoking_status."""
        # Data without 'Unknown' in smoking_status
        data_without_unknown = pd.DataFrame({
            "id": [1, 2],
            "gender": ["Male", "Female"],
            "age": [50.0, 45.0],
            "hypertension": [0, 0],
            "heart_disease": [0, 0],
            "ever_married": ["Yes", "Yes"],
            "work_type": ["Private", "Private"],
            "Residence_type": ["Urban", "Rural"],
            "avg_glucose_level": [100.0, 95.0],
            "bmi": [25.0, 27.0],
            "smoking_status": ["never smoked", "smokes"],  # No 'Unknown'
            "stroke": [0, 0]
        })
        
        csv_path = tmp_path / "no_unknown.csv"
        data_without_unknown.to_csv(csv_path, index=False)
        
        with pytest.raises(ValueError) as exc_info:
            load_stroke_data(csv_path)
        
        assert "Unknown" in str(exc_info.value)


# =============================================================================
# Test Cases: Glucose Outlier Warning
# =============================================================================

class TestGlucoseOutlierWarning:
    """Tests for glucose outlier warnings."""
    
    def test_glucose_above_300_triggers_warning(self, tmp_path):
        """Test that glucose values > 300 trigger a warning."""
        data_with_outlier = pd.DataFrame({
            "id": [1, 2, 3],
            "gender": ["Male", "Female", "Male"],
            "age": [50.0, 45.0, 60.0],
            "hypertension": [0, 0, 1],
            "heart_disease": [0, 0, 0],
            "ever_married": ["Yes", "Yes", "Yes"],
            "work_type": ["Private", "Private", "Private"],
            "Residence_type": ["Urban", "Rural", "Urban"],
            "avg_glucose_level": [350.0, 95.0, 400.0],  # Outliers > 300
            "bmi": [25.0, 27.0, 30.0],
            "smoking_status": ["never smoked", "smokes", "Unknown"],
            "stroke": [0, 0, 1]
        })
        
        csv_path = tmp_path / "outlier_glucose.csv"
        data_with_outlier.to_csv(csv_path, index=False)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = load_stroke_data(csv_path)
            
            # Check that a warning was raised
            glucose_warnings = [
                warning for warning in w 
                if "glucose" in str(warning.message).lower() or "300" in str(warning.message)
            ]
            assert len(glucose_warnings) >= 1


# =============================================================================
# Test Cases: Data Sanitization
# =============================================================================

class TestDataSanitization:
    """Tests for data sanitization."""
    
    def test_shape_assertion_after_id_drop(self, valid_csv_file):
        """Test that shape is correctly asserted after dropping id."""
        df = load_stroke_data(valid_csv_file)
        
        # Original had 12 columns, after dropping 'id' should have 11
        assert df.shape[1] == 11
        # Row count should remain the same
        assert df.shape[0] == 5
    
    def test_no_id_column_in_output(self, valid_csv_file):
        """Test that id column is never in final output."""
        df = load_stroke_data(valid_csv_file)
        assert "id" not in df.columns


# =============================================================================
# Test Cases: Data Summary
# =============================================================================

class TestDataSummary:
    """Tests for the get_data_summary function."""
    
    def test_summary_contains_required_keys(self, valid_sample_data):
        """Test that summary contains all required information."""
        df = valid_sample_data.drop(columns=["id"])
        validated_df = validate_stroke_data(df)
        summary = get_data_summary(validated_df)
        
        required_keys = [
            "shape", "columns", "missing_values", "stroke_distribution",
            "age_stats", "glucose_stats", "smoking_status_levels", "bmi_missing_count"
        ]
        
        for key in required_keys:
            assert key in summary, f"Missing key: {key}"
    
    def test_summary_bmi_missing_count(self, valid_sample_data):
        """Test that BMI missing count is correctly calculated."""
        df = valid_sample_data.drop(columns=["id"])
        validated_df = validate_stroke_data(df)
        summary = get_data_summary(validated_df)
        
        # We have 1 NaN BMI in our fixture
        assert summary["bmi_missing_count"] == 1


# =============================================================================
# Test Cases: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_age_exactly_zero_raises_error(self, tmp_path):
        """Test that age = 0 raises an error (must be > 0)."""
        data_with_zero_age = pd.DataFrame({
            "id": [1],
            "gender": ["Male"],
            "age": [0.0],  # INVALID: Must be > 0
            "hypertension": [0],
            "heart_disease": [0],
            "ever_married": ["No"],
            "work_type": ["children"],
            "Residence_type": ["Urban"],
            "avg_glucose_level": [80.0],
            "bmi": [15.0],
            "smoking_status": ["Unknown"],
            "stroke": [0]
        })
        
        csv_path = tmp_path / "zero_age.csv"
        data_with_zero_age.to_csv(csv_path, index=False)
        
        with pytest.raises((SchemaError, SchemaErrors)):
            load_stroke_data(csv_path)
    
    def test_very_young_age_valid(self, tmp_path):
        """Test that very young ages (e.g., infants) are valid."""
        data_with_infant = pd.DataFrame({
            "id": [1],
            "gender": ["Female"],
            "age": [0.5],  # 6 months old - valid
            "hypertension": [0],
            "heart_disease": [0],
            "ever_married": ["No"],
            "work_type": ["children"],
            "Residence_type": ["Urban"],
            "avg_glucose_level": [70.0],
            "bmi": [15.0],
            "smoking_status": ["Unknown"],
            "stroke": [0]
        })
        
        csv_path = tmp_path / "infant_age.csv"
        data_with_infant.to_csv(csv_path, index=False)
        
        df = load_stroke_data(csv_path)
        assert df["age"].iloc[0] == 0.5
    
    def test_all_valid_smoking_statuses(self, tmp_path):
        """Test that all valid smoking statuses are accepted."""
        data_with_all_statuses = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "gender": ["Male", "Female", "Male", "Female"],
            "age": [50.0, 45.0, 60.0, 55.0],
            "hypertension": [0, 0, 1, 0],
            "heart_disease": [0, 0, 0, 1],
            "ever_married": ["Yes", "Yes", "Yes", "No"],
            "work_type": ["Private", "Private", "Govt_job", "Self-employed"],
            "Residence_type": ["Urban", "Rural", "Urban", "Rural"],
            "avg_glucose_level": [100.0, 95.0, 110.0, 85.0],
            "bmi": [25.0, 27.0, 30.0, 22.0],
            "smoking_status": ["formerly smoked", "never smoked", "smokes", "Unknown"],
            "stroke": [0, 0, 1, 0]
        })
        
        csv_path = tmp_path / "all_smoking_statuses.csv"
        data_with_all_statuses.to_csv(csv_path, index=False)
        
        df = load_stroke_data(csv_path)
        assert set(df["smoking_status"].unique()) == set(VALID_SMOKING_STATUS)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
