"""
Unit Tests for Preprocessing Pipeline
======================================
Tests for data leakage prevention and pipeline correctness.

Author: Senior ML Engineer
Project: Healthcare AI - Stroke Prediction
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing_pipeline import (
    create_preprocessing_pipeline,
    create_numerical_pipeline,
    create_categorical_pipeline,
    split_data,
    preprocess_data,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame mimicking the stroke dataset structure."""
    np.random.seed(42)
    n_samples = 500
    
    return pd.DataFrame({
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'age': np.random.uniform(1, 80, n_samples),
        'hypertension': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'heart_disease': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'ever_married': np.random.choice(['Yes', 'No'], n_samples),
        'work_type': np.random.choice(
            ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'],
            n_samples
        ),
        'Residence_type': np.random.choice(['Urban', 'Rural'], n_samples),
        'avg_glucose_level': np.random.uniform(50, 300, n_samples),
        # Include some NaN values in BMI to test imputation
        'bmi': np.where(
            np.random.random(n_samples) < 0.05,  # 5% missing
            np.nan,
            np.random.uniform(15, 50, n_samples)
        ),
        'smoking_status': np.random.choice(
            ['formerly smoked', 'never smoked', 'smokes', 'Unknown'],
            n_samples
        ),
        'stroke': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    })


@pytest.fixture
def sample_dataframe_with_more_missing():
    """Create DataFrame with higher missing rate for robust testing."""
    np.random.seed(42)
    n_samples = 200
    
    df = pd.DataFrame({
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'age': np.random.uniform(1, 80, n_samples),
        'hypertension': np.random.choice([0, 1], n_samples),
        'heart_disease': np.random.choice([0, 1], n_samples),
        'ever_married': np.random.choice(['Yes', 'No'], n_samples),
        'work_type': np.random.choice(['Private', 'Self-employed', 'Govt_job'], n_samples),
        'Residence_type': np.random.choice(['Urban', 'Rural'], n_samples),
        'avg_glucose_level': np.random.uniform(50, 300, n_samples),
        'bmi': np.where(
            np.random.random(n_samples) < 0.15,  # 15% missing
            np.nan,
            np.random.uniform(15, 50, n_samples)
        ),
        'smoking_status': np.random.choice(
            ['formerly smoked', 'never smoked', 'smokes', 'Unknown'],
            n_samples
        ),
        'stroke': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    })
    return df


# =============================================================================
# Test: Data Splitting
# =============================================================================

class TestDataSplitting:
    """Tests for stratified data splitting."""
    
    def test_split_maintains_proportions(self, sample_dataframe):
        """Test that stratified split maintains class proportions."""
        X_train, X_test, y_train, y_test = split_data(sample_dataframe)
        
        original_ratio = sample_dataframe['stroke'].mean()
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        
        # Ratios should be very close (within 2% of each other)
        assert abs(original_ratio - train_ratio) < 0.02
        assert abs(original_ratio - test_ratio) < 0.02
    
    def test_split_sizes_correct(self, sample_dataframe):
        """Test that split produces correct sizes (80/20)."""
        X_train, X_test, y_train, y_test = split_data(
            sample_dataframe, test_size=0.2
        )
        
        total = len(sample_dataframe)
        assert len(X_train) == pytest.approx(total * 0.8, rel=0.02)
        assert len(X_test) == pytest.approx(total * 0.2, rel=0.02)
    
    def test_split_no_data_overlap(self, sample_dataframe):
        """Test that train and test sets don't overlap."""
        X_train, X_test, y_train, y_test = split_data(sample_dataframe)
        
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        
        assert len(train_indices.intersection(test_indices)) == 0


# =============================================================================
# Test: Pipeline Leakage Prevention (CRITICAL TEST)
# =============================================================================

class TestPipelineLeakage:
    """
    Tests to verify that the pipeline prevents data leakage.
    
    Data leakage occurs when information from outside the training dataset
    is used to create the model. In preprocessing, this happens when:
    - Scaling uses statistics from test data
    - Imputation uses values from test data
    - Encoding learns categories from test data
    
    These tests verify that the sklearn Pipeline architecture properly
    isolates training and test data during fit/transform operations.
    """
    
    def test_pipeline_leakage(self, sample_dataframe):
        """
        CRITICAL TEST: Verify that outliers in test set don't affect training transformation.
        
        Test Protocol:
        ==============
        1. Split data into train/test
        2. Fit pipeline on training data and transform it -> R1
        3. Add extreme outliers to test set
        4. Re-transform training data using the same fitted pipeline -> R2
        5. Assert R1 == R2 (bitwise identical)
        
        If the test fails, it means test data is leaking into the training
        transformation, which would invalidate any model performance metrics.
        """
        # Step 1: Split the data
        X_train, X_test, y_train, y_test = split_data(
            sample_dataframe, test_size=0.2, random_state=42
        )
        
        # Step 2: Create and fit pipeline on training data ONLY
        preprocessor = create_preprocessing_pipeline()
        preprocessor.fit(X_train)
        
        # Step 3: Transform training data -> R1
        R1 = preprocessor.transform(X_train)
        
        # Step 4: Create extreme outliers and add to test set
        # These outliers should NOT affect the training transformation
        extreme_outliers = pd.DataFrame({
            'gender': ['Male', 'Female'],
            'age': [999.0, 1000.0],  # Extreme age outliers
            'hypertension': [1, 1],
            'heart_disease': [1, 1],
            'ever_married': ['Yes', 'No'],
            'work_type': ['Private', 'Private'],
            'Residence_type': ['Urban', 'Rural'],
            'avg_glucose_level': [9999.0, 10000.0],  # Extreme glucose outliers
            'bmi': [999.0, 1000.0],  # Extreme BMI outliers
            'smoking_status': ['Unknown', 'smokes']
        })
        
        # Append outliers to test set (simulating new extreme data)
        X_test_with_outliers = pd.concat([X_test, extreme_outliers], ignore_index=True)
        
        # Step 5: Transform test set with outliers (should not affect preprocessor state)
        _ = preprocessor.transform(X_test_with_outliers)
        
        # Step 6: Re-transform training data -> R2
        R2 = preprocessor.transform(X_train)
        
        # Step 7: CRITICAL ASSERTION - R1 and R2 must be IDENTICAL
        # If this fails, test data is leaking into the training transformation
        np.testing.assert_array_equal(
            R1, R2,
            err_msg=(
                "DATA LEAKAGE DETECTED!\n"
                "Training set transformation changed after processing test set.\n"
                "This indicates that test data statistics are leaking into "
                "the training transformation, which invalidates model evaluation."
            )
        )
    
    def test_pipeline_leakage_with_new_categories(self, sample_dataframe):
        """
        Test that new categories in test set don't affect training transformation.
        
        This verifies that OneHotEncoder's handle_unknown='ignore' works correctly
        and doesn't retroactively affect the training transformation.
        """
        # Split data
        X_train, X_test, y_train, y_test = split_data(
            sample_dataframe, test_size=0.2, random_state=42
        )
        
        # Fit pipeline
        preprocessor = create_preprocessing_pipeline()
        preprocessor.fit(X_train)
        
        # Transform training data
        R1 = preprocessor.transform(X_train)
        
        # Create data with unseen categories
        unseen_categories = pd.DataFrame({
            'gender': ['Other', 'Unknown_Gender'],  # Unseen gender
            'age': [50.0, 60.0],
            'hypertension': [0, 1],
            'heart_disease': [0, 0],
            'ever_married': ['Maybe', 'Unknown_Marriage'],  # Unseen marriage status
            'work_type': ['Freelancer', 'Student'],  # Unseen work types
            'Residence_type': ['Suburban', 'Remote'],  # Unseen residence
            'avg_glucose_level': [100.0, 150.0],
            'bmi': [25.0, 30.0],
            'smoking_status': ['vapes', 'quit_recently']  # Unseen smoking status
        })
        
        # This should not raise an error due to handle_unknown='ignore'
        _ = preprocessor.transform(unseen_categories)
        
        # Re-transform training data
        R2 = preprocessor.transform(X_train)
        
        # Verify no change
        np.testing.assert_array_equal(R1, R2)
    
    def test_fit_transform_equivalence(self, sample_dataframe):
        """
        Test that fit_transform gives same result as fit then transform.
        
        This verifies the internal consistency of the pipeline.
        """
        X_train, X_test, y_train, y_test = split_data(
            sample_dataframe, test_size=0.2, random_state=42
        )
        
        # Method 1: fit_transform
        preprocessor1 = create_preprocessing_pipeline()
        R1 = preprocessor1.fit_transform(X_train)
        
        # Method 2: fit then transform
        preprocessor2 = create_preprocessing_pipeline()
        preprocessor2.fit(X_train)
        R2 = preprocessor2.transform(X_train)
        
        np.testing.assert_array_almost_equal(R1, R2, decimal=10)
    
    def test_scaler_statistics_from_train_only(self, sample_dataframe):
        """
        Test that RobustScaler statistics come only from training data.
        
        This directly verifies that the scaler's center_ and scale_ parameters
        are not influenced by test data.
        """
        X_train, X_test, y_train, y_test = split_data(
            sample_dataframe, test_size=0.2, random_state=42
        )
        
        # Fit pipeline
        preprocessor = create_preprocessing_pipeline()
        preprocessor.fit(X_train)
        
        # Get scaler from pipeline
        scaler = preprocessor.named_transformers_['numerical'].named_steps['scaler']
        
        # Record the learned statistics
        original_center = scaler.center_.copy()
        original_scale = scaler.scale_.copy()
        
        # Create extreme data and transform it
        extreme_data = pd.DataFrame({
            'gender': ['Male'],
            'age': [999999.0],  # Way outside training range
            'hypertension': [1],
            'heart_disease': [1],
            'ever_married': ['Yes'],
            'work_type': ['Private'],
            'Residence_type': ['Urban'],
            'avg_glucose_level': [999999.0],  # Way outside training range
            'bmi': [999999.0],  # Way outside training range
            'smoking_status': ['Unknown']
        })
        
        # Transform (should not affect scaler)
        _ = preprocessor.transform(extreme_data)
        
        # Verify statistics unchanged
        np.testing.assert_array_equal(
            scaler.center_, original_center,
            err_msg="Scaler center changed after transforming extreme data!"
        )
        np.testing.assert_array_equal(
            scaler.scale_, original_scale,
            err_msg="Scaler scale changed after transforming extreme data!"
        )
    
    def test_imputer_neighbors_from_train_only(self, sample_dataframe_with_more_missing):
        """
        Test that KNNImputer uses only training data for neighbor calculations.
        """
        X_train, X_test, y_train, y_test = split_data(
            sample_dataframe_with_more_missing, test_size=0.2, random_state=42
        )
        
        # Create a row with missing BMI for testing
        test_row_with_missing = X_train.iloc[[0]].copy()
        test_row_with_missing['bmi'] = np.nan
        
        # Fit pipeline
        preprocessor = create_preprocessing_pipeline()
        preprocessor.fit(X_train)
        
        # Transform the row with missing BMI
        R1 = preprocessor.transform(test_row_with_missing)
        imputed_bmi_1 = R1[0, 2]  # BMI is the 3rd numerical feature (index 2)
        
        # Add extreme values to test set and transform them
        extreme_test = X_test.copy()
        extreme_test['bmi'] = 999999.0  # Extreme BMI values
        _ = preprocessor.transform(extreme_test)
        
        # Re-transform the same row
        R2 = preprocessor.transform(test_row_with_missing)
        imputed_bmi_2 = R2[0, 2]
        
        # The imputed value should be identical
        assert imputed_bmi_1 == imputed_bmi_2, (
            f"Imputed BMI changed: {imputed_bmi_1} -> {imputed_bmi_2}\n"
            "This indicates test data is affecting the imputation!"
        )


# =============================================================================
# Test: Transformation Correctness
# =============================================================================

class TestTransformationCorrectness:
    """Tests for correct transformation behavior."""
    
    def test_no_missing_after_transformation(self, sample_dataframe_with_more_missing):
        """Test that KNNImputer eliminates all missing values."""
        X_train, X_test, y_train, y_test = split_data(
            sample_dataframe_with_more_missing, test_size=0.2, random_state=42
        )
        
        # Verify there are missing values before transformation
        assert X_train['bmi'].isna().sum() > 0, "Test setup: should have missing BMI"
        
        # Transform
        preprocessor = create_preprocessing_pipeline()
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        # Verify no missing values after transformation
        assert np.isnan(X_train_transformed).sum() == 0
        assert np.isnan(X_test_transformed).sum() == 0
    
    def test_unknown_smoking_status_encoded(self, sample_dataframe):
        """Test that 'Unknown' smoking status is properly encoded."""
        X_train, X_test, y_train, y_test = split_data(
            sample_dataframe, test_size=0.2, random_state=42
        )
        
        preprocessor = create_preprocessing_pipeline()
        preprocessor.fit(X_train)
        
        # Get encoder
        encoder = preprocessor.named_transformers_['categorical'].named_steps['encoder']
        
        # Find smoking_status categories
        smoking_idx = CATEGORICAL_FEATURES.index('smoking_status')
        smoking_categories = encoder.categories_[smoking_idx] 
        
        # 'Unknown' should be in the categories
        assert 'Unknown' in smoking_categories, (
            f"'Unknown' not found in smoking_status categories: {smoking_categories}"
        )
    
    def test_output_shape_consistency(self, sample_dataframe):
        """Test that output shape is consistent across transforms."""
        X_train, X_test, y_train, y_test = split_data(
            sample_dataframe, test_size=0.2, random_state=42
        )
        
        preprocessor = create_preprocessing_pipeline()
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        # Same number of features
        assert X_train_transformed.shape[1] == X_test_transformed.shape[1]
        
        # Correct number of samples
        assert X_train_transformed.shape[0] == len(X_train)
        assert X_test_transformed.shape[0] == len(X_test)
    
    def test_reproducibility(self, sample_dataframe):
        """Test that pipeline produces reproducible results."""
        X_train, X_test, y_train, y_test = split_data(
            sample_dataframe, test_size=0.2, random_state=42
        )
        
        # First run
        preprocessor1 = create_preprocessing_pipeline()
        R1 = preprocessor1.fit_transform(X_train)
        
        # Second run with same data
        preprocessor2 = create_preprocessing_pipeline()
        R2 = preprocessor2.fit_transform(X_train)
        
        np.testing.assert_array_almost_equal(R1, R2, decimal=10)


# =============================================================================
# Test: Integration with Real Data
# =============================================================================

class TestRealDataIntegration:
    """Integration tests with actual stroke dataset."""
    
    @pytest.fixture
    def real_dataframe(self):
        """Load real stroke dataset if available."""
        from data_ingestion import load_stroke_data
        data_path = Path(__file__).parent.parent / "healthcare-dataset-stroke-data.csv"
        if data_path.exists():
            return load_stroke_data(data_path)
        pytest.skip("Real dataset not available")
    
    def test_real_data_preprocessing(self, real_dataframe):
        """Test preprocessing on real stroke dataset."""
        (X_train, X_test, y_train, y_test,
         preprocessor, feature_names) = preprocess_data(real_dataframe)
        
        # Verify shapes
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert X_train.shape[1] == X_test.shape[1]
        
        # Verify no missing values
        assert np.isnan(X_train).sum() == 0
        assert np.isnan(X_test).sum() == 0
        
        # Verify feature names
        assert len(feature_names) == X_train.shape[1]
    
    def test_real_data_leakage(self, real_dataframe):
        """Run leakage test on real data."""
        X_train, X_test, y_train, y_test = split_data(
            real_dataframe, test_size=0.2, random_state=42
        )
        
        preprocessor = create_preprocessing_pipeline()
        preprocessor.fit(X_train)
        
        R1 = preprocessor.transform(X_train)
        
        # Add extreme outliers
        extreme = pd.DataFrame({
            'gender': ['Male'],
            'age': [999.0],
            'hypertension': [1],
            'heart_disease': [1],
            'ever_married': ['Yes'],
            'work_type': ['Private'],
            'Residence_type': ['Urban'],
            'avg_glucose_level': [9999.0],
            'bmi': [999.0],
            'smoking_status': ['Unknown']
        })
        
        X_test_extreme = pd.concat([X_test, extreme], ignore_index=True)
        _ = preprocessor.transform(X_test_extreme)
        
        R2 = preprocessor.transform(X_train)
        
        np.testing.assert_array_equal(R1, R2)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
