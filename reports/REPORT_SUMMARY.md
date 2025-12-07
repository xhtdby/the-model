# Stroke Prediction Model - EDA & Preprocessing Justification Report

**Author**: Healthcare AI Team  
**Date**: December 6, 2025  
**Dataset**: Healthcare Stroke Dataset (5,110 samples)

---

## Executive Summary

This report provides comprehensive exploratory data analysis (EDA) and preprocessing justification for the stroke prediction model. The analysis addresses:

1. **Missing Values Analysis** - Only BMI has 3.93% missing values
2. **Feature Cardinality** - 7 low-cardinality features identified
3. **Target Imbalance** - Severe class imbalance (19.52:1 ratio)
4. **Feature Classification** - 3 continuous, 2 binary, 5 categorical features
5. **Multicollinearity** - No concerning correlations detected
6. **Distribution Analysis** - All continuous features are non-normal
7. **Preprocessing Strategy** - RobustScaler + KNN Imputation + One-Hot Encoding

---

## 1. Missing Values Analysis

### Findings

| Feature | Missing Count | Missing % | Data Type |
|---------|---------------|-----------|-----------|
| **bmi** | 201 | 3.93% | float64 |

**All other features**: Complete (0% missing)

### Statistical Test: Little's MCAR Test

From `statistical_analysis.py`, the Little's MCAR test was performed:

- **Test Statistic**: χ² = 36.79
- **P-value**: < 0.001
- **Conclusion**: Data is **NOT Missing Completely At Random (MCAR)**
- **Implication**: BMI missingness depends on other variables (likely correlated with age, health conditions)

### Justification for KNN Imputation

**Method Selected**: KNN Imputation (k=5)

**Rationale**:
1. **Preserves Relationships**: Since data is MAR (Missing At Random), KNN uses relationships between features
2. **Superior to Simple Imputation**: Mean/median imputation would ignore correlations
3. **Distance-Based**: Uses Euclidean distance to find 5 nearest neighbors
4. **Applied After Scaling**: RobustScaler applied first to ensure fair distance calculation

**Alternative Methods Rejected**:
- ❌ **Mean/Median**: Ignores multivariate relationships
- ❌ **Deletion**: Would lose 3.93% of data (201 samples)
- ❌ **MICE**: Overcomplicated for single feature with <5% missing

**Visualization**: `figures/eda/missing_values_analysis.png`

---

## 2. Feature Cardinality & Richness Analysis

### Low Cardinality Features (Ratio < 1%)

| Feature | Unique Values | Cardinality Ratio | Type |
|---------|---------------|-------------------|------|
| heart_disease | 2 | 0.0391% | Binary |
| hypertension | 2 | 0.0391% | Binary |
| Residence_type | 2 | 0.0391% | Categorical |
| ever_married | 2 | 0.0391% | Categorical |
| gender | 3 | 0.0587% | Categorical |
| smoking_status | 4 | 0.0783% | Categorical |
| work_type | 5 | 0.0978% | Categorical |

### Feature Richness Assessment

**Continuous Features** (High Information):
- `avg_glucose_level`: 3,979 unique values (77.9% cardinality) ✅
- `bmi`: 418 unique values (8.2% cardinality) ✅
- `age`: 104 unique values (2.0% cardinality) ✅

**Categorical Features** (Low Cardinality):
- All categorical features have <10 unique values
- **No need for dimensionality reduction** (e.g., grouping rare categories)
- **One-Hot Encoding is appropriate** (won't create excessive features)

### Grouping Analysis

**Gender Feature**:
- Female: 2,994 (58.6%)
- Male: 2,115 (41.4%)
- Other: 1 (0.02%) ⚠️

**Recommendation**: Group 'Other' with majority class or drop (1 sample insufficient for analysis)

**Smoking Status**:
- Never smoked: 1,892 (37.0%)
- Unknown: 1,544 (30.2%)
- Formerly smoked: 885 (17.3%)
- Smokes: 789 (15.4%)

**Recommendation**: Keep all categories (each >15% representation)

**Visualization**: `figures/eda/feature_cardinality.png`

---

## 3. Target Variable Analysis (Class Imbalance)

### Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| **No Stroke (0)** | 4,861 | 95.13% |
| **Stroke (1)** | 249 | 4.87% |

### Imbalance Metrics

- **Imbalance Ratio**: 19.52:1
- **Minority Class**: 4.87%
- **Severity**: **SEVERE** (ratio > 10:1)

### Long-Tail Analysis

**Distribution Shape**: Heavily skewed toward negative class

**Clinical Context**:
- Realistic representation of stroke incidence (~5% in general population)
- Minority class has sufficient samples (n=249) for learning
- Not an extreme case (e.g., 1000:1 would require SMOTE)

### Handling Strategy

**Selected Approach**: Cost-Sensitive Learning

**Implementation**:
```python
scale_pos_weight = n_negative / n_positive = 4861 / 249 = 19.52
```

**Justification**:
1. **XGBoost Native Support**: `scale_pos_weight` parameter penalizes misclassifying minority class
2. **No Synthetic Data**: Avoids SMOTE's risk of overfitting to synthetic samples
3. **Preserves Distribution**: Maintains realistic class probabilities for calibration
4. **Stratified CV**: Ensures both classes represented in each fold

**Alternative Methods Rejected**:
- ❌ **SMOTE**: Risk of generating unrealistic synthetic patients
- ❌ **Undersampling**: Would discard 95% of negative class data
- ❌ **No Action**: Model would achieve 95% accuracy by predicting all negatives

**Visualization**: `figures/eda/target_imbalance.png`

---

## 4. Feature Type Classification & Encoding Requirements

### Feature Inventory

#### Continuous Features (n=3)

| Feature | Min | Max | Mean | Median | Std |
|---------|-----|-----|------|--------|-----|
| **age** | 0.08 | 82.00 | 43.23 | 45.00 | 22.61 |
| **avg_glucose_level** | 55.12 | 271.74 | 106.15 | 91.88 | 45.28 |
| **bmi** | 10.30 | 97.60 | 28.89 | 28.10 | 7.85 |

**Encoding**: None required (already numeric)  
**Preprocessing**: RobustScaler (see Section 6)

#### Discrete Features (n=0)

No discrete numeric features in this dataset.

#### Binary Features (n=2)

| Feature | Values | Distribution |
|---------|--------|--------------|
| **hypertension** | [0, 1] | 0: 90.3%, 1: 9.7% |
| **heart_disease** | [0, 1] | 0: 94.6%, 1: 5.4% |

**Encoding**: None required (already 0/1)

#### Categorical Features (n=5)

| Feature | Categories | Encoding Method |
|---------|-----------|-----------------|
| **gender** | 3 | One-Hot (drop='first') |
| **ever_married** | 2 | One-Hot (drop='first') |
| **work_type** | 5 | One-Hot (drop='first') |
| **Residence_type** | 2 | One-Hot (drop='first') |
| **smoking_status** | 4 | One-Hot (drop='first') |

### Encoding Justification

**One-Hot Encoding** (selected for all categorical features):

**Advantages**:
1. ✅ **No Ordinal Assumption**: Categories are nominal (no inherent order)
2. ✅ **Tree-Based Compatibility**: XGBoost handles sparse binary features efficiently
3. ✅ **Interpretability**: Each category becomes an explicit feature for SHAP analysis
4. ✅ **drop='first'**: Prevents dummy variable trap (multicollinearity)

**Alternatives Rejected**:
- ❌ **Label Encoding** (1, 2, 3, ...): Implies false ordinal relationship
- ❌ **Target Encoding**: Risk of data leakage and overfitting
- ❌ **Frequency Encoding**: Loses categorical semantics

**Output Dimensions**:
- gender: 3 categories → 2 binary features (drop 'Female')
- ever_married: 2 categories → 1 binary feature (drop 'No')
- work_type: 5 categories → 4 binary features (drop 'Govt_job')
- Residence_type: 2 categories → 1 binary feature (drop 'Rural')
- smoking_status: 4 categories → 3 binary features (drop 'Unknown')

**Total Features After Encoding**: 3 (continuous) + 2 (binary) + 11 (one-hot) = **16 features**

**Saved Analysis**: `reports/feature_classification.csv`

---

## 5. Multicollinearity Analysis

### Correlation with Target (Stroke)

| Feature | Correlation | Strength | Direction |
|---------|-------------|----------|-----------|
| **age** | 0.2453 | Weak | Positive |
| **heart_disease** | 0.1349 | Weak | Positive |
| **avg_glucose_level** | 0.1319 | Weak | Positive |
| **hypertension** | 0.1279 | Weak | Positive |
| **bmi** | 0.0424 | Weak | Positive |

**Key Insight**: Age is the strongest predictor (r=0.25), but all correlations are weak (<0.3)

### Pairwise Feature Correlations

**Threshold**: |r| > 0.8 (high multicollinearity)

**Result**: ✅ **No multicollinearity detected**

**Highest Pairwise Correlations** (all below threshold):
- All numeric features have |r| < 0.5
- No redundant features requiring removal

### Justification for Keeping 'id' Column

**REMOVED from analysis**

**Correlation with target**: r ≈ 0.00 (negligible)

**Reasoning**:
1. ❌ Sequential identifier (1, 2, 3, ..., 5110)
2. ❌ No biological relationship to stroke risk
3. ❌ Would cause overfitting if model memorizes patient IDs
4. ❌ Not present in production data (new patients have new IDs)

**Visualizations**:
- `figures/eda/correlation_matrix.png` (full heatmap)
- `figures/eda/target_correlation.png` (target-focused)
- `reports/correlation_matrix.csv` (numeric values)

---

## 6. Distribution Analysis & Normalization Needs

### Normality Tests

| Feature | Test | P-value | Normal? | Skewness | Kurtosis |
|---------|------|---------|---------|----------|----------|
| **age** | D'Agostino | <0.0001 | ❌ No | -0.14 | -0.99 |
| **avg_glucose_level** | D'Agostino | <0.0001 | ❌ No | **1.57** | 1.68 |
| **bmi** | Shapiro-Wilk | <0.0001 | ❌ No | **1.06** | 3.36 |

**Result**: **All 3 continuous features are non-normal** (p < 0.05)

### Distribution Characteristics

#### Age
- **Shape**: Nearly uniform (slight left skew)
- **Issue**: Platykurtic (kurtosis = -0.99, flatter than normal)
- **Range**: Spans entire lifespan (0.08 to 82 years)

#### Average Glucose Level
- **Shape**: Right-skewed (skewness = 1.57)
- **Issue**: Long tail of high glucose values (diabetes patients)
- **Clinical Significance**: Skewness reflects real population distribution

#### BMI
- **Shape**: Right-skewed (skewness = 1.06)
- **Issue**: Outliers in obese range (max BMI = 97.6)
- **Clinical Significance**: Reflects obesity epidemic

### Normalization Strategy

**Selected Method**: **RobustScaler**

**Mathematical Transformation**:
```
X_scaled = (X - median(X)) / IQR(X)
```

Where IQR = Q3 - Q1 (interquartile range)

**Justification**:

| Aspect | StandardScaler | RobustScaler | Winner |
|--------|----------------|--------------|--------|
| **Assumption** | Normal distribution | No assumption | ✅ RobustScaler |
| **Outlier Sensitivity** | High (uses mean/std) | Low (uses median/IQR) | ✅ RobustScaler |
| **Our Data** | All non-normal | Skewed + outliers | ✅ RobustScaler |
| **KNN Compatibility** | Yes | Yes | ✅ Both |

**Why NOT Log Transformation?**

1. ✅ **XGBoost is Invariant**: Tree-based models split on thresholds, not absolute values
2. ✅ **RobustScaler Handles Skew**: Already addresses non-normality
3. ❌ **Interpretation**: Log transform complicates SHAP explanations
4. ❌ **Zero Values**: Age starts at 0.08 (would need log1p)

**Use Cases for Log Transform** (not applicable here):
- Linear models requiring normality
- Extreme skewness (>2)
- Multiple orders of magnitude (e.g., 1 to 10,000)

**Visualizations**:
- `figures/eda/distributions.png` (histograms with KDE)
- `figures/eda/qq_plots.png` (quantile-quantile plots)
- `reports/normality_analysis.csv` (detailed statistics)

---

## 7. Preprocessing Pipeline Justification

### Final Pipeline Architecture

```
sklearn.Pipeline
│
├── ColumnTransformer
│   │
│   ├── Numerical Pipeline (age, avg_glucose_level, bmi)
│   │   ├── 1. RobustScaler()           → median=0, IQR=1
│   │   └── 2. KNNImputer(n_neighbors=5) → fill missing BMI
│   │
│   └── Categorical Pipeline (gender, ever_married, work_type, Residence_type, smoking_status)
│       └── 1. OneHotEncoder(drop='first', handle_unknown='ignore')
│
└── XGBoostClassifier(scale_pos_weight=19.52, monotone_constraints=...)
```

### Step-by-Step Justification

#### Step 1: RobustScaler (Numerical Features)

**Applied First** (before imputation)

**Why RobustScaler?**
- 3/3 continuous features are non-normal (p<0.0001)
- Uses median/IQR instead of mean/std (robust to outliers)
- Prevents glucose (~200) from dominating age (~50) in distance calculations

**Why BEFORE KNN Imputation?**
- KNN uses Euclidean distance: `d = sqrt((x1-y1)² + (x2-y2)² + ...)`
- Unscaled: Glucose differences (±100) dwarf age differences (±10)
- Scaled: All features contribute equally to neighbor selection

#### Step 2: KNN Imputation (Missing BMI)

**Applied After Scaling**

**Parameters**: `n_neighbors=5`

**Justification**:
- BMI is MAR (Missing At Random, p<0.001)
- Preserves correlations (e.g., age ↔ BMI, glucose ↔ BMI)
- k=5 balances bias-variance tradeoff

**Alternative Rejected**: SimpleImputer(strategy='median')
- Would ignore relationships with other features
- Produces identical BMI for all missing cases (unrealistic)

#### Step 3: One-Hot Encoding (Categorical Features)

**Applied in Parallel** (separate pipeline)

**Parameters**:
- `drop='first'`: Prevents multicollinearity (n-1 encoding)
- `handle_unknown='ignore'`: Production robustness (new categories → all zeros)

**Output**: 11 binary features from 5 categorical features

#### Step 4: XGBoost with Cost-Sensitive Learning

**Class Weighting**: `scale_pos_weight = 19.52`

**Monotonic Constraints**: Age ↑ → Stroke ↑, Glucose ↑ → Stroke ↑

**Hyperparameter Tuning**: RandomizedSearchCV with stratified 5-fold CV

### Leakage Prevention

**CRITICAL**: All transformations fitted ONLY on training data

**Implementation**:
```python
X_train, X_test = train_test_split(X, y, stratify=y, test_size=0.2)

pipeline.fit(X_train, y_train)        # Fit on train set only
y_pred = pipeline.predict(X_test)      # Transform test set using train statistics
```

**What Gets Learned on Training Data**:
1. RobustScaler: median_train, IQR_train
2. KNNImputer: imputation model from train set
3. OneHotEncoder: categories_train
4. XGBoost: tree structures, split points

**Why This Matters**:
- ❌ **Leakage Example**: Scaling on full data → test statistics influence train scaling
- ✅ **Our Approach**: Test set is "unseen" during fitting

### Benefits of sklearn.Pipeline

1. ✅ **Prevents Leakage**: `fit()` only on train, `transform()` on test
2. ✅ **Reproducibility**: Serialize entire pipeline with `joblib`
3. ✅ **Cross-Validation**: `cross_val_score()` applies pipeline to each fold
4. ✅ **Production Deployment**: Single `.predict()` call handles all preprocessing

**Saved Report**: `reports/preprocessing_justification.txt`

---

## 8. Generated Outputs

### Visualizations (figures/eda/)

| Filename | Description | Key Insights |
|----------|-------------|--------------|
| `missing_values_analysis.png` | Bar chart of missing % | Only BMI missing (3.93%) |
| `feature_cardinality.png` | Cardinality ratios + type distribution | 7 low-cardinality features |
| `target_imbalance.png` | Bar + pie charts of class distribution | 19.52:1 imbalance |
| `correlation_matrix.png` | Lower-triangle heatmap | No multicollinearity |
| `target_correlation.png` | Horizontal bar chart | Age is strongest predictor |
| `distributions.png` | Histograms + KDE + mean/median | All non-normal, glucose/BMI right-skewed |
| `qq_plots.png` | Q-Q plots for normality | Confirms non-normality |

### Data Tables (reports/)

| Filename | Columns | Rows | Purpose |
|----------|---------|------|---------|
| `feature_classification.csv` | Feature, Type, Unique_Values, Encoding_Method | 10 | Encoding roadmap |
| `correlation_matrix.csv` | age, bmi, glucose, hypertension, heart_disease, stroke | 6×6 | Numeric correlation values |
| `normality_analysis.csv` | Feature, Mean, Median, Std, Skewness, Kurtosis, P_Value, Recommendation | 3 | Distribution statistics |
| `preprocessing_justification.txt` | N/A | Text | Full preprocessing rationale |

---

## 9. Key Findings Summary

### Data Quality
✅ High quality dataset with only 3.93% missing values  
✅ No duplicates or invalid entries  
✅ No extreme outliers requiring removal  

### Feature Characteristics
✅ 3 continuous features (all non-normal, right-skewed)  
✅ 2 binary features (already encoded 0/1)  
✅ 5 categorical features (low cardinality, <10 categories each)  
⚠️ 'id' feature removed (no predictive value)  

### Target Variable
⚠️ Severe class imbalance (19.52:1)  
✅ Sufficient minority class samples (n=249)  
✅ Realistic clinical distribution (~5% stroke incidence)  

### Relationships
✅ No multicollinearity detected (all |r| < 0.8)  
✅ Age is strongest predictor (r=0.25)  
✅ BMI missingness is MAR (depends on other variables)  

### Preprocessing Requirements
1. ✅ RobustScaler for continuous features (non-normal distributions)
2. ✅ KNN Imputation for missing BMI (preserves relationships)
3. ✅ One-Hot Encoding for categorical features (drop='first')
4. ✅ Cost-sensitive learning for class imbalance (scale_pos_weight=19.52)
5. ✅ Stratified cross-validation (ensure minority class in each fold)

---

## 10. Recommendations for Report

### Tables to Include

1. **Table 1**: Feature Inventory (from `feature_classification.csv`)
2. **Table 2**: Missing Values Summary (Section 1)
3. **Table 3**: Target Distribution (Section 3)
4. **Table 4**: Correlation with Target (Section 5)
5. **Table 5**: Normality Tests (Section 6)

### Figures to Include

1. **Figure 1**: Missing values bar chart (`missing_values_analysis.png`)
2. **Figure 2**: Target imbalance visualization (`target_imbalance.png`)
3. **Figure 3**: Correlation heatmap (`correlation_matrix.png`)
4. **Figure 4**: Distribution histograms (`distributions.png`)
5. **Figure 5**: Q-Q plots (`qq_plots.png`)

### Justification Sections

Copy directly from `reports/preprocessing_justification.txt`:
- Missing values handling
- Categorical encoding
- Numerical scaling
- Log transformation decision
- Feature removal
- Pipeline structure

---

## 11. References for Report

### Statistical Tests
1. Little, R. J. A. (1988). A test of missing completely at random for multivariate data with missing values. *Journal of the American Statistical Association*, 83(404), 1198-1202.
2. D'Agostino, R. B., & Pearson, E. S. (1973). Tests for departure from normality. *Biometrika*, 60(3), 613-622.
3. Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test for normality. *Biometrika*, 52(3/4), 591-611.

### Preprocessing Methods
4. Troyanskaya, O., et al. (2001). Missing value estimation methods for DNA microarrays. *Bioinformatics*, 17(6), 520-525. (KNN Imputation)
5. Loh, W. Y. (2011). Classification and regression trees. *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, 1(14-23). (RobustScaler for tree models)

### Class Imbalance
6. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*, 785-794. (scale_pos_weight parameter)
7. He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284.

---

## Appendix: Code Execution

**Generated by**: `src/eda_report.py`  
**Execution time**: ~15 seconds  
**Python version**: 3.13  
**Key libraries**: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn

**To regenerate all analyses**:
```bash
python src/eda_report.py
```

**Output locations**:
- Figures: `figures/eda/`
- Reports: `reports/`

---

**End of Report**
