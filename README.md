# Clinical Stroke Prediction Engine

**A Production-Grade Machine Learning System for Cerebrovascular Accident Risk Stratification**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-34%20passed-green.svg)]()
[![Recall](https://img.shields.io/badge/recall-82%25-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Abstract

This project implements a clinically-oriented stroke prediction system optimized for **high-sensitivity screening** in emergency and primary care settings. Unlike conventional accuracy-focused models, we prioritize **recall (82%)** to minimize false negatives—a critical requirement when missed diagnoses carry severe consequences including disability and mortality.

The system incorporates methodological safeguards addressing common pitfalls in healthcare ML: data leakage prevention via proper cross-validation pipelines, monotonic constraints encoding established clinical knowledge, algorithmic fairness auditing across demographic groups, and schema-enforced data validation.

**Keywords:** Stroke prediction, XGBoost, Clinical decision support, Imbalanced classification, Algorithmic fairness, SHAP explainability

---

## Table of Contents

1. [Background & Motivation](#background--motivation)
2. [Dataset Description](#dataset-description)
3. [Methodology](#methodology)
4. [System Architecture](#system-architecture)
5. [Results & Evaluation](#results--evaluation)
6. [Fairness Audit](#fairness-audit)
7. [Explainability Analysis](#explainability-analysis)
8. [Installation & Usage](#installation--usage)
9. [Repository Structure](#repository-structure)
10. [Limitations & Future Work](#limitations--future-work)
11. [References](#references)

---

## Background & Motivation

### Clinical Context

Stroke remains the **second leading cause of death globally** and a primary cause of long-term disability (WHO, 2023). Early identification of high-risk individuals enables preventive interventions including lifestyle modifications, anticoagulation therapy, and closer monitoring. However, traditional risk scores (e.g., CHA2DS2-VASc, Framingham) rely on linear assumptions that may underfit complex, non-linear biological interactions.

### Machine Learning Opportunity

Gradient boosting methods, particularly XGBoost, have demonstrated superior performance in tabular medical data by:
- Capturing non-linear feature interactions
- Handling mixed data types (continuous, categorical)
- Providing native missing value handling
- Supporting monotonic constraints for domain knowledge integration

### Design Philosophy

This project adopts a **"safety-first"** engineering approach:

| Principle | Implementation |
|-----------|----------------|
| **High Sensitivity** | Optimize for recall over precision; accept more false positives to catch true strokes |
| **Clinical Interpretability** | Monotonic constraints ensure predictions align with medical knowledge |
| **Algorithmic Fairness** | Audit for disparate impact across gender and residence type |
| **Reproducibility** | Deterministic pipelines, version-controlled artifacts, comprehensive testing |

---

## Dataset Description

**Source:** [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

### Sample Characteristics

| Characteristic | Value |
|---------------|-------|
| Total Samples | 5,110 |
| Stroke Cases (Positive) | 249 (4.87%) |
| Non-Stroke Cases (Negative) | 4,861 (95.13%) |
| Class Imbalance Ratio | 1:19.5 |

### Feature Inventory

| Feature | Type | Description | Missing |
|---------|------|-------------|---------|
| `age` | Continuous | Patient age in years | 0% |
| `avg_glucose_level` | Continuous | Average blood glucose (mg/dL) | 0% |
| `bmi` | Continuous | Body Mass Index (kg/m²) | 3.9% |
| `gender` | Categorical | Male / Female / Other | 0% |
| `hypertension` | Binary | History of hypertension (0/1) | 0% |
| `heart_disease` | Binary | History of heart disease (0/1) | 0% |
| `ever_married` | Binary | Marital status (Yes/No) | 0% |
| `work_type` | Categorical | Employment category | 0% |
| `Residence_type` | Categorical | Urban / Rural | 0% |
| `smoking_status` | Categorical | Smoking history (including "Unknown") | 0%* |

*Note: 30.2% of samples have `smoking_status = "Unknown"`, treated as a distinct category rather than missing data.

### Missingness Analysis

We performed **Little's MCAR Test** on BMI missingness:

- **Result:** Chi-squared = 45.23, p < 0.001
- **Interpretation:** BMI is **not Missing Completely At Random (MCAR)**
- **Implication:** Simple mean imputation would introduce bias; we use KNN imputation to preserve covariate relationships

---

## Methodology

### 1. Data Validation Layer

Schema validation using `pandera` ensures data integrity at ingestion:

```python
STROKE_DATA_SCHEMA = pa.DataFrameSchema({
    "age": pa.Column(float, checks=[pa.Check.ge(0), pa.Check.le(120)]),
    "avg_glucose_level": pa.Column(float, checks=pa.Check.ge(0)),
    "bmi": pa.Column(float, nullable=True, checks=pa.Check.in_range(10, 100)),
    "smoking_status": pa.Column(str, checks=pa.Check.isin([
        "formerly smoked", "never smoked", "smokes", "Unknown"
    ])),
    "stroke": pa.Column(int, checks=pa.Check.isin([0, 1]))
})
```

### 2. Preprocessing Pipeline

We construct a **leakage-free** sklearn `Pipeline` with `ColumnTransformer`:

```
+-------------------------------------------------------------+
|                    ColumnTransformer                        |
+-------------------------------------------------------------+
|  Numerical Features          |  Categorical Features        |
|  (age, glucose, bmi)         |  (gender, work_type, etc.)   |
|  +---------------------+     |  +---------------------+     |
|  | 1. RobustScaler     |     |  | OneHotEncoder       |     |
|  | 2. KNNImputer(k=5)  |     |  | (drop='first')      |     |
|  +---------------------+     |  +---------------------+     |
+-------------------------------------------------------------+
```

**Key Design Decisions:**

| Decision | Rationale |
|----------|-----------|
| **RobustScaler before KNNImputer** | KNN uses Euclidean distance; unscaled features (e.g., glucose ~200 vs. age ~50) would dominate distance calculations. RobustScaler (median/IQR) is chosen over StandardScaler for outlier robustness. |
| **KNN Imputation (k=5)** | Preserves multivariate relationships in BMI missingness (MAR mechanism). Mean imputation would underestimate variance and distort correlations. |
| **"Unknown" as Category** | Chi-square analysis shows `smoking_status="Unknown"` has a statistically distinct stroke rate (Chi-squared = 30.1, p < 0.001). Treating it as missing would discard informative signal. |

### 3. Model Architecture: XGBoost with Clinical Constraints

#### Cost-Sensitive Learning

The 1:19.5 class imbalance is addressed via `scale_pos_weight`:

```
scale_pos_weight = N_negative / N_positive = 3889 / 199 ≈ 19.54
```

This penalizes false negatives proportionally to class imbalance, pushing the model toward higher recall.

#### Monotonic Constraints

We encode established clinical knowledge as **monotonic constraints**:

| Feature | Constraint | Clinical Basis |
|---------|------------|----------------|
| `age` | +1 (increasing) | Stroke risk increases monotonically with age (Framingham Heart Study) |
| `avg_glucose_level` | +1 (increasing) | Hyperglycemia is an established stroke risk factor (ADA Guidelines) |

This prevents the model from learning spurious inverse relationships (e.g., "older patients have lower risk") that may exist due to sampling bias or confounding.

#### Hyperparameter Optimization

**Method:** RandomizedSearchCV with 5-fold Stratified Cross-Validation

**Search Space:**

| Parameter | Distribution | Range |
|-----------|--------------|-------|
| `learning_rate` | Log-uniform | [0.01, 0.1] |
| `max_depth` | Uniform int | [3, 6] |
| `n_estimators` | Uniform int | [100, 500] |
| `reg_alpha` (L1) | Uniform | [0, 10] |
| `reg_lambda` (L2) | Uniform | [1, 10] |
| `subsample` | Uniform | [0.6, 0.9] |
| `colsample_bytree` | Uniform | [0.6, 0.9] |

**Optimization Metric:** ROC-AUC (threshold-independent, appropriate for imbalanced data)

---

## System Architecture

```
+--------------------------------------------------------------------+
|                    STROKE PREDICTION SYSTEM                        |
+--------------------------------------------------------------------+
|                                                                    |
|  +--------------+    +--------------+    +--------------+          |
|  | Data Input   | -> | Schema       | -> | Preprocessing|          |
|  | (CSV/Dict)   |    | Validation   |    | Pipeline     |          |
|  +--------------+    | (Pandera)    |    | (sklearn)    |          |
|                      +--------------+    +--------------+          |
|                                                 |                  |
|                                                 v                  |
|  +--------------+    +--------------+    +--------------+          |
|  | Risk Output  | <- | Calibrated   | <- | XGBoost      |          |
|  | + Alerts     |    | Probability  |    | Classifier   |          |
|  +--------------+    +--------------+    +--------------+          |
|                                                                    |
+--------------------------------------------------------------------+
```

### Module Responsibilities

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `data_ingestion.py` | Schema validation, data loading | `load_stroke_data()`, `validate_stroke_data()` |
| `preprocessing_pipeline.py` | Feature transformation | `create_preprocessing_pipeline()`, `split_data()` |
| `statistical_analysis.py` | Missingness & association tests | `littles_mcar_test()`, `perform_chi_square_test()` |
| `xgboost_model.py` | Model training with constraints | `train_stroke_model()`, `get_monotonic_constraints()` |
| `model_evaluation.py` | Performance & fairness metrics | `run_fairness_audit()`, `generate_shap_analysis()` |
| `inference.py` | Production inference API | `StrokePredictor` class |
| `train_pipeline.py` | End-to-end orchestration | `run_training_pipeline()` |

---

## Results & Evaluation

### Performance on Held-Out Test Set (n=1,022)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **ROC-AUC** | 0.850 | Excellent discrimination |
| **Average Precision** | 0.318 | Good given 4.9% prevalence |
| **Recall (Sensitivity)** | 82.0% | Detects 41/50 stroke cases |
| **Specificity** | 71.2% | Acceptable for screening |
| **PPV (Precision)** | 12.8% | Expected for low-prevalence screening |
| **NPV** | 98.7% | High confidence in negative predictions |

### Confusion Matrix

```
                    Predicted
                 No Stroke  Stroke
Actual No Stroke    692      280    (FP: False Alarms)
       Stroke         9       41    (TP: Detected)
```

### Clinical Interpretation

- **9 Missed Strokes (FN):** These represent the most critical errors. Further investigation of these cases could inform model improvements.
- **280 False Alarms (FP):** In a screening context, these patients would receive additional workup. The cost of follow-up testing is typically acceptable compared to missed strokes.
- **NPV of 98.7%:** When the model predicts "low risk," there is only a 1.3% chance of stroke—useful for triaging emergency department patients.

### Precision-Recall Trade-off

At the default threshold (0.5), we achieve 82% recall. The threshold can be adjusted based on clinical context:

| Threshold | Recall | Precision | Use Case |
|-----------|--------|-----------|----------|
| 0.3 | 90%+ | ~8% | Emergency screening (maximize detection) |
| 0.5 | 82% | 12.8% | Balanced screening (default) |
| 0.7 | ~65% | ~20% | Confirmatory testing (higher confidence) |

---

## Fairness Audit

We evaluate algorithmic fairness using two standard metrics:

### 1. Equalized Odds (Gender)

**Definition:** True Positive Rate (TPR) should be equal across demographic groups.

| Group | TPR (Recall) | Support | Disparity |
|-------|--------------|---------|-----------|
| Female | 82.76% | n=599 (29 stroke) | — |
| Male | 80.95% | n=423 (21 stroke) | **1.81%** |

**Result:** Gender disparity (1.81%) is below the 10% threshold. PASSED

### 2. Demographic Parity (Residence)

**Definition:** Positive prediction rate should be equal across groups.

| Group | Selection Rate | Support | Disparity |
|-------|---------------|---------|-----------|
| Rural | 31.92% | n=495 | — |
| Urban | 30.93% | n=527 | **0.99%** |

**Result:** Residence disparity (0.99%) is below the 10% threshold. PASSED

### Fairness Conclusion

The model does not exhibit significant disparate impact across the audited demographic dimensions. However, fairness auditing should be an ongoing process as the model is deployed to new populations.

---

## Explainability Analysis

### SHAP Feature Importance (Global)

| Rank | Feature | Mean SHAP | Direction |
|------|---------|-----------|-----------|
| 1 | `age` | 0.426 | Higher age -> Higher risk |
| 2 | `avg_glucose_level` | 0.080 | Higher glucose -> Higher risk |
| 3 | `hypertension` | 0.055 | Hypertension -> Higher risk |
| 4 | `bmi` | 0.046 | Mixed effect |
| 5 | `heart_disease` | 0.044 | Heart disease -> Higher risk |

### Smoking Status Analysis

| Category | Mean SHAP | Effect |
|----------|-----------|--------|
| `smokes` | +0.066 | Raises risk |
| `formerly smoked` | +0.036 | Raises risk |
| `never smoked` | -0.041 | Lowers risk |
| `Unknown` | -0.017 | Minimal effect |

**Finding:** The `Unknown` smoking status has minimal differential effect on predictions, suggesting the model treats missing smoking data conservatively rather than assuming worst-case scenarios.

---

## Installation & Usage

### Requirements

- Python 3.10+
- Dependencies: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `pandera`, `shap`, `matplotlib`

### Installation

```bash
git clone https://github.com/yourusername/stroke-prediction.git
cd stroke-prediction
pip install -r requirements.txt
```

### Training

```bash
# Full training (50 iterations, ~2 minutes)
python main.py --train

# Quick training for testing (10 iterations)
python main.py --train --quick
```

### Inference

```bash
# Run demo predictions
python main.py --predict --demo

# Evaluate model performance and fairness
python main.py --evaluate
```

### Programmatic Usage

```python
from src.inference import StrokePredictor

# Initialize predictor (loads saved model)
predictor = StrokePredictor()

# Single patient prediction
result = predictor.predict({
    "age": 72,
    "gender": "Male",
    "hypertension": 1,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Self-employed",
    "Residence_type": "Rural",
    "avg_glucose_level": 228.0,
    "bmi": 36.6,
    "smoking_status": "smokes"
})

print(result)
# Output: Stroke Risk: 83.0% (CRITICAL)
# CRITICAL ALERT: Patient has extremely high stroke risk (>80%).
# Immediate clinical evaluation recommended.
```

---

## Repository Structure

```
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py         # Pandera schemas & data loading
│   ├── statistical_analysis.py   # Little's MCAR, Chi-square tests
│   ├── preprocessing_pipeline.py # ColumnTransformer, KNN imputation
│   ├── xgboost_model.py          # Model training, monotonic constraints
│   ├── model_evaluation.py       # Metrics, fairness audit, SHAP
│   ├── inference.py              # StrokePredictor production class
│   └── train_pipeline.py         # End-to-end orchestrator
│
├── tests/
│   ├── test_data_ingestion.py    # 20 tests for schema validation
│   └── test_preprocessing_pipeline.py  # 14 tests for leakage prevention
│
├── models/
│   ├── stroke_xgboost_model.joblib  # Serialized model
│   ├── best_hyperparameters.json    # Tuned parameters
│   └── training_metadata.json       # Training provenance
│
├── figures/
│   ├── confusion_matrix.png
│   ├── precision_recall_curve.png
│   ├── shap_summary_plot.png
│   └── shap_dependence_bmi_age.png
│
├── main.py                       # CLI entry point
├── requirements.txt
└── README.md
```

---

## Limitations & Future Work

### Current Limitations

1. **Single Dataset:** Model trained on Kaggle dataset; generalization to other populations (e.g., different healthcare systems, ethnicities) requires validation.

2. **Temporal Validation:** No time-based train/test split; real-world deployment should validate on temporally distinct data.

3. **Feature Limitations:** Dataset lacks key clinical predictors (e.g., atrial fibrillation, cholesterol levels, family history) that would improve performance.

4. **Calibration:** Probability outputs may not be perfectly calibrated; Platt scaling or isotonic regression could be applied.

### Future Enhancements

- [ ] External validation on MIMIC-III or eICU datasets
- [ ] Temporal cross-validation for deployment readiness
- [ ] Integration with electronic health record (EHR) systems
- [ ] Prospective clinical trial evaluation
- [ ] Model monitoring for data drift detection

---

## References

1. Fedesoriano. (2021). Stroke Prediction Dataset. Kaggle. https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

2. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.

3. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems*.

4. Little, R. J. A. (1988). A Test of Missing Completely at Random for Multivariate Data with Missing Values. *Journal of the American Statistical Association*, 83(404), 1198-1202.

5. Hardt, M., Price, E., & Srebro, N. (2016). Equality of Opportunity in Supervised Learning. *Advances in Neural Information Processing Systems*.

6. World Health Organization. (2023). Global Health Estimates: Stroke Factsheet. https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Citation

If you use this work in academic research, please cite:

```bibtex
@software{stroke_prediction_engine,
  title={Clinical Stroke Prediction Engine},
  author={Healthcare AI Team},
  year={2025},
  url={https://github.com/yourusername/stroke-prediction}
}
```

---

*Developed as part of AI in Healthcare coursework. This is a research prototype and should not be used for clinical decision-making without proper validation and regulatory approval.*
