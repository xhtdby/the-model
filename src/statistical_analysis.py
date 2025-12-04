"""
Stroke Prediction - Statistical Analysis Module
================================================
Deep-dive analysis of dataset anomalies including:
- Smoking status analysis with Chi-Square test
- BMI missingness mechanism (Little's MCAR Test)

Author: Senior ML Engineer
Project: Healthcare AI - Stroke Prediction
"""

import warnings
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2


# =============================================================================
# Smoking Status Analysis
# =============================================================================

def calculate_stroke_prevalence_by_smoking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate stroke prevalence rate for each smoking status category.
    
    Args:
        df: DataFrame with 'smoking_status' and 'stroke' columns
        
    Returns:
        DataFrame with stroke prevalence statistics per smoking category
    """
    prevalence = df.groupby('smoking_status').agg(
        total_count=('stroke', 'count'),
        stroke_count=('stroke', 'sum'),
        no_stroke_count=('stroke', lambda x: (x == 0).sum())
    ).reset_index()
    
    prevalence['stroke_prevalence_rate'] = (
        prevalence['stroke_count'] / prevalence['total_count'] * 100
    )
    
    # Sort by prevalence rate descending
    prevalence = prevalence.sort_values('stroke_prevalence_rate', ascending=False)
    
    return prevalence


def perform_chi_square_test(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform Chi-Square test of independence between smoking_status and stroke.
    
    Tests whether the 'Unknown' group's stroke risk is statistically distinct
    from other smoking status groups.
    
    Args:
        df: DataFrame with 'smoking_status' and 'stroke' columns
        
    Returns:
        Dictionary containing:
        - contingency_table: Cross-tabulation of smoking_status vs stroke
        - chi2_statistic: Chi-square test statistic
        - p_value: P-value for the test
        - degrees_of_freedom: Degrees of freedom
        - expected_frequencies: Expected frequencies under null hypothesis
        - is_significant: Boolean indicating if p < 0.05
        - interpretation: Text interpretation of results
    """
    # Create contingency table
    contingency_table = pd.crosstab(
        df['smoking_status'], 
        df['stroke'],
        margins=True,
        margins_name='Total'
    )
    contingency_table.columns = ['No Stroke', 'Stroke', 'Total']
    
    # Perform Chi-Square test (excluding margins)
    observed = pd.crosstab(df['smoking_status'], df['stroke'])
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed)
    
    # Create expected frequencies DataFrame
    expected_df = pd.DataFrame(
        expected,
        index=observed.index,
        columns=['No Stroke (Expected)', 'Stroke (Expected)']
    )
    
    is_significant = p_value < 0.05
    
    # Generate interpretation
    if is_significant:
        interpretation = (
            f"SIGNIFICANT RESULT (p = {p_value:.6f} < 0.05):\n"
            f"The smoking status categories (including 'Unknown') show statistically "
            f"significant differences in stroke risk. This provides justification for "
            f"treating 'Unknown' as a meaningful feature category rather than missing data.\n"
            f"The 'Unknown' status may represent patients who refused to disclose their "
            f"smoking habits, which itself could be an informative signal."
        )
    else:
        interpretation = (
            f"NON-SIGNIFICANT RESULT (p = {p_value:.6f} >= 0.05):\n"
            f"No statistically significant association found between smoking status "
            f"(including 'Unknown') and stroke risk. The 'Unknown' category may be "
            f"treated as missing data or kept as a separate category based on domain knowledge."
        )
    
    return {
        'contingency_table': contingency_table,
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'expected_frequencies': expected_df,
        'is_significant': is_significant,
        'interpretation': interpretation
    }


def pairwise_chi_square_unknown_vs_others(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Perform pairwise Chi-Square tests comparing 'Unknown' against each other
    smoking status category.
    
    Args:
        df: DataFrame with 'smoking_status' and 'stroke' columns
        
    Returns:
        Dictionary with pairwise comparison results
    """
    other_categories = ['smokes', 'formerly smoked', 'never smoked']
    results = {}
    
    unknown_df = df[df['smoking_status'] == 'Unknown']
    
    for category in other_categories:
        category_df = df[df['smoking_status'] == category]
        combined = pd.concat([unknown_df, category_df])
        
        contingency = pd.crosstab(combined['smoking_status'], combined['stroke'])
        
        if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
            chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency)
            
            results[f'Unknown_vs_{category}'] = {
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'is_significant': p_value < 0.05,
                'unknown_stroke_rate': unknown_df['stroke'].mean() * 100,
                'other_stroke_rate': category_df['stroke'].mean() * 100
            }
    
    return results


# =============================================================================
# Little's MCAR Test Implementation
# =============================================================================

def littles_mcar_test(df: pd.DataFrame, 
                      columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Implement Little's MCAR (Missing Completely At Random) Test.
    
    This test evaluates whether the missing data mechanism is MCAR by comparing
    the means of observed values across different missing data patterns.
    
    Null Hypothesis (H0): Data is Missing Completely At Random (MCAR)
    Alternative Hypothesis (H1): Data is NOT MCAR (MAR or MNAR)
    
    Args:
        df: DataFrame to test
        columns: Columns to include in the test (numeric columns only).
                 If None, all numeric columns with missing values are used.
    
    Returns:
        Dictionary containing:
        - chi2_statistic: Test statistic
        - degrees_of_freedom: Degrees of freedom
        - p_value: P-value
        - is_mcar: Boolean (True if p > 0.05, suggesting MCAR)
        - interpretation: Text interpretation
    """
    # Select numeric columns
    if columns is None:
        numeric_df = df.select_dtypes(include=[np.number])
        # Only include columns with missing values and columns without
        columns_with_missing = numeric_df.columns[numeric_df.isnull().any()].tolist()
        columns_without_missing = numeric_df.columns[~numeric_df.isnull().any()].tolist()
        columns = columns_with_missing + columns_without_missing[:5]  # Limit to avoid computational issues
    
    test_df = df[columns].copy()
    
    # Handle case with no missing data
    if not test_df.isnull().any().any():
        return {
            'chi2_statistic': 0.0,
            'degrees_of_freedom': 0,
            'p_value': 1.0,
            'is_mcar': True,
            'interpretation': "No missing data found in the selected columns."
        }
    
    # Create missing data pattern indicator
    missing_pattern = test_df.isnull().astype(int)
    pattern_str = missing_pattern.apply(lambda row: ''.join(row.astype(str)), axis=1)
    unique_patterns = pattern_str.unique()
    
    # If only one pattern, cannot perform test
    if len(unique_patterns) <= 1:
        return {
            'chi2_statistic': np.nan,
            'degrees_of_freedom': 0,
            'p_value': np.nan,
            'is_mcar': None,
            'interpretation': "Cannot perform MCAR test: Only one missing data pattern found."
        }
    
    # Calculate overall means and covariance for complete data columns
    complete_cols = test_df.columns[~test_df.isnull().any()].tolist()
    
    if len(complete_cols) == 0:
        # Use pairwise complete observations
        overall_means = test_df.mean()
        overall_cov = test_df.cov()
    else:
        overall_means = test_df.mean()
        overall_cov = test_df.cov()
    
    # Calculate chi-square statistic
    chi2_stat = 0.0
    total_df = 0
    
    for pattern in unique_patterns:
        if pattern == '0' * len(columns):  # Complete cases
            continue
            
        pattern_mask = pattern_str == pattern
        n_pattern = pattern_mask.sum()
        
        if n_pattern < 2:
            continue
        
        # Get observed columns for this pattern
        pattern_array = np.array([int(x) for x in pattern])
        observed_cols = [columns[i] for i in range(len(columns)) if pattern_array[i] == 0]
        
        if len(observed_cols) == 0:
            continue
        
        # Calculate pattern-specific means
        pattern_data = test_df.loc[pattern_mask, observed_cols]
        pattern_means = pattern_data.mean()
        
        # Calculate difference from overall means
        mean_diff = pattern_means - overall_means[observed_cols]
        
        # Get submatrix of covariance
        cov_submatrix = overall_cov.loc[observed_cols, observed_cols]
        
        try:
            # Add small regularization for numerical stability
            cov_submatrix_reg = cov_submatrix + np.eye(len(observed_cols)) * 1e-6
            cov_inv = np.linalg.inv(cov_submatrix_reg)
            
            # Calculate contribution to chi-square
            contribution = n_pattern * mean_diff.values @ cov_inv @ mean_diff.values
            chi2_stat += contribution
            total_df += len(observed_cols)
        except np.linalg.LinAlgError:
            continue
    
    # Calculate p-value
    if total_df > 0:
        p_value = 1 - chi2.cdf(chi2_stat, total_df)
    else:
        p_value = np.nan
    
    is_mcar = p_value > 0.05 if not np.isnan(p_value) else None
    
    # Generate interpretation
    if is_mcar is None:
        interpretation = "Unable to determine MCAR status due to insufficient data patterns."
    elif is_mcar:
        interpretation = (
            f"MCAR SUPPORTED (p = {p_value:.6f} > 0.05):\n"
            f"The null hypothesis (MCAR) cannot be rejected. The missing data appears to be "
            f"Missing Completely At Random. Standard imputation methods (mean, median, or "
            f"more sophisticated methods like KNN or MICE) are appropriate and will not "
            f"introduce systematic bias."
        )
    else:
        interpretation = (
            f"MCAR REJECTED (p = {p_value:.6f} <= 0.05):\n"
            f"The null hypothesis (MCAR) is rejected. The missing data is likely Missing "
            f"At Random (MAR) or Missing Not At Random (MNAR). Simple imputation methods "
            f"may introduce bias. Consider:\n"
            f"  1. Multiple Imputation (MICE) that accounts for uncertainty\n"
            f"  2. Model-based approaches that explicitly handle MAR/MNAR\n"
            f"  3. Investigating which variables predict missingness"
        )
    
    return {
        'chi2_statistic': chi2_stat,
        'degrees_of_freedom': total_df,
        'p_value': p_value,
        'is_mcar': is_mcar,
        'interpretation': interpretation,
        'n_patterns': len(unique_patterns),
        'missing_patterns': dict(pattern_str.value_counts())
    }


def analyze_bmi_missingness_correlations(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze correlations between BMI missingness and other variables.
    
    This helps identify if BMI is Missing At Random (MAR) by checking
    if missingness correlates with observed variables.
    
    Args:
        df: DataFrame with 'bmi' column and other variables
        
    Returns:
        Dictionary with correlation analysis results
    """
    # Create missingness indicator
    df_analysis = df.copy()
    df_analysis['bmi_missing'] = df_analysis['bmi'].isnull().astype(int)
    
    # Variables to test correlation with
    numeric_vars = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'stroke']
    
    correlations = {}
    point_biserial_results = {}
    
    for var in numeric_vars:
        if var in df_analysis.columns:
            # Point-biserial correlation (for binary missingness indicator)
            valid_mask = ~df_analysis[var].isnull()
            if valid_mask.sum() > 0:
                corr, p_value = stats.pointbiserialr(
                    df_analysis.loc[valid_mask, 'bmi_missing'],
                    df_analysis.loc[valid_mask, var]
                )
                correlations[var] = corr
                point_biserial_results[var] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'is_significant': p_value < 0.05
                }
    
    # T-tests comparing variable means between BMI missing vs not missing
    t_test_results = {}
    for var in numeric_vars:
        if var in df_analysis.columns:
            missing_group = df_analysis.loc[df_analysis['bmi_missing'] == 1, var].dropna()
            not_missing_group = df_analysis.loc[df_analysis['bmi_missing'] == 0, var].dropna()
            
            if len(missing_group) > 1 and len(not_missing_group) > 1:
                t_stat, p_value = stats.ttest_ind(missing_group, not_missing_group)
                t_test_results[var] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'mean_when_bmi_missing': missing_group.mean(),
                    'mean_when_bmi_present': not_missing_group.mean(),
                    'is_significant': p_value < 0.05
                }
    
    # Chi-square test for categorical variables
    categorical_vars = ['smoking_status', 'gender', 'ever_married', 'work_type', 'Residence_type']
    chi_square_results = {}
    
    for var in categorical_vars:
        if var in df_analysis.columns:
            contingency = pd.crosstab(df_analysis[var], df_analysis['bmi_missing'])
            chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency)
            chi_square_results[var] = {
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'degrees_of_freedom': dof,
                'is_significant': p_value < 0.05
            }
    
    # Summary statistics
    bmi_missing_count = df_analysis['bmi_missing'].sum()
    bmi_missing_pct = bmi_missing_count / len(df_analysis) * 100
    
    return {
        'bmi_missing_count': bmi_missing_count,
        'bmi_missing_percentage': bmi_missing_pct,
        'point_biserial_correlations': point_biserial_results,
        't_test_results': t_test_results,
        'chi_square_results': chi_square_results,
        'correlation_summary': correlations
    }


# =============================================================================
# Main Analysis Function
# =============================================================================

def analyze_missingness(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive analysis of dataset missingness and anomalies.
    
    This function performs:
    1. Smoking Status Analysis with Chi-Square test
    2. BMI Missingness Mechanism analysis (Little's MCAR Test)
    3. Correlation analysis for BMI missingness
    
    Args:
        df: Stroke prediction DataFrame
        
    Returns:
        Dictionary containing all analysis results and recommendations
    """
    results = {
        'analysis_timestamp': pd.Timestamp.now().isoformat(),
        'dataset_shape': df.shape,
    }
    
    # ==========================================================================
    # 1. Smoking Status Analysis
    # ==========================================================================
    print("=" * 80)
    print("SMOKING STATUS ANALYSIS")
    print("=" * 80)
    
    # Calculate stroke prevalence
    prevalence = calculate_stroke_prevalence_by_smoking(df)
    results['smoking_prevalence'] = prevalence.to_dict('records')
    
    print("\nStroke Prevalence by Smoking Status:")
    print("-" * 50)
    for _, row in prevalence.iterrows():
        print(f"  {row['smoking_status']:20s}: {row['stroke_prevalence_rate']:6.2f}% "
              f"({row['stroke_count']}/{row['total_count']})")
    
    # Chi-Square test
    chi_square_results = perform_chi_square_test(df)
    results['chi_square_test'] = {
        'chi2_statistic': chi_square_results['chi2_statistic'],
        'p_value': chi_square_results['p_value'],
        'degrees_of_freedom': chi_square_results['degrees_of_freedom'],
        'is_significant': chi_square_results['is_significant']
    }
    
    print(f"\nChi-Square Test of Independence:")
    print("-" * 50)
    print(f"  Chi-Square Statistic: {chi_square_results['chi2_statistic']:.4f}")
    print(f"  Degrees of Freedom:   {chi_square_results['degrees_of_freedom']}")
    print(f"  P-Value:              {chi_square_results['p_value']:.6f}")
    print(f"\n{chi_square_results['interpretation']}")
    
    # Pairwise comparisons
    pairwise_results = pairwise_chi_square_unknown_vs_others(df)
    results['pairwise_chi_square'] = pairwise_results
    
    print("\nPairwise Comparisons (Unknown vs Others):")
    print("-" * 50)
    for comparison, data in pairwise_results.items():
        sig_marker = "*" if data['is_significant'] else ""
        print(f"  {comparison}:")
        print(f"    Chi2={data['chi2_statistic']:.4f}, p={data['p_value']:.6f}{sig_marker}")
        print(f"    Unknown stroke rate: {data['unknown_stroke_rate']:.2f}%")
        print(f"    Other stroke rate:   {data['other_stroke_rate']:.2f}%")
    
    # ==========================================================================
    # 2. BMI Missingness Analysis
    # ==========================================================================
    print("\n" + "=" * 80)
    print("BMI MISSINGNESS MECHANISM ANALYSIS")
    print("=" * 80)
    
    # Little's MCAR Test
    mcar_test_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']
    available_cols = [col for col in mcar_test_cols if col in df.columns]
    
    mcar_results = littles_mcar_test(df, columns=available_cols)
    results['littles_mcar_test'] = {
        'chi2_statistic': mcar_results['chi2_statistic'],
        'degrees_of_freedom': mcar_results['degrees_of_freedom'],
        'p_value': mcar_results['p_value'],
        'is_mcar': mcar_results['is_mcar'],
        'n_patterns': mcar_results.get('n_patterns', 0)
    }
    
    print(f"\nLittle's MCAR Test:")
    print("-" * 50)
    print(f"  Chi-Square Statistic: {mcar_results['chi2_statistic']:.4f}")
    print(f"  Degrees of Freedom:   {mcar_results['degrees_of_freedom']}")
    print(f"  P-Value:              {mcar_results['p_value']:.6f}" if not np.isnan(mcar_results['p_value']) else "  P-Value:              N/A")
    print(f"  Number of Patterns:   {mcar_results.get('n_patterns', 'N/A')}")
    print(f"\n{mcar_results['interpretation']}")
    
    # Correlation analysis for BMI missingness
    correlation_results = analyze_bmi_missingness_correlations(df)
    results['bmi_missingness_correlations'] = correlation_results
    
    print(f"\nBMI Missingness Statistics:")
    print("-" * 50)
    print(f"  Missing BMI count:      {correlation_results['bmi_missing_count']}")
    print(f"  Missing BMI percentage: {correlation_results['bmi_missing_percentage']:.2f}%")
    
    print(f"\nCorrelation of BMI Missingness with Numeric Variables:")
    print("-" * 50)
    for var, data in correlation_results['point_biserial_correlations'].items():
        sig_marker = "*" if data['is_significant'] else ""
        print(f"  {var:20s}: r={data['correlation']:+.4f}, p={data['p_value']:.6f}{sig_marker}")
    
    print(f"\nT-Tests (comparing means: BMI missing vs BMI present):")
    print("-" * 50)
    for var, data in correlation_results['t_test_results'].items():
        sig_marker = "*" if data['is_significant'] else ""
        print(f"  {var}:")
        print(f"    Mean when BMI missing:  {data['mean_when_bmi_missing']:.2f}")
        print(f"    Mean when BMI present:  {data['mean_when_bmi_present']:.2f}")
        print(f"    t={data['t_statistic']:.4f}, p={data['p_value']:.6f}{sig_marker}")
    
    print(f"\nChi-Square Tests (BMI missingness vs categorical variables):")
    print("-" * 50)
    for var, data in correlation_results['chi_square_results'].items():
        sig_marker = "*" if data['is_significant'] else ""
        print(f"  {var:20s}: Chi2={data['chi2_statistic']:.4f}, p={data['p_value']:.6f}{sig_marker}")
    
    # ==========================================================================
    # 3. Generate Recommendations
    # ==========================================================================
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    recommendations = []
    
    # Smoking status recommendation
    if chi_square_results['is_significant']:
        recommendations.append(
            "SMOKING STATUS: Treat 'Unknown' as a valid categorical feature, not missing data. "
            "The significant Chi-Square test indicates that stroke risk varies across smoking "
            "categories including 'Unknown'. This could represent patients who refuse to disclose "
            "smoking habits, which may itself be predictive."
        )
    else:
        recommendations.append(
            "SMOKING STATUS: Consider either keeping 'Unknown' as a category or imputing based "
            "on other features. The non-significant result suggests no strong evidence that "
            "'Unknown' represents a distinct risk group."
        )
    
    # BMI imputation recommendation
    if mcar_results['is_mcar'] is True:
        recommendations.append(
            "BMI IMPUTATION: Safe to use standard imputation methods (mean, median, KNN, or MICE). "
            "Little's MCAR test supports the assumption that BMI is Missing Completely At Random."
        )
    elif mcar_results['is_mcar'] is False or (mcar_results['p_value'] is not None and mcar_results['p_value'] < 0.05):
        # Check which variables correlate with missingness
        sig_correlations = [
            var for var, data in correlation_results['point_biserial_correlations'].items()
            if data['is_significant']
        ]
        if sig_correlations:
            recommendations.append(
                f"BMI IMPUTATION: Use caution with simple imputation. MCAR is REJECTED (p < 0.05). "
                f"BMI missingness correlates significantly with: {', '.join(sig_correlations)}. "
                f"This indicates MAR (Missing At Random) mechanism. Consider:\n"
                f"  - Multiple Imputation by Chained Equations (MICE)\n"
                f"  - Model-based imputation conditioning on correlated variables\n"
                f"  - Creating a 'bmi_missing' indicator as an additional feature"
            )
        else:
            recommendations.append(
                "BMI IMPUTATION: Although MCAR is rejected, no strong correlations found with "
                "missingness. Consider using MICE or predictive mean matching for imputation."
            )
    else:
        recommendations.append(
            "BMI IMPUTATION: Unable to determine missingness mechanism. Recommend using robust "
            "imputation methods like MICE that handle uncertainty appropriately."
        )
    
    results['recommendations'] = recommendations
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec}")
    
    print("\n" + "=" * 80)
    print("(* indicates p < 0.05)")
    print("=" * 80)
    
    return results


def generate_analysis_report(df: pd.DataFrame, output_path: Optional[str] = None) -> str:
    """
    Generate a comprehensive text report of the missingness analysis.
    
    Args:
        df: Stroke prediction DataFrame
        output_path: Optional path to save the report
        
    Returns:
        Report text as string
    """
    import io
    import sys
    
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    
    # Run analysis
    results = analyze_missingness(df)
    
    # Get captured output
    report = buffer.getvalue()
    sys.stdout = old_stdout
    
    # Save if path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {output_path}")
    
    return report


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent))
    from data_ingestion import load_stroke_data
    
    # Default path
    default_path = Path(__file__).parent.parent / "healthcare-dataset-stroke-data.csv"
    
    filepath = sys.argv[1] if len(sys.argv) > 1 else default_path
    
    try:
        print(f"Loading data from: {filepath}")
        df = load_stroke_data(filepath)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")
        
        # Run analysis
        results = analyze_missingness(df)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
