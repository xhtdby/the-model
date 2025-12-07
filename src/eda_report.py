"""
Exploratory Data Analysis & Preprocessing Justification Report
===============================================================

This module generates comprehensive analysis and visualizations for the
research report, covering:

1. Missing Values Analysis
2. Feature Cardinality & Richness
3. Target Variable Analysis (Imbalance)
4. Feature Type Classification
5. Multicollinearity Analysis
6. Distribution Analysis (Normalization needs)
7. Preprocessing Justification

Outputs:
- Statistical summaries
- Visualizations (correlation matrix, distributions, etc.)
- Preprocessing recommendations

Author: Healthcare AI Team
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
from scipy import stats
from scipy.stats import chi2_contingency, normaltest, shapiro
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures" / "eda"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# 1. Missing Values Analysis
# =============================================================================

def analyze_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive missing values analysis.
    
    Returns:
        Dict with missing statistics and recommendations
    """
    print("\n" + "="*70)
    print("1. MISSING VALUES ANALYSIS")
    print("="*70)
    
    missing_stats = []
    
    for col in df.columns:
        n_missing = df[col].isna().sum()
        pct_missing = (n_missing / len(df)) * 100
        
        if n_missing > 0:
            missing_stats.append({
                'Feature': col,
                'Missing_Count': n_missing,
                'Missing_Percentage': pct_missing,
                'Data_Type': str(df[col].dtype)
            })
    
    missing_df = pd.DataFrame(missing_stats)
    
    if len(missing_df) > 0:
        missing_df = missing_df.sort_values('Missing_Percentage', ascending=False)
        print("\nFeatures with Missing Values:")
        print(missing_df.to_string(index=False))
        
        # Visualize
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=missing_df, x='Missing_Percentage', y='Feature', 
                    palette='Reds_r', ax=ax)
        ax.set_xlabel('Missing Percentage (%)')
        ax.set_title('Missing Values by Feature', fontweight='bold')
        ax.axvline(x=5, color='orange', linestyle='--', label='5% threshold')
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'missing_values_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {FIGURES_DIR / 'missing_values_analysis.png'}")
        plt.close()
    else:
        print("\n✓ No missing values detected in dataset")
    
    # Recommendation
    print("\n📊 Missing Value Recommendations:")
    for _, row in missing_df.iterrows() if len(missing_df) > 0 else []:
        if row['Missing_Percentage'] < 5:
            method = "KNN Imputation (preserves relationships)"
        elif row['Missing_Percentage'] < 20:
            method = "KNN Imputation or Multiple Imputation"
        else:
            method = "Consider dropping or separate 'Unknown' category"
        print(f"  • {row['Feature']}: {method}")
    
    return {
        'missing_summary': missing_df if len(missing_df) > 0 else None,
        'total_features_with_missing': len(missing_df)
    }


# =============================================================================
# 2. Feature Cardinality & Richness Analysis
# =============================================================================

def analyze_feature_cardinality(df: pd.DataFrame, target_col: str = 'stroke') -> Dict[str, Any]:
    """
    Analyze feature cardinality to identify low-information features.
    
    Returns:
        Dict with cardinality statistics and grouping recommendations
    """
    print("\n" + "="*70)
    print("2. FEATURE CARDINALITY & RICHNESS ANALYSIS")
    print("="*70)
    
    cardinality_stats = []
    
    for col in df.columns:
        if col == target_col:
            continue
            
        n_unique = df[col].nunique()
        n_total = len(df)
        cardinality_ratio = n_unique / n_total
        
        # Classify feature type
        if df[col].dtype in ['int64', 'float64']:
            if n_unique == 2:
                feature_type = 'Binary'
            elif n_unique < 10:
                feature_type = 'Discrete'
            else:
                feature_type = 'Continuous'
        else:
            if n_unique < 10:
                feature_type = 'Categorical (Low)'
            elif n_unique < 50:
                feature_type = 'Categorical (Medium)'
            else:
                feature_type = 'Categorical (High)'
        
        cardinality_stats.append({
            'Feature': col,
            'Unique_Values': n_unique,
            'Total_Samples': n_total,
            'Cardinality_Ratio': cardinality_ratio,
            'Feature_Type': feature_type,
            'Top_3_Values': df[col].value_counts().head(3).to_dict()
        })
    
    card_df = pd.DataFrame(cardinality_stats).sort_values('Cardinality_Ratio')
    
    print("\nFeature Cardinality Summary:")
    print(card_df[['Feature', 'Unique_Values', 'Cardinality_Ratio', 'Feature_Type']].to_string(index=False))
    
    # Identify low-cardinality features
    low_card = card_df[card_df['Cardinality_Ratio'] < 0.01]
    
    if len(low_card) > 0:
        print("\n⚠️  Low Cardinality Features (ratio < 1%):")
        for _, row in low_card.iterrows():
            print(f"  • {row['Feature']}: {row['Unique_Values']} unique values")
            print(f"    Top values: {row['Top_3_Values']}")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Cardinality ratio
    sns.barplot(data=card_df, y='Feature', x='Cardinality_Ratio', 
                palette='viridis', ax=ax1)
    ax1.set_xlabel('Cardinality Ratio (Unique/Total)')
    ax1.set_title('Feature Cardinality Analysis', fontweight='bold')
    ax1.axvline(x=0.01, color='red', linestyle='--', label='Low cardinality threshold')
    ax1.legend()
    
    # Feature type distribution
    type_counts = card_df['Feature_Type'].value_counts()
    ax2.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
            startangle=90, colors=sns.color_palette('Set2'))
    ax2.set_title('Feature Type Distribution', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'feature_cardinality.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {FIGURES_DIR / 'feature_cardinality.png'}")
    plt.close()
    
    return {
        'cardinality_summary': card_df,
        'low_cardinality_features': low_card['Feature'].tolist() if len(low_card) > 0 else []
    }


# =============================================================================
# 3. Target Variable Analysis (Class Imbalance)
# =============================================================================

def analyze_target_imbalance(df: pd.DataFrame, target_col: str = 'stroke') -> Dict[str, Any]:
    """
    Analyze target variable distribution and imbalance.
    
    Returns:
        Dict with imbalance metrics and recommendations
    """
    print("\n" + "="*70)
    print("3. TARGET VARIABLE IMBALANCE ANALYSIS")
    print("="*70)
    
    # Class distribution
    class_counts = df[target_col].value_counts().sort_index()
    class_props = df[target_col].value_counts(normalize=True).sort_index()
    
    n_negative = class_counts[0]
    n_positive = class_counts[1]
    imbalance_ratio = n_negative / n_positive
    minority_pct = (n_positive / len(df)) * 100
    
    print(f"\nClass Distribution:")
    print(f"  Negative (No Stroke): {n_negative:,} ({class_props[0]*100:.2f}%)")
    print(f"  Positive (Stroke):    {n_positive:,} ({class_props[1]*100:.2f}%)")
    print(f"\n  Imbalance Ratio: {imbalance_ratio:.2f}:1")
    print(f"  Minority Class %: {minority_pct:.2f}%")
    
    # Classify imbalance severity
    if imbalance_ratio < 3:
        severity = "Mild"
        recommendation = "Standard cross-validation sufficient"
    elif imbalance_ratio < 10:
        severity = "Moderate"
        recommendation = "Use stratified sampling + class weights"
    else:
        severity = "Severe"
        recommendation = "Use cost-sensitive learning (scale_pos_weight) + stratified CV"
    
    print(f"\n  Imbalance Severity: {severity}")
    print(f"  📊 Recommendation: {recommendation}")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart
    colors = ['#2ecc71', '#e74c3c']
    bars = ax1.bar(['No Stroke', 'Stroke'], class_counts.values, color=colors, 
                   edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Count')
    ax1.set_title('Target Variable Distribution (Absolute)', fontweight='bold')
    
    # Add count labels
    for bar, count in zip(bars, class_counts.values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({count/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    ax2.pie(class_counts.values, labels=['No Stroke', 'Stroke'], 
            autopct='%1.1f%%', startangle=90, colors=colors,
            explode=(0, 0.1), shadow=True)
    ax2.set_title('Target Variable Distribution (Relative)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'target_imbalance.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {FIGURES_DIR / 'target_imbalance.png'}")
    plt.close()
    
    return {
        'n_negative': int(n_negative),
        'n_positive': int(n_positive),
        'imbalance_ratio': float(imbalance_ratio),
        'minority_percentage': float(minority_pct),
        'severity': severity,
        'recommendation': recommendation
    }


# =============================================================================
# 4. Feature Type Classification & Encoding Needs
# =============================================================================

def classify_features(df: pd.DataFrame, target_col: str = 'stroke') -> Dict[str, List[str]]:
    """
    Classify features by type and identify encoding requirements.
    
    Returns:
        Dict with feature classifications
    """
    print("\n" + "="*70)
    print("4. FEATURE TYPE CLASSIFICATION & ENCODING REQUIREMENTS")
    print("="*70)
    
    continuous_features = []
    discrete_features = []
    binary_features = []
    categorical_features = []
    
    for col in df.columns:
        if col == target_col:
            continue
        
        n_unique = df[col].nunique()
        dtype = df[col].dtype
        
        if dtype in ['int64', 'float64']:
            if n_unique == 2:
                binary_features.append(col)
            elif n_unique < 10:
                discrete_features.append(col)
            else:
                continuous_features.append(col)
        else:
            categorical_features.append(col)
    
    print("\n📊 Feature Classification:")
    print(f"\n  Continuous Features ({len(continuous_features)}):")
    for f in continuous_features:
        print(f"    • {f}: {df[f].min():.2f} to {df[f].max():.2f}")
    
    print(f"\n  Discrete Features ({len(discrete_features)}):")
    for f in discrete_features:
        print(f"    • {f}: {sorted(df[f].unique())}")
    
    print(f"\n  Binary Features ({len(binary_features)}):")
    for f in binary_features:
        print(f"    • {f}: {sorted(df[f].unique())}")
    
    print(f"\n  Categorical Features ({len(categorical_features)}):")
    for f in categorical_features:
        categories = df[f].value_counts()
        print(f"    • {f}: {len(categories)} categories - {list(categories.index[:3])}...")
    
    # Encoding recommendations
    print("\n🔧 Encoding Recommendations:")
    
    if binary_features:
        print(f"\n  Binary features: {binary_features}")
        print("    → Already numeric (0/1), no encoding needed")
    
    if categorical_features:
        print(f"\n  Categorical features: {categorical_features}")
        for feat in categorical_features:
            n_cat = df[feat].nunique()
            if n_cat == 2:
                print(f"    → {feat}: Label Encoding (2 categories)")
            elif n_cat < 10:
                print(f"    → {feat}: One-Hot Encoding (drop='first' to avoid multicollinearity)")
            else:
                print(f"    → {feat}: Consider Target Encoding or Frequency Encoding ({n_cat} categories)")
    
    # Create summary table
    summary_data = []
    for col in df.columns:
        if col == target_col:
            continue
        
        if col in continuous_features:
            ftype = "Continuous"
            encoding = "None (RobustScaler for normalization)"
        elif col in discrete_features:
            ftype = "Discrete"
            encoding = "None (keep as numeric)"
        elif col in binary_features:
            ftype = "Binary"
            encoding = "None (already 0/1)"
        else:
            ftype = "Categorical"
            n_cat = df[col].nunique()
            if n_cat <= 10:
                encoding = "One-Hot Encoding (drop='first')"
            else:
                encoding = "Target/Frequency Encoding"
        
        summary_data.append({
            'Feature': col,
            'Type': ftype,
            'Unique_Values': df[col].nunique(),
            'Encoding_Method': encoding
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary table
    summary_df.to_csv(REPORTS_DIR / 'feature_classification.csv', index=False)
    print(f"\n✓ Saved: {REPORTS_DIR / 'feature_classification.csv'}")
    
    return {
        'continuous': continuous_features,
        'discrete': discrete_features,
        'binary': binary_features,
        'categorical': categorical_features,
        'summary': summary_df
    }


# =============================================================================
# 5. Multicollinearity Analysis
# =============================================================================

def analyze_multicollinearity(df: pd.DataFrame, target_col: str = 'stroke') -> Dict[str, Any]:
    """
    Analyze multicollinearity using correlation matrix for ALL features.
    
    Encodes categorical features temporarily to compute correlations across
    all features (continuous, discrete, binary, and categorical).
    
    Returns:
        Dict with correlation analysis results
    """
    print("\n" + "="*70)
    print("5. MULTICOLLINEARITY ANALYSIS")
    print("="*70)
    
    # Create a copy for encoding
    df_encoded = df.copy()
    
    # Encode categorical features as numeric for correlation analysis
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        # Use label encoding for correlation purposes
        df_encoded[col] = pd.factorize(df_encoded[col])[0]
    
    # Now all columns are numeric
    numeric_cols = df_encoded.columns.tolist()
    
    # Calculate correlation matrix for ALL features
    corr_matrix = df_encoded[numeric_cols].corr()
    
    print(f"\nCorrelation with Target Variable '{target_col}':")
    target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
    
    for feat, corr in target_corr.items():
        strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
        direction = "positive" if corr > 0 else "negative"
        print(f"  • {feat:30s}: {corr:7.4f} ({strength} {direction})")
    
    # Identify highly correlated feature pairs
    high_corr_pairs = []
    threshold = 0.8
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr_pairs.append({
                    'Feature_1': corr_matrix.columns[i],
                    'Feature_2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
    
    if high_corr_pairs:
        print(f"\n⚠️  Highly Correlated Pairs (|r| > {threshold}):")
        for pair in high_corr_pairs:
            print(f"  • {pair['Feature_1']} <-> {pair['Feature_2']}: {pair['Correlation']:.4f}")
    else:
        print(f"\n✓ No multicollinearity detected (all |r| < {threshold})")
    
    # Visualize correlation matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Draw heatmap
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title('Correlation Matrix Heatmap\n(Lower Triangle)', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {FIGURES_DIR / 'correlation_matrix.png'}")
    plt.close()
    
    # Focused plot: Target correlation
    fig, ax = plt.subplots(figsize=(10, 8))
    
    target_corr_sorted = target_corr.abs().sort_values(ascending=True)
    colors = ['green' if x > 0 else 'red' for x in target_corr[target_corr_sorted.index]]
    
    ax.barh(range(len(target_corr_sorted)), target_corr[target_corr_sorted.index].values,
            color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(target_corr_sorted)))
    ax.set_yticklabels(target_corr_sorted.index)
    ax.set_xlabel('Correlation Coefficient')
    ax.set_title(f'Feature Correlation with Target ({target_col})', fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'target_correlation.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {FIGURES_DIR / 'target_correlation.png'}")
    plt.close()
    
    # Save correlation matrix
    corr_matrix.to_csv(REPORTS_DIR / 'correlation_matrix.csv')
    print(f"✓ Saved: {REPORTS_DIR / 'correlation_matrix.csv'}")
    
    return {
        'correlation_matrix': corr_matrix,
        'target_correlation': target_corr,
        'high_corr_pairs': high_corr_pairs,
        'multicollinearity_detected': len(high_corr_pairs) > 0
    }


# =============================================================================
# 6. Distribution Analysis (Normalization Needs)
# =============================================================================

def analyze_distributions(df: pd.DataFrame, continuous_features: List[str]) -> Dict[str, Any]:
    """
    Analyze distribution of continuous features to determine normalization needs.
    
    Returns:
        Dict with normality tests and recommendations
    """
    print("\n" + "="*70)
    print("6. DISTRIBUTION ANALYSIS & NORMALIZATION NEEDS")
    print("="*70)
    
    normality_results = []
    
    for col in continuous_features:
        data = df[col].dropna()
        
        # Descriptive statistics
        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()
        skew_val = data.skew()
        kurt_val = data.kurtosis()
        
        # Normality test (Shapiro-Wilk for small samples, D'Agostino for large)
        if len(data) < 5000:
            stat, p_value = shapiro(data)
            test_name = "Shapiro-Wilk"
        else:
            stat, p_value = normaltest(data)
            test_name = "D'Agostino"
        
        is_normal = p_value > 0.05
        
        # Recommendation
        if is_normal:
            recommendation = "StandardScaler (data is normally distributed)"
        elif abs(skew_val) > 1:
            recommendation = "RobustScaler or Log Transform (highly skewed)"
        else:
            recommendation = "RobustScaler (not normal, use median/IQR)"
        
        normality_results.append({
            'Feature': col,
            'Mean': mean_val,
            'Median': median_val,
            'Std': std_val,
            'Skewness': skew_val,
            'Kurtosis': kurt_val,
            'Normality_Test': test_name,
            'P_Value': p_value,
            'Is_Normal': is_normal,
            'Recommendation': recommendation
        })
    
    norm_df = pd.DataFrame(normality_results)
    
    print("\nDistribution Analysis:")
    for _, row in norm_df.iterrows():
        print(f"\n  {row['Feature']}:")
        print(f"    Mean: {row['Mean']:.2f}, Median: {row['Median']:.2f}, Std: {row['Std']:.2f}")
        print(f"    Skewness: {row['Skewness']:.2f}, Kurtosis: {row['Kurtosis']:.2f}")
        print(f"    {row['Normality_Test']} p-value: {row['P_Value']:.4f}")
        print(f"    Normal? {row['Is_Normal']} → {row['Recommendation']}")
    
    # Visualize distributions
    n_features = len(continuous_features)
    n_rows = (n_features + 1) // 2
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 5*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, col in enumerate(continuous_features):
        ax = axes[idx]
        data = df[col].dropna()
        
        # Histogram with KDE
        ax.hist(data, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        
        # Overlay KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 100)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        
        # Add mean and median lines
        ax.axvline(data.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.1f}')
        ax.axvline(data.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {data.median():.1f}')
        
        ax.set_xlabel(col)
        ax.set_ylabel('Density')
        ax.set_title(f'{col} Distribution\nSkewness: {data.skew():.2f}', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'distributions.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {FIGURES_DIR / 'distributions.png'}")
    plt.close()
    
    # Q-Q plots for normality assessment
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 5*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, col in enumerate(continuous_features):
        ax = axes[idx]
        data = df[col].dropna()
        
        stats.probplot(data, dist="norm", plot=ax)
        ax.set_title(f'Q-Q Plot: {col}', fontweight='bold')
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'qq_plots.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {FIGURES_DIR / 'qq_plots.png'}")
    plt.close()
    
    # Save analysis
    norm_df.to_csv(REPORTS_DIR / 'normality_analysis.csv', index=False)
    print(f"✓ Saved: {REPORTS_DIR / 'normality_analysis.csv'}")
    
    return {
        'normality_summary': norm_df,
        'non_normal_features': norm_df[~norm_df['Is_Normal']]['Feature'].tolist()
    }


# =============================================================================
# 7. Preprocessing Justification Summary
# =============================================================================

def generate_preprocessing_summary(
    missing_analysis: Dict,
    feature_classification: Dict,
    correlation_analysis: Dict,
    distribution_analysis: Dict
) -> None:
    """
    Generate comprehensive preprocessing justification report.
    """
    print("\n" + "="*70)
    print("7. PREPROCESSING JUSTIFICATION SUMMARY")
    print("="*70)
    
    report = []
    
    report.append("="*70)
    report.append("PREPROCESSING JUSTIFICATION REPORT")
    report.append("="*70)
    report.append("")
    
    # Missing Values
    report.append("1. MISSING VALUES HANDLING")
    report.append("-" * 70)
    if missing_analysis['total_features_with_missing'] > 0:
        report.append(f"Features with missing values: {missing_analysis['total_features_with_missing']}")
        report.append("")
        report.append("Strategy: KNN Imputation (k=5)")
        report.append("Justification:")
        report.append("  • Preserves multivariate relationships between features")
        report.append("  • Superior to mean/median imputation for MAR data")
        report.append("  • Uses k=5 nearest neighbors based on Euclidean distance")
        report.append("  • Applied AFTER RobustScaler to ensure fair distance calculation")
    else:
        report.append("✓ No missing values detected")
    report.append("")
    
    # Categorical Encoding
    report.append("2. CATEGORICAL FEATURE ENCODING")
    report.append("-" * 70)
    cat_features = feature_classification['categorical']
    if cat_features:
        report.append("Strategy: One-Hot Encoding with drop='first'")
        report.append("Justification:")
        report.append("  • Converts nominal categories into binary indicators")
        report.append("  • drop='first' prevents perfect multicollinearity (dummy variable trap)")
        report.append("  • Scikit-learn's OneHotEncoder handles unseen categories gracefully")
        report.append("")
        report.append(f"Categorical features to encode: {cat_features}")
    else:
        report.append("No categorical features require encoding")
    report.append("")
    
    # Numerical Scaling
    report.append("3. NUMERICAL FEATURE SCALING")
    report.append("-" * 70)
    continuous_features = feature_classification['continuous']
    non_normal = distribution_analysis['non_normal_features']
    
    report.append("Strategy: RobustScaler (median and IQR)")
    report.append("Justification:")
    report.append(f"  • {len(non_normal)}/{len(continuous_features)} continuous features are non-normal")
    report.append("  • RobustScaler uses median/IQR instead of mean/std")
    report.append("  • More robust to outliers than StandardScaler")
    report.append("  • Critical for distance-based algorithms (KNN imputation)")
    report.append("")
    report.append("Applied BEFORE KNN Imputation to ensure:")
    report.append("  • Fair distance calculations across features with different scales")
    report.append("  • e.g., glucose (~200) vs age (~50) have equal weight")
    report.append("")
    
    # Log Transformation
    report.append("4. LOG TRANSFORMATION")
    report.append("-" * 70)
    report.append("NOT APPLIED in this project")
    report.append("Justification:")
    report.append("  • Tree-based models (XGBoost) are invariant to monotonic transformations")
    report.append("  • RobustScaler already handles skewness via median/IQR")
    report.append("  • Log transform would be needed for linear models or when:")
    report.append("    - Feature has extreme right skew (skewness > 2)")
    report.append("    - Feature spans multiple orders of magnitude")
    report.append("")
    
    # Feature Removal
    report.append("5. FEATURE REMOVAL")
    report.append("-" * 70)
    report.append("Removed: 'id' column")
    report.append("Justification:")
    target_corr = correlation_analysis['target_correlation']
    if 'id' in target_corr.index:
        report.append(f"  • Correlation with target: {target_corr.get('id', 0):.4f} (negligible)")
    report.append("  • Sequential identifier with no predictive value")
    report.append("  • Would cause overfitting if included")
    report.append("")
    
    # Multicollinearity
    report.append("6. MULTICOLLINEARITY CHECK")
    report.append("-" * 70)
    if correlation_analysis['multicollinearity_detected']:
        report.append("⚠️  High correlation detected between:")
        for pair in correlation_analysis['high_corr_pairs']:
            report.append(f"  • {pair['Feature_1']} <-> {pair['Feature_2']}: {pair['Correlation']:.4f}")
        report.append("")
        report.append("Action: Monitor during feature importance analysis")
    else:
        report.append("✓ No multicollinearity detected (all pairwise |r| < 0.8)")
    report.append("")
    
    # Pipeline Structure
    report.append("7. FINAL PIPELINE STRUCTURE")
    report.append("-" * 70)
    report.append("sklearn.Pipeline with ColumnTransformer:")
    report.append("")
    report.append("  Numerical Features:")
    report.append("    1. RobustScaler() → scales to median=0, IQR=1")
    report.append("    2. KNNImputer(n_neighbors=5) → fills missing BMI")
    report.append("")
    report.append("  Categorical Features:")
    report.append("    1. OneHotEncoder(drop='first', handle_unknown='ignore')")
    report.append("")
    report.append("Benefits:")
    report.append("  • Prevents data leakage (fit only on training data)")
    report.append("  • Ensures reproducibility")
    report.append("  • Compatible with scikit-learn cross-validation")
    report.append("  • Serializable for production deployment")
    report.append("")
    
    report.append("="*70)
    
    # Print and save
    full_report = "\n".join(report)
    print("\n" + full_report)
    
    with open(REPORTS_DIR / 'preprocessing_justification.txt', 'w', encoding='utf-8') as f:
        f.write(full_report)
    
    print(f"\n✓ Saved: {REPORTS_DIR / 'preprocessing_justification.txt'}")


# =============================================================================
# Main Execution
# =============================================================================

def run_full_eda_report(data_path: str = None) -> Dict[str, Any]:
    """
    Execute complete EDA and preprocessing justification analysis.
    
    Args:
        data_path: Path to dataset CSV
        
    Returns:
        Dict with all analysis results
    """
    if data_path is None:
        data_path = PROJECT_ROOT / "healthcare-dataset-stroke-data.csv"
    
    print("="*70)
    print("EXPLORATORY DATA ANALYSIS & PREPROCESSING JUSTIFICATION")
    print("="*70)
    print(f"Dataset: {data_path}")
    print(f"Output directory: {FIGURES_DIR}")
    print("="*70)
    
    # Load data
    df_original = pd.read_csv(data_path)
    print(f"\nDataset loaded: {len(df_original)} rows, {len(df_original.columns)} columns")
    
    # Keep original df for correlation analysis (includes 'id')
    # Create working df without 'id' for other analyses
    df = df_original.copy()
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
        print("Dropped 'id' column for analyses (kept for correlation matrix)")
    
    # Run analyses
    results = {}
    
    results['missing'] = analyze_missing_values(df)
    results['cardinality'] = analyze_feature_cardinality(df)
    results['target'] = analyze_target_imbalance(df)
    results['classification'] = classify_features(df)
    # Use original df with 'id' for correlation analysis
    results['correlation'] = analyze_multicollinearity(df_original)
    results['distribution'] = analyze_distributions(df, results['classification']['continuous'])
    
    # Generate summary report
    generate_preprocessing_summary(
        results['missing'],
        results['classification'],
        results['correlation'],
        results['distribution']
    )
    
    print("\n" + "="*70)
    print("EDA REPORT GENERATION COMPLETE")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Reports: {REPORTS_DIR}")
    
    return results


if __name__ == "__main__":
    results = run_full_eda_report()
