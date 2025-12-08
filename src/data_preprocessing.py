"""
Data Preprocessing Utilities

This module contains functions for:
- Loading data
- Handling missing values
- Detecting outliers
- Data cleaning and validation

Educational Notes are included throughout to explain concepts!
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


def load_data(data_path: str = 'data',
              train_file: str = 'application_train.csv',
              test_file: str = 'application_test.csv',
              use_all_data_sources: bool = True,
              validate: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and test datasets with comprehensive validation.

    Educational Note:
    -----------------
    Always load train and test data separately to avoid data leakage.
    Data leakage = using information from test set during training,
    which leads to overly optimistic performance estimates.

    When use_all_data_sources=True, this function will aggregate features from:
    - bureau.csv: Credit bureau data
    - bureau_balance.csv: Monthly credit bureau balances
    - previous_application.csv: Previous loan applications
    - POS_CASH_balance.csv: Point of sale and cash loan balances
    - credit_card_balance.csv: Credit card balances
    - installments_payments.csv: Payment installment history

    Parameters:
    -----------
    data_path : str
        Path to data directory
    train_file : str
        Training data filename
    test_file : str
        Test data filename
    use_all_data_sources : bool
        If True, load and aggregate all data sources
    validate : bool
        If True, run comprehensive data validation

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        Training and test dataframes

    Raises:
    -------
    FileNotFoundError : If data files don't exist
    ValueError : If data validation fails

    Example:
    --------
    >>> train_df, test_df = load_data(use_all_data_sources=True)
    >>> print(f"Train shape: {train_df.shape}")
    """
    from src.validation import (
        validate_file_exists, validate_dataframe_schema,
        validate_id_column, validate_target_column
    )

    path = Path(data_path)

    # Robust path resolution with proper error handling
    if not path.exists():
        # Try parent directory
        parent_path = Path('..') / data_path
        if parent_path.exists():
            path = parent_path
            data_path = str(path)
        else:
            # Try project root
            project_root = Path(__file__).parent.parent / data_path
            if project_root.exists():
                path = project_root
                data_path = str(path)
            else:
                raise FileNotFoundError(
                    f"Data directory not found: {data_path}\n"
                    f"Current working directory: {Path.cwd()}\n"
                    f"Tried paths:\n"
                    f"  - {Path(data_path).absolute()}\n"
                    f"  - {parent_path.absolute()}\n"
                    f"  - {project_root.absolute()}"
                )

    print(f"Loading data from: {path.absolute()}\n")

    try:
        if use_all_data_sources:
            # Use comprehensive feature aggregation from all data sources
            from src.feature_aggregation import load_and_aggregate_all_data
            train_df, test_df = load_and_aggregate_all_data(data_dir=data_path)
        else:
            # Validate files exist
            train_path = path / train_file
            test_path = path / test_file

            if validate:
                validate_file_exists(train_path, "Training file")
                validate_file_exists(test_path, "Test file")

            # Load only main application tables
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            print(f"[OK] Training data loaded: {train_df.shape}")
            print(f"[OK] Test data loaded: {test_df.shape}")

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Failed to load data: {e}")
    except pd.errors.EmptyDataError:
        raise ValueError("Data file is empty")
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse CSV file: {e}")

    # Comprehensive validation
    if validate:
        print("\nValidating data integrity...")

        # Required columns
        validate_dataframe_schema(
            train_df,
            required_columns=['SK_ID_CURR', 'TARGET'],
            data_description="Training data"
        )
        validate_dataframe_schema(
            test_df,
            required_columns=['SK_ID_CURR'],
            data_description="Test data"
        )

        # ID column validation
        validate_id_column(train_df, data_description="Training data")
        validate_id_column(test_df, data_description="Test data")

        # Target validation (binary classification)
        validate_target_column(
            train_df,
            target_column='TARGET',
            expected_values=[0, 1],
            data_description="Training data"
        )

        # Check for data leakage (no overlap in IDs)
        train_ids = set(train_df['SK_ID_CURR'])
        test_ids = set(test_df['SK_ID_CURR'])
        overlap = train_ids & test_ids

        if overlap:
            raise ValueError(
                f"Data leakage detected! {len(overlap)} IDs appear in both train and test sets.\n"
                f"First 5 overlapping IDs: {list(overlap)[:5]}"
            )

        print("[OK] Data validation passed")

    return train_df, test_df


def analyze_missing_values(df: pd.DataFrame,
                           threshold: float = 0.0,
                           verbose: bool = True) -> pd.DataFrame:
    """
    Analyze missing values in a dataframe.

    Educational Note:
    -----------------
    Missing values are common in real-world data. Understanding WHY data is missing
    helps you choose the right strategy:

    1. **Missing Completely at Random (MCAR):** Random, no pattern
       → Safe to drop or impute with mean/median

    2. **Missing at Random (MAR):** Missing depends on other variables
       → Impute using related features

    3. **Missing Not at Random (MNAR):** Missing has a pattern (e.g., people don't report high income)
       → Create "is_missing" indicator feature!

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    threshold : float
        Only show features with missing % > threshold (default: 0)
    verbose : bool
        Print detailed output

    Returns:
    --------
    pd.DataFrame
        Summary of missing values

    Example:
    --------
    >>> missing_summary = analyze_missing_values(train_df, threshold=10)
    >>> # Shows only features with >10% missing
    """
    # Calculate missing values
    total = df.isnull().sum()
    percent = (df.isnull().sum() / len(df)) * 100
    dtypes = df.dtypes

    # Create summary dataframe
    missing_data = pd.DataFrame({
        'column': df.columns,
        'missing_count': total.values,
        'missing_percent': percent.values,
        'dtype': dtypes.values
    })

    # Filter by threshold
    missing_data = missing_data[missing_data['missing_percent'] > threshold]
    missing_data = missing_data.sort_values('missing_percent', ascending=False).reset_index(drop=True)

    if verbose and len(missing_data) > 0:
        print("=" * 80)
        print("MISSING VALUES ANALYSIS")
        print("=" * 80)
        print(f"Total features: {len(df.columns)}")
        print(f"Features with missing > {threshold}%: {len(missing_data)}")
        print(f"\\nTop features with missing values:")
        print(missing_data.head(15).to_string(index=False))

        # Categorize severity
        high_missing = len(missing_data[missing_data['missing_percent'] > 50])
        medium_missing = len(missing_data[(missing_data['missing_percent'] > 20) &
                                         (missing_data['missing_percent'] <= 50)])
        low_missing = len(missing_data[missing_data['missing_percent'] <= 20])

        print(f"\\n📊 Severity Breakdown:")
        print(f"   High (>50%): {high_missing} features")
        print(f"   Medium (20-50%): {medium_missing} features")
        print(f"   Low (<20%): {low_missing} features")

    elif verbose:
        print(f"✅ No features with missing values > {threshold}%")

    return missing_data


def handle_missing_values(df: pd.DataFrame,
                          strategy: Dict[str, str],
                          create_indicators: bool = True) -> pd.DataFrame:
    """
    Handle missing values using specified strategies.

    Educational Note:
    -----------------
    Different imputation strategies work better for different scenarios:

    - **drop:** Remove feature entirely (if >70% missing and not informative)
    - **median:** For numerical features with skewed distribution
    - **mean:** For numerical features with normal distribution
    - **mode:** For categorical features (most frequent value)
    - **constant:** Fill with a specific value (e.g., 0, 'Unknown')
    - **forward_fill/back_fill:** For time-series data

    Creating "is_missing" indicators preserves information about missingness!

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    strategy : Dict[str, str]
        Dictionary mapping feature names to imputation strategy
        Example: {'feature1': 'median', 'feature2': 'drop', 'feature3': 'constant_0'}
    create_indicators : bool
        Whether to create binary "was_missing" indicator features

    Returns:
    --------
    pd.DataFrame
        Dataframe with missing values handled

    Example:
    --------
    >>> strategy = {
    ...     'AMT_ANNUITY': 'median',
    ...     'AMT_GOODS_PRICE': 'median',
    ...     'VERY_SPARSE_FEATURE': 'drop'
    ... }
    >>> df_clean = handle_missing_values(df, strategy)
    """
    df_copy = df.copy()

    for feature, method in strategy.items():
        if feature not in df_copy.columns:
            print(f"⚠️  Feature '{feature}' not found, skipping...")
            continue

        # Create missing indicator if requested
        if create_indicators and df_copy[feature].isnull().sum() > 0:
            indicator_name = f'{feature}_WAS_MISSING'
            df_copy[indicator_name] = df_copy[feature].isnull().astype(int)
            print(f"   Created indicator: {indicator_name}")

        # Apply imputation strategy
        if method == 'drop':
            df_copy = df_copy.drop(columns=[feature])
            print(f"✅ Dropped feature: {feature}")

        elif method == 'median':
            median_val = df_copy[feature].median()
            df_copy[feature].fillna(median_val, inplace=True)
            print(f"✅ Imputed {feature} with median: {median_val}")

        elif method == 'mean':
            mean_val = df_copy[feature].mean()
            df_copy[feature].fillna(mean_val, inplace=True)
            print(f"✅ Imputed {feature} with mean: {mean_val}")

        elif method == 'mode':
            mode_val = df_copy[feature].mode()[0] if len(df_copy[feature].mode()) > 0 else 'Unknown'
            df_copy[feature].fillna(mode_val, inplace=True)
            print(f"✅ Imputed {feature} with mode: {mode_val}")

        elif method.startswith('constant_'):
            constant_val = method.split('_')[1]
            # Try to convert to appropriate type
            try:
                if df_copy[feature].dtype in ['int64', 'float64']:
                    constant_val = float(constant_val)
            except:
                pass
            df_copy[feature].fillna(constant_val, inplace=True)
            print(f"✅ Imputed {feature} with constant: {constant_val}")

        else:
            print(f"⚠️  Unknown strategy '{method}' for feature '{feature}', skipping...")

    return df_copy


def detect_outliers(df: pd.DataFrame,
                   features: List[str],
                   method: str = 'iqr',
                   threshold: float = 3.0) -> pd.DataFrame:
    """
    Detect outliers in numerical features.

    Educational Note:
    -----------------
    Outliers are data points that differ significantly from other observations.
    They can be:
    - **Errors:** Data entry mistakes, sensor malfunction
    - **Genuine:** Real extreme values (e.g., billionaire's income)

    Two common detection methods:

    1. **IQR (Interquartile Range):**
       - Outlier if: value < Q1 - 1.5*IQR or value > Q3 + 1.5*IQR
       - Robust to extreme values
       - Good for skewed distributions

    2. **Z-score:**
       - Outlier if: |z-score| > threshold (usually 3)
       - Assumes normal distribution
       - Sensitive to extreme values

    **Important:** Don't automatically remove outliers! Investigate first.
    In credit scoring, extreme values might be informative (e.g., very high debt).

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    features : List[str]
        List of numerical features to check
    method : str
        Detection method: 'iqr' or 'zscore'
    threshold : float
        Threshold for outlier detection (for z-score method)

    Returns:
    --------
    pd.DataFrame
        Summary of outliers per feature

    Example:
    --------
    >>> numerical_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']
    >>> outlier_summary = detect_outliers(df, numerical_cols, method='iqr')
    """
    outlier_summary = []

    for feature in features:
        if feature not in df.columns:
            continue

        if method == 'iqr':
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((df[feature] < lower_bound) | (df[feature] > upper_bound))
            n_outliers = outliers.sum()
            pct_outliers = (n_outliers / len(df)) * 100

            outlier_summary.append({
                'feature': feature,
                'method': 'IQR',
                'n_outliers': n_outliers,
                'pct_outliers': pct_outliers,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            })

        elif method == 'zscore':
            mean = df[feature].mean()
            std = df[feature].std()
            z_scores = np.abs((df[feature] - mean) / std)

            outliers = z_scores > threshold
            n_outliers = outliers.sum()
            pct_outliers = (n_outliers / len(df)) * 100

            outlier_summary.append({
                'feature': feature,
                'method': 'Z-score',
                'n_outliers': n_outliers,
                'pct_outliers': pct_outliers,
                'threshold': threshold
            })

    summary_df = pd.DataFrame(outlier_summary)
    if len(summary_df) > 0:
        summary_df = summary_df.sort_values('pct_outliers', ascending=False)

        print("=" * 80)
        print(f"OUTLIER DETECTION - {method.upper()} METHOD")
        print("=" * 80)
        print(summary_df.to_string(index=False))

    return summary_df


def validate_data_quality(df: pd.DataFrame, target_col: str = 'TARGET') -> Dict[str, any]:
    """
    Comprehensive data quality validation.

    Educational Note:
    -----------------
    Data quality checks are essential before modeling!
    Common issues:
    - Duplicate rows
    - Constant features (no variance)
    - Inf values
    - High cardinality categoricals (too many unique values)
    - Target leakage (features that shouldn't be available at prediction time)

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of target column

    Returns:
    --------
    Dict[str, any]
        Dictionary with validation results

    Example:
    --------
    >>> quality_report = validate_data_quality(train_df)
    >>> if quality_report['has_issues']:
    ...     print("⚠️ Data quality issues found!")
    """
    results = {
        'has_issues': False,
        'issues': []
    }

    print("=" * 80)
    print("DATA QUALITY VALIDATION")
    print("=" * 80)

    # Check for duplicates
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        results['has_issues'] = True
        results['issues'].append(f"Found {n_duplicates} duplicate rows")
        print(f"⚠️  Duplicate rows: {n_duplicates}")
    else:
        print(f"✅ No duplicate rows")

    # Check for constant features (no variance)
    constant_features = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].nunique() <= 1:
            constant_features.append(col)

    if constant_features:
        results['has_issues'] = True
        results['issues'].append(f"Constant features: {constant_features}")
        print(f"⚠️  Constant features (remove these): {constant_features}")
    else:
        print(f"✅ No constant features")

    # Check for inf values
    inf_features = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if np.isinf(df[col]).any():
            inf_features.append(col)

    if inf_features:
        results['has_issues'] = True
        results['issues'].append(f"Inf values in: {inf_features}")
        print(f"⚠️  Inf values found in: {inf_features}")
    else:
        print(f"✅ No inf values")

    # Check target distribution
    if target_col in df.columns:
        target_dist = df[target_col].value_counts(normalize=True)
        if target_dist.min() < 0.01:
            results['has_issues'] = True
            results['issues'].append(f"Severe class imbalance: {target_dist.to_dict()}")
            print(f"⚠️  Severe class imbalance detected")
        print(f"✅ Target distribution: {target_dist.to_dict()}")

    # Check high cardinality categoricals
    high_card_features = []
    for col in df.select_dtypes(include=['object']).columns:
        n_unique = df[col].nunique()
        if n_unique > 50:  # Arbitrary threshold
            high_card_features.append((col, n_unique))

    if high_card_features:
        print(f"⚠️  High cardinality categorical features (may need special encoding):")
        for feat, n in high_card_features:
            print(f"   - {feat}: {n} unique values")

    print("=" * 80)
    if not results['has_issues']:
        print("✅ All data quality checks passed!")
    else:
        print("⚠️  Some issues found - review and address before modeling")

    return results


# Example usage
if __name__ == "__main__":
    print("Data Preprocessing Utilities")
    print("=" * 80)
    print("\\nThis module provides functions for:")
    print("  ✅ Loading data")
    print("  ✅ Analyzing missing values")
    print("  ✅ Handling missing values")
    print("  ✅ Detecting outliers")
    print("  ✅ Validating data quality")
    print("\\nImport in notebooks:")
    print("  from src.data_preprocessing import load_data, analyze_missing_values")
