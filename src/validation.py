"""
Data Validation Utilities

This module provides comprehensive validation functions for data integrity,
schema validation, and quality checks throughout the ML pipeline.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings


class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class SchemaValidationError(Exception):
    """Custom exception for schema validation errors."""
    pass


def validate_file_exists(file_path: Path, file_description: str = "File") -> None:
    """
    Validate that a file exists and is readable.

    Args:
        file_path: Path to file
        file_description: Description for error message

    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file isn't readable
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_description} not found: {file_path}\n"
            f"Current working directory: {Path.cwd()}"
        )

    if not file_path.is_file():
        raise ValueError(f"{file_description} is not a file: {file_path}")

    # Check readability
    try:
        with open(file_path, 'r') as f:
            f.read(1)
    except PermissionError:
        raise PermissionError(f"{file_description} is not readable: {file_path}")


def validate_dataframe_schema(
    df: pd.DataFrame,
    required_columns: List[str],
    optional_columns: Optional[List[str]] = None,
    data_description: str = "DataFrame"
) -> None:
    """
    Validate DataFrame schema has required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        optional_columns: List of optional column names
        data_description: Description for error messages

    Raises:
        SchemaValidationError: If required columns are missing
    """
    if df is None or df.empty:
        raise DataValidationError(f"{data_description} is empty or None")

    missing_required = set(required_columns) - set(df.columns)
    if missing_required:
        raise SchemaValidationError(
            f"{data_description} is missing required columns: {sorted(missing_required)}\n"
            f"Available columns: {sorted(df.columns.tolist())[:20]}..."
        )

    # Log optional columns that are missing (not an error)
    if optional_columns:
        missing_optional = set(optional_columns) - set(df.columns)
        if missing_optional:
            warnings.warn(
                f"{data_description} is missing optional columns: {sorted(missing_optional)}"
            )


def validate_id_column(
    df: pd.DataFrame,
    id_column: str = 'SK_ID_CURR',
    allow_duplicates: bool = False,
    data_description: str = "DataFrame"
) -> None:
    """
    Validate ID column integrity.

    Args:
        df: DataFrame to validate
        id_column: Name of ID column
        allow_duplicates: Whether to allow duplicate IDs
        data_description: Description for error messages

    Raises:
        DataValidationError: If validation fails
    """
    if id_column not in df.columns:
        raise SchemaValidationError(
            f"{data_description} missing ID column: {id_column}"
        )

    # Check for nulls
    null_count = df[id_column].isna().sum()
    if null_count > 0:
        raise DataValidationError(
            f"{data_description} has {null_count} null values in {id_column}"
        )

    # Check for duplicates
    if not allow_duplicates:
        duplicate_count = df[id_column].duplicated().sum()
        if duplicate_count > 0:
            duplicate_ids = df[df[id_column].duplicated()][id_column].tolist()[:5]
            raise DataValidationError(
                f"{data_description} has {duplicate_count} duplicate IDs in {id_column}\n"
                f"First 5 duplicates: {duplicate_ids}"
            )


def validate_target_column(
    df: pd.DataFrame,
    target_column: str = 'TARGET',
    expected_values: Optional[List] = None,
    data_description: str = "DataFrame"
) -> None:
    """
    Validate target column for classification.

    Args:
        df: DataFrame to validate
        target_column: Name of target column
        expected_values: Expected unique values (e.g., [0, 1] for binary)
        data_description: Description for error messages

    Raises:
        DataValidationError: If validation fails
    """
    if target_column not in df.columns:
        raise SchemaValidationError(
            f"{data_description} missing target column: {target_column}"
        )

    # Check data type
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise DataValidationError(
            f"{data_description} target column {target_column} must be numeric, "
            f"got {df[target_column].dtype}"
        )

    # Check for nulls
    null_count = df[target_column].isna().sum()
    if null_count > 0:
        raise DataValidationError(
            f"{data_description} has {null_count} null values in target column {target_column}"
        )

    # Check values
    unique_values = sorted(df[target_column].unique())
    if expected_values is not None:
        unexpected = set(unique_values) - set(expected_values)
        if unexpected:
            raise DataValidationError(
                f"{data_description} target has unexpected values: {unexpected}\n"
                f"Expected: {expected_values}, Got: {unique_values}"
            )

    # Log class distribution
    value_counts = df[target_column].value_counts().to_dict()
    print(f"[OK] Target distribution in {data_description}: {value_counts}")


def validate_prediction_probabilities(
    probabilities: np.ndarray,
    data_description: str = "Predictions"
) -> None:
    """
    Validate prediction probabilities are in valid range.

    Args:
        probabilities: Array of probabilities
        data_description: Description for error messages

    Raises:
        DataValidationError: If validation fails
    """
    if probabilities is None:
        raise DataValidationError(f"{data_description} is None")

    if len(probabilities) == 0:
        raise DataValidationError(f"{data_description} is empty")

    # Check for NaN/Inf
    if np.isnan(probabilities).any():
        nan_count = np.isnan(probabilities).sum()
        raise DataValidationError(
            f"{data_description} contains {nan_count} NaN values"
        )

    if np.isinf(probabilities).any():
        inf_count = np.isinf(probabilities).sum()
        raise DataValidationError(
            f"{data_description} contains {inf_count} infinite values"
        )

    # Check range [0, 1]
    if (probabilities < 0).any() or (probabilities > 1).any():
        out_of_range = ((probabilities < 0) | (probabilities > 1)).sum()
        min_val = probabilities.min()
        max_val = probabilities.max()
        raise DataValidationError(
            f"{data_description} has {out_of_range} values outside [0, 1] range\n"
            f"Min: {min_val}, Max: {max_val}"
        )


def validate_feature_names_match(
    df: pd.DataFrame,
    expected_features: List[str],
    allow_subset: bool = False,
    data_description: str = "DataFrame"
) -> None:
    """
    Validate DataFrame features match expected feature names.

    Args:
        df: DataFrame to validate
        expected_features: List of expected feature names
        allow_subset: If True, allow df to have subset of expected features
        data_description: Description for error messages

    Raises:
        SchemaValidationError: If features don't match
    """
    df_features = set(df.columns)
    expected_set = set(expected_features)

    if allow_subset:
        # Check df is subset of expected
        extra_features = df_features - expected_set
        if extra_features:
            raise SchemaValidationError(
                f"{data_description} has unexpected features: {sorted(extra_features)[:10]}...\n"
                f"Expected features: {sorted(expected_features)[:10]}..."
            )
    else:
        # Check exact match
        missing_features = expected_set - df_features
        extra_features = df_features - expected_set

        if missing_features or extra_features:
            error_msg = f"{data_description} features don't match expected features.\n"
            if missing_features:
                error_msg += f"Missing: {sorted(missing_features)[:10]}...\n"
            if extra_features:
                error_msg += f"Extra: {sorted(extra_features)[:10]}...\n"
            raise SchemaValidationError(error_msg)


def validate_no_constant_features(
    df: pd.DataFrame,
    threshold: float = 0.99,
    data_description: str = "DataFrame"
) -> List[str]:
    """
    Identify features with constant or near-constant values.

    Args:
        df: DataFrame to validate
        threshold: Fraction above which a feature is considered constant
        data_description: Description for warnings

    Returns:
        List of constant feature names

    Warns:
        If constant features are found
    """
    constant_features = []

    for col in df.columns:
        if df[col].dtype in ['object', 'category']:
            # For categorical, check most common value frequency
            most_common_freq = df[col].value_counts(normalize=True).iloc[0]
            if most_common_freq >= threshold:
                constant_features.append(col)
        else:
            # For numeric, check if std is near zero
            if df[col].std() < 1e-10:
                constant_features.append(col)

    if constant_features:
        warnings.warn(
            f"{data_description} has {len(constant_features)} constant/near-constant features: "
            f"{constant_features[:5]}... (showing first 5)"
        )

    return constant_features


def validate_data_quality_summary(
    df: pd.DataFrame,
    data_description: str = "DataFrame"
) -> Dict[str, any]:
    """
    Comprehensive data quality check and summary.

    Args:
        df: DataFrame to validate
        data_description: Description for reporting

    Returns:
        Dictionary with quality metrics
    """
    print(f"\n{'='*80}")
    print(f"DATA QUALITY SUMMARY: {data_description}")
    print(f"{'='*80}")

    quality = {}

    # Shape
    quality['n_rows'] = len(df)
    quality['n_columns'] = len(df.columns)
    print(f"Shape: {df.shape}")

    # Missing values
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    cols_with_missing = (missing > 0).sum()
    quality['columns_with_missing'] = cols_with_missing
    quality['max_missing_pct'] = missing_pct.max()

    print(f"\nMissing Values:")
    print(f"  Columns with missing: {cols_with_missing}/{len(df.columns)}")
    print(f"  Max missing %: {missing_pct.max():.2f}%")

    if cols_with_missing > 0:
        print(f"  Top 5 columns by missing %:")
        top_missing = missing_pct[missing_pct > 0].sort_values(ascending=False).head(5)
        for col, pct in top_missing.items():
            print(f"    - {col}: {pct:.2f}%")

    # Data types
    dtype_counts = df.dtypes.value_counts()
    quality['dtypes'] = dtype_counts.to_dict()
    print(f"\nData Types:")
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")

    # Duplicates
    duplicate_rows = df.duplicated().sum()
    quality['duplicate_rows'] = duplicate_rows
    if duplicate_rows > 0:
        print(f"\n[WARNING] Duplicate rows: {duplicate_rows}")
    else:
        print(f"\n[OK] No duplicate rows")

    # Memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    quality['memory_mb'] = round(memory_mb, 2)
    print(f"\nMemory Usage: {memory_mb:.2f} MB")

    print(f"{'='*80}\n")

    return quality
