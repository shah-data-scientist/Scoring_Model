"""
Feature Engineering Utilities

This module contains functions for:
- Categorical encoding
- Feature scaling
- Column name cleaning
"""

import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple

def encode_categorical_features(train_df: pd.DataFrame, 
                              test_df: pd.DataFrame, 
                              cardinality_limit: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-hot encode categorical features with low cardinality.
    Drops features with cardinality higher than the limit.
    """
    # Identify categorical columns
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    print(f"Found {len(categorical_cols)} categorical columns")

    # One-hot encode low cardinality categoricals
    low_cardinality_cols = [col for col in categorical_cols
                            if train_df[col].nunique() < cardinality_limit]

    print(f"\nOne-hot encoding {len(low_cardinality_cols)} low-cardinality features...")

    # Apply one-hot encoding
    train_df = pd.get_dummies(train_df, columns=low_cardinality_cols, drop_first=True, dtype=int)
    test_df = pd.get_dummies(test_df, columns=low_cardinality_cols, drop_first=True, dtype=int)

    # Align columns (ensure train and test have same columns)
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    # Add missing columns to test
    for col in train_cols - test_cols:
        if col != 'TARGET':
            test_df[col] = 0

    # Remove extra columns from test
    test_df = test_df[[col for col in train_df.columns if col in test_df.columns]]

    # Handle remaining high-cardinality categoricals (drop them)
    remaining_categorical = train_df.select_dtypes(include=['object']).columns.tolist()
    if remaining_categorical:
        print(f"\n[WARNING] {len(remaining_categorical)} high-cardinality features remain: {remaining_categorical}")
        print("Dropping these features as they have too many categories for one-hot encoding...")
        train_df = train_df.drop(columns=remaining_categorical)
        test_df = test_df.drop(columns=[col for col in remaining_categorical if col in test_df.columns])
        print(f"  [OK] Dropped {len(remaining_categorical)} high-cardinality features")
    
    return train_df, test_df

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names to remove special characters (for LightGBM/XGBoost compatibility).
    """
    def _clean_name(col_name):
        # Keep only alphanumeric, underscores, and hyphens
        cleaned = re.sub(r'[^a-zA-Z0-9_-]', '_', str(col_name))
        # Replace multiple underscores with single underscore
        cleaned = re.sub(r'_+', '_', cleaned)
        # Remove leading/trailing underscores
        cleaned = cleaned.strip('_')
        return cleaned

    df.columns = [_clean_name(col) for col in df.columns]
    return df

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scale features using StandardScaler.
    Returns DataFrames with columns and indices preserved.
    """
    print("Scaling features with StandardScaler...")
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    print("[OK] Scaling complete")
    return X_train_scaled, X_test_scaled
