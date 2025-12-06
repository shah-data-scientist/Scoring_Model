"""
Polynomial feature engineering for credit scoring model.

This module creates polynomial features (interactions and squares) for
selected numeric features to capture non-linear relationships.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from typing import List, Tuple


def select_features_for_polynomial(df: pd.DataFrame,
                                   max_features: int = 15) -> List[str]:
    """
    Select the most relevant numeric features for polynomial expansion.

    Args:
        df: DataFrame with features
        max_features: Maximum number of features to select

    Returns:
        List of feature names selected for polynomial expansion
    """
    # Priority features based on domain knowledge
    priority_features = [
        # Financial features
        'AMT_INCOME_TOTAL',
        'AMT_CREDIT',
        'AMT_ANNUITY',
        'AMT_GOODS_PRICE',

        # External scores (highly predictive)
        'EXT_SOURCE_1',
        'EXT_SOURCE_2',
        'EXT_SOURCE_3',

        # Time-based features
        'DAYS_BIRTH',
        'DAYS_EMPLOYED',
        'DAYS_ID_PUBLISH',
        'DAYS_REGISTRATION',

        # Credit bureau features (if available)
        'BUREAU_DAYS_CREDIT_MEAN',
        'BUREAU_DAYS_CREDIT_UPDATE_MEAN',
        'BUREAU_AMT_CREDIT_SUM_MEAN',
        'BUREAU_AMT_CREDIT_SUM_DEBT_MEAN'
    ]

    # Find which priority features exist in the dataframe
    available_features = [f for f in priority_features if f in df.columns]

    # If we don't have enough, add high-variance numeric features
    if len(available_features) < max_features:
        # Get all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Calculate variance for each numeric column
        variances = df[numeric_cols].var().sort_values(ascending=False)

        # Add high-variance features not already in our list
        for col in variances.index:
            if col not in available_features and len(available_features) < max_features:
                available_features.append(col)

    return available_features[:max_features]


def create_polynomial_features(df: pd.DataFrame,
                               feature_list: List[str] = None,
                               degree: int = 2,
                               interaction_only: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create polynomial features for selected numeric columns.

    Args:
        df: Input DataFrame
        feature_list: List of features to expand (None = auto-select)
        degree: Polynomial degree (2 = interactions + squares)
        interaction_only: If True, only interactions (no squares)

    Returns:
        Tuple of (DataFrame with polynomial features, list of new feature names)
    """
    print("Creating polynomial features...")

    # Auto-select features if not provided
    if feature_list is None:
        feature_list = select_features_for_polynomial(df)
        print(f"  Auto-selected {len(feature_list)} features for polynomial expansion")

    # Verify all features exist
    missing_features = [f for f in feature_list if f not in df.columns]
    if missing_features:
        print(f"  Warning: {len(missing_features)} features not found, skipping them")
        feature_list = [f for f in feature_list if f in df.columns]

    if len(feature_list) == 0:
        print("  Warning: No valid features for polynomial expansion!")
        return df, []

    print(f"  Creating polynomial features from {len(feature_list)} base features")
    print(f"  Degree: {degree}, Interaction only: {interaction_only}")

    # Extract selected features
    X_poly = df[feature_list].copy()

    # Handle missing values (polynomial features can't handle NaN)
    # Fill with median for now
    X_poly = X_poly.fillna(X_poly.median())

    # Create polynomial features
    poly = PolynomialFeatures(
        degree=degree,
        interaction_only=interaction_only,
        include_bias=False  # Don't include constant term
    )

    poly_features = poly.fit_transform(X_poly)

    # Get feature names
    poly_feature_names = poly.get_feature_names_out(feature_list)

    # Create DataFrame with polynomial features
    poly_df = pd.DataFrame(
        poly_features,
        columns=poly_feature_names,
        index=df.index
    )

    # Remove original features (they're already in the main df)
    # Only keep the new polynomial features
    original_feature_set = set(feature_list)
    new_poly_features = [col for col in poly_df.columns
                         if col not in original_feature_set]

    poly_df = poly_df[new_poly_features]

    # Rename features to be more readable
    # Replace '^' with '_pow_' and ' ' with '_x_'
    poly_df.columns = [
        col.replace('^', '_pow_').replace(' ', '_x_')
        for col in poly_df.columns
    ]

    # Add POLY_ prefix for clarity
    poly_df.columns = ['POLY_' + col for col in poly_df.columns]

    print(f"  Created {len(poly_df.columns)} new polynomial features")

    # Combine with original dataframe
    result_df = pd.concat([df, poly_df], axis=1)

    return result_df, poly_df.columns.tolist()


def create_interaction_features(df: pd.DataFrame,
                                feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Create specific interaction features between feature pairs.

    Args:
        df: Input DataFrame
        feature_pairs: List of (feature1, feature2) tuples to interact

    Returns:
        DataFrame with interaction features added
    """
    result_df = df.copy()
    new_features = []

    for feat1, feat2 in feature_pairs:
        if feat1 in df.columns and feat2 in df.columns:
            interaction_name = f'INTERACT_{feat1}_x_{feat2}'
            result_df[interaction_name] = df[feat1] * df[feat2]
            new_features.append(interaction_name)

    print(f"Created {len(new_features)} specific interaction features")
    return result_df


# Predefined important interactions for credit scoring
IMPORTANT_INTERACTIONS = [
    ('AMT_INCOME_TOTAL', 'AMT_CREDIT'),
    ('AMT_INCOME_TOTAL', 'AMT_ANNUITY'),
    ('EXT_SOURCE_1', 'EXT_SOURCE_2'),
    ('EXT_SOURCE_2', 'EXT_SOURCE_3'),
    ('DAYS_BIRTH', 'DAYS_EMPLOYED'),
]


if __name__ == "__main__":
    # Example usage
    print("Polynomial Features Module")
    print("=" * 80)
    print("\nExample: Creating polynomial features")

    # Create sample data
    sample_df = pd.DataFrame({
        'AMT_INCOME_TOTAL': np.random.rand(100) * 100000,
        'AMT_CREDIT': np.random.rand(100) * 500000,
        'EXT_SOURCE_1': np.random.rand(100),
        'EXT_SOURCE_2': np.random.rand(100),
        'DAYS_BIRTH': -np.random.rand(100) * 20000
    })

    print(f"Original features: {sample_df.shape[1]}")

    # Create polynomial features
    result_df, new_features = create_polynomial_features(
        sample_df,
        degree=2,
        interaction_only=False
    )

    print(f"After polynomial expansion: {result_df.shape[1]}")
    print(f"New polynomial features: {len(new_features)}")
    print(f"\nSample new features:")
    for feat in new_features[:5]:
        print(f"  - {feat}")
