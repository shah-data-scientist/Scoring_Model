"""
Feature Selection Utilities

This module contains functions to select the most important features
and reduce dimensionality.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

def select_features(X_train: pd.DataFrame, 
                    X_test: pd.DataFrame, 
                    variance_threshold: float = 0.01, 
                    correlation_threshold: float = 0.95) -> pd.DataFrame:
    """
    Perform feature selection using variance threshold and correlation.

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    variance_threshold : float
        Threshold for removing low-variance features
    correlation_threshold : float
        Threshold for removing highly correlated features

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        Selected X_train and X_test
    """
    print(f"Features before selection: {X_train.shape[1]}")

    # 1. Remove low-variance features
    print("\n1. Removing low-variance features...")
    selector = VarianceThreshold(threshold=variance_threshold)
    selector.fit(X_train)

    low_var_features = X_train.columns[~selector.get_support()].tolist()
    print(f"   Found {len(low_var_features)} low-variance features to remove")

    X_train = X_train[X_train.columns[selector.get_support()]]
    X_test = X_test[X_test.columns[selector.get_support()]]

    # 2. Remove highly correlated features
    print(f"\n2. Removing highly correlated features (>{{correlation_threshold}})...")
    corr_matrix = X_train.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    highly_corr_features = [column for column in upper_triangle.columns
                            if any(upper_triangle[column] > correlation_threshold)]
    print(f"   Found {len(highly_corr_features)} highly correlated features to remove")

    X_train = X_train.drop(columns=highly_corr_features)
    X_test = X_test.drop(columns=highly_corr_features)

    print(f"\nFeatures after selection: {X_train.shape[1]}")
    
    return X_train, X_test
