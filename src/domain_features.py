"""
Domain Knowledge Feature Engineering

This module contains functions to create domain-specific features
based on financial knowledge and business logic.
"""

import pandas as pd
import numpy as np

def create_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-based features using financial knowledge.

    Educational Note:
    -----------------
    These features capture important financial relationships:
    - Ratios: Normalize values and capture proportions
    - Flags: Binary indicators of important conditions
    - Transformations: Handle skewed distributions

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    pd.DataFrame
        Dataframe with new domain features
    """
    df = df.copy()
    print("Creating domain features...")

    # 1. AGE FEATURES
    # Convert DAYS_BIRTH to years (more interpretable)
    if 'DAYS_BIRTH' in df.columns:
        df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365
        print("  [OK] Age features created")

    # 2. EMPLOYMENT FEATURES
    if 'DAYS_EMPLOYED' in df.columns:
        # Convert DAYS_EMPLOYED to years (handle anomalies)
        df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)  # Anomaly in data
        df['EMPLOYMENT_YEARS'] = -df['DAYS_EMPLOYED'] / 365
        df['EMPLOYMENT_YEARS'] = df['EMPLOYMENT_YEARS'].clip(lower=0)  # No negative
        df['IS_EMPLOYED'] = (df['EMPLOYMENT_YEARS'] > 0).astype(int)
        print("  [OK] Employment features created")

    # 3. INCOME FEATURES
    if 'AMT_INCOME_TOTAL' in df.columns and 'CNT_FAM_MEMBERS' in df.columns:
        # Income per family member
        df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1e-5)
        print("  [OK] Income features created")

    # 4. CREDIT FEATURES
    if 'AMT_CREDIT' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        # Debt-to-Income Ratio (KEY FEATURE!)
        df['DEBT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1e-5)
    
    if 'AMT_CREDIT' in df.columns and 'AMT_GOODS_PRICE' in df.columns:
        # Credit to goods price ratio
        df['CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1e-5)
        # Credit utilization
        df['CREDIT_UTILIZATION'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1e-5)

    if 'AMT_ANNUITY' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        # Annuity to income ratio (payment burden)
        df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1e-5)

    print("  [OK] Credit features created")

    # 5. FAMILY FEATURES
    if 'CNT_CHILDREN' in df.columns:
        df['HAS_CHILDREN'] = (df['CNT_CHILDREN'] > 0).astype(int)
        if 'CNT_FAM_MEMBERS' in df.columns:
            df['CHILDREN_RATIO'] = df['CNT_CHILDREN'] / (df['CNT_FAM_MEMBERS'] + 1e-5)
        print("  [OK] Family features created")

    # 6. DOCUMENT FLAGS (combine related documents)
    doc_cols = [col for col in df.columns if col.startswith('FLAG_DOCUMENT_')]
    if doc_cols:
        df['TOTAL_DOCUMENTS_PROVIDED'] = df[doc_cols].sum(axis=1)
        print("  [OK] Document features created")

    # 7. EXTERNAL SOURCE FEATURES (if available)
    ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    ext_sources_present = [col for col in ext_sources if col in df.columns]

    if ext_sources_present:
        df['EXT_SOURCE_MEAN'] = df[ext_sources_present].mean(axis=1)
        df['EXT_SOURCE_MAX'] = df[ext_sources_present].max(axis=1)
        df['EXT_SOURCE_MIN'] = df[ext_sources_present].min(axis=1)
        print("  [OK] External source features created")

    # 8. REGIONAL FEATURES
    if 'REGION_RATING_CLIENT' in df.columns and 'REGION_RATING_CLIENT_W_CITY' in df.columns:
        df['REGION_RATING_COMBINED'] = (df['REGION_RATING_CLIENT'] +
                                         df['REGION_RATING_CLIENT_W_CITY']) / 2
        print("  [OK] Regional features created")

    return df
