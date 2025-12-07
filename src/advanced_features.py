"""
Advanced Feature Engineering for Credit Scoring - Target: 0.82 ROC-AUC

This module implements sophisticated features based on:
- Home Credit Default Risk competition winners
- Financial domain expertise
- Feature interactions and transformations

Key Principles:
1. EXT_SOURCE features are CRITICAL - create rich interactions
2. Missing value patterns are informative
3. Nonlinear transformations capture relationships
4. Bureau data aggregations are highly predictive
5. Credit behavior patterns (payment burden, utilization)
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')


def create_advanced_ext_source_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create advanced EXT_SOURCE features.

    EXT_SOURCE_1, 2, 3 are the MOST PREDICTIVE features in credit scoring.
    Winners of Kaggle competitions created 20-30 features from these alone.
    """
    df = df.copy()
    print("  Creating advanced EXT_SOURCE features...")

    ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    available = [col for col in ext_sources if col in df.columns]

    if len(available) >= 2:
        # Pairwise products (interactions)
        for i, source1 in enumerate(available):
            for source2 in available[i+1:]:
                df[f'{source1}_x_{source2}'] = df[source1] * df[source2]

        # Pairwise ratios
        for i, source1 in enumerate(available):
            for source2 in available[i+1:]:
                df[f'{source1}_div_{source2}'] = df[source1] / (df[source2] + 1e-5)

        # Pairwise differences
        for i, source1 in enumerate(available):
            for source2 in available[i+1:]:
                df[f'{source1}_minus_{source2}'] = df[source1] - df[source2]

    if len(available) >= 1:
        # Statistical aggregations
        df['EXT_SOURCE_MEAN'] = df[available].mean(axis=1)
        df['EXT_SOURCE_STD'] = df[available].std(axis=1)
        df['EXT_SOURCE_MAX'] = df[available].max(axis=1)
        df['EXT_SOURCE_MIN'] = df[available].min(axis=1)
        df['EXT_SOURCE_MEDIAN'] = df[available].median(axis=1)
        df['EXT_SOURCE_RANGE'] = df['EXT_SOURCE_MAX'] - df['EXT_SOURCE_MIN']

        # Weighted combinations (based on typical importance)
        if 'EXT_SOURCE_2' in available and 'EXT_SOURCE_3' in available:
            # EXT_SOURCE_2 and 3 are usually most important
            df['EXT_SOURCE_WEIGHTED'] = (0.1 * df.get('EXT_SOURCE_1', 0) +
                                         0.5 * df['EXT_SOURCE_2'] +
                                         0.4 * df['EXT_SOURCE_3'])

        # Squared terms (capture non-linearity)
        for source in available:
            df[f'{source}_SQUARED'] = df[source] ** 2
            df[f'{source}_CUBED'] = df[source] ** 3
            df[f'{source}_SQRT'] = np.sqrt(df[source].clip(lower=0))

    print(f"    Created {len([c for c in df.columns if 'EXT_SOURCE' in c]) - len(available)} new EXT_SOURCE features")
    return df


def create_missing_value_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Missing values are INFORMATIVE in credit data.
    Clients who don't provide certain information often have different risk profiles.
    """
    df = df.copy()
    print("  Creating missing value indicator features...")

    # Key columns where missing values matter
    important_cols = [
        'AMT_GOODS_PRICE', 'AMT_ANNUITY', 'CNT_FAM_MEMBERS',
        'DAYS_EMPLOYED', 'OWN_CAR_AGE', 'EXT_SOURCE_1',
        'EXT_SOURCE_2', 'EXT_SOURCE_3', 'OCCUPATION_TYPE',
        'ORGANIZATION_TYPE'
    ]

    available_cols = [col for col in important_cols if col in df.columns]

    for col in available_cols:
        df[f'{col}_IS_MISSING'] = df[col].isnull().astype(int)

    # Total missing count
    df['TOTAL_MISSING_COUNT'] = df[[f'{col}_IS_MISSING' for col in available_cols
                                     if f'{col}_IS_MISSING' in df.columns]].sum(axis=1)

    # Missing rate
    if len(available_cols) > 0:
        df['MISSING_RATE'] = df['TOTAL_MISSING_COUNT'] / len(available_cols)

    print(f"    Created {len(available_cols) + 2} missing value features")
    return df


def create_time_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Time-based features capture life stage and stability.
    """
    df = df.copy()
    print("  Creating time-based features...")

    # Age groups (different risk profiles)
    if 'DAYS_BIRTH' in df.columns:
        df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365
        df['AGE_GROUP'] = pd.cut(df['AGE_YEARS'],
                                  bins=[0, 25, 35, 50, 65, 100],
                                  labels=[1, 2, 3, 4, 5]).astype(float)
        df['AGE_SQUARED'] = df['AGE_YEARS'] ** 2
        df['AGE_LOG'] = np.log1p(df['AGE_YEARS'])

    # Employment stability
    if 'DAYS_EMPLOYED' in df.columns:
        df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
        df['EMPLOYMENT_YEARS'] = -df['DAYS_EMPLOYED'] / 365
        df['EMPLOYMENT_YEARS'] = df['EMPLOYMENT_YEARS'].clip(lower=0)

        # Employment to age ratio (career stability indicator)
        if 'AGE_YEARS' in df.columns:
            df['EMPLOYMENT_TO_AGE_RATIO'] = df['EMPLOYMENT_YEARS'] / (df['AGE_YEARS'] + 1e-5)
            df['EMPLOYMENT_TO_AGE_RATIO'] = df['EMPLOYMENT_TO_AGE_RATIO'].clip(0, 1)

    # Account age features
    time_cols = ['DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE']
    for col in time_cols:
        if col in df.columns:
            df[f'{col}_YEARS'] = -df[col] / 365
            # Freshness indicator (recently changed = potentially risky)
            df[f'{col}_IS_RECENT'] = (df[col] > -365).astype(int)

    # Car age features
    if 'OWN_CAR_AGE' in df.columns:
        df['CAR_AGE_SQUARED'] = df['OWN_CAR_AGE'] ** 2
        df['HAS_OLD_CAR'] = (df['OWN_CAR_AGE'] > 10).astype(int)
        df['HAS_NEW_CAR'] = (df['OWN_CAR_AGE'] < 3).astype(int)

    print(f"    Created ~{15} time-based features")
    return df


def create_credit_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Advanced credit behavior features - payment burden, efficiency.
    """
    df = df.copy()
    print("  Creating credit behavior features...")

    # Payment burden (annuity to income)
    if 'AMT_ANNUITY' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        df['PAYMENT_BURDEN'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1e-5)
        df['PAYMENT_BURDEN_SQUARED'] = df['PAYMENT_BURDEN'] ** 2

        # High burden flag (>40% of income)
        df['HIGH_PAYMENT_BURDEN'] = (df['PAYMENT_BURDEN'] > 0.4).astype(int)

    # Credit amount patterns
    if 'AMT_CREDIT' in df.columns:
        df['AMT_CREDIT_LOG'] = np.log1p(df['AMT_CREDIT'])
        df['AMT_CREDIT_SQRT'] = np.sqrt(df['AMT_CREDIT'])

        if 'AMT_INCOME_TOTAL' in df.columns:
            df['CREDIT_TO_INCOME'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1e-5)
            df['CREDIT_TO_INCOME_SQUARED'] = df['CREDIT_TO_INCOME'] ** 2

            # High leverage flag (credit > 10x income)
            df['HIGH_LEVERAGE'] = (df['CREDIT_TO_INCOME'] > 10).astype(int)

        if 'AMT_GOODS_PRICE' in df.columns:
            # Down payment proxy
            df['DOWN_PAYMENT'] = df['AMT_GOODS_PRICE'] - df['AMT_CREDIT']
            df['DOWN_PAYMENT_RATE'] = df['DOWN_PAYMENT'] / (df['AMT_GOODS_PRICE'] + 1e-5)
            df['DOWN_PAYMENT_RATE'] = df['DOWN_PAYMENT_RATE'].clip(-1, 1)

            # Overfinancing (credit > goods price)
            df['IS_OVERFINANCED'] = (df['AMT_CREDIT'] > df['AMT_GOODS_PRICE']).astype(int)

    # Income patterns
    if 'AMT_INCOME_TOTAL' in df.columns:
        df['INCOME_LOG'] = np.log1p(df['AMT_INCOME_TOTAL'])
        df['INCOME_SQRT'] = np.sqrt(df['AMT_INCOME_TOTAL'])
        df['INCOME_PER_FAMILY_MEMBER'] = df['AMT_INCOME_TOTAL'] / (df.get('CNT_FAM_MEMBERS', 1) + 1e-5)

        # Income groups
        df['INCOME_GROUP'] = pd.cut(df['AMT_INCOME_TOTAL'],
                                      bins=[0, 100000, 200000, 300000, np.inf],
                                      labels=[1, 2, 3, 4]).astype(float)

    print(f"    Created ~{20} credit behavior features")
    return df


def create_bureau_aggregation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Advanced aggregations from bureau data (if available from feature_aggregation).
    """
    df = df.copy()
    print("  Creating advanced bureau aggregation features...")

    # Bureau credit features (if aggregated data exists)
    bureau_cols = [col for col in df.columns if 'BUREAU_' in col]

    if len(bureau_cols) > 0:
        # Total bureau debt
        debt_cols = [col for col in bureau_cols if 'DEBT' in col or 'SUM' in col]
        if debt_cols:
            df['BUREAU_TOTAL_DEBT'] = df[debt_cols].sum(axis=1)

            if 'AMT_INCOME_TOTAL' in df.columns:
                df['BUREAU_DEBT_TO_INCOME'] = df['BUREAU_TOTAL_DEBT'] / (df['AMT_INCOME_TOTAL'] + 1e-5)

        # Bureau credit activity
        credit_cols = [col for col in bureau_cols if 'AMT_CREDIT' in col]
        if len(credit_cols) > 1:
            df['BUREAU_CREDIT_MEAN'] = df[credit_cols].mean(axis=1)
            df['BUREAU_CREDIT_STD'] = df[credit_cols].std(axis=1)
            df['BUREAU_CREDIT_MAX'] = df[credit_cols].max(axis=1)

        # Bureau time features
        days_cols = [col for col in bureau_cols if 'DAYS_CREDIT' in col]
        if days_cols:
            df['BUREAU_AVG_CREDIT_AGE'] = df[days_cols].mean(axis=1) / -365

        print(f"    Created ~{len(df.columns) - len(bureau_cols) - len(df.columns.difference(bureau_cols))} bureau features")
    else:
        print("    No bureau data found - skipping bureau features")

    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interactions between highly predictive features.
    """
    df = df.copy()
    print("  Creating interaction features...")

    # Key feature pairs for interaction
    interactions = [
        ('EXT_SOURCE_2', 'EXT_SOURCE_3'),
        ('AMT_CREDIT', 'AMT_INCOME_TOTAL'),
        ('AMT_ANNUITY', 'AMT_INCOME_TOTAL'),
        ('AGE_YEARS', 'EMPLOYMENT_YEARS'),
        ('AMT_CREDIT', 'AMT_GOODS_PRICE'),
    ]

    count = 0
    for feat1, feat2 in interactions:
        if feat1 in df.columns and feat2 in df.columns:
            # Product
            df[f'{feat1}_x_{feat2}_INT'] = df[feat1] * df[feat2]
            # Ratio
            df[f'{feat1}_div_{feat2}_INT'] = df[feat1] / (df[feat2] + 1e-5)
            # Sum
            df[f'{feat1}_plus_{feat2}_INT'] = df[feat1] + df[feat2]
            count += 3

    print(f"    Created {count} interaction features")
    return df


def create_categorical_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregations within categorical groups.
    """
    df = df.copy()
    print("  Creating categorical aggregation features...")

    # Income by occupation/organization
    if 'AMT_INCOME_TOTAL' in df.columns:
        for cat_col in ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE', 'NAME_EDUCATION_TYPE']:
            if cat_col in df.columns:
                # Group mean
                group_means = df.groupby(cat_col)['AMT_INCOME_TOTAL'].transform('mean')
                df[f'INCOME_VS_{cat_col}_MEAN'] = df['AMT_INCOME_TOTAL'] / (group_means + 1e-5)

    # Credit amount by contract type
    if 'AMT_CREDIT' in df.columns and 'NAME_CONTRACT_TYPE' in df.columns:
        group_means = df.groupby('NAME_CONTRACT_TYPE')['AMT_CREDIT'].transform('mean')
        df['CREDIT_VS_CONTRACT_MEAN'] = df['AMT_CREDIT'] / (group_means + 1e-5)

    print(f"    Created ~{5} categorical aggregation features")
    return df


def create_all_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ALL advanced features for maximum performance.

    This function orchestrates all advanced feature engineering techniques
    to help reach the target of 0.82 ROC-AUC.
    """
    print("="*80)
    print("CREATING ADVANCED FEATURES - TARGET: 0.82 ROC-AUC")
    print("="*80)

    initial_features = df.shape[1]

    df = create_advanced_ext_source_features(df)
    df = create_missing_value_features(df)
    df = create_time_based_features(df)
    df = create_credit_behavior_features(df)
    df = create_bureau_aggregation_features(df)
    df = create_interaction_features(df)
    df = create_categorical_aggregations(df)

    final_features = df.shape[1]
    new_features = final_features - initial_features

    print("="*80)
    print(f"FEATURE ENGINEERING COMPLETE")
    print(f"  Initial features: {initial_features}")
    print(f"  Final features:   {final_features}")
    print(f"  New features:     {new_features}")
    print("="*80)

    return df


if __name__ == "__main__":
    # Example usage
    print("Advanced Feature Engineering Module")
    print("Target: 0.82 ROC-AUC")
    print("\nKey Techniques:")
    print("1. Advanced EXT_SOURCE interactions (20+ features)")
    print("2. Missing value indicators (informative patterns)")
    print("3. Time-based features (life stage, stability)")
    print("4. Credit behavior (payment burden, efficiency)")
    print("5. Bureau aggregations (credit history)")
    print("6. Feature interactions (capture relationships)")
    print("7. Categorical aggregations (group patterns)")
