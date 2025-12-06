"""
Script to create processed data with comprehensive features.
This script loads all data sources, performs feature engineering, and saves processed data.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
import gc

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_preprocessing import load_data

# Constants
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def create_domain_features(df):
    """Create domain-based features."""
    df = df.copy()
    print("Creating domain features...")

    # Age features
    df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365

    # Employment features
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    df['EMPLOYMENT_YEARS'] = -df['DAYS_EMPLOYED'] / 365
    df['EMPLOYMENT_YEARS'] = df['EMPLOYMENT_YEARS'].clip(lower=0)
    df['IS_EMPLOYED'] = (df['EMPLOYMENT_YEARS'] > 0).astype(int)

    # Income features
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1e-5)

    # Credit features
    df['DEBT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1e-5)
    df['CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1e-5)
    df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1e-5)
    df['CREDIT_UTILIZATION'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1e-5)

    # Family features
    df['HAS_CHILDREN'] = (df['CNT_CHILDREN'] > 0).astype(int)
    df['CHILDREN_RATIO'] = df['CNT_CHILDREN'] / (df['CNT_FAM_MEMBERS'] + 1e-5)

    # Document flags
    doc_cols = [col for col in df.columns if col.startswith('FLAG_DOCUMENT_')]
    df['TOTAL_DOCUMENTS_PROVIDED'] = df[doc_cols].sum(axis=1)

    # External source features
    ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    ext_sources_present = [col for col in ext_sources if col in df.columns]
    if ext_sources_present:
        df['EXT_SOURCE_MEAN'] = df[ext_sources_present].mean(axis=1)
        df['EXT_SOURCE_MAX'] = df[ext_sources_present].max(axis=1)
        df['EXT_SOURCE_MIN'] = df[ext_sources_present].min(axis=1)

    # Regional features
    df['REGION_RATING_COMBINED'] = (df['REGION_RATING_CLIENT'] +
                                     df['REGION_RATING_CLIENT_W_CITY']) / 2

    print(f"  Created domain features. New shape: {df.shape}")
    return df

def main():
    print("="*80)
    print("CREATING PROCESSED DATA WITH COMPREHENSIVE FEATURES")
    print("="*80)

    # Step 1: Load data
    print("\n1. Loading comprehensive data...")
    train_df, test_df = load_data(data_path='data', use_all_data_sources=True)
    print(f"   Train: {train_df.shape}, Test: {test_df.shape}")

    # Step 2: Drop high missing columns
    print("\n2. Handling missing values...")
    missing_pct = (train_df.isnull().sum() / len(train_df)) * 100
    high_missing = missing_pct[missing_pct > 70].index.tolist()
    print(f"   Dropping {len(high_missing)} columns with >70% missing")
    train_df = train_df.drop(columns=high_missing)
    test_df = test_df.drop(columns=[col for col in high_missing if col in test_df.columns])

    # Step 3: Create domain features
    print("\n3. Creating domain features...")
    train_df = create_domain_features(train_df)
    test_df = create_domain_features(test_df)

    # Step 4: Encode categorical variables
    print("\n4. Encoding categorical variables...")
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    low_cardinality_cols = [col for col in categorical_cols
                            if train_df[col].nunique() < 10]
    print(f"   One-hot encoding {len(low_cardinality_cols)} low-cardinality features")

    train_df = pd.get_dummies(train_df, columns=low_cardinality_cols, drop_first=True, dtype=int)
    test_df = pd.get_dummies(test_df, columns=low_cardinality_cols, drop_first=True, dtype=int)

    # Align columns
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    for col in train_cols - test_cols:
        if col != 'TARGET':
            test_df[col] = 0
    test_df = test_df[[col for col in train_df.columns if col in test_df.columns]]

    # Drop remaining categorical
    remaining_categorical = train_df.select_dtypes(include=['object']).columns.tolist()
    if remaining_categorical:
        print(f"   Dropping {len(remaining_categorical)} high-cardinality features")
        train_df = train_df.drop(columns=remaining_categorical)
        test_df = test_df.drop(columns=[col for col in remaining_categorical if col in test_df.columns])

    # Clean column names
    import re
    def clean_column_name(col_name):
        cleaned = re.sub(r'[^a-zA-Z0-9_-]', '_', str(col_name))
        cleaned = re.sub(r'_+', '_', cleaned)
        return cleaned.strip('_')

    train_df.columns = [clean_column_name(col) for col in train_df.columns]
    test_df.columns = [clean_column_name(col) for col in test_df.columns]

    # Step 5: Impute missing values
    print("\n5. Imputing missing values...")
    X_train = train_df.drop(columns=['SK_ID_CURR', 'TARGET'])
    y_train = train_df['TARGET']
    X_test = test_df.drop(columns=['SK_ID_CURR'])

    missing_cols = X_train.columns[X_train.isnull().any()].tolist()
    if missing_cols:
        print(f"   Imputing {len(missing_cols)} columns with median")
        imputer = SimpleImputer(strategy='median')
        X_train[missing_cols] = imputer.fit_transform(X_train[missing_cols])
        X_test[missing_cols] = imputer.transform(X_test[missing_cols])

    # Step 6: Feature selection
    print("\n6. Performing feature selection...")
    print(f"   Features before selection: {X_train.shape[1]}")

    # Remove low variance
    selector = VarianceThreshold(threshold=0.01)
    selector.fit(X_train)
    X_train = X_train[X_train.columns[selector.get_support()]]
    X_test = X_test[X_test.columns[selector.get_support()]]

    # Remove highly correlated
    corr_matrix = X_train.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    highly_corr = [column for column in upper_triangle.columns
                   if any(upper_triangle[column] > 0.95)]
    print(f"   Removing {len(highly_corr)} highly correlated features")
    X_train = X_train.drop(columns=highly_corr)
    X_test = X_test.drop(columns=highly_corr)

    print(f"   Features after selection: {X_train.shape[1]}")

    # Step 7: Scale features
    print("\n7. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    # Step 8: Train-validation split
    print("\n8. Creating train-validation split...")
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_scaled, y_train,
        test_size=0.3,
        stratify=y_train,
        random_state=RANDOM_STATE
    )

    print(f"   Training: {X_train_split.shape}")
    print(f"   Validation: {X_val_split.shape}")
    print(f"   Test: {X_test_scaled.shape}")

    # Step 9: Save processed data
    print("\n9. Saving processed data...")
    processed_dir = Path('data/processed')
    processed_dir.mkdir(exist_ok=True, parents=True)

    X_train_split.to_csv(processed_dir / 'X_train.csv', index=False)
    X_val_split.to_csv(processed_dir / 'X_val.csv', index=False)
    X_test_scaled.to_csv(processed_dir / 'X_test.csv', index=False)

    y_train_split.to_csv(processed_dir / 'y_train.csv', index=False, header=True)
    y_val_split.to_csv(processed_dir / 'y_val.csv', index=False, header=True)

    pd.DataFrame({'feature': X_train_split.columns}).to_csv(
        processed_dir / 'feature_names.csv', index=False
    )

    train_df[['SK_ID_CURR']].iloc[X_train_split.index].to_csv(
        processed_dir / 'train_ids.csv', index=False
    )
    train_df[['SK_ID_CURR']].iloc[X_val_split.index].to_csv(
        processed_dir / 'val_ids.csv', index=False
    )
    test_df[['SK_ID_CURR']].to_csv(processed_dir / 'test_ids.csv', index=False)

    print("\n" + "="*80)
    print("SUCCESS! Processed data created")
    print("="*80)
    print(f"\nFinal dataset:")
    print(f"  - Training samples: {X_train_split.shape[0]:,}")
    print(f"  - Validation samples: {X_val_split.shape[0]:,}")
    print(f"  - Test samples: {X_test_scaled.shape[0]:,}")
    print(f"  - Features: {X_train_split.shape[1]}")
    print(f"  - Files saved to: {processed_dir}")

    print(f"\nClass distribution:")
    print(f"  - Training: {y_train_split.value_counts(normalize=True).to_dict()}")
    print(f"  - Validation: {y_val_split.value_counts(normalize=True).to_dict()}")

if __name__ == "__main__":
    main()
