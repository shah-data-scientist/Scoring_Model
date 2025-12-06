"""
Feature aggregation from multiple data sources for Home Credit Default Risk.

This module handles joining and aggregating features from:
- bureau.csv: Credit bureau data
- bureau_balance.csv: Monthly credit bureau balances
- previous_application.csv: Previous loan applications
- POS_CASH_balance.csv: Point of sale and cash loan balances
- credit_card_balance.csv: Credit card balances
- installments_payments.csv: Payment installment history
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple


def aggregate_bureau(bureau_df: pd.DataFrame, bureau_balance_agg: pd.DataFrame = None) -> pd.DataFrame:
    """
    Aggregate credit bureau data by customer.

    Args:
        bureau_df: Credit bureau records
        bureau_balance_agg: Pre-aggregated bureau balance data (by SK_ID_BUREAU)

    Returns:
        DataFrame aggregated by SK_ID_CURR
    """
    print("  Aggregating bureau data...")

    # Merge with pre-aggregated bureau balance if provided
    if bureau_balance_agg is not None and not bureau_balance_agg.empty:
        bureau_df = bureau_df.merge(bureau_balance_agg, on='SK_ID_BUREAU', how='left')

    # Aggregate bureau by customer
    bureau_agg = bureau_df.groupby('SK_ID_CURR').agg({
        # Counts
        'SK_ID_BUREAU': 'count',

        # Days
        'DAYS_CREDIT': ['min', 'max', 'mean'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['min', 'max', 'mean'],

        # Amounts
        'AMT_CREDIT_SUM': ['sum', 'mean', 'max'],
        'AMT_CREDIT_SUM_DEBT': ['sum', 'mean', 'max'],
        'AMT_CREDIT_SUM_LIMIT': ['sum', 'mean', 'max'],
        'AMT_CREDIT_SUM_OVERDUE': ['sum', 'mean', 'max'],
        'AMT_ANNUITY': ['sum', 'mean', 'max'],

        # Overdue
        'CREDIT_DAY_OVERDUE': ['sum', 'mean', 'max'],
        'CNT_CREDIT_PROLONG': ['sum', 'mean', 'max'],
    }).reset_index()

    # Flatten column names
    bureau_agg.columns = ['SK_ID_CURR'] + [
        f'BUREAU_{col[0]}_{col[1].upper()}' if col[1] else f'BUREAU_{col[0]}'
        for col in bureau_agg.columns[1:]
    ]

    # Separately aggregate credit status
    credit_status = bureau_df.groupby('SK_ID_CURR')['CREDIT_ACTIVE'].value_counts().unstack(fill_value=0)
    credit_status.columns = [f'BUREAU_CREDIT_{col.upper()}_COUNT' for col in credit_status.columns]
    credit_status = credit_status.reset_index()

    # Merge credit status
    bureau_agg = bureau_agg.merge(credit_status, on='SK_ID_CURR', how='left')

    # Additional engineered features
    bureau_agg['BUREAU_DEBT_TO_CREDIT_RATIO'] = (
        bureau_agg['BUREAU_AMT_CREDIT_SUM_DEBT_SUM'] /
        (bureau_agg['BUREAU_AMT_CREDIT_SUM_SUM'] + 1)
    )
    bureau_agg['BUREAU_OVERDUE_RATIO'] = (
        bureau_agg['BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM'] /
        (bureau_agg['BUREAU_AMT_CREDIT_SUM_SUM'] + 1)
    )

    print(f"    Created {len(bureau_agg.columns) - 1} bureau features")
    return bureau_agg


def aggregate_previous_applications(prev_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate previous loan applications by customer.

    Args:
        prev_df: Previous applications data

    Returns:
        DataFrame aggregated by SK_ID_CURR
    """
    print("  Aggregating previous applications...")

    # First, aggregate standard features
    prev_agg = prev_df.groupby('SK_ID_CURR').agg({
        # Counts
        'SK_ID_PREV': 'count',

        # Amounts
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean', 'sum'],
        'AMT_CREDIT': ['min', 'max', 'mean', 'sum'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean', 'sum'],

        # Days
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'DAYS_FIRST_DRAWING': ['min', 'max', 'mean'],
        'DAYS_FIRST_DUE': ['min', 'max', 'mean'],
        'DAYS_LAST_DUE': ['min', 'max', 'mean'],
        'DAYS_TERMINATION': ['min', 'max', 'mean'],

        # Rates
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'RATE_INTEREST_PRIMARY': ['min', 'max', 'mean'],
        'RATE_INTEREST_PRIVILEGED': ['min', 'max', 'mean'],

        # Other
        'CNT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'NFLAG_INSURED_ON_APPROVAL': ['mean', 'sum'],
    }).reset_index()

    # Flatten column names
    prev_agg.columns = ['SK_ID_CURR'] + [
        f'PREV_{col[0]}_{col[1].upper()}' if col[1] else f'PREV_{col[0]}'
        for col in prev_agg.columns[1:]
    ]

    # Separately aggregate status counts
    status_counts = prev_df.groupby('SK_ID_CURR')['NAME_CONTRACT_STATUS'].value_counts().unstack(fill_value=0)
    status_counts.columns = [f'PREV_STATUS_{col.upper()}_COUNT' for col in status_counts.columns]
    status_counts = status_counts.reset_index()

    # Merge status counts
    prev_agg = prev_agg.merge(status_counts, on='SK_ID_CURR', how='left')

    # Engineered features
    prev_agg['PREV_APP_CREDIT_RATIO'] = (
        prev_agg['PREV_AMT_APPLICATION_MEAN'] /
        (prev_agg['PREV_AMT_CREDIT_MEAN'] + 1)
    )

    # Calculate approval rate if 'Approved' column exists
    if 'PREV_STATUS_APPROVED_COUNT' in prev_agg.columns:
        prev_agg['PREV_APPROVAL_RATE'] = (
            prev_agg['PREV_STATUS_APPROVED_COUNT'] /
            (prev_agg['PREV_SK_ID_PREV_COUNT'] + 1)
        )

    print(f"    Created {len(prev_agg.columns) - 1} previous application features")
    return prev_agg


def aggregate_pos_cash(pos_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate POS and cash loan balances by customer.

    Args:
        pos_df: POS cash balance data

    Returns:
        DataFrame aggregated by SK_ID_CURR
    """
    print("  Aggregating POS/cash balances...")

    pos_agg = pos_df.groupby('SK_ID_CURR').agg({
        'SK_ID_PREV': 'count',
        'MONTHS_BALANCE': ['min', 'max', 'mean'],
        'CNT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
        'CNT_INSTALMENT_FUTURE': ['min', 'max', 'mean', 'sum'],
        'SK_DPD': ['min', 'max', 'mean', 'sum'],
        'SK_DPD_DEF': ['min', 'max', 'mean', 'sum'],
    }).reset_index()

    # Flatten column names
    pos_agg.columns = ['SK_ID_CURR'] + [
        f'POS_{col[0]}_{col[1].upper()}' if col[1] else f'POS_{col[0]}'
        for col in pos_agg.columns[1:]
    ]

    print(f"    Created {len(pos_agg.columns) - 1} POS/cash features")
    return pos_agg


def aggregate_credit_card(cc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate credit card balances by customer.

    Args:
        cc_df: Credit card balance data

    Returns:
        DataFrame aggregated by SK_ID_CURR
    """
    print("  Aggregating credit card balances...")

    cc_agg = cc_df.groupby('SK_ID_CURR').agg({
        'SK_ID_PREV': 'count',
        'MONTHS_BALANCE': ['min', 'max', 'mean'],
        'AMT_BALANCE': ['min', 'max', 'mean'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['min', 'max', 'mean'],
        'AMT_DRAWINGS_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_DRAWINGS_ATM_CURRENT': ['sum', 'mean'],
        'AMT_DRAWINGS_POS_CURRENT': ['sum', 'mean'],
        'AMT_PAYMENT_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_PAYMENT_TOTAL_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_RECEIVABLE_PRINCIPAL': ['min', 'max', 'mean', 'sum'],
        'AMT_TOTAL_RECEIVABLE': ['min', 'max', 'mean', 'sum'],
        'CNT_DRAWINGS_CURRENT': ['min', 'max', 'mean', 'sum'],
        'CNT_INSTALMENT_MATURE_CUM': ['min', 'max', 'mean', 'sum'],
        'SK_DPD': ['min', 'max', 'mean', 'sum'],
        'SK_DPD_DEF': ['min', 'max', 'mean', 'sum'],
    }).reset_index()

    # Flatten column names
    cc_agg.columns = ['SK_ID_CURR'] + [
        f'CC_{col[0]}_{col[1].upper()}' if col[1] else f'CC_{col[0]}'
        for col in cc_agg.columns[1:]
    ]

    # Engineered features
    cc_agg['CC_BALANCE_TO_LIMIT_RATIO'] = (
        cc_agg['CC_AMT_BALANCE_MEAN'] /
        (cc_agg['CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN'] + 1)
    )
    cc_agg['CC_DRAWING_TO_LIMIT_RATIO'] = (
        cc_agg['CC_AMT_DRAWINGS_CURRENT_MEAN'] /
        (cc_agg['CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN'] + 1)
    )

    print(f"    Created {len(cc_agg.columns) - 1} credit card features")
    return cc_agg


def aggregate_installments(inst_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate installment payments by customer.

    Args:
        inst_df: Installment payments data

    Returns:
        DataFrame aggregated by SK_ID_CURR
    """
    print("  Aggregating installment payments...")

    # Calculate payment difference
    inst_df['PAYMENT_DIFF'] = inst_df['AMT_PAYMENT'] - inst_df['AMT_INSTALMENT']
    inst_df['PAYMENT_RATIO'] = inst_df['AMT_PAYMENT'] / (inst_df['AMT_INSTALMENT'] + 1)
    inst_df['DPD'] = inst_df['DAYS_ENTRY_PAYMENT'] - inst_df['DAYS_INSTALMENT']
    inst_df['IS_LATE'] = (inst_df['DPD'] > 0).astype(int)

    inst_agg = inst_df.groupby('SK_ID_CURR').agg({
        'SK_ID_PREV': 'count',
        'NUM_INSTALMENT_NUMBER': ['min', 'max'],
        'NUM_INSTALMENT_VERSION': 'nunique',
        'DAYS_INSTALMENT': ['min', 'max', 'mean'],
        'DAYS_ENTRY_PAYMENT': ['min', 'max', 'mean'],
        'AMT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'PAYMENT_DIFF': ['min', 'max', 'mean', 'sum'],
        'PAYMENT_RATIO': ['min', 'max', 'mean'],
        'DPD': ['min', 'max', 'mean'],
        'IS_LATE': ['sum', 'mean'],
    }).reset_index()

    # Flatten column names
    inst_agg.columns = ['SK_ID_CURR'] + [
        f'INST_{col[0]}_{col[1].upper()}' if col[1] else f'INST_{col[0]}'
        for col in inst_agg.columns[1:]
    ]

    # Engineered features
    inst_agg['INST_LATE_PAYMENT_RATIO'] = inst_agg['INST_IS_LATE_MEAN']

    print(f"    Created {len(inst_agg.columns) - 1} installment features")
    return inst_agg


def load_and_aggregate_all_data(data_dir: str = 'data') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and aggregate all data sources.

    Args:
        data_dir: Path to data directory

    Returns:
        Tuple of (train_df, test_df) with all aggregated features
    """
    data_path = Path(data_dir)

    print("="*80)
    print("LOADING AND AGGREGATING ALL DATA SOURCES")
    print("="*80)

    # Load main tables
    print("\n1. Loading main application tables...")
    train_df = pd.read_csv(data_path / 'application_train.csv')
    test_df = pd.read_csv(data_path / 'application_test.csv')
    print(f"   Train: {train_df.shape}, Test: {test_df.shape}")

    # Load and aggregate bureau data with chunked processing
    print("\n2. Loading bureau data...")
    bureau_df = pd.read_csv(data_path / 'bureau.csv')
    print(f"   Bureau: {bureau_df.shape}")

    # Process bureau_balance in chunks (27M+ rows - memory intensive)
    print(f"   Processing bureau_balance.csv in chunks...")
    chunk_size = 1_000_000
    chunk_list = []
    chunk_num = 0

    for chunk in pd.read_csv(data_path / 'bureau_balance.csv', chunksize=chunk_size):
        chunk_num += 1
        # Aggregate each chunk by SK_ID_BUREAU
        chunk_agg = chunk.groupby('SK_ID_BUREAU').agg({
            'MONTHS_BALANCE': ['min', 'max', 'size'],
            'STATUS': lambda x: (x == 'C').sum(),
        }).reset_index()
        chunk_agg.columns = ['SK_ID_BUREAU', 'BB_MONTHS_MIN', 'BB_MONTHS_MAX',
                             'BB_COUNT', 'BB_STATUS_CLOSED_COUNT']
        chunk_list.append(chunk_agg)
        print(f"     Processed chunk {chunk_num} ({len(chunk):,} rows)")

    # Combine all chunks and re-aggregate by SK_ID_BUREAU
    print(f"   Combining {len(chunk_list)} chunks...")
    bureau_balance_combined = pd.concat(chunk_list, ignore_index=True)
    bureau_balance_agg = bureau_balance_combined.groupby('SK_ID_BUREAU').agg({
        'BB_MONTHS_MIN': 'min',
        'BB_MONTHS_MAX': 'max',
        'BB_COUNT': 'sum',
        'BB_STATUS_CLOSED_COUNT': 'sum',
    }).reset_index()

    print(f"   Bureau Balance aggregated: {bureau_balance_agg.shape}")
    bureau_agg = aggregate_bureau(bureau_df, bureau_balance_agg)

    # Load and aggregate previous applications
    print("\n3. Loading previous applications...")
    prev_df = pd.read_csv(data_path / 'previous_application.csv')
    print(f"   Previous applications: {prev_df.shape}")
    prev_agg = aggregate_previous_applications(prev_df)

    # Load and aggregate POS cash
    print("\n4. Loading POS/cash balances...")
    pos_df = pd.read_csv(data_path / 'POS_CASH_balance.csv')
    print(f"   POS/cash: {pos_df.shape}")
    pos_agg = aggregate_pos_cash(pos_df)

    # Load and aggregate credit card
    print("\n5. Loading credit card balances...")
    cc_df = pd.read_csv(data_path / 'credit_card_balance.csv')
    print(f"   Credit card: {cc_df.shape}")
    cc_agg = aggregate_credit_card(cc_df)

    # Load and aggregate installments with chunked processing
    print("\n6. Loading installment payments...")
    print(f"   Processing installments_payments.csv in chunks...")
    chunk_size = 2_000_000
    chunk_list = []
    chunk_num = 0

    for chunk in pd.read_csv(data_path / 'installments_payments.csv', chunksize=chunk_size):
        chunk_num += 1
        # Calculate derived features for chunk
        chunk['PAYMENT_DIFF'] = chunk['AMT_PAYMENT'] - chunk['AMT_INSTALMENT']
        chunk['PAYMENT_RATIO'] = chunk['AMT_PAYMENT'] / (chunk['AMT_INSTALMENT'] + 1)
        chunk['DPD'] = chunk['DAYS_ENTRY_PAYMENT'] - chunk['DAYS_INSTALMENT']
        chunk['IS_LATE'] = (chunk['DPD'] > 0).astype(int)

        # Aggregate chunk by customer
        chunk_agg = chunk.groupby('SK_ID_CURR').agg({
            'SK_ID_PREV': 'count',
            'NUM_INSTALMENT_NUMBER': ['min', 'max'],
            'NUM_INSTALMENT_VERSION': 'nunique',
            'DAYS_INSTALMENT': ['min', 'max', 'mean'],
            'DAYS_ENTRY_PAYMENT': ['min', 'max', 'mean'],
            'AMT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
            'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
            'PAYMENT_DIFF': ['min', 'max', 'mean', 'sum'],
            'PAYMENT_RATIO': ['min', 'max', 'mean'],
            'DPD': ['min', 'max', 'mean'],
            'IS_LATE': ['sum', 'mean'],
        }).reset_index()

        chunk_list.append(chunk_agg)
        print(f"     Processed chunk {chunk_num} ({len(chunk):,} rows)")

    # Combine chunks and flatten column names
    print(f"   Combining {len(chunk_list)} chunks...")
    inst_combined = pd.concat(chunk_list, ignore_index=True)

    # Flatten the multi-index columns from the chunks
    inst_combined.columns = ['SK_ID_CURR'] + [
        f'{col[0]}_{col[1]}' if isinstance(col, tuple) else col
        for col in inst_combined.columns[1:]
    ]

    # Re-aggregate by customer (some customers may appear in multiple chunks)
    # For min/max we keep the min/max, for sums we sum, for means we average
    agg_dict = {}
    for col in inst_combined.columns:
        if col == 'SK_ID_CURR':
            continue
        elif 'count' in col or 'sum' in col:
            agg_dict[col] = 'sum'
        elif 'min' in col:
            agg_dict[col] = 'min'
        elif 'max' in col:
            agg_dict[col] = 'max'
        else:  # mean, nunique, etc
            agg_dict[col] = 'mean'

    inst_agg = inst_combined.groupby('SK_ID_CURR').agg(agg_dict).reset_index()

    # Rename columns with INST_ prefix
    inst_agg.columns = ['SK_ID_CURR'] + [
        f'INST_{col.upper()}' for col in inst_agg.columns[1:]
    ]

    # Engineered features
    if 'INST_IS_LATE_MEAN' in inst_agg.columns:
        inst_agg['INST_LATE_PAYMENT_RATIO'] = inst_agg['INST_IS_LATE_MEAN']

    print(f"   Installments aggregated: {inst_agg.shape}")
    print(f"    Created {len(inst_agg.columns) - 1} installment features")

    # Merge all aggregated data
    print("\n7. Merging all aggregated features...")
    train_df = train_df.merge(bureau_agg, on='SK_ID_CURR', how='left')
    train_df = train_df.merge(prev_agg, on='SK_ID_CURR', how='left')
    train_df = train_df.merge(pos_agg, on='SK_ID_CURR', how='left')
    train_df = train_df.merge(cc_agg, on='SK_ID_CURR', how='left')
    train_df = train_df.merge(inst_agg, on='SK_ID_CURR', how='left')

    test_df = test_df.merge(bureau_agg, on='SK_ID_CURR', how='left')
    test_df = test_df.merge(prev_agg, on='SK_ID_CURR', how='left')
    test_df = test_df.merge(pos_agg, on='SK_ID_CURR', how='left')
    test_df = test_df.merge(cc_agg, on='SK_ID_CURR', how='left')
    test_df = test_df.merge(inst_agg, on='SK_ID_CURR', how='left')

    print(f"\n[SUCCESS] Final shapes:")
    print(f"   Train: {train_df.shape}")
    print(f"   Test: {test_df.shape}")
    print(f"   Total features: {train_df.shape[1] - 2} (excluding SK_ID_CURR and TARGET)")
    print("="*80)

    return train_df, test_df
