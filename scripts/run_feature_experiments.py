"""
Comprehensive Feature Engineering Experiment Suite

Systematically tests 12 configurations:
- 4 feature strategies (baseline, domain, polynomial, combined)
- 3 sampling methods (balanced, SMOTE, undersample)

All experiments tracked in MLflow with no duplicates.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, accuracy_score
)
from lightgbm import LGBMClassifier
import mlflow
import mlflow.sklearn
import time
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_preprocessing import load_data
from src.polynomial_features import create_polynomial_features
from src.sampling_strategies import get_sampling_strategy

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# MLflow configuration
EXPERIMENT_NAME = "credit_scoring_feature_engineering"
MLFLOW_TRACKING_URI = "sqlite:///notebooks/mlruns/mlflow.db"

# Base model parameters
BASE_MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': -1
}


def create_domain_features(df):
    """Create domain-knowledge features."""
    df = df.copy()
    print("  Creating domain features...")

    # Age features
    if 'DAYS_BIRTH' in df.columns:
        df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365

    # Employment features
    if 'DAYS_EMPLOYED' in df.columns:
        df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
        df['EMPLOYMENT_YEARS'] = -df['DAYS_EMPLOYED'] / 365
        df['EMPLOYMENT_YEARS'] = df['EMPLOYMENT_YEARS'].clip(lower=0)
        df['IS_EMPLOYED'] = (df['EMPLOYMENT_YEARS'] > 0).astype(int)

    # Income features
    if 'AMT_INCOME_TOTAL' in df.columns and 'CNT_FAM_MEMBERS' in df.columns:
        df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1e-5)

    # Credit features
    if all(col in df.columns for col in ['AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE', 'AMT_ANNUITY']):
        df['DEBT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1e-5)
        df['CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1e-5)
        df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1e-5)
        df['CREDIT_UTILIZATION'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1e-5)

    # Family features
    if 'CNT_CHILDREN' in df.columns:
        df['HAS_CHILDREN'] = (df['CNT_CHILDREN'] > 0).astype(int)
        if 'CNT_FAM_MEMBERS' in df.columns:
            df['CHILDREN_RATIO'] = df['CNT_CHILDREN'] / (df['CNT_FAM_MEMBERS'] + 1e-5)

    # Document flags
    doc_cols = [col for col in df.columns if col.startswith('FLAG_DOCUMENT_')]
    if doc_cols:
        df['TOTAL_DOCUMENTS_PROVIDED'] = df[doc_cols].sum(axis=1)

    # External source features
    ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    ext_sources_present = [col for col in ext_sources if col in df.columns]
    if ext_sources_present:
        df['EXT_SOURCE_MEAN'] = df[ext_sources_present].mean(axis=1)
        df['EXT_SOURCE_MAX'] = df[ext_sources_present].max(axis=1)
        df['EXT_SOURCE_MIN'] = df[ext_sources_present].min(axis=1)

    # Regional features
    if all(col in df.columns for col in ['REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY']):
        df['REGION_RATING_COMBINED'] = (df['REGION_RATING_CLIENT'] +
                                         df['REGION_RATING_CLIENT_W_CITY']) / 2

    print(f"  Added {len(df.columns) - len(df.columns)} domain features")
    return df


def prepare_features(X, y, feature_strategy='baseline'):
    """
    Prepare features according to strategy.

    Args:
        X: Feature DataFrame
        y: Target Series
        feature_strategy: 'baseline', 'domain', 'polynomial', or 'combined'

    Returns:
        Tuple of (X_processed, feature_list)
    """
    print(f"\nPreparing features with strategy: {feature_strategy}")
    X_processed = X.copy()
    feature_list = X.columns.tolist()

    if feature_strategy == 'baseline':
        # No additional features
        print("  Using baseline features only")

    elif feature_strategy == 'domain':
        # Add domain features
        X_processed = create_domain_features(X_processed)
        feature_list = X_processed.columns.tolist()

    elif feature_strategy == 'polynomial':
        # Add polynomial features
        X_processed, poly_features = create_polynomial_features(X_processed, degree=2)
        feature_list = X_processed.columns.tolist()

    elif feature_strategy == 'combined':
        # Add both domain and polynomial
        X_processed = create_domain_features(X_processed)
        X_processed, poly_features = create_polynomial_features(X_processed, degree=2)
        feature_list = X_processed.columns.tolist()

    print(f"  Final feature count: {len(feature_list)}")
    return X_processed, feature_list


def run_single_experiment(X_train, y_train, X_val, y_val,
                         feature_strategy, sampling_strategy,
                         experiment_number):
    """
    Run a single experiment with specific configuration.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        feature_strategy: Feature engineering strategy
        sampling_strategy: Sampling method
        experiment_number: Experiment number (1-12)

    Returns:
        Dict of metrics
    """
    print("="*80)
    print(f"EXPERIMENT {experiment_number}/12")
    print(f"Feature Strategy: {feature_strategy}")
    print(f"Sampling Strategy: {sampling_strategy}")
    print("="*80)

    # Prepare run name
    run_name = f"exp{experiment_number:02d}_lgbm_{feature_strategy}_{sampling_strategy}"

    # Check if run already exists to avoid duplicates
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment:
        existing_runs = client.search_runs(
            [experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_name}'"
        )
        if len(existing_runs) > 0:
            print(f"[SKIP] Run '{run_name}' already exists. Skipping to avoid duplicate.")
            return None

    # Prepare features
    X_train_proc, train_features = prepare_features(X_train, y_train, feature_strategy)
    X_val_proc, _ = prepare_features(X_val, y_val, feature_strategy)

    # Apply sampling strategy
    X_train_sampled, y_train_sampled, sampling_metadata = get_sampling_strategy(
        sampling_strategy, X_train_proc, y_train, random_state=RANDOM_STATE
    )

    # Configure model based on sampling strategy
    model_params = BASE_MODEL_PARAMS.copy()
    if sampling_strategy == 'balanced':
        model_params['class_weight'] = 'balanced'
    # SMOTE and undersample don't need class_weight (data is already balanced)

    # Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        # Log tags
        mlflow.set_tag("feature_strategy", feature_strategy)
        mlflow.set_tag("sampling_strategy", sampling_strategy)
        mlflow.set_tag("experiment_number", experiment_number)
        mlflow.set_tag("model_type", "lgbm")
        mlflow.set_tag("data_version", "v2_comprehensive_318features")

        # Log parameters
        mlflow.log_params(model_params)
        mlflow.log_param("feature_count", len(train_features))
        mlflow.log_param("original_train_samples", len(X_train))
        mlflow.log_param("resampled_train_samples", len(X_train_sampled))

        # Log sampling metadata
        for key, value in sampling_metadata.items():
            mlflow.log_param(f"sampling_{key}", value)

        # Save feature list
        feature_list_path = Path('artifacts/feature_lists')
        feature_list_path.mkdir(exist_ok=True, parents=True)
        feature_file = feature_list_path / f"{run_name}_features.csv"
        pd.DataFrame({'feature': train_features}).to_csv(feature_file, index=False)
        mlflow.log_artifact(str(feature_file))

        # Train model
        print("\nTraining model...")
        start_time = time.time()

        model = LGBMClassifier(**model_params)
        model.fit(X_train_sampled, y_train_sampled)

        training_time = time.time() - start_time
        mlflow.log_metric("training_time_seconds", training_time)
        print(f"  Training completed in {training_time:.2f} seconds")

        # Predictions
        y_pred = model.predict(X_val_proc)
        y_pred_proba = model.predict_proba(X_val_proc)[:, 1]

        # Calculate metrics
        metrics = {
            'roc_auc': roc_auc_score(y_val, y_pred_proba),
            'pr_auc': average_precision_score(y_val, y_pred_proba),
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1_score': f1_score(y_val, y_pred)
        }

        # Calculate business metrics
        tn = ((y_val == 0) & (y_pred == 0)).sum()
        fp = ((y_val == 0) & (y_pred == 1)).sum()
        fn = ((y_val == 1) & (y_pred == 0)).sum()
        tp = ((y_val == 1) & (y_pred == 1)).sum()

        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0

        # Log all metrics
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Print results
        print("\nResults:")
        print(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:     {metrics['pr_auc']:.4f}")
        print(f"  F1-Score:   {metrics['f1_score']:.4f}")
        print(f"  Precision:  {metrics['precision']:.4f}")
        print(f"  Recall:     {metrics['recall']:.4f}")
        print(f"  FPR:        {metrics['false_positive_rate']:.4f}")
        print(f"  FNR:        {metrics['false_negative_rate']:.4f}")

        return metrics


def main():
    """Run all 12 experiments."""
    print("="*80)
    print("COMPREHENSIVE FEATURE ENGINEERING EXPERIMENT SUITE")
    print("="*80)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total experiments: 12")
    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Experiment name: {EXPERIMENT_NAME}")

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load data
    print("\nLoading processed data...")
    data_dir = Path('data/processed')
    X_train = pd.read_csv(data_dir / 'X_train.csv')
    X_val = pd.read_csv(data_dir / 'X_val.csv')
    y_train = pd.read_csv(data_dir / 'y_train.csv').squeeze()
    y_val = pd.read_csv(data_dir / 'y_val.csv').squeeze()

    print(f"  Training: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Baseline features: {X_train.shape[1]}")

    # Define experiment matrix
    feature_strategies = ['baseline', 'domain', 'polynomial', 'combined']
    sampling_strategies = ['balanced', 'smote', 'undersample']

    experiments = []
    exp_num = 1
    for feat_strat in feature_strategies:
        for samp_strat in sampling_strategies:
            experiments.append({
                'number': exp_num,
                'feature_strategy': feat_strat,
                'sampling_strategy': samp_strat
            })
            exp_num += 1

    # Run all experiments
    results = []
    start_time = time.time()

    for exp in experiments:
        try:
            metrics = run_single_experiment(
                X_train, y_train, X_val, y_val,
                exp['feature_strategy'],
                exp['sampling_strategy'],
                exp['number']
            )

            if metrics:  # None if skipped due to duplicate
                results.append({
                    'experiment_number': exp['number'],
                    'feature_strategy': exp['feature_strategy'],
                    'sampling_strategy': exp['sampling_strategy'],
                    **metrics
                })

        except Exception as e:
            print(f"\n[ERROR] Experiment {exp['number']} failed: {e}")
            continue

    total_time = time.time() - start_time

    # Create results summary
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('roc_auc', ascending=False)

        # Save results
        results_path = Path('results/feature_engineering_comparison.csv')
        results_path.parent.mkdir(exist_ok=True, parents=True)
        results_df.to_csv(results_path, index=False)

        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        print(f"\nCompleted {len(results)} experiments in {total_time/60:.2f} minutes")
        print(f"\nTop 5 Configurations:")
        print(results_df[['experiment_number', 'feature_strategy', 'sampling_strategy', 'roc_auc', 'pr_auc', 'f1_score']].head())
        print(f"\nFull results saved to: {results_path}")
        print(f"\nBest configuration:")
        best = results_df.iloc[0]
        print(f"  Feature Strategy: {best['feature_strategy']}")
        print(f"  Sampling Strategy: {best['sampling_strategy']}")
        print(f"  ROC-AUC: {best['roc_auc']:.4f}")
        print("="*80)


if __name__ == "__main__":
    main()
