"""
5-Fold Cross-Validation Experiments - Focused Approach

Tests 4 configurations:
- 2 feature strategies: baseline, domain
- 2 sampling methods: balanced, undersample

Skipping:
- SMOTE (calibration issues)
- Polynomial features (no benefit shown, adds complexity)
- Combined features (redundant with domain)

Goal: Identify best configuration for hyperparameter optimization
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier
import mlflow
import mlflow.sklearn
import time
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.config import MLFLOW_TRACKING_URI, RANDOM_STATE
from src.domain_features import create_domain_features
from src.sampling_strategies import get_sampling_strategy

# Configuration
np.random.seed(RANDOM_STATE)
N_FOLDS = 5

# MLflow setup
EXPERIMENT_NAME = "credit_scoring_feature_engineering_cv"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

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

def prepare_features(X, feature_strategy='baseline'):
    """Prepare features according to strategy."""
    X_processed = X.copy()

    if feature_strategy == 'domain':
        X_processed = create_domain_features(X_processed)

    return X_processed

def run_cv_experiment(X, y, feature_strategy, sampling_strategy, exp_num):
    """Run 5-fold CV for a specific configuration."""
    print("="*80)
    print(f"EXPERIMENT {exp_num}/4: {feature_strategy} + {sampling_strategy}")
    print("="*80)

    run_name = f"exp{exp_num:02d}_cv_{feature_strategy}_{sampling_strategy}"

    # Prepare features
    print("Preparing features...")
    X_proc = prepare_features(X, feature_strategy)
    print(f"Features: {X_proc.shape[1]}")

    # Cross-validation
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_results = []

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("feature_strategy", feature_strategy)
        mlflow.set_tag("sampling_strategy", sampling_strategy)
        mlflow.set_tag("validation", f"{N_FOLDS}-fold CV")
        mlflow.log_param("n_features", X_proc.shape[1])

        print(f"\nRunning {N_FOLDS}-fold CV...")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_proc, y)):
            X_train_fold = X_proc.iloc[train_idx]
            X_val_fold = X_proc.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]

            # Apply sampling strategy
            try:
                X_train_resampled, y_train_resampled, _ = get_sampling_strategy(
                    sampling_strategy, X_train_fold, y_train_fold, random_state=RANDOM_STATE
                )
            except Exception as e:
                print(f"  Fold {fold+1} failed during sampling: {e}")
                continue

            # Train model
            model_params = BASE_MODEL_PARAMS.copy()
            if sampling_strategy == 'balanced':
                model_params['class_weight'] = 'balanced'

            model = LGBMClassifier(**model_params)
            model.fit(X_train_resampled, y_train_resampled)

            # Evaluate on validation fold
            y_pred_proba = model.predict_proba(X_val_fold)[:, 1]

            # Calculate metrics
            roc_auc = roc_auc_score(y_val_fold, y_pred_proba)
            pr_auc = average_precision_score(y_val_fold, y_pred_proba)

            fold_results.append({
                'fold': fold + 1,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc
            })

            print(f"  Fold {fold+1}: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")

        # Aggregate results
        if fold_results:
            df_folds = pd.DataFrame(fold_results)
            mean_roc_auc = df_folds['roc_auc'].mean()
            std_roc_auc = df_folds['roc_auc'].std()
            mean_pr_auc = df_folds['pr_auc'].mean()
            std_pr_auc = df_folds['pr_auc'].std()

            # Log metrics
            mlflow.log_metric("mean_roc_auc", mean_roc_auc)
            mlflow.log_metric("std_roc_auc", std_roc_auc)
            mlflow.log_metric("mean_pr_auc", mean_pr_auc)
            mlflow.log_metric("std_pr_auc", std_pr_auc)
            mlflow.log_metric("roc_auc", mean_roc_auc)  # For sorting

            print(f"\nCV Results:")
            print(f"  ROC-AUC: {mean_roc_auc:.4f} +/- {std_roc_auc:.4f}")
            print(f"  PR-AUC:  {mean_pr_auc:.4f} +/- {std_pr_auc:.4f}")

            return {
                'feature_strategy': feature_strategy,
                'sampling_strategy': sampling_strategy,
                'mean_roc_auc': mean_roc_auc,
                'std_roc_auc': std_roc_auc,
                'mean_pr_auc': mean_pr_auc,
                'std_pr_auc': std_pr_auc
            }
        else:
            print("All folds failed!")
            return None

def main():
    """Run all CV experiments."""
    print("="*80)
    print("5-FOLD CROSS-VALIDATION EXPERIMENTS")
    print("="*80)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Experiments: 4 (2 features x 2 sampling)")
    print(f"MLflow: {MLFLOW_TRACKING_URI}")
    print(f"Experiment: {EXPERIMENT_NAME}")

    # Setup MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    data_dir = Path('data/processed')
    X_train = pd.read_csv(data_dir / 'X_train.csv')
    y_train = pd.read_csv(data_dir / 'y_train.csv').squeeze()

    print(f"Training: {X_train.shape}")
    print(f"Class distribution: {(y_train==0).sum():,} / {(y_train==1).sum():,}")

    # Define experiments
    experiments = [
        {'feature': 'baseline', 'sampling': 'balanced', 'num': 1},
        {'feature': 'baseline', 'sampling': 'undersample', 'num': 2},
        {'feature': 'domain', 'sampling': 'balanced', 'num': 3},
        {'feature': 'domain', 'sampling': 'undersample', 'num': 4},
    ]

    results = []
    for exp in experiments:
        try:
            result = run_cv_experiment(
                X_train, y_train,
                exp['feature'], exp['sampling'], exp['num']
            )
            if result:
                results.append(result)
        except Exception as e:
            print(f"\nExperiment {exp['num']} failed: {e}")
            continue

    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    if results:
        df_results = pd.DataFrame(results).sort_values('mean_roc_auc', ascending=False)
        print("\n", df_results.to_string(index=False))

        # Save results
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        df_results.to_csv(results_dir / 'cv_experiments_summary.csv', index=False)

        # Best configuration
        best = df_results.iloc[0]
        print("\n" + "="*80)
        print("BEST CONFIGURATION")
        print("="*80)
        print(f"Features: {best['feature_strategy']}")
        print(f"Sampling: {best['sampling_strategy']}")
        print(f"ROC-AUC: {best['mean_roc_auc']:.4f} +/- {best['std_roc_auc']:.4f}")
        print(f"\nGap to 0.82 target: {0.82 - best['mean_roc_auc']:.4f}")

        if best['mean_roc_auc'] >= 0.82:
            print("\nSUCCESS! Target achieved with CV!")
        else:
            print("\nNEXT STEP: Hyperparameter optimization on best configuration")
    else:
        print("No results to display - all experiments failed")

    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
