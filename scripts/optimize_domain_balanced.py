"""
Hyperparameter Optimization for Domain + Balanced Configuration

Based on 5-fold CV results:
- Best configuration: domain features + balanced class weights
- CV ROC-AUC: 0.7761 +/- 0.0064
- Target: 0.82 ROC-AUC

This script:
1. Runs RandomizedSearchCV with 100 iterations
2. Uses domain features
3. Uses balanced class weights
4. Logs training and validation metrics (overfitting detection)
5. Aims to reach 0.82 ROC-AUC target
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, make_scorer
from lightgbm import LGBMClassifier
import mlflow
import mlflow.sklearn
import time
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.config import MLFLOW_TRACKING_URI, RANDOM_STATE
from src.domain_features import create_domain_features

# Configuration
np.random.seed(RANDOM_STATE)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

EXPERIMENT_NAME = "credit_scoring_hyperparameter_optimization"
N_ITER = 50  # Hyperparameter combinations to try (reduced to avoid resource issues)
CV_FOLDS = 3  # CV folds for RandomizedSearchCV

# Hyperparameter search space
PARAM_SPACE = {
    'n_estimators': [100, 150, 200, 250, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 7, 8, 10],
    'learning_rate': [0.01, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5, 7, 10, 15],
    'min_child_samples': [10, 20, 30, 50],
    'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
    'reg_lambda': [0, 0.01, 0.1, 0.5, 1.0],
}

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate comprehensive metrics."""
    return {
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'pr_auc': average_precision_score(y_true, y_pred_proba),
    }

def main():
    """Run hyperparameter optimization."""
    print("="*80)
    print("HYPERPARAMETER OPTIMIZATION - DOMAIN + BALANCED")
    print("="*80)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration: domain features + balanced class weights")
    print(f"Search iterations: {N_ITER}")
    print(f"CV folds: {CV_FOLDS}")
    print(f"Target: 0.82 ROC-AUC")

    # Setup MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    data_dir = Path('data/processed')
    X_train_base = pd.read_csv(data_dir / 'X_train.csv')
    X_val_base = pd.read_csv(data_dir / 'X_val.csv')
    y_train = pd.read_csv(data_dir / 'y_train.csv').squeeze()
    y_val = pd.read_csv(data_dir / 'y_val.csv').squeeze()

    # Create domain features
    print("\nCreating domain features...")
    X_train = create_domain_features(X_train_base.copy())
    X_val = create_domain_features(X_val_base.copy())

    print(f"Training: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Features: {X_train.shape[1]}")

    # Run RandomizedSearchCV
    print("\n" + "="*80)
    print("RANDOMIZED SEARCH")
    print("="*80)

    base_model = LGBMClassifier(
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=1,  # Sequential to avoid resource issues
        verbose=-1
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_SPACE,
        n_iter=N_ITER,
        scoring='roc_auc',  # Use string instead of make_scorer
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=2,  # Reduce parallelism to avoid resource exhaustion
        verbose=1,
        return_train_score=True
    )

    with mlflow.start_run(run_name="hyperparam_opt_domain_balanced") as parent_run:
        mlflow.set_tag("feature_strategy", "domain")
        mlflow.set_tag("sampling_strategy", "balanced")
        mlflow.set_tag("optimization_method", "random_search")
        mlflow.log_param("n_iterations", N_ITER)
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("n_features", X_train.shape[1])

        print("\nStarting hyperparameter search...")
        start_time = time.time()

        random_search.fit(X_train, y_train)

        search_time = time.time() - start_time
        mlflow.log_metric("search_time_seconds", search_time)
        print(f"\nSearch completed in {search_time/60:.2f} minutes")

        # Best parameters
        best_params = random_search.best_params_
        best_cv_score = random_search.best_score_

        print("\n" + "="*80)
        print("BEST PARAMETERS")
        print("="*80)
        for param, value in best_params.items():
            print(f"{param:20s}: {value}")
            mlflow.log_param(f"best_{param}", value)

        print(f"\nBest CV ROC-AUC: {best_cv_score:.4f}")
        mlflow.log_metric("best_cv_roc_auc", best_cv_score)

        # Train final model
        print("\n" + "="*80)
        print("FINAL MODEL EVALUATION")
        print("="*80)

        final_model = LGBMClassifier(
            **best_params,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=1,  # Sequential to avoid resource issues
            verbose=-1
        )

        final_model.fit(X_train, y_train)

        # Training metrics
        y_train_pred = final_model.predict(X_train)
        y_train_pred_proba = final_model.predict_proba(X_train)[:, 1]
        train_metrics = calculate_metrics(y_train, y_train_pred, y_train_pred_proba)

        print("\nTraining Metrics:")
        for metric, value in train_metrics.items():
            print(f"  {metric:15s}: {value:.4f}")
            mlflow.log_metric(f"train_{metric}", value)

        # Validation metrics
        y_val_pred = final_model.predict(X_val)
        y_val_pred_proba = final_model.predict_proba(X_val)[:, 1]
        val_metrics = calculate_metrics(y_val, y_val_pred, y_val_pred_proba)

        print("\nValidation Metrics:")
        for metric, value in val_metrics.items():
            print(f"  {metric:15s}: {value:.4f}")
            mlflow.log_metric(f"val_{metric}", value)
            mlflow.log_metric(metric, value)

        # Overfitting analysis
        print("\n" + "="*80)
        print("OVERFITTING ANALYSIS")
        print("="*80)

        roc_gap = train_metrics['roc_auc'] - val_metrics['roc_auc']
        mlflow.log_metric("roc_auc_gap", roc_gap)

        print(f"\nROC-AUC Gap: {roc_gap:.4f}")
        print(f"  Training:   {train_metrics['roc_auc']:.4f}")
        print(f"  Validation: {val_metrics['roc_auc']:.4f}")

        if roc_gap > 0.05:
            status = "OVERFITTING"
            mlflow.set_tag("overfitting_status", "overfitting")
        elif roc_gap < 0.02:
            status = "GOOD FIT"
            mlflow.set_tag("overfitting_status", "good_fit")
        else:
            status = "MINOR OVERFITTING"
            mlflow.set_tag("overfitting_status", "minor_overfitting")

        print(f"  Status: {status}")

        # Target analysis
        print("\n" + "="*80)
        print("TARGET ANALYSIS")
        print("="*80)
        print(f"\nValidation ROC-AUC: {val_metrics['roc_auc']:.4f}")
        print(f"Target ROC-AUC:     0.8200")
        print(f"Gap to target:      {0.82 - val_metrics['roc_auc']:.4f}")

        if val_metrics['roc_auc'] >= 0.82:
            print("\nSUCCESS! 0.82 target achieved!")
            mlflow.set_tag("target_achieved", "yes")
        elif val_metrics['roc_auc'] >= 0.80:
            print("\nCLOSE! Additional tuning may reach 0.82")
            mlflow.set_tag("target_achieved", "close")
        else:
            print("\nNeed further optimization")
            mlflow.set_tag("target_achieved", "no")

        # Save model
        mlflow.sklearn.log_model(final_model, "model")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)

        importance_path = Path('artifacts/feature_importance_optimized_domain.csv')
        importance_path.parent.mkdir(exist_ok=True, parents=True)
        feature_importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(str(importance_path))

        print("\n" + "="*80)
        print("TOP 10 FEATURES")
        print("="*80)
        print("\n", feature_importance.head(10).to_string(index=False))

        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE")
        print("="*80)
        print(f"\nRun ID: {parent_run.info.run_id}")

if __name__ == "__main__":
    main()
