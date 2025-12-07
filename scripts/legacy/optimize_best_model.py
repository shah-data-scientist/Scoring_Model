"""
Hyperparameter Optimization for Best Model Configuration

Based on feature engineering experiments:
- Best configuration: baseline features + balanced class weights
- ROC-AUC: 0.7783

This script:
1. Runs RandomizedSearchCV with 50 iterations
2. Logs BOTH training and validation metrics (to detect overfitting)
3. Tracks all trials in MLflow
4. Identifies best hyperparameters
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, make_scorer
)
from lightgbm import LGBMClassifier
import mlflow
import mlflow.sklearn
import time
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import MLFLOW_TRACKING_URI, RANDOM_STATE

# Configuration
np.random.seed(RANDOM_STATE)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

EXPERIMENT_NAME = "credit_scoring_hyperparameter_optimization"
N_ITER = 50  # Number of hyperparameter combinations to try
CV_FOLDS = 3  # Cross-validation folds

# Hyperparameter search space
PARAM_SPACE = {
    'n_estimators': [100, 150, 200, 250, 300],
    'max_depth': [3, 5, 7, 10, 15],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5, 7, 10],
    'reg_alpha': [0, 0.1, 0.5, 1.0, 2.0],
    'reg_lambda': [0, 0.1, 0.5, 1.0, 2.0],
}

def calculate_all_metrics(y_true, y_pred, y_pred_proba):
    """Calculate comprehensive metrics."""
    metrics = {
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'pr_auc': average_precision_score(y_true, y_pred_proba),
        'accuracy': (y_pred == y_true).mean(),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }

    # Confusion matrix metrics
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tp = ((y_true == 1) & (y_pred == 1)).sum()

    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0

    return metrics

def main():
    """Run hyperparameter optimization with training metrics logging."""
    print("="*80)
    print("HYPERPARAMETER OPTIMIZATION - BEST MODEL CONFIGURATION")
    print("="*80)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration: baseline features + balanced class weights")
    print(f"Search iterations: {N_ITER}")
    print(f"Cross-validation folds: {CV_FOLDS}")
    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Experiment name: {EXPERIMENT_NAME}")

    # Setup MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    data_dir = Path('data/processed')
    X_train = pd.read_csv(data_dir / 'X_train.csv')
    X_val = pd.read_csv(data_dir / 'X_val.csv')
    y_train = pd.read_csv(data_dir / 'y_train.csv').squeeze()
    y_val = pd.read_csv(data_dir / 'y_val.csv').squeeze()

    print(f"\nTraining: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Class distribution (train): {(y_train==0).sum():,} / {(y_train==1).sum():,}")

    # Run RandomizedSearchCV
    print("\n" + "="*80)
    print("RUNNING RANDOMIZED SEARCH")
    print("="*80)

    # Base model with balanced class weight
    base_model = LGBMClassifier(
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )

    # Define scoring metric
    scorer = make_scorer(roc_auc_score, needs_proba=True)

    # Cross-validation strategy
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_SPACE,
        n_iter=N_ITER,
        scoring=scorer,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=2,
        return_train_score=True  # IMPORTANT: Get training scores for overfitting detection
    )

    # Start parent run
    with mlflow.start_run(run_name="hyperparameter_optimization_baseline_balanced") as parent_run:
        # Log parent run info
        mlflow.set_tag("feature_strategy", "baseline")
        mlflow.set_tag("sampling_strategy", "balanced")
        mlflow.set_tag("model_type", "lgbm")
        mlflow.set_tag("optimization_method", "random_search")
        mlflow.set_tag("data_version", "v2_comprehensive_318features")

        mlflow.log_param("n_iterations", N_ITER)
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("feature_count", X_train.shape[1])

        # Run search
        print("\nStarting hyperparameter search...")
        start_time = time.time()

        random_search.fit(X_train, y_train)

        search_time = time.time() - start_time
        mlflow.log_metric("search_time_seconds", search_time)
        print(f"\nSearch completed in {search_time/60:.2f} minutes")

        # Get best parameters
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

        # Train final model with best parameters on full training set
        print("\n" + "="*80)
        print("TRAINING FINAL MODEL")
        print("="*80)

        final_model = LGBMClassifier(
            **best_params,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )

        final_model.fit(X_train, y_train)

        # TRAINING METRICS (to detect overfitting)
        y_train_pred = final_model.predict(X_train)
        y_train_pred_proba = final_model.predict_proba(X_train)[:, 1]
        train_metrics = calculate_all_metrics(y_train, y_train_pred, y_train_pred_proba)

        print("\nTraining Metrics:")
        for metric, value in train_metrics.items():
            print(f"  {metric:25s}: {value:.4f}")
            mlflow.log_metric(f"train_{metric}", value)

        # VALIDATION METRICS
        y_val_pred = final_model.predict(X_val)
        y_val_pred_proba = final_model.predict_proba(X_val)[:, 1]
        val_metrics = calculate_all_metrics(y_val, y_val_pred, y_val_pred_proba)

        print("\nValidation Metrics:")
        for metric, value in val_metrics.items():
            print(f"  {metric:25s}: {value:.4f}")
            mlflow.log_metric(f"val_{metric}", value)
            mlflow.log_metric(metric, value)  # Also log without prefix for compatibility

        # OVERFITTING DETECTION
        print("\n" + "="*80)
        print("OVERFITTING ANALYSIS")
        print("="*80)

        roc_gap = train_metrics['roc_auc'] - val_metrics['roc_auc']
        mlflow.log_metric("roc_auc_gap", roc_gap)

        print(f"\nROC-AUC:")
        print(f"  Training:   {train_metrics['roc_auc']:.4f}")
        print(f"  Validation: {val_metrics['roc_auc']:.4f}")
        print(f"  Gap:        {roc_gap:.4f}")

        if roc_gap > 0.05:
            status = "OVERFITTING DETECTED"
            print(f"\n  Status: {status}")
            print(f"  Recommendation: Add more regularization or reduce model complexity")
            mlflow.set_tag("overfitting_status", "overfitting")
        elif roc_gap < 0.02:
            status = "GOOD FIT"
            print(f"\n  Status: {status}")
            mlflow.set_tag("overfitting_status", "good_fit")
        else:
            status = "MINOR OVERFITTING"
            print(f"\n  Status: {status}")
            print(f"  Acceptable level of overfitting")
            mlflow.set_tag("overfitting_status", "minor_overfitting")

        # Log model
        mlflow.sklearn.log_model(final_model, "model")

        # Log feature importances
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)

        feature_importance_path = Path('artifacts/feature_importance_optimized.csv')
        feature_importance_path.parent.mkdir(exist_ok=True, parents=True)
        feature_importance.to_csv(feature_importance_path, index=False)
        mlflow.log_artifact(str(feature_importance_path))

        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE")
        print("="*80)
        print(f"\nFinal Validation ROC-AUC: {val_metrics['roc_auc']:.4f}")
        print(f"Overfitting Status: {status}")
        print(f"\nMLflow Run ID: {parent_run.info.run_id}")
        print(f"View in UI: http://localhost:5000/#/experiments/{parent_run.info.experiment_id}/runs/{parent_run.info.run_id}")

if __name__ == "__main__":
    main()
