"""
Model Training Utilities

This module contains functions for training, evaluating, and logging models with MLflow.
"""

import time
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple

from src.evaluation import (
    evaluate_model,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_feature_importance
)

def train_and_evaluate_model(model: Any, 
                           model_name: str, 
                           params: Dict[str, Any], 
                           X_train: pd.DataFrame, 
                           y_train: pd.Series, 
                           X_val: pd.DataFrame, 
                           y_val: pd.Series) -> Tuple[Dict[str, float], Any]:
    """
    Train a model and log everything to MLflow.

    Educational Note:
    -----------------
    This function demonstrates professional ML workflow:
    1. Track training time
    2. Make predictions
    3. Evaluate with multiple metrics
    4. Log parameters, metrics, model, and artifacts
    5. Return results for comparison

    Parameters:
    -----------
    model : estimator
        The initialized model (sklearn interface)
    model_name : str
        Name of the model for logging
    params : dict
        Hyperparameters used
    X_train, y_train : 
        Training data
    X_val, y_val : 
        Validation data

    Returns:
    --------
    Tuple[Dict, Any]
        Metrics dictionary and trained model
    """
    print("="*80)
    print(f"Training: {model_name}")
    print("="*80)

    # Create plots directory if it doesn't exist
    Path('plots').mkdir(exist_ok=True)

    with mlflow.start_run(run_name=model_name) as run:
        # Log parameters
        mlflow.log_params(params)
        mlflow.set_tag("model_type", model_name)

        # Train model
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        mlflow.log_metric("training_time_seconds", training_time)
        print(f"[OK] Training completed in {training_time:.2f} seconds")

        # Predictions
        y_pred = model.predict(X_val)
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_val)[:, 1]
        else:
            # For models without predict_proba (e.g. some SVMs), use decision function or just 0/1
            y_pred_proba = y_pred 

        # Evaluate
        metrics = evaluate_model(y_val, y_pred, y_pred_proba, model_name)

        # Log all metrics to MLflow
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(metric_name, value)

        # Create and log visualizations
        # 1. ROC Curve
        fig = plot_roc_curve(y_val, y_pred_proba, model_name)
        roc_path = f"plots/{model_name}_roc_curve.png"
        fig.savefig(roc_path, dpi=100, bbox_inches='tight')
        mlflow.log_artifact(roc_path)
        plt.close()

        # 2. Precision-Recall Curve
        fig = plot_precision_recall_curve(y_val, y_pred_proba, model_name)
        pr_path = f"plots/{model_name}_pr_curve.png"
        fig.savefig(pr_path, dpi=100, bbox_inches='tight')
        mlflow.log_artifact(pr_path)
        plt.close()

        # 3. Confusion Matrix
        fig = plot_confusion_matrix(y_val, y_pred, model_name, normalize=True)
        cm_path = f"plots/{model_name}_confusion_matrix.png"
        fig.savefig(cm_path, dpi=100, bbox_inches='tight')
        mlflow.log_artifact(cm_path)
        plt.close()

        # 4. Feature Importance (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            fig = plot_feature_importance(
                X_train.columns.tolist(),
                model.feature_importances_,
                top_n=20,
                model_name=model_name
            )
            fi_path = f"plots/{model_name}_feature_importance.png"
            fig.savefig(fi_path, dpi=100, bbox_inches='tight')
            mlflow.log_artifact(fi_path)
            plt.close()
            print(f"[OK] Feature importance plot saved")

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"[OK] All metrics and artifacts logged to MLflow")
        print(f"Run ID: {run.info.run_id}")

        return metrics, model
