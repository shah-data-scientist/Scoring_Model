"""
Model Training Utilities

This module contains functions for training, evaluating, and
logging models with MLflow.
"""

import time
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd

from src.evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_roc_curve,
)

# Plots written here regardless of the notebook's working directory.
_PLOTS_DIR = Path(__file__).resolve().parent.parent / "results" / "plots"


def train_and_evaluate_model(
    model: Any,
    model_name: str,
    params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    log_model_artifact: bool = False,
) -> Tuple[Dict[str, float], Any]:
    """
    Train a model and log everything to MLflow.

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
    log_model_artifact : bool, default False
        Whether to serialize and log the model to the MLflow artifact store.
        Skip for baseline/exploratory runs; set True only for the final model.

    Returns:
    --------
    Tuple[Dict, Any]
        Metrics dictionary and trained model
    """
    print("=" * 80)
    print(f"Training: {model_name}")
    print("=" * 80)

    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(params)
        mlflow.set_tag("model_type", model_name)

        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        mlflow.log_metric("training_time_seconds", training_time)
        print(f"[OK] Training completed in {training_time:.2f} seconds")

        y_pred = model.predict(X_val)
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_val)[:, 1]
        else:
            y_pred_proba = y_pred

        metrics = evaluate_model(y_val, y_pred, y_pred_proba, model_name)

        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(metric_name, value)

        # 1. ROC Curve
        fig = plot_roc_curve(y_val, y_pred_proba, model_name)
        roc_path = str(_PLOTS_DIR / f"{model_name}_roc_curve.png")
        fig.savefig(roc_path, dpi=100, bbox_inches="tight")
        mlflow.log_artifact(roc_path)
        plt.close()

        # 2. Precision-Recall Curve
        fig = plot_precision_recall_curve(y_val, y_pred_proba, model_name)
        pr_path = str(_PLOTS_DIR / f"{model_name}_pr_curve.png")
        fig.savefig(pr_path, dpi=100, bbox_inches="tight")
        mlflow.log_artifact(pr_path)
        plt.close()

        # 3. Confusion Matrix
        fig = plot_confusion_matrix(y_val, y_pred, model_name, normalize=True)
        cm_path = str(_PLOTS_DIR / f"{model_name}_confusion_matrix.png")
        fig.savefig(cm_path, dpi=100, bbox_inches="tight")
        mlflow.log_artifact(cm_path)
        plt.close()

        # 4. Feature Importance (tree-based models only)
        if hasattr(model, "feature_importances_"):
            fig = plot_feature_importance(
                X_train.columns.tolist(),
                model.feature_importances_,
                top_n=20,
                model_name=model_name,
            )
            fi_path = str(
                _PLOTS_DIR / f"{model_name}_feature_importance.png"
            )
            fig.savefig(fi_path, dpi=100, bbox_inches="tight")
            mlflow.log_artifact(fi_path)
            plt.close()
            print("[OK] Feature importance plot saved")

        if log_model_artifact:
            mlflow.sklearn.log_model(model, "model")

        print("[OK] All metrics and artifacts logged to MLflow")
        print(f"Run ID: {run.info.run_id}")

        return metrics, model
