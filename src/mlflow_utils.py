"""
MLflow utility functions for the Credit Scoring Model project.

This module provides helper functions for standardized MLflow tracking,
experiment management, and model registry operations.
"""
import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime

from .config import (
    MLFLOW_TRACKING_URI,
    EXPERIMENTS,
    REGISTERED_MODELS,
    get_baseline_tags,
    get_optimization_tags,
    get_production_tags,
    get_artifact_path
)


# ============================================================================
# MLflow Setup
# ============================================================================

def setup_mlflow(experiment_key: str = "baseline"):
    """
    Setup MLflow with standardized configuration.

    Args:
        experiment_key: Key from EXPERIMENTS dict ('baseline', 'optimization', etc.)

    Returns:
        Experiment object

    Example:
        >>> exp = setup_mlflow('baseline')
        >>> print(f"Experiment: {exp.name}")
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    experiment_name = EXPERIMENTS.get(experiment_key, experiment_key)
    mlflow.set_experiment(experiment_name)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    print(f"[MLflow] Experiment: {experiment_name}")
    print(f"[MLflow] Experiment ID: {experiment.experiment_id}")
    print(f"[MLflow] Tracking URI: {MLFLOW_TRACKING_URI}")

    return experiment


# ============================================================================
# Run Management
# ============================================================================

def start_run_with_tags(run_name: str, tags: Dict[str, str], **kwargs):
    """
    Start MLflow run with standardized tags.

    Args:
        run_name: Name of the run
        tags: Dictionary of tags
        **kwargs: Additional arguments for mlflow.start_run()

    Returns:
        MLflow run context

    Example:
        >>> tags = get_baseline_tags('lgbm', author='john_doe')
        >>> with start_run_with_tags('lgbm_v1_baseline', tags) as run:
        ...     # Training code here
    """
    run = mlflow.start_run(run_name=run_name, **kwargs)

    # Set all tags
    for key, value in tags.items():
        mlflow.set_tag(key, value)

    return run


def log_model_with_signature(model, model_name: str, X_sample: pd.DataFrame):
    """
    Log model with input/output signature for better tracking.

    Args:
        model: Trained model
        model_name: Name/path for the model artifact
        X_sample: Sample of input data for signature inference

    Example:
        >>> log_model_with_signature(model, "model", X_train.head())
    """
    from mlflow.models.signature import infer_signature

    # Infer signature
    signature = infer_signature(X_sample, model.predict(X_sample))

    # Log model with signature
    mlflow.sklearn.log_model(
        model,
        model_name,
        signature=signature,
        input_example=X_sample.iloc[:5]
    )


def log_metrics_with_prefix(metrics: Dict[str, float], prefix: str = ""):
    """
    Log multiple metrics with optional prefix.

    Args:
        metrics: Dictionary of metric names and values
        prefix: Prefix to add to all metric names (e.g., 'train_', 'val_')

    Example:
        >>> metrics = {'roc_auc': 0.78, 'f1': 0.29}
        >>> log_metrics_with_prefix(metrics, prefix='val_')
        # Logs: val_roc_auc, val_f1
    """
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            metric_name = f"{prefix}{name}" if prefix else name
            mlflow.log_metric(metric_name, value)


def log_params_clean(params: Dict[str, Any]):
    """
    Log parameters, handling non-serializable types.

    Args:
        params: Dictionary of parameters

    Example:
        >>> log_params_clean({'max_depth': 6, 'class_weight': 'balanced'})
    """
    clean_params = {}
    for key, value in params.items():
        # Convert non-serializable types
        if value is None:
            clean_params[key] = "None"
        elif isinstance(value, (list, tuple)):
            clean_params[key] = str(value)
        elif isinstance(value, dict):
            clean_params[key] = str(value)
        else:
            clean_params[key] = value

    mlflow.log_params(clean_params)


# ============================================================================
# Artifact Management
# ============================================================================

def log_plot_artifact(fig, model_name: str, plot_type: str):
    """
    Save and log a matplotlib figure as an artifact.

    Args:
        fig: Matplotlib figure
        model_name: Name of the model (for filename)
        plot_type: Type of plot ('roc_curve', 'feature_importance', etc.)

    Example:
        >>> fig = plot_roc_curve(y_val, y_pred_proba, 'lgbm_v1')
        >>> log_plot_artifact(fig, 'lgbm_v1_baseline', 'roc_curve')
    """
    # Create plots directory if it doesn't exist
    Path('plots').mkdir(exist_ok=True)

    # Get standardized path
    artifact_path = get_artifact_path(model_name, plot_type)

    # Save figure
    fig.savefig(artifact_path, dpi=150, bbox_inches='tight')

    # Log to MLflow
    mlflow.log_artifact(artifact_path)

    # Close figure to free memory
    import matplotlib.pyplot as plt
    plt.close(fig)


def log_dataframe_artifact(df: pd.DataFrame, model_name: str, artifact_type: str):
    """
    Save and log a pandas DataFrame as an artifact.

    Args:
        df: DataFrame to save
        model_name: Name of the model (for filename)
        artifact_type: Type of artifact ('predictions', 'metrics', etc.)

    Example:
        >>> predictions = pd.DataFrame({'y_true': y_val, 'y_pred': y_pred})
        >>> log_dataframe_artifact(predictions, 'lgbm_v1_baseline', 'predictions')
    """
    # Create data directory if it doesn't exist
    Path('data').mkdir(exist_ok=True)

    # Get standardized path
    artifact_path = get_artifact_path(model_name, artifact_type, extension='csv')

    # Save DataFrame
    df.to_csv(artifact_path, index=False)

    # Log to MLflow
    mlflow.log_artifact(artifact_path)


# ============================================================================
# Model Registry
# ============================================================================

def register_model(run_id: str, model_type: str, description: str = "",
                   stage: str = "None") -> Any:
    """
    Register a model in MLflow Model Registry.

    Args:
        run_id: MLflow run ID
        model_type: Type of model ('lgbm', 'xgboost', etc.)
        description: Model description
        stage: Initial stage ('None', 'Staging', 'Production', 'Archived')

    Returns:
        ModelVersion object

    Example:
        >>> run_id = mlflow.active_run().info.run_id
        >>> register_model(run_id, 'lgbm', 'Optimized with Random Search', 'Staging')
    """
    # Get standardized model name
    model_name = REGISTERED_MODELS.get(model_type, f"credit_scoring_{model_type}")

    # Register model
    model_uri = f"runs:/{run_id}/model"
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )

    # Update description
    client = mlflow.tracking.MlflowClient()
    client.update_model_version(
        name=model_name,
        version=model_version.version,
        description=description
    )

    # Set stage
    if stage != "None":
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage=stage
        )

    print(f"[Model Registry] Registered: {model_name} (v{model_version.version})")
    print(f"[Model Registry] Stage: {stage}")

    return model_version


def promote_model_to_production(model_type: str, version: int,
                                archive_existing: bool = True):
    """
    Promote a model version to Production stage.

    Args:
        model_type: Type of model ('lgbm', 'xgboost', etc.)
        version: Version number to promote
        archive_existing: Whether to archive existing production versions

    Example:
        >>> promote_model_to_production('lgbm', version=3, archive_existing=True)
    """
    model_name = REGISTERED_MODELS.get(model_type, f"credit_scoring_{model_type}")
    client = mlflow.tracking.MlflowClient()

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=archive_existing
    )

    print(f"[Model Registry] Promoted {model_name} v{version} to Production")


def get_production_model(model_type: str):
    """
    Load the current production model.

    Args:
        model_type: Type of model ('lgbm', 'xgboost', etc.)

    Returns:
        Loaded model

    Example:
        >>> model = get_production_model('lgbm')
    """
    model_name = REGISTERED_MODELS.get(model_type, f"credit_scoring_{model_type}")
    model_uri = f"models:/{model_name}/Production"

    try:
        model = mlflow.sklearn.load_model(model_uri)
        print(f"[Model Registry] Loaded production model: {model_name}")
        return model
    except Exception as e:
        print(f"[Model Registry] Error loading production model: {e}")
        return None


# ============================================================================
# Experiment Analysis
# ============================================================================

def get_best_run(experiment_name: str, metric: str = "roc_auc",
                ascending: bool = False) -> Optional[Any]:
    """
    Get the best run from an experiment based on a metric.

    Args:
        experiment_name: Name of the experiment
        metric: Metric to sort by
        ascending: Sort order (False = descending, higher is better)

    Returns:
        Best run object or None

    Example:
        >>> best_run = get_best_run('credit_scoring_01_baseline', 'roc_auc')
        >>> print(f"Best ROC-AUC: {best_run.data.metrics['roc_auc']:.4f}")
    """
    client = mlflow.tracking.MlflowClient()

    # Get experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"[Warning] Experiment '{experiment_name}' not found")
        return None

    # Get all runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"]
    )

    if not runs:
        print(f"[Warning] No runs found in experiment '{experiment_name}'")
        return None

    return runs[0]


def compare_experiment_runs(experiment_name: str,
                           metrics: List[str] = None) -> pd.DataFrame:
    """
    Get a comparison DataFrame of all runs in an experiment.

    Args:
        experiment_name: Name of the experiment
        metrics: List of metrics to include (None = all metrics)

    Returns:
        DataFrame with run comparisons

    Example:
        >>> df = compare_experiment_runs('credit_scoring_01_baseline',
        ...                             metrics=['roc_auc', 'f1_score'])
    """
    client = mlflow.tracking.MlflowClient()

    # Get experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        return pd.DataFrame()

    # Get all runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"]
    )

    # Extract data
    data = []
    for run in runs:
        row = {
            'run_name': run.data.tags.get('mlflow.runName', 'Unknown'),
            'run_id': run.info.run_id,
            'status': run.info.status,
            'start_time': datetime.fromtimestamp(run.info.start_time / 1000)
        }

        # Add metrics
        if metrics:
            for metric in metrics:
                row[metric] = run.data.metrics.get(metric, None)
        else:
            row.update(run.data.metrics)

        data.append(row)

    return pd.DataFrame(data)


# ============================================================================
# Cleanup
# ============================================================================

def delete_experiment_runs(experiment_name: str, keep_latest: int = 0):
    """
    Delete runs from an experiment, optionally keeping the latest N runs.

    Args:
        experiment_name: Name of the experiment
        keep_latest: Number of latest runs to keep (0 = delete all)

    Example:
        >>> delete_experiment_runs('credit_scoring_01_baseline', keep_latest=5)
    """
    client = mlflow.tracking.MlflowClient()

    # Get experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"[Warning] Experiment '{experiment_name}' not found")
        return

    # Get all runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"]
    )

    # Delete runs beyond keep_latest
    deleted = 0
    for i, run in enumerate(runs):
        if i >= keep_latest:
            client.delete_run(run.info.run_id)
            deleted += 1
            print(f"[Cleanup] Deleted run: {run.data.tags.get('mlflow.runName', 'Unknown')}")

    print(f"[Cleanup] Deleted {deleted} runs from '{experiment_name}'")
    print(f"[Cleanup] Kept {min(keep_latest, len(runs))} latest runs")


# ============================================================================
# Utilities
# ============================================================================

def print_run_info(run_id: Optional[str] = None):
    """
    Print information about a run.

    Args:
        run_id: Run ID (None = active run)

    Example:
        >>> with mlflow.start_run():
        ...     print_run_info()
    """
    if run_id is None:
        run = mlflow.active_run()
    else:
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)

    if run is None:
        print("[Warning] No active run")
        return

    print("="*80)
    print("RUN INFORMATION")
    print("="*80)
    print(f"Run ID: {run.info.run_id}")
    print(f"Run Name: {run.data.tags.get('mlflow.runName', 'N/A')}")
    print(f"Experiment ID: {run.info.experiment_id}")
    print(f"Status: {run.info.status}")
    print(f"\nTags:")
    for key, value in run.data.tags.items():
        if not key.startswith('mlflow.'):
            print(f"  {key}: {value}")
    print(f"\nMetrics:")
    for key, value in run.data.metrics.items():
        print(f"  {key}: {value:.4f}")
    print("="*80)


if __name__ == "__main__":
    # Example usage
    exp = setup_mlflow('baseline')
    print("\n[OK] MLflow utilities loaded successfully!")
