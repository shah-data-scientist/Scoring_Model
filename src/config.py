"""
Centralized MLflow configuration for the Credit Scoring Model project.

This module provides standardized naming conventions, paths, and settings
for MLflow experiment tracking across all notebooks and scripts.
"""
from pathlib import Path
from datetime import datetime

# ============================================================================
# PROJECT SETTINGS
# ============================================================================

PROJECT_NAME = "credit_scoring"
PROJECT_ROOT = Path(__file__).parent.parent
DATA_VERSION = "v2_comprehensive_318features"
RANDOM_STATE = 42

# ============================================================================
# MLFLOW SETTINGS
# ============================================================================

# Tracking backend
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT}/mlruns/mlflow.db"
MLFLOW_ARTIFACT_ROOT = str(PROJECT_ROOT / "mlruns")

# ============================================================================
# EXPERIMENT NAMES (Standardized)
# ============================================================================

EXPERIMENTS = {
    "baseline": f"{PROJECT_NAME}_01_baseline",
    "optimization": f"{PROJECT_NAME}_02_optimization",
    "final_evaluation": f"{PROJECT_NAME}_03_final_evaluation",
    "production": f"{PROJECT_NAME}_04_production"
}

# Legacy experiment names (for migration reference)
LEGACY_EXPERIMENTS = {
    "credit_scoring_baseline_models": "baseline",
    "credit_scoring_hyperparameter_optimization": "optimization"
}

# ============================================================================
# MODEL REGISTRY NAMES
# ============================================================================

REGISTERED_MODELS = {
    "lgbm": f"{PROJECT_NAME}_lgbm",
    "xgboost": f"{PROJECT_NAME}_xgboost",
    "random_forest": f"{PROJECT_NAME}_random_forest",
    "logistic_regression": f"{PROJECT_NAME}_logistic_regression",
    "ensemble": f"{PROJECT_NAME}_ensemble"
}

# ============================================================================
# RUN NAME TEMPLATES
# ============================================================================

def get_baseline_run_name(model_type, version=1):
    """
    Generate standardized run name for baseline models.

    Args:
        model_type: Model type (e.g., 'lgbm', 'xgboost')
        version: Version number (default: 1)

    Returns:
        Standardized run name

    Example:
        >>> get_baseline_run_name('lgbm', 1)
        'lgbm_v1_baseline'
    """
    return f"{model_type}_v{version}_baseline"


def get_optimization_run_name(model_type, key_params=None, version=1):
    """
    Generate standardized run name for optimized models.

    Args:
        model_type: Model type (e.g., 'lgbm', 'xgboost')
        key_params: Dict of key parameters to include in name
        version: Version number (default: 1)

    Returns:
        Standardized run name

    Example:
        >>> get_optimization_run_name('lgbm', {'lr': 0.05, 'depth': 5}, 2)
        'lgbm_lr0.05_depth5_v2_optimized'
    """
    if key_params:
        param_str = "_".join([f"{k}{v}" for k, v in key_params.items()])
        return f"{model_type}_{param_str}_v{version}_optimized"
    return f"{model_type}_v{version}_optimized"


def get_production_run_name(model_type, version=1, date=None):
    """
    Generate standardized run name for production models.

    Args:
        model_type: Model type (e.g., 'lgbm', 'xgboost')
        version: Version number (default: 1)
        date: Date string (YYYYMMDD) or None for today

    Returns:
        Standardized run name

    Example:
        >>> get_production_run_name('lgbm', 3, '20231206')
        'lgbm_v3_production_20231206'
    """
    if date is None:
        date = datetime.now().strftime("%Y%m%d")
    return f"{model_type}_v{version}_production_{date}"


# ============================================================================
# STANDARD TAGS
# ============================================================================

STANDARD_TAGS = {
    "project": PROJECT_NAME,
    "data_version": DATA_VERSION,
}


def get_baseline_tags(model_type, author=None):
    """Get standard tags for baseline runs."""
    tags = STANDARD_TAGS.copy()
    tags.update({
        "stage": "baseline",
        "model_type": model_type,
        "purpose": "baseline_comparison"
    })
    if author:
        tags["author"] = author
    return tags


def get_optimization_tags(model_type, method="random_search", cv_folds=5, author=None):
    """Get standard tags for optimization runs."""
    tags = STANDARD_TAGS.copy()
    tags.update({
        "stage": "optimization",
        "model_type": model_type,
        "purpose": "hyperparameter_tuning",
        "optimization_method": method,
        "cv_folds": str(cv_folds)
    })
    if author:
        tags["author"] = author
    return tags


def get_production_tags(model_type, deployment_env="production", author=None):
    """Get standard tags for production runs."""
    tags = STANDARD_TAGS.copy()
    tags.update({
        "stage": "production",
        "model_type": model_type,
        "purpose": "deployment",
        "deployment_env": deployment_env,
        "deployment_date": datetime.now().strftime("%Y-%m-%d")
    })
    if author:
        tags["author"] = author
    return tags


# ============================================================================
# ARTIFACT PATHS
# ============================================================================

def get_artifact_path(model_name, artifact_type, extension="png"):
    """
    Generate standardized artifact path.

    Args:
        model_name: Model run name
        artifact_type: Type of artifact (e.g., 'roc_curve', 'feature_importance')
        extension: File extension (default: 'png')

    Returns:
        Standardized artifact path

    Example:
        >>> get_artifact_path('lgbm_v1_baseline', 'roc_curve')
        'plots/lgbm_v1_baseline_roc_curve.png'
    """
    artifact_dirs = {
        'roc_curve': 'plots',
        'pr_curve': 'plots',
        'confusion_matrix': 'plots',
        'feature_importance': 'plots',
        'model': 'models',
        'report': 'reports',
        'predictions': 'data'
    }

    dir_name = artifact_dirs.get(artifact_type, 'artifacts')
    return f"{dir_name}/{model_name}_{artifact_type}.{extension}"


# ============================================================================
# MODEL PARAMETERS PRESETS
# ============================================================================

# Default parameters for baseline models
BASELINE_PARAMS = {
    "lgbm": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 50,
        "min_samples_leaf": 20,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": 0
    },
    "logistic_regression": {
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    },
    "dummy": {
        "strategy": "most_frequent",
        "random_state": RANDOM_STATE
    }
}

# Hyperparameter search spaces
SEARCH_SPACES = {
    "lgbm": {
        "n_estimators": [100, 150, 200, 250, 300],
        "max_depth": [3, 5, 7, 10, 15],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "reg_alpha": [0, 0.1, 0.5, 1.0],
        "reg_lambda": [0, 0.1, 0.5, 1.0]
    },
    "xgboost": {
        "n_estimators": [100, 150, 200, 250, 300],
        "max_depth": [3, 5, 7, 10, 15],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0, 0.1, 0.2, 0.3]
    }
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_short_name(model_type):
    """
    Get short name for model type.

    Example:
        >>> get_model_short_name('lightgbm')
        'lgbm'
    """
    name_mapping = {
        "lightgbm": "lgbm",
        "xgboost": "xgboost",
        "random_forest": "rf",
        "logistic_regression": "lr",
        "dummy_classifier": "dummy"
    }
    return name_mapping.get(model_type.lower(), model_type.lower())


def print_config():
    """Print current configuration."""
    print("="*80)
    print("MLFLOW CONFIGURATION")
    print("="*80)
    print(f"\nProject: {PROJECT_NAME}")
    print(f"Data Version: {DATA_VERSION}")
    print(f"Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"\nExperiments:")
    for key, name in EXPERIMENTS.items():
        print(f"  {key:20s}: {name}")
    print(f"\nRegistered Models:")
    for key, name in REGISTERED_MODELS.items():
        print(f"  {key:20s}: {name}")
    print("="*80)


if __name__ == "__main__":
    print_config()
