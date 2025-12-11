"""
Configuration Loader

Loads project configuration from config.yaml.
"""
import yaml
from pathlib import Path
import os

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    # Robustly find project root relative to this file (src/config.py)
    # src/config.py -> parent = src -> parent = Project Root
    project_root = Path(__file__).resolve().parent.parent
    path = project_root / config_path
    
    if not path.exists():
        # Fallback to CWD
        path = Path.cwd() / config_path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found. Searched at:\n1. {project_root / config_path}\n2. {Path.cwd() / config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)
        
    return config

# Singleton config object
# We remove the try-except block that masks errors. 
# The application requires configuration to run; strictly failing is better than silently failing.
CONFIG = load_config()

# Helper accessors
def get_data_path():
    return Path(CONFIG.get('paths', {}).get('data', 'data/processed'))

def get_mlflow_uri():
    return CONFIG.get('mlflow', {}).get('tracking_uri', 'sqlite:///mlruns/mlflow.db')

def get_random_state():
    return CONFIG.get('project', {}).get('random_state', 42)

# Project root path
PROJECT_ROOT = Path(__file__).parent.parent

# MLflow Configuration
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT}/mlruns/mlflow.db"
MLFLOW_ARTIFACT_ROOT = str(PROJECT_ROOT / "mlruns")

# Experiment names from config
EXPERIMENTS = CONFIG.get('mlflow', {}).get('experiment_names', {
    'model_selection': 'credit_scoring_model_selection',
    'feature_engineering': 'credit_scoring_feature_engineering_cv',
    'optimization': 'credit_scoring_optimization_fbeta',
    'final_delivery': 'credit_scoring_final_delivery',
})

# Registered model names by model type
REGISTERED_MODELS = {
    'lgbm': 'credit_scoring_lightgbm',
    'xgboost': 'credit_scoring_xgboost',
    'random_forest': 'credit_scoring_random_forest',
    'production': 'credit_scoring_production_model',
}

# Path configurations
DATA_DIR = PROJECT_ROOT / CONFIG.get('paths', {}).get('data', 'data/processed')
MODELS_DIR = PROJECT_ROOT / CONFIG.get('paths', {}).get('models', 'models')
RESULTS_DIR = PROJECT_ROOT / CONFIG.get('paths', {}).get('results', 'results')
MLRUNS_DIR = PROJECT_ROOT / CONFIG.get('paths', {}).get('mlruns', 'mlruns')

def get_baseline_tags(model_name: str, **kwargs) -> dict:
    """
    Get standardized tags for baseline experiments.

    Args:
        model_name: Name of the model
        **kwargs: Additional custom tags

    Returns:
        Dictionary of tags
    """
    tags = {
        'stage': 'baseline',
        'model': model_name,
        'project': 'credit_scoring',
    }
    tags.update(kwargs)
    return tags

def get_optimization_tags(model_name: str, optimization_type: str = 'optuna', **kwargs) -> dict:
    """
    Get standardized tags for optimization experiments.

    Args:
        model_name: Name of the model
        optimization_type: Type of optimization (optuna, grid_search, etc.)
        **kwargs: Additional custom tags

    Returns:
        Dictionary of tags
    """
    tags = {
        'stage': 'optimization',
        'model': model_name,
        'optimization': optimization_type,
        'project': 'credit_scoring',
    }
    tags.update(kwargs)
    return tags

def get_production_tags(model_name: str, version: str = '1.0', **kwargs) -> dict:
    """
    Get standardized tags for production models.

    Args:
        model_name: Name of the model
        version: Model version
        **kwargs: Additional custom tags

    Returns:
        Dictionary of tags
    """
    tags = {
        'stage': 'production',
        'model': model_name,
        'version': version,
        'project': 'credit_scoring',
    }
    tags.update(kwargs)
    return tags

def get_artifact_path(model_name: str, artifact_type: str) -> str:
    """
    Get standardized artifact path.

    Args:
        model_name: Name of the model
        artifact_type: Type of artifact (plot, data, model, etc.)

    Returns:
        Standardized artifact path string
    """
    return f"{artifact_type}/{model_name}_{artifact_type}"