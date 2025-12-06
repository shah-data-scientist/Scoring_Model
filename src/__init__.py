"""
Credit Scoring Model - Source Code Package

This package contains reusable utilities for:
- Data preprocessing
- Feature engineering
- Model training
- Evaluation and metrics
- Visualization

Educational Note:
-----------------
Organizing code into modules makes it:
1. Reusable across notebooks
2. Easier to test
3. More maintainable
4. Professional and production-ready

Usage:
------
from src.data_preprocessing import load_data, analyze_missing_values
from src.feature_engineering import create_domain_features
from src.model_training import train_with_mlflow
from src.evaluation import evaluate_model, plot_roc_curve
"""

__version__ = "0.1.0"
__author__ = "Shahul SHAIK"

# Make key functions easily accessible
__all__ = [
    'data_preprocessing',
    'feature_engineering',
    'model_training',
    'evaluation',
    'visualization'
]
