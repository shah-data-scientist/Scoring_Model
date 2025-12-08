"""
Add artifacts to MLflow runs with feature type categorization.

This script retroactively adds visualization artifacts to existing MLflow runs,
with feature importance plots that color-code features by type (baseline, domain, polynomial, aggregated).
"""
import warnings
warnings.filterwarnings('ignore')

import mlflow
from mlflow import MlflowClient
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    roc_auc_score, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
MLRUNS_DIR = PROJECT_ROOT / 'mlruns'

# MLflow setup
mlflow.set_tracking_uri(f"sqlite:///{MLRUNS_DIR}/mlflow.db")


def categorize_features(feature_names):
    """
    Categorize features by type.

    Returns:
        dict: Dictionary mapping feature types to lists of feature names
    """
    categories = {
        'baseline': [],
        'domain': [],
        'polynomial': [],
        'aggregated': []
    }

    # Domain feature patterns (from domain_features.py)
    domain_patterns = [
        'AGE_YEARS', 'EMPLOYMENT_YEARS', 'IS_EMPLOYED',
        'INCOME_PER_PERSON', 'DEBT_TO_INCOME_RATIO',
        'CREDIT_TO_GOODS_RATIO', 'CREDIT_UTILIZATION',
        'ANNUITY_TO_INCOME_RATIO', 'HAS_CHILDREN', 'CHILDREN_RATIO',
        'TOTAL_DOCUMENTS_PROVIDED', 'EXT_SOURCE_MEAN', 'EXT_SOURCE_MAX',
        'EXT_SOURCE_MIN', 'REGION_RATING_COMBINED'
    ]

    # Aggregated feature prefixes (from feature_aggregation.py)
    aggregated_prefixes = [
        'BUREAU_', 'PREV_APP_', 'POS_CASH_', 'INSTALL_',
        'CREDIT_CARD_', 'AGG_'
    ]

    for feature in feature_names:
        # Check domain features
        if any(pattern in feature for pattern in domain_patterns):
            categories['domain'].append(feature)
        # Check aggregated features
        elif any(feature.startswith(prefix) for prefix in aggregated_prefixes):
            categories['aggregated'].append(feature)
        # Check polynomial features (contains ' ' or '^')
        elif ' ' in feature or '^' in feature:
            categories['polynomial'].append(feature)
        # Otherwise baseline
        else:
            categories['baseline'].append(feature)

    return categories


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Create confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix (Row-Normalized)')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curve_viz(y_true, y_proba, save_path):
    """Create ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pr_curve(y_true, y_proba, save_path):
    """Create Precision-Recall curve plot."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {pr_auc:.3f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance_with_types(model, feature_names, save_path):
    """
    Create feature importance plot with color-coding by feature type.

    Args:
        model: Trained model with feature_importances_
        feature_names: List of feature names
        save_path: Path to save plot
    """
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:30]  # Top 30 features

    # Categorize features
    categories = categorize_features(feature_names)

    # Create color map
    colors = []
    color_map = {
        'baseline': '#1f77b4',  # Blue
        'domain': '#ff7f0e',    # Orange
        'polynomial': '#2ca02c',  # Green
        'aggregated': '#d62728'  # Red
    }

    feature_labels = []
    for idx in indices:
        feature = feature_names[idx]
        # Determine feature type
        feature_type = 'baseline'
        for cat, features in categories.items():
            if feature in features:
                feature_type = cat
                break
        colors.append(color_map[feature_type])

        # Create label with type indicator
        type_abbrev = {'baseline': '[B]', 'domain': '[D]', 'polynomial': '[P]', 'aggregated': '[A]'}
        feature_labels.append(f"{type_abbrev[feature_type]} {feature}")

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 12))
    y_pos = np.arange(len(indices))
    ax.barh(y_pos, importances[indices], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_labels[i] for i in range(len(indices))], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top 30 Feature Importances by Type\n[B]=Baseline, [D]=Domain, [P]=Polynomial, [A]=Aggregated')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map['baseline'], label='Baseline'),
        Patch(facecolor=color_map['domain'], label='Domain'),
        Patch(facecolor=color_map['polynomial'], label='Polynomial'),
        Patch(facecolor=color_map['aggregated'], label='Aggregated')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Also save feature importance CSV with type labels
    feature_categories = categorize_features(feature_names)
    feature_types = {}
    for cat, features in feature_categories.items():
        for feature in features:
            feature_types[feature] = cat

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
        'type': [feature_types.get(f, 'baseline') for f in feature_names]
    }).sort_values('importance', ascending=False)

    csv_path = str(save_path).replace('.png', '.csv')
    importance_df.to_csv(csv_path, index=False)


def load_training_data():
    """Load training data with features."""
    print("Loading training data...")
    X_train = pd.read_csv(DATA_DIR / 'X_train.csv')
    y_train = pd.read_csv(DATA_DIR / 'y_train.csv')

    # Handle different column structures
    if 'TARGET' in y_train.columns:
        y_train = y_train['TARGET']
    elif 'SK_ID_CURR' in y_train.columns and len(y_train.columns) == 2:
        # Assume second column is target
        y_train = y_train.iloc[:, 1]
    else:
        y_train = y_train.iloc[:, 0]

    # Remove ID column if present
    if 'SK_ID_CURR' in X_train.columns:
        X_train = X_train.drop('SK_ID_CURR', axis=1)

    print(f"  Data shape: X={X_train.shape}, y={y_train.shape}")
    return X_train, y_train


def add_artifacts_to_run(run_id, run_name, X, y, params, feature_strategy):
    """
    Generate and add artifacts to a specific MLflow run.

    Args:
        run_id: MLflow run ID
        run_name: Name of the run
        X: Training features
        y: Training labels
        params: Model parameters from the run
        feature_strategy: Feature strategy used (for logging)
    """
    print(f"\n[OK] Processing run: {run_name}")
    print(f"  Run ID: {run_id}")
    print(f"  Feature strategy: {feature_strategy}")

    # Check if artifacts already exist
    client = MlflowClient()
    existing_artifacts = client.list_artifacts(run_id)
    if len(existing_artifacts) >= 4:
        print(f"  [SKIP] Run already has {len(existing_artifacts)} artifacts")
        return

    try:
        # Extract model parameters (handle different parameter sets)
        model_params = {
            'n_estimators': int(params.get('n_estimators', 100)),
            'max_depth': int(params.get('max_depth', 6)),
            'learning_rate': float(params.get('learning_rate', 0.1)),
            'subsample': float(params.get('subsample', 0.8)),
            'colsample_bytree': float(params.get('colsample_bytree', 0.8)),
            'random_state': int(params.get('random_state', 42)),
            'n_jobs': int(params.get('n_jobs', -1)),
            'verbose': int(params.get('verbose', -1)),
            'class_weight': params.get('class_weight', None)
        }

        # Train model with 5-fold CV to get predictions
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred_proba = np.zeros(len(y))

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

            model = LGBMClassifier(**model_params)
            model.fit(X_fold_train, y_fold_train)
            y_pred_proba[val_idx] = model.predict_proba(X_fold_val)[:, 1]

        # Use final model for feature importance
        final_model = LGBMClassifier(**model_params)
        final_model.fit(X, y)

        # Generate predictions for confusion matrix
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Create artifact directory
        artifact_dir = MLRUNS_DIR / run_id[:2] / run_id / 'artifacts'
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Generate and save artifacts
        print(f"  Generating artifacts...")

        # 1. Confusion Matrix
        cm_path = artifact_dir / 'confusion_matrix.png'
        plot_confusion_matrix(y, y_pred, cm_path)

        # 2. ROC Curve
        roc_path = artifact_dir / 'roc_curve.png'
        plot_roc_curve_viz(y, y_pred_proba, roc_path)

        # 3. PR Curve
        pr_path = artifact_dir / 'pr_curve.png'
        plot_pr_curve(y, y_pred_proba, pr_path)

        # 4. Feature Importance with type labels
        fi_path = artifact_dir / 'feature_importance.png'
        plot_feature_importance_with_types(final_model, X.columns.tolist(), fi_path)

        print(f"  [OK] Created 5 artifacts (4 plots + 1 CSV)")

    except Exception as e:
        print(f"  [ERROR] Failed to add artifacts: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main execution function."""
    print("=" * 80)
    print("ADD ARTIFACTS TO MLFLOW RUNS WITH FEATURE TYPE LABELING")
    print("=" * 80)

    # Load training data
    X_train, y_train = load_training_data()

    # Get runs from feature engineering experiment (ID=2)
    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=['2'],
        order_by=["metrics.mean_roc_auc DESC", "metrics.cv_mean_roc_auc DESC"],
        max_results=10
    )

    print(f"\nFound {len(runs)} runs in feature engineering experiment")
    print(f"Processing top 10 runs...")

    # Process each run
    for i, run in enumerate(runs, 1):
        run_name = run.data.tags.get('mlflow.runName', 'Unnamed')
        feature_strategy = run.data.tags.get('feature_strategy', 'unknown')

        add_artifacts_to_run(
            run_id=run.info.run_id,
            run_name=run_name,
            X=X_train,
            y=y_train,
            params=run.data.params,
            feature_strategy=feature_strategy
        )

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print("\nAll artifacts have been added with feature type categorization.")
    print("Feature types: [B]=Baseline, [D]=Domain, [P]=Polynomial, [A]=Aggregated")
    print("\nYou can now view them in MLflow UI: http://localhost:5000")


if __name__ == '__main__':
    main()
