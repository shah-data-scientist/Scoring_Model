"""
Add artifacts to top performing MLflow runs retroactively

For each top run:
1. Load the data and recreate the model
2. Generate visualizations (confusion matrix, ROC curve, PR curve, feature importance)
3. Upload artifacts to the existing run
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from lightgbm import LGBMClassifier

from src.config import MLFLOW_TRACKING_URI, RANDOM_STATE, PROJECT_ROOT
from src.domain_features import create_domain_features
from src.sampling_strategies import get_sampling_strategy

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# Load data
print('='*80)
print('ADDING ARTIFACTS TO TOP RUNS')
print('='*80)
print('\nLoading training data...')

data_dir = PROJECT_ROOT / 'data' / 'processed'
X_train = pd.read_csv(data_dir / 'X_train.csv')
y_train = pd.read_csv(data_dir / 'y_train.csv').squeeze()

print(f'Data loaded: {X_train.shape}')

# Find top runs from feature engineering experiment
exp = client.get_experiment_by_name('credit_scoring_feature_engineering_cv')

if not exp:
    print('ERROR: credit_scoring_feature_engineering_cv experiment not found!')
    sys.exit(1)

runs = client.search_runs([exp.experiment_id], order_by=["metrics.mean_roc_auc DESC"], max_results=5)

print(f'\nFound {len(runs)} top runs to enhance with artifacts')

for idx, run in enumerate(runs, 1):
    run_name = run.data.tags.get('mlflow.runName', 'Unnamed')
    run_id = run.info.run_id
    roc_auc = run.data.metrics.get('mean_roc_auc', 0)

    print(f'\n{"="*80}')
    print(f'Processing Run {idx}/5: {run_name}')
    print(f'ROC-AUC: {roc_auc:.4f}')
    print(f'Run ID: {run_id}')
    print('='*80)

    # Check if artifacts already exist
    existing_artifacts = client.list_artifacts(run_id)
    if len(existing_artifacts) > 0:
        print(f'  [SKIP] Run already has {len(existing_artifacts)} artifacts. Skipping...')
        continue

    # Get run configuration
    feature_strategy = run.data.tags.get('feature_strategy', 'baseline')
    sampling_strategy = run.data.tags.get('sampling_strategy', 'balanced')

    print(f'  Feature Strategy: {feature_strategy}')
    print(f'  Sampling Strategy: {sampling_strategy}')

    # Prepare features
    print('  Preparing features...')
    X_proc = X_train.copy()

    if feature_strategy == 'domain':
        X_proc = create_domain_features(X_proc)
    # Add other feature strategies if needed

    # Get model parameters from run
    model_params = {
        'n_estimators': int(run.data.params.get('n_estimators', 100)),
        'max_depth': int(run.data.params.get('max_depth', 6)),
        'learning_rate': float(run.data.params.get('learning_rate', 0.1)),
        'subsample': float(run.data.params.get('subsample', 0.8)),
        'colsample_bytree': float(run.data.params.get('colsample_bytree', 0.8)),
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1
    }

    if sampling_strategy == 'balanced':
        model_params['class_weight'] = 'balanced'

    # Train model with 5-fold CV to get predictions
    print('  Running 5-fold CV...')
    model = LGBMClassifier(**model_params)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Get CV predictions
    y_pred_proba = cross_val_predict(
        model, X_proc, y_train,
        cv=skf,
        method='predict_proba',
        n_jobs=-1
    )[:, 1]

    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Generate artifacts
    print('  Generating artifacts...')

    artifact_dir = Path(f'mlruns/{exp.experiment_id}/{run_id}/artifacts')
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # 1. Confusion Matrix
    cm = confusion_matrix(y_train, y_pred)
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_pct, annot=True, fmt='.2%', cmap='Greens', ax=ax1,
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    ax1.set_ylabel('Actual')
    ax1.set_xlabel('Predicted')
    ax1.set_title(f'Confusion Matrix - {run_name}')
    plt.tight_layout()
    plt.savefig(artifact_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_train, y_pred_proba)
    roc_auc_score_val = roc_auc_score(y_train, y_pred_proba)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_score_val:.4f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title(f'ROC Curve - {run_name}')
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(artifact_dir / 'roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_train, y_pred_proba)

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.plot(recall, precision, color='blue', lw=2)
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title(f'Precision-Recall Curve - {run_name}')
    ax3.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(artifact_dir / 'pr_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Feature Importance (train final model)
    print('  Training final model for feature importance...')
    final_model = LGBMClassifier(**model_params)
    final_model.fit(X_proc, y_train)

    feature_importance = pd.DataFrame({
        'feature': X_proc.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Save feature importance CSV
    feature_importance.to_csv(artifact_dir / 'feature_importance.csv', index=False)

    # Plot top 20 features
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    top_features = feature_importance.head(20)
    ax4.barh(range(len(top_features)), top_features['importance'])
    ax4.set_yticks(range(len(top_features)))
    ax4.set_yticklabels(top_features['feature'], fontsize=9)
    ax4.set_xlabel('Importance')
    ax4.set_title(f'Top 20 Feature Importances - {run_name}')
    ax4.invert_yaxis()
    plt.tight_layout()
    plt.savefig(artifact_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f'  [OK] Created 4 artifacts in {artifact_dir}')
    print(f'    - confusion_matrix.png')
    print(f'    - roc_curve.png')
    print(f'    - pr_curve.png')
    print(f'    - feature_importance.csv/png')

print('\n' + '='*80)
print('ARTIFACT GENERATION COMPLETE')
print('='*80)
print('\nArtifacts have been added to MLflow runs.')
print('Refresh MLflow UI (http://localhost:5000) to view them.')
