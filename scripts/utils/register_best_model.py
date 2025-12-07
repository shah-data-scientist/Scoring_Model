"""
Register Best Model in MLflow Model Registry

Registers the best performing model (exp05_cv_domain_balanced) with full metadata
for production deployment tracking.
"""
import sys
from pathlib import Path
# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, fbeta_score

from src.config import CONFIG
from src.domain_features import create_domain_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_TRACKING_URI = CONFIG['mlflow']['tracking_uri']
RANDOM_STATE = CONFIG['project']['random_state']

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

print('='*80)
print('REGISTERING BEST MODEL IN MLFLOW MODEL REGISTRY')
print('='*80)

# Find best run
exp = client.get_experiment_by_name('credit_scoring_feature_engineering_cv')
runs = client.search_runs([exp.experiment_id], order_by=["metrics.mean_roc_auc DESC"], max_results=1)

if not runs:
    print('ERROR: No runs found!')
    sys.exit(1)

best_run = runs[0]
run_id = best_run.info.run_id
run_name = best_run.data.tags.get('mlflow.runName', 'Unnamed')
roc_auc = best_run.data.metrics.get('mean_roc_auc', 0)

print(f'\nBest Run Identified:')
print(f'  Name: {run_name}')
print(f'  Run ID: {run_id}')
print(f'  ROC-AUC: {roc_auc:.4f}')
print(f'  Feature Strategy: {best_run.data.tags.get("feature_strategy")}')
print(f'  Sampling Strategy: {best_run.data.tags.get("sampling_strategy")}')

# Load data and train final model
print('\nTraining final model for registration...')

data_dir = PROJECT_ROOT / 'data' / 'processed'
X_train = pd.read_csv(data_dir / 'X_train.csv')
y_train = pd.read_csv(data_dir / 'y_train.csv').squeeze()

# Apply domain features
X_proc = create_domain_features(X_train)

# Model parameters (from best run)
model_params = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': -1
}

# Train final model on full training set
model = LGBMClassifier(**model_params)
model.fit(X_proc, y_train)

print('  Model trained successfully')

# Log model with MLflow
print('\nRegistering model with MLflow...')

model_name = "credit_scoring_production_model"

# Set experiment for new run
mlflow.set_experiment('credit_scoring_feature_engineering_cv')

with mlflow.start_run(run_name=f"{run_name}_registered") as run:
    # Log parameters
    mlflow.log_params(model_params)
    mlflow.log_param("n_features", X_proc.shape[1])
    mlflow.log_param("training_samples", len(y_train))

    # Log metrics (from CV)
    mlflow.log_metric("cv_mean_roc_auc", roc_auc)
    mlflow.log_metric("cv_std_roc_auc", best_run.data.metrics.get('std_roc_auc', 0))

    # Calculate training metrics
    y_train_pred_proba = model.predict_proba(X_proc)[:, 1]
    y_train_pred = (y_train_pred_proba >= 0.5).astype(int)

    train_roc_auc = roc_auc_score(y_train, y_train_pred_proba)
    train_pr_auc = average_precision_score(y_train, y_train_pred_proba)
    train_f1 = f1_score(y_train, y_train_pred)
    train_fbeta = fbeta_score(y_train, y_train_pred, beta=3.2)

    mlflow.log_metric("train_roc_auc", train_roc_auc)
    mlflow.log_metric("train_pr_auc", train_pr_auc)
    mlflow.log_metric("train_f1", train_f1)
    mlflow.log_metric("train_fbeta", train_fbeta)

    # Log tags
    mlflow.set_tag("model_type", "LightGBM")
    mlflow.set_tag("feature_strategy", "domain")
    mlflow.set_tag("sampling_strategy", "balanced")
    mlflow.set_tag("validation", "5-fold CV")
    mlflow.set_tag("status", "production_candidate")
    mlflow.set_tag("optimal_threshold", "0.3282")

    # Log model
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name=model_name
    )

    registered_run_id = run.info.run_id

print(f'\n[OK] Model registered successfully!')
print(f'  Model Name: {model_name}')
print(f'  Run ID: {registered_run_id}')

# Get registered model details
try:
    registered_model = client.get_registered_model(model_name)
    latest_version = client.get_latest_versions(model_name)[0]

    print(f'\n  Model Version: {latest_version.version}')
    print(f'  Stage: {latest_version.current_stage}')
    print(f'  Status: {latest_version.status}')

    # Transition to Staging
    print('\nTransitioning model to Staging...')
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version.version,
        stage="Staging"
    )

    print('[OK] Model transitioned to Staging stage')
    print('\nModel is now ready for testing in staging environment.')
    print('After validation, transition to Production stage using:')
    print(f'  client.transition_model_version_stage(name="{model_name}", version={latest_version.version}, stage="Production")')

except Exception as e:
    print(f'\nWARNING: Could not transition model stage: {e}')

print('\n' + '='*80)
print('MODEL REGISTRATION COMPLETE')
print('='*80)
print(f'\nAccess in MLflow UI: http://localhost:5000/#/models/{model_name}')
