"""
Comprehensive Feature Engineering Experiment Suite with Cross-Validation

Systematically tests 16 configurations:
- 4 feature strategies (baseline, domain, polynomial, combined)
- 4 sampling methods (balanced, SMOTE, undersample, SMOTE+Undersample)

Methodology:
- 5-Fold Stratified Cross-Validation.
- Metrics: ROC-AUC, PR-AUC, F3.2-Score (Recall-focused), Business Cost.
- Optimizes threshold for F-beta per fold.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, fbeta_score,
    accuracy_score, confusion_matrix, precision_recall_curve
)
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
import mlflow
import mlflow.sklearn
import time
from datetime import datetime
import re

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_preprocessing import load_data
from src.polynomial_features import create_polynomial_features
from src.domain_features import create_domain_features
from src.sampling_strategies import get_sampling_strategy
from src.config import CONFIG

# Configuration from YAML
RANDOM_STATE = CONFIG['project']['random_state']
N_FOLDS = CONFIG['project']['n_folds']
BETA = CONFIG['business']['f_beta']
EXPERIMENT_NAME = CONFIG['mlflow']['experiment_names']['feature_engineering']
MLFLOW_TRACKING_URI = CONFIG['mlflow']['tracking_uri']
DATA_DIR = Path(CONFIG['paths']['data'])

# Base model parameters
BASE_MODEL_PARAMS = CONFIG['model']['base_lgbm'].copy()
BASE_MODEL_PARAMS['random_state'] = RANDOM_STATE
BASE_MODEL_PARAMS['n_jobs'] = -1

def get_optimal_fbeta_metrics(y_true, y_proba, beta=BETA):
    """
    Find the threshold that maximizes F-beta score and calculate metrics at that threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Calculate F-beta for all thresholds
    numerator = (1 + beta**2) * precision * recall
    denominator = (beta**2 * precision) + recall
    
    # Avoid division by zero
    fbeta = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    
    # Find index of max F-beta
    ix = np.argmax(fbeta)
    best_score = fbeta[ix]
    
    # Get best threshold (handle edge case where ix is last index)
    best_thresh = thresholds[ix] if ix < len(thresholds) else 0.5
    
    # Calculate Business Cost at this threshold
    y_pred = (y_proba >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Cost: 10 * FN + 1 * FP
    cost = CONFIG['business']['cost_fn'] * fn + CONFIG['business']['cost_fp'] * fp
    
    # Normalized Cost (Cost per applicant)
    avg_cost = cost / len(y_true)
    
    return best_score, cost, avg_cost, best_thresh

def prepare_features(X, feature_strategy='baseline'):
    """Prepare features according to strategy."""
    X_processed = X.copy()
    
    if feature_strategy == 'baseline':
        pass
    elif feature_strategy == 'domain':
        X_processed = create_domain_features(X_processed)
    elif feature_strategy == 'polynomial':
        X_processed, _ = create_polynomial_features(X_processed, degree=2)
    elif feature_strategy == 'combined':
        X_processed = create_domain_features(X_processed)
        X_processed, _ = create_polynomial_features(X_processed, degree=2)
    
    # Clean column names for LightGBM
    X_processed = X_processed.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    
    return X_processed

def run_cv_experiment(X, y, feature_strategy, sampling_strategy, experiment_number):
    """Run 5-fold Cross-Validation for a specific configuration."""
    print("="*80)
    print(f"EXPERIMENT {experiment_number}/16: {feature_strategy} + {sampling_strategy}")
    print("="*80)

    run_name = f"exp{experiment_number:02d}_cv_{feature_strategy}_{sampling_strategy}"
    
    print("Preparing features...")
    X_proc = prepare_features(X, feature_strategy)
    feature_names = X_proc.columns.tolist()
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []
    
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("feature_strategy", feature_strategy)
        mlflow.set_tag("sampling_strategy", sampling_strategy)
        mlflow.set_tag("validation", f"{N_FOLDS}-fold CV")
        mlflow.log_param("beta", BETA)
        
        print(f"Starting {N_FOLDS}-fold CV...")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_proc, y)):
            X_train_fold, X_val_fold = X_proc.iloc[train_idx], X_proc.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Sampling
            X_train_resampled, y_train_resampled, _ = get_sampling_strategy(
                sampling_strategy, X_train_fold, y_train_fold, random_state=RANDOM_STATE
            )
            
            # Model
            model_params = BASE_MODEL_PARAMS.copy()
            if sampling_strategy == 'balanced':
                model_params['class_weight'] = 'balanced'
            
            model = LGBMClassifier(**model_params)
            model.fit(X_train_resampled, y_train_resampled)
            
            # Evaluate
            y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
            
            # Calculate Standard Metrics
            roc_auc = roc_auc_score(y_val_fold, y_pred_proba)
            pr_auc = average_precision_score(y_val_fold, y_pred_proba)
            
            # Calculate F-Beta and Cost (Optimized Threshold)
            fbeta, cost, avg_cost, thresh = get_optimal_fbeta_metrics(y_val_fold, y_pred_proba, beta=BETA)
            
            # Metrics at optimized threshold
            y_pred = (y_pred_proba >= thresh).astype(int)
            
            scores = {
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'fbeta_score': fbeta,
                'accuracy': accuracy_score(y_val_fold, y_pred),
                'precision': precision_score(y_val_fold, y_pred),
                'recall': recall_score(y_val_fold, y_pred),
                'f1_score': f1_score(y_val_fold, y_pred),
                'business_cost': cost,
                'avg_cost': avg_cost,
                'best_threshold': thresh
            }
            fold_metrics.append(scores)
            print(f"  Fold {fold+1}: ROC={roc_auc:.4f}, F{BETA}={fbeta:.4f}, Recall={scores['recall']:.4f}")

        # Aggregate Metrics
        avg_metrics = pd.DataFrame(fold_metrics).mean().to_dict()
        std_metrics = pd.DataFrame(fold_metrics).std().to_dict()
        
        # Log Metrics
        for metric, value in avg_metrics.items():
            mlflow.log_metric(f"mean_{metric}", value)
            mlflow.log_metric(f"std_{metric}", std_metrics[metric])
            
        print(f"\nAverage Results:")
        print(f"  ROC-AUC: {avg_metrics['roc_auc']:.4f}")
        print(f"  F{BETA}-Score: {avg_metrics['fbeta_score']:.4f}")
        print(f"  Recall: {avg_metrics['recall']:.4f}")
        print(f"  Avg Cost: {avg_metrics['avg_cost']:.4f}")
        
        return avg_metrics

def run_dummy_baseline(X, y):
    """Run dummy classifier to set a baseline."""
    print("\nRunning Dummy Baseline...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    metrics = []
    
    for train_idx, val_idx in skf.split(X, y):
        dummy = DummyClassifier(strategy='most_frequent')
        dummy.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_proba = dummy.predict_proba(X.iloc[val_idx])[:, 1]
        
        fbeta, cost, avg_cost, _ = get_optimal_fbeta_metrics(y.iloc[val_idx], y_proba, beta=BETA)
        
        metrics.append({
            'roc_auc': roc_auc_score(y.iloc[val_idx], y_proba),
            'fbeta_score': fbeta,
            'avg_cost': avg_cost
        })
    
    avg = pd.DataFrame(metrics).mean()
    print(f"Dummy Baseline: ROC={avg['roc_auc']:.4f}, F{BETA}={avg['fbeta_score']:.4f}, Cost={avg['avg_cost']:.4f}")
    return avg

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    print(f"Loading data from {DATA_DIR}...")
    X_train = pd.read_csv(DATA_DIR / 'X_train.csv')
    y_train = pd.read_csv(DATA_DIR / 'y_train.csv').squeeze()
    
    run_dummy_baseline(X_train, y_train)
    
    feature_strategies = ['baseline', 'domain', 'polynomial', 'combined']
    sampling_strategies = ['balanced', 'smote', 'undersample', 'smote_undersample']
    
    results = []
    exp_num = 1
    
    for feat in feature_strategies:
        for samp in sampling_strategies:
            try:
                metrics = run_cv_experiment(X_train, y_train, feat, samp, exp_num)
                results.append({
                    'feature_strategy': feat,
                    'sampling_strategy': samp,
                    **metrics
                })
                exp_num += 1
            except Exception as e:
                print(f"Experiment {exp_num} failed: {e}")
                exp_num += 1

    if results:
        # Sort by Business Cost (Lower is better)
        res_df = pd.DataFrame(results).sort_values('avg_cost', ascending=True)
        # Save to results dir from config
        res_path = Path(CONFIG['paths']['results']) / 'cv_experiment_summary_fbeta.csv'
        res_df.to_csv(res_path, index=False)
        print("\nTop 3 Configurations (Lowest Cost):")
        print(res_df[['feature_strategy', 'sampling_strategy', 'fbeta_score', 'avg_cost']].head(3))

if __name__ == "__main__":
    main()