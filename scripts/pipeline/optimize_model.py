"""
Hyperparameter Optimization for Credit Scoring Model

Goal: Maximize Recall while maintaining reasonable Precision.
Metric: F-beta score (beta=3.2), reflecting 10x cost of False Negatives vs False Positives.
Method: Optuna (Bayesian Optimization) with 5-fold CV.
Model: LightGBM with Domain Features + Balanced Class Weights.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import optuna
import mlflow
import mlflow.sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, accuracy_score, f1_score,
    roc_auc_score, average_precision_score
)
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import json
import re

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.domain_features import create_domain_features
from src.config import CONFIG

# Configuration from YAML
RANDOM_STATE = CONFIG['project']['random_state']
N_FOLDS = CONFIG['project']['n_folds']
N_TRIALS = CONFIG['model']['optimization']['n_trials']
BETA = CONFIG['business']['f_beta']
EXPERIMENT_NAME = CONFIG['mlflow']['experiment_names']['optimization']
MLFLOW_TRACKING_URI = CONFIG['mlflow']['tracking_uri']
DATA_DIR = Path(CONFIG['paths']['data'])
RESULTS_DIR = Path(CONFIG['paths']['results'])

def get_optimal_fbeta_metrics(y_true, y_proba, beta=BETA):
    """Find optimal threshold and calculate F-beta metrics."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Calculate F-beta for all thresholds
    numerator = (1 + beta**2) * precision * recall
    denominator = (beta**2 * precision) + recall
    fbeta = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    
    ix = np.argmax(fbeta)
    best_score = fbeta[ix]
    best_thresh = thresholds[ix] if ix < len(thresholds) else 0.5
    
    return best_score, best_thresh

def plot_confusion_matrix(cm, title='Confusion Matrix'):
    """Create confusion matrix plots (Counts and Percentages)."""
    import seaborn as sns
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')
    axes[0].set_title(f'{title} (Counts)')
    
    # 2. Percentages (Normalized by Row - Recall/Specificity)
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_pct, annot=True, fmt='.2%', cmap='Greens', ax=axes[1],
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    axes[1].set_ylabel('Actual')
    axes[1].set_xlabel('Predicted')
    axes[1].set_title(f'{title} (Row %)')
    
    plt.tight_layout()
    return fig

def objective(trial, X, y):
    """Optuna objective function."""
    
    # Define hyperparameter search space from CONFIG
    search_space = CONFIG['model']['optimization']['search_space']
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', *search_space['n_estimators']),
        'learning_rate': trial.suggest_float('learning_rate', *search_space['learning_rate'], log=True),
        'num_leaves': trial.suggest_int('num_leaves', *search_space['num_leaves']),
        'max_depth': trial.suggest_int('max_depth', *search_space['max_depth']),
        'min_child_samples': trial.suggest_int('min_child_samples', *search_space['min_child_samples']),
        'subsample': trial.suggest_float('subsample', *search_space['subsample']),
        'colsample_bytree': trial.suggest_float('colsample_bytree', *search_space['colsample_bytree']),
        'reg_alpha': trial.suggest_float('reg_alpha', *search_space['reg_alpha']),
        'reg_lambda': trial.suggest_float('reg_lambda', *search_space['reg_lambda']),
        'class_weight': 'balanced',
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1
    }

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    
    # 5-Fold CV
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        
        y_proba = model.predict_proba(X_val)[:, 1]
        
        # Optimize threshold for this fold
        best_fbeta, _ = get_optimal_fbeta_metrics(y_val, y_proba, beta=BETA)
        cv_scores.append(best_fbeta)

    mean_fbeta = np.mean(cv_scores)
    
    # Log to MLflow (nested run)
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("fbeta_score", mean_fbeta)
        mlflow.set_tag("model", "LightGBM")
        
    return mean_fbeta

def main():
    print("="*80)
    print("HYPERPARAMETER OPTIMIZATION (F-BETA)")
    print("="*80)
    print(f"Target: Maximize F{BETA}-Score")
    print(f"Strategy: Domain Features + Balanced Weights")
    
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Load Data
    print(f"Loading data from {DATA_DIR}...")
    X = pd.read_csv(DATA_DIR / 'X_train.csv') 
    y = pd.read_csv(DATA_DIR / 'y_train.csv').squeeze()
    
    # Create Domain Features
    print("Creating domain features...")
    X_proc = create_domain_features(X)
    
    # Clean column names for LightGBM
    X_proc = X_proc.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    
    # Create Optuna Study
    study = optuna.create_study(direction='maximize')
    
    print(f"\nStarting optimization with {N_TRIALS} trials...")
    with mlflow.start_run(run_name="optuna_optimization_domain_balanced") as parent_run:
        mlflow.log_param("beta", BETA)
        mlflow.log_param("n_trials", N_TRIALS)
        mlflow.log_param("feature_strategy", "domain")
        mlflow.log_param("sampling_strategy", "balanced")
        
        # Optimize
        study.optimize(lambda trial: objective(trial, X_proc, y), n_trials=N_TRIALS)
        
        # Log best results
        best_params = study.best_params
        best_score = study.best_value
        
        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE")
        print("="*80)
        print(f"Best F{BETA}-Score: {best_score:.4f}")
        
        # --- RETRAIN AND EVALUATE BEST MODEL TO GET PLOTS ---
        print("\nRetraining best model to generate artifacts...")
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        y_probas = np.zeros(len(y))
        
        # Add fixed params
        final_params = best_params.copy()
        final_params['class_weight'] = 'balanced'
        final_params['random_state'] = RANDOM_STATE
        final_params['n_jobs'] = -1
        final_params['verbose'] = -1
        
        for train_idx, val_idx in cv.split(X_proc, y):
            model = LGBMClassifier(**final_params)
            model.fit(X_proc.iloc[train_idx], y.iloc[train_idx])
            y_probas[val_idx] = model.predict_proba(X_proc.iloc[val_idx])[:, 1]
            
        # Find Optimal Threshold on the aggregated predictions
        precision, recall, thresholds = precision_recall_curve(y, y_probas)
        numerator = (1 + BETA**2) * precision * recall
        denominator = (BETA**2 * precision) + recall
        fbeta = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
        
        ix = np.argmax(fbeta)
        best_thresh = thresholds[ix] if ix < len(thresholds) else 0.5
        
        # Metrics at Best Threshold
        y_pred_opt = (y_probas >= best_thresh).astype(int)
        cm = confusion_matrix(y, y_pred_opt)
        tn, fp, fn, tp = cm.ravel()
        
        business_cost = CONFIG['business']['cost_fn'] * fn + CONFIG['business']['cost_fp'] * fp
        accuracy = accuracy_score(y, y_pred_opt)
        f1 = f1_score(y, y_pred_opt)
        roc_auc = roc_auc_score(y, y_probas)
        pr_auc = average_precision_score(y, y_probas)
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Log Metrics & Artifacts
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_fbeta_score", best_score)
        mlflow.log_metric("optimal_threshold", best_thresh)
        
        mlflow.log_metric("business_cost", business_cost)
        mlflow.log_metric("recall", recall[ix])
        mlflow.log_metric("precision", precision[ix])
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("false_positive_rate", fpr)
        mlflow.log_metric("false_negative_rate", fnr)
        
        # Plot CM
        fig_cm = plot_confusion_matrix(cm, title=f'Confusion Matrix @ Threshold {best_thresh:.3f}')
        cm_path = RESULTS_DIR / "confusion_matrix_optimal.png"
        fig_cm.savefig(cm_path)
        mlflow.log_artifact(str(cm_path))
        plt.close(fig_cm)
        
        print(f"Optimal Threshold: {best_thresh:.4f}")
        print("Confusion Matrix saved to MLflow.")
        print(f"Metrics at Optimal Threshold:")
        print(f"  Business Cost: {business_cost} (FN={fn}, FP={fp})")
        print(f"  Recall: {recall[ix]:.4f}")
        print(f"  Precision: {precision[ix]:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  F3.2-Score: {best_score:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  PR-AUC: {pr_auc:.4f}")
        print(f"  FPR: {fpr:.4f}")
        print(f"  FNR: {fnr:.4f}")
        
        # Save best params
        with open(RESULTS_DIR / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=4)
        print(f"\nBest parameters saved to {RESULTS_DIR}/best_params.json")

if __name__ == "__main__":
    main()
