"""
Model Selection Experiment

Compares different model architectures using the Baseline Feature set (no feature engineering).
Goal: Confirm the best model architecture before final optimization.

Models:
1. Dummy Classifier (Baseline)
2. Logistic Regression (Linear)
3. Random Forest (Ensemble)
4. XGBoost (Gradient Boosting)
5. LightGBM (Gradient Boosting)

Settings:
- Baseline Features (Raw processed data)
- Balanced Class Weights
- 5-Fold CV
- F3.2 Score (Recall priority)
- Logs comprehensive metrics (Accuracy, Precision, Recall, F1, PR-AUC)
- Saves ROC and PR curves for comparison.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, fbeta_score, precision_recall_curve, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score, average_precision_score,
    roc_curve, auc
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import re

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import CONFIG

# Configuration from YAML
RANDOM_STATE = CONFIG['project']['random_state']
N_FOLDS = CONFIG['project']['n_folds']
BETA = CONFIG['business']['f_beta']
EXPERIMENT_NAME = CONFIG['mlflow']['experiment_names']['model_selection']
MLFLOW_TRACKING_URI = CONFIG['mlflow']['tracking_uri']
DATA_DIR = Path(CONFIG['paths']['data'])
RESULTS_DIR = Path(CONFIG['paths']['results'])

def get_optimal_fbeta(y_true, y_proba, beta=BETA):
    """Find max F-beta and threshold."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    numerator = (1 + beta**2) * precision * recall
    denominator = (beta**2 * precision) + recall
    with np.errstate(divide='ignore', invalid='ignore'):
        fbeta = np.divide(numerator, denominator)
    fbeta = np.nan_to_num(fbeta)
    
    ix = np.argmax(fbeta)
    best_score = fbeta[ix]
    best_thresh = thresholds[ix] if ix < len(thresholds) else 0.5
    
    # Cost
    y_pred = (y_proba >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cost = CONFIG['business']['cost_fn'] * fn + CONFIG['business']['cost_fp'] * fp
    
    return best_score, cost, best_thresh

def plot_roc_curve(y_true, y_proba, title='ROC Curve'):
    """Plot ROC Curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    return fig

def plot_pr_curve(y_true, y_proba, title='Precision-Recall Curve'):
    """Plot Precision-Recall Curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='blue', lw=2, label='PR curve')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc="lower left")
    return fig

def run_model_cv(model, model_name, X, y):
    """Run CV for a model."""
    print(f"\nRunning {model_name}...")
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    metrics = {
        'roc_auc': [],
        'fbeta': [],
        'cost': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'pr_auc': []
    }
    
    # Collect OOF predictions for plotting
    y_oof = np.zeros(len(y))
    
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("feature_set", "Baseline Features")
        mlflow.log_param("beta", BETA)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_train, y_train)
            
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_val)[:, 1]
            else:
                y_proba = model.predict(X_val)
            
            # Store predictions
            y_oof[val_idx] = y_proba
                
            # Metrics at Optimal Threshold (per fold)
            fbeta, cost, thresh = get_optimal_fbeta(y_val, y_proba, BETA)
            y_pred = (y_proba >= thresh).astype(int)
            
            metrics['roc_auc'].append(roc_auc_score(y_val, y_proba))
            metrics['fbeta'].append(fbeta)
            metrics['cost'].append(cost)
            metrics['pr_auc'].append(average_precision_score(y_val, y_proba))
            metrics['accuracy'].append(accuracy_score(y_val, y_pred))
            metrics['precision'].append(precision_score(y_val, y_pred))
            metrics['recall'].append(recall_score(y_val, y_pred))
            metrics['f1'].append(f1_score(y_val, y_pred))
            
            print(f"  Fold {fold+1}: ROC={metrics['roc_auc'][-1]:.4f}, F{BETA}={fbeta:.4f}")
            
        # Log Model
        model.fit(X, y)
        mlflow.sklearn.log_model(model, "model")
        
        # Log Plots (Using OOF predictions)
        RESULTS_DIR.mkdir(exist_ok=True)
        fig_roc = plot_roc_curve(y, y_oof, title=f'ROC Curve - {model_name}')
        fig_roc.savefig(RESULTS_DIR / f'{model_name}_roc.png')
        mlflow.log_artifact(str(RESULTS_DIR / f'{model_name}_roc.png'))
        plt.close(fig_roc)
        
        fig_pr = plot_pr_curve(y, y_oof, title=f'PR Curve - {model_name}')
        fig_pr.savefig(RESULTS_DIR / f'{model_name}_pr.png')
        mlflow.log_artifact(str(RESULTS_DIR / f'{model_name}_pr.png'))
        plt.close(fig_pr)
            
        # Log Aggregated Metrics
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        
        mlflow.log_metric("mean_roc_auc", avg_metrics['roc_auc'])
        mlflow.log_metric("mean_fbeta_score", avg_metrics['fbeta'])
        mlflow.log_metric("mean_business_cost", avg_metrics['cost'])
        mlflow.log_metric("mean_pr_auc", avg_metrics['pr_auc'])
        mlflow.log_metric("mean_accuracy", avg_metrics['accuracy'])
        mlflow.log_metric("mean_precision", avg_metrics['precision'])
        mlflow.log_metric("mean_recall", avg_metrics['recall'])
        mlflow.log_metric("mean_f1_score", avg_metrics['f1'])
        
        print(f"  Average: ROC={avg_metrics['roc_auc']:.4f}, F{BETA}={avg_metrics['fbeta']:.4f}")
        return avg_metrics['roc_auc'], avg_metrics['fbeta'], avg_metrics['cost'], y_oof

def plot_combined_curves(results_data, y_true):
    """Plot combined ROC and PR curves for all models."""
    # ROC
    plt.figure(figsize=(10, 8))
    for model_name, y_oof in results_data:
        fpr, tpr, _ = roc_curve(y_true, y_oof)
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Model Comparison - ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig(RESULTS_DIR / 'model_comparison_roc.png')
    print(f"Saved combined ROC curve to {RESULTS_DIR / 'model_comparison_roc.png'}")
    
    # PR
    plt.figure(figsize=(10, 8))
    for model_name, y_oof in results_data:
        precision, recall, _ = precision_recall_curve(y_true, y_oof)
        avg_prec = average_precision_score(y_true, y_oof)
        plt.plot(recall, precision, lw=2, label=f'{model_name} (AP = {avg_prec:.3f})')
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Model Comparison - Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.savefig(RESULTS_DIR / 'model_comparison_pr.png')
    print(f"Saved combined PR curve to {RESULTS_DIR / 'model_comparison_pr.png'}")

def main():
    print("="*80)
    print("MODEL SELECTION EXPERIMENT (Configured)")
    print("="*80)
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # 1. Load Data
    print(f"Loading data from {DATA_DIR}...")
    X = pd.read_csv(DATA_DIR / 'X_train.csv')
    y = pd.read_csv(DATA_DIR / 'y_train.csv').squeeze()
    
    # 2. Features (Baseline only)
    print("Using Baseline Features (no domain engineering)...")
    
    # Clean names for LGBM/XGB
    X = X.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    
    models = [
        (
            "Dummy Classifier",
            DummyClassifier(strategy='most_frequent')
        ),
        (
            "Logistic Regression",
            Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE))
            ])
        ),
        (
            "Random Forest",
            Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('clf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1))
            ])
        ),
        (
            "XGBoost",
            XGBClassifier(
                n_estimators=100, 
                scale_pos_weight=(len(y[y==0])/len(y[y==1])), # Manual balanced weight for XGB
                random_state=RANDOM_STATE, 
                n_jobs=-1,
                verbosity=0
            )
        ),
        (
            "LightGBM",
            LGBMClassifier(
                n_estimators=100, 
                class_weight='balanced', 
                random_state=RANDOM_STATE, 
                n_jobs=-1, 
                verbose=-1
            )
        )
    ]
    
    results = []
    model_predictions = [] # Store OOF predictions for combined plotting
    
    for name, model in models:
        roc, fbeta, cost, y_oof = run_model_cv(model, name, X, y)
        results.append({'Model': name, 'ROC-AUC': roc, 'F3.2': fbeta, 'Cost': cost})
        model_predictions.append((name, y_oof))
        
    # Create combined plots
    plot_combined_curves(model_predictions, y)
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(pd.DataFrame(results).sort_values('F3.2', ascending=False))

if __name__ == "__main__":
    main()
