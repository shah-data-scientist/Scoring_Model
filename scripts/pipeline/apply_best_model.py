"""
Apply Optimized Model to Test Data

1. Loads best hyperparameters.
2. Retrains model on full training data (Train + Validation).
3. Determines optimal threshold using cross-validation on training data.
4. Generates predictions for the Kaggle test set (X_test).
5. Saves predictions and feature importance.
6. Logs all artifacts (including ROC/PR curves and Confusion Matrix) to MLflow.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import mlflow
import mlflow.sklearn
import re

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.domain_features import create_domain_features
from src.config import CONFIG

# Configuration from YAML
RANDOM_STATE = CONFIG['project']['random_state']
N_FOLDS = CONFIG['project']['n_folds']
BETA = CONFIG['business']['f_beta']
EXPERIMENT_NAME = CONFIG['mlflow']['experiment_names']['final_delivery']
MLFLOW_TRACKING_URI = CONFIG['mlflow']['tracking_uri']
DATA_DIR = Path(CONFIG['paths']['data'])
RESULTS_DIR = Path(CONFIG['paths']['results'])

def get_optimal_threshold(y_true, y_proba, beta=BETA):
    """Find optimal threshold maximizing F-beta."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Calculate F-beta for all thresholds
    numerator = (1 + beta**2) * precision * recall
    denominator = (beta**2 * precision) + recall
    
    # Avoid div by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        fbeta = np.divide(numerator, denominator)
    fbeta = np.nan_to_num(fbeta)
    
    ix = np.argmax(fbeta)
    return thresholds[ix] if ix < len(thresholds) else 0.5

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

def plot_confusion_matrix(cm, title='Confusion Matrix'):
    """Create confusion matrix plots (Counts and Percentages)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')
    axes[0].set_title(f'{title} (Counts)')
    
    # 2. Percentages (Normalized by Row)
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_pct, annot=True, fmt='.2%', cmap='Greens', ax=axes[1],
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    axes[1].set_ylabel('Actual')
    axes[1].set_xlabel('Predicted')
    axes[1].set_title(f'{title} (Row %)')
    
    plt.tight_layout()
    return fig

def main():
    print("="*80)
    print("APPLYING BEST MODEL TO TEST DATA")
    print("="*80)
    
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Ensure artifact path is absolute and correct URI
    artifact_path = Path("mlruns").resolve().as_uri()
    
    try:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_path)
    except mlflow.exceptions.MlflowException:
        pass # Experiment already exists
        
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name="final_model_application") as run:
        # 1. Load Data
        print(f"Loading data from {DATA_DIR}...")
        RESULTS_DIR.mkdir(exist_ok=True)
        
        X_train = pd.read_csv(DATA_DIR / 'X_train.csv')
        y_train = pd.read_csv(DATA_DIR / 'y_train.csv').squeeze()
        X_val = pd.read_csv(DATA_DIR / 'X_val.csv')
        y_val = pd.read_csv(DATA_DIR / 'y_val.csv').squeeze()
        X_test = pd.read_csv(DATA_DIR / 'X_test.csv')
        test_ids = pd.read_csv(DATA_DIR / 'test_ids.csv')
        
        # Combine Train and Val for final training
        print("Combining training and validation sets for final training...")
        X_full = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
        y_full = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
        
        # 2. Feature Engineering
        print("Applying feature engineering...")
        X_full_proc = create_domain_features(X_full)
        X_test_proc = create_domain_features(X_test)
        
        clean_name = lambda x: re.sub('[^A-Za-z0-9_]+', '', x)
        X_full_proc = X_full_proc.rename(columns=clean_name)
        X_test_proc = X_test_proc.rename(columns=clean_name)
        
        common_cols = X_full_proc.columns.intersection(X_test_proc.columns)
        X_full_proc = X_full_proc[common_cols]
        X_test_proc = X_test_proc[common_cols]
        
        # 3. Load Best Hyperparameters
        params_path = RESULTS_DIR / 'best_params.json'
        if not params_path.exists():
            print(f"[ERROR] {params_path} not found.")
            return
            
        with open(params_path, 'r') as f:
            best_params = json.load(f)
        
        best_params['class_weight'] = 'balanced'
        best_params['random_state'] = RANDOM_STATE
        best_params['n_jobs'] = 4
        best_params['verbose'] = -1
        
        mlflow.log_params(best_params)
        
        # 4. Determine Optimal Threshold
        print("\nDetermining optimal threshold using 5-fold CV on full training data...")
        model = LGBMClassifier(**best_params)
        
        # Use n_jobs=1 to avoid WinError 1450
        y_proba_cv = cross_val_predict(
            model, X_full_proc, y_full, 
            cv=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            method='predict_proba',
            n_jobs=1
        )[:, 1]
        
        optimal_threshold = get_optimal_threshold(y_full, y_proba_cv, beta=BETA)
        print(f"Optimal Threshold: {optimal_threshold:.4f}")
        mlflow.log_metric("optimal_threshold", optimal_threshold)
        
        # --- SAVE TRAIN PREDICTIONS FOR DASHBOARD ---
        print("Saving training predictions for dashboard...")
        train_preds_df = pd.DataFrame({
            'TARGET': y_full,
            'PROBABILITY': y_proba_cv
        })
        train_preds_df.to_csv(RESULTS_DIR / 'train_predictions.csv', index=False)
        mlflow.log_artifact(str(RESULTS_DIR / 'train_predictions.csv'))
        # --------------------------------------------
        
        # --- PLOTS ---
        print("Generating evaluation plots...")
        
        # Confusion Matrix
        y_pred_cv = (y_proba_cv >= optimal_threshold).astype(int)
        cm = confusion_matrix(y_full, y_pred_cv)
        fig_cm = plot_confusion_matrix(cm, title=f'Confusion Matrix @ {optimal_threshold:.3f}')
        fig_cm.savefig(RESULTS_DIR / 'confusion_matrix.png')
        mlflow.log_artifact(str(RESULTS_DIR / 'confusion_matrix.png'))
        plt.close(fig_cm)
        
        # ROC Curve
        fig_roc = plot_roc_curve(y_full, y_proba_cv)
        fig_roc.savefig(RESULTS_DIR / 'roc_curve.png')
        mlflow.log_artifact(str(RESULTS_DIR / 'roc_curve.png'))
        plt.close(fig_roc)
        
        # PR Curve
        fig_pr = plot_pr_curve(y_full, y_proba_cv)
        fig_pr.savefig(RESULTS_DIR / 'pr_curve.png')
        mlflow.log_artifact(str(RESULTS_DIR / 'pr_curve.png'))
        plt.close(fig_pr)
        
        # 5. Train Final Model
        print("\nTraining final model on all data...")
        model.fit(X_full_proc, y_full)
        
        # Log and Register Model
        mlflow.sklearn.log_model(
            model, 
            "final_model",
            registered_model_name="CreditScoringModel"
        )
        print("Model registered as 'CreditScoringModel'.")
        
        # 6. Generate Predictions on Test Set
        print("Generating predictions on test set...")
        test_proba = model.predict_proba(X_test_proc)[:, 1]
        test_pred = (test_proba >= optimal_threshold).astype(int)
        
        # 7. Save Submission
        submission = pd.DataFrame({
            'SK_ID_CURR': test_ids['SK_ID_CURR'],
            'TARGET': test_proba,
            'PREDICTION': test_pred
        })
        
        submission_path = RESULTS_DIR / 'submission.csv'
        submission.to_csv(submission_path, index=False)
        print(f"\nPredictions saved to: {submission_path}")
        mlflow.log_artifact(str(submission_path))
        
        # 8. Feature Importance
        print("\nSaving feature importance...")
        
        # Known domain engineered features
        domain_cols = {
            'AGE_YEARS', 'EMPLOYMENT_YEARS', 'IS_EMPLOYED', 'INCOME_PER_PERSON', 
            'DEBT_TO_INCOME_RATIO', 'CREDIT_TO_GOODS_RATIO', 'ANNUITY_TO_INCOME_RATIO', 
            'CREDIT_UTILIZATION', 'HAS_CHILDREN', 'CHILDREN_RATIO', 
            'TOTAL_DOCUMENTS_PROVIDED', 'EXT_SOURCE_MEAN', 'EXT_SOURCE_MAX', 
            'EXT_SOURCE_MIN', 'REGION_RATING_COMBINED'
        }
        
        # Aggregation prefixes
        agg_prefixes = ('BUREAU_', 'PREV_', 'POS_', 'CC_', 'INST_', 'BB_')
        
        feature_names = []
        for col in X_full_proc.columns:
            if col in domain_cols:
                feature_names.append(f"{col} (Domain)")
            elif col.startswith(agg_prefixes):
                feature_names.append(f"{col} (Aggregated)")
            else:
                feature_names.append(f"{col} (Raw)")
        
        importance = pd.DataFrame({
            'feature': feature_names,
            'raw_feature': X_full_proc.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fi_csv_path = RESULTS_DIR / 'feature_importance.csv'
        importance[['feature', 'importance']].head(50).to_csv(fi_csv_path, index=False)
        mlflow.log_artifact(str(fi_csv_path))
        
        plt.figure(figsize=(12, 10))
        sns.barplot(data=importance.head(20), x='importance', y='feature', palette='viridis')
        plt.title('Top 20 Features (with Type)')
        plt.tight_layout()
        fi_png_path = RESULTS_DIR / 'feature_importance.png'
        plt.savefig(fi_png_path)
        mlflow.log_artifact(str(fi_png_path))
        
        print(f"\n[SUCCESS] All artifacts logged to MLflow run: {run.info.run_id}")

if __name__ == "__main__":
    main()