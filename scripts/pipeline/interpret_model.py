"""Model Interpretation Script

Generates Global and Local interpretation artifacts using SHAP and standard metrics.
Replicates logic from notebooks/05_model_interpretation.ipynb.

1. Loads the best model (local file or MLflow).
2. Performs comprehensive evaluation (ROC, PR, Confusion Matrix).
3. Generates Feature Importance plots.
4. Calculates SHAP values on a sample.
5. Generates Global SHAP plots (Summary, Bar).
6. Generates Local Explanations for 5 specific examples (Risky, Safe, Borderline).
7. Logs everything to MLflow.
"""

import sys
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import shap
import matplotlib.pyplot as plt
import re
import warnings

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import CONFIG
from src.evaluation import (
    evaluate_model,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_feature_importance
)

# Configuration
warnings.filterwarnings('ignore')
RANDOM_STATE = CONFIG['project']['random_state']
EXPERIMENT_NAME = CONFIG['mlflow']['experiment_names']['final_delivery']
MLFLOW_TRACKING_URI = CONFIG['mlflow']['tracking_uri']
DATA_DIR = Path(CONFIG['paths']['data'])
RESULTS_DIR = Path(CONFIG['paths']['results']) / 'interpretation'
MODELS_DIR = Path(CONFIG['paths']['models'])

def main():
    print("="*80)
    print("MODEL INTERPRETATION & EVALUATION")
    print("="*80)
    
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    
    # 1. Load Data
    print(f"Loading data from {DATA_DIR}...")
    # Using validation set as test set for this demonstration, as in the notebook
    X_train = pd.read_csv(DATA_DIR / 'X_train.csv')
    X_val = pd.read_csv(DATA_DIR / 'X_val.csv')
    y_val = pd.read_csv(DATA_DIR / 'y_val.csv').squeeze()
    
    # X_test_eval = X_val
    # y_test = y_val
    
    print(f"Data loaded. Evaluation set shape: {X_val.shape}")
    
    # 2. Load Model
    print("Loading model...")
    model_path = MODELS_DIR / 'best_xgboost_model.pkl'
    model = None
    
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            print(f"Loaded model from local file: {model_path}")
        except Exception as e:
            print(f"Failed to load local model: {e}")
    
    if model is None:
        print("Attempting to load from MLflow...")
        try:
            client = mlflow.tracking.MlflowClient()
            exp = client.get_experiment_by_name(EXPERIMENT_NAME)
            runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"], max_results=1)
            if runs:
                run_id = runs[0].info.run_id
                model_uri = f"runs:/{run_id}/final_model"
                model = mlflow.sklearn.load_model(model_uri)
                print(f"Loaded model from run {run_id}")
            else:
                print("No MLflow runs found.")
        except Exception as e:
            print(f"Failed to load from MLflow: {e}")
            
    if model is None:
        print("ERROR: Could not load any model. Exiting.")
        return

    with mlflow.start_run(run_name="model_interpretation_script") as run:
        
        # Debug: Check feature mismatch
        if hasattr(model, 'feature_names_in_'):
            print("\nDEBUG: Feature Verification")
            print(f"Model expects {len(model.feature_names_in_)} features.")
            print(f"Data has {len(X_val.columns)} features.")
            
            missing_in_data = set(model.feature_names_in_) - set(X_val.columns)
            if missing_in_data:
                print(f"FATAL: Data is missing {len(missing_in_data)} features expected by model:")
                print(list(missing_in_data)[:10], "...")
            else:
                print("OK: All model features are present in data.")
                
            # Align columns to match model order
            X_val = X_val[model.feature_names_in_]
            print("Reordered data columns to match model.")
        
        # 3. Comprehensive Evaluation
        print("\n" + "="*80)
        print("FINAL MODEL EVALUATION")
        print("="*80)
        
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        metrics = evaluate_model(y_val, y_pred, y_pred_proba, "Final_XGBoost")
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Save plots
        print("\nGenerating evaluation plots...")
        
        # ROC
        fig = plot_roc_curve(y_val, y_pred_proba, "Final Model")
        plt.savefig(RESULTS_DIR / "final_roc_curve.png", bbox_inches='tight')
        mlflow.log_artifact(str(RESULTS_DIR / "final_roc_curve.png"))
        plt.close()
        
        # PR
        fig = plot_precision_recall_curve(y_val, y_pred_proba, "Final Model")
        plt.savefig(RESULTS_DIR / "final_pr_curve.png", bbox_inches='tight')
        mlflow.log_artifact(str(RESULTS_DIR / "final_pr_curve.png"))
        plt.close()
        
        # Confusion Matrix
        fig = plot_confusion_matrix(y_val, y_pred, "Final Model", normalize=True)
        plt.savefig(RESULTS_DIR / "final_confusion_matrix.png", bbox_inches='tight')
        mlflow.log_artifact(str(RESULTS_DIR / "final_confusion_matrix.png"))
        plt.close()
        
        # 4. Feature Importance (Built-in)
        print("\nGenerating Feature Importance...")
        if hasattr(model, 'feature_importances_'):
            # Use model feature names if available, or fall back to X_val columns (which are now aligned)
            feat_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else X_val.columns.tolist()
            
            fig = plot_feature_importance(
                feat_names,
                model.feature_importances_,
                top_n=20,
                model_name="Optimized XGBoost"
            )
            plt.savefig(RESULTS_DIR / "feature_importance.png", bbox_inches='tight')
            mlflow.log_artifact(str(RESULTS_DIR / "feature_importance.png"))
            plt.close()
            
            # Save CSV
            fi_df = pd.DataFrame({
                'feature': feat_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            fi_df.to_csv(RESULTS_DIR / "feature_importance_ranking.csv", index=False)
            mlflow.log_artifact(str(RESULTS_DIR / "feature_importance_ranking.csv"))
        else:
            print("Model does not support built-in feature importance.")

        # 5. SHAP Analysis
        print("\n" + "="*80)
        print("SHAP ANALYSIS")
        print("="*80)
        
        print("Calculating SHAP values (1000 samples)...")
        explainer = shap.TreeExplainer(model)
        
        # Sample background data for speed
        sample_idx = np.random.choice(len(X_val), min(1000, len(X_val)), replace=False)
        X_explain = X_val.iloc[sample_idx]
        y_explain = y_val.iloc[sample_idx]
        
        shap_values = explainer.shap_values(X_explain)
        
        # Handle list vs array output (binary class)
        if isinstance(shap_values, list):
            shap_values_target = shap_values[1]
        else:
            shap_values_target = shap_values
            
        # Global Summary
        print("Generating Global SHAP Summary...")
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values_target, X_explain, show=False)
        plt.title("SHAP Summary Plot")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "shap_summary_plot.png")
        mlflow.log_artifact(str(RESULTS_DIR / "shap_summary_plot.png"))
        plt.close()
        
        # Bar Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_target, X_explain, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance (Mean |SHAP|)")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "shap_bar_plot.png")
        mlflow.log_artifact(str(RESULTS_DIR / "shap_bar_plot.png"))
        plt.close()
        
        # 6. Local Analysis (5 Examples)
        print("\n" + "="*80)
        print("LOCAL ANALYSIS (5 EXAMPLES)")
        print("="*80)
        
        # We need probabilities for the explanation set
        y_proba_explain = model.predict_proba(X_explain)[:, 1]
        
        # Select examples: 2 High Risk, 2 Low Risk, 1 Borderline
        # High Risk: Highest prob
        idx_high = np.argsort(y_proba_explain)[-2:]
        # Low Risk: Lowest prob
        idx_low = np.argsort(y_proba_explain)[:2]
        # Borderline: Closest to 0.35 (approx threshold) or median
        threshold = 0.35
        idx_border = np.argsort(np.abs(y_proba_explain - threshold))[:1]
        
        examples_indices = np.concatenate([idx_high[::-1], idx_low, idx_border]) # High desc, Low asc, Border
        labels = ['High Risk 1', 'High Risk 2', 'Safe 1', 'Safe 2', 'Borderline']
        
        # Prepare CSV data
        local_results = []
        
        for i, idx in enumerate(examples_indices):
            label = labels[i]
            prob = y_proba_explain[idx]
            print(f"Analyzing {label} (Prob: {prob:.4f})...")
            
            # Waterfall Plot
            exp = shap.Explanation(
                values=shap_values_target[idx],
                base_values=explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1],
                data=X_explain.iloc[idx],
                feature_names=X_explain.columns
            )
            
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(exp, show=False, max_display=10)
            plt.title(f"Local Explanation: {label}\nProb: {prob:.2%}")
            plt.tight_layout()
            
            filename = f"local_shap_{label.replace(' ', '_').lower()}.png"
            plt.savefig(RESULTS_DIR / filename)
            mlflow.log_artifact(str(RESULTS_DIR / filename))
            plt.close()
            
            # Save data to list
            # Get top 5 contributing features
            feature_impacts = pd.DataFrame({
                'feature': X_explain.columns,
                'shap_value': shap_values_target[idx],
                'feature_value': X_explain.iloc[idx].values
            })
            # Sort by absolute impact
            feature_impacts['abs_shap'] = feature_impacts['shap_value'].abs()
            top_features = feature_impacts.sort_values('abs_shap', ascending=False).head(5)
            
            top_feats_str = "; ".join([f"{row['feature']} ({row['feature_value']:.2f}): {row['shap_value']:.2f}" for _, row in top_features.iterrows()])
            
            local_results.append({
                'Example Type': label,
                'Probability': prob,
                'Prediction': int(prob > threshold), # Assuming default threshold for now
                'Top Contributing Features': top_feats_str
            })
            
        # Save Local Analysis CSV
        local_df = pd.DataFrame(local_results)
        local_df.to_csv(RESULTS_DIR / "local_examples_analysis.csv", index=False)
        mlflow.log_artifact(str(RESULTS_DIR / "local_examples_analysis.csv"))
        
        print(f"\n[SUCCESS] Local analysis saved to {RESULTS_DIR}/local_examples_analysis.csv")
        print("\nAll interpretation tasks completed successfully.")

if __name__ == "__main__":
    main()