"""
Comprehensive Overfitting/Underfitting Analysis

Analyzes all MLflow experiments to check for:
1. Overfitting: Training performance >> Validation performance
2. Underfitting: Both training and validation performance are poor
3. Good fit: Training and validation performance are similar

Also answers key questions:
- Did we use SMOTE undersampling? (Clarification: SMOTE is OVERSAMPLING)
- Which sampling strategies were tested?
- Was hyperparameter optimization done on the best model?
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import MLFLOW_TRACKING_URI

# Configuration
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()


def analyze_experiment_for_overfitting(experiment_name):
    """Analyze an experiment for overfitting/underfitting."""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*80}")

    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Experiment '{experiment_name}' not found!")
        return None

    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=["metrics.roc_auc DESC"]
    )

    if not runs:
        print(f"No runs found in experiment '{experiment_name}'")
        return None

    results = []
    for run in runs:
        run_name = run.data.tags.get("mlflow.runName", "Unknown")
        metrics = run.data.metrics
        params = run.data.params

        # Check if we have training metrics (some runs might not have them)
        train_roc_auc = metrics.get("train_roc_auc", metrics.get("training_roc_auc", None))
        val_roc_auc = metrics.get("roc_auc", metrics.get("val_roc_auc", metrics.get("validation_roc_auc", None)))

        if train_roc_auc is not None and val_roc_auc is not None:
            # Calculate overfitting gap
            gap = train_roc_auc - val_roc_auc

            # Categorize
            if gap > 0.05:  # More than 5% difference
                status = "OVERFITTING"
            elif val_roc_auc < 0.70:  # Poor validation performance
                status = "UNDERFITTING"
            else:
                status = "GOOD FIT"

            results.append({
                'run_name': run_name,
                'train_roc_auc': train_roc_auc,
                'val_roc_auc': val_roc_auc,
                'gap': gap,
                'status': status,
                'feature_strategy': run.data.tags.get("feature_strategy", "N/A"),
                'sampling_strategy': run.data.tags.get("sampling_strategy", "N/A"),
            })
        else:
            # No training metrics available
            results.append({
                'run_name': run_name,
                'train_roc_auc': None,
                'val_roc_auc': val_roc_auc,
                'gap': None,
                'status': "NO TRAIN METRICS",
                'feature_strategy': run.data.tags.get("feature_strategy", "N/A"),
                'sampling_strategy': run.data.tags.get("sampling_strategy", "N/A"),
            })

    if results:
        df = pd.DataFrame(results)
        print(f"\nTotal runs: {len(df)}")
        print(f"\nOverfitting Analysis:")
        print(df[['run_name', 'train_roc_auc', 'val_roc_auc', 'gap', 'status']].to_string(index=False))

        # Summary statistics
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        status_counts = df['status'].value_counts()
        for status, count in status_counts.items():
            print(f"{status:20s}: {count} runs")

        # Overfitting runs
        if "OVERFITTING" in status_counts:
            print(f"\nOVERFITTING DETECTED in {status_counts['OVERFITTING']} runs:")
            overfitting_runs = df[df['status'] == 'OVERFITTING']
            for _, row in overfitting_runs.iterrows():
                print(f"  {row['run_name']:50s} | Gap: {row['gap']:.4f} | Train: {row['train_roc_auc']:.4f} | Val: {row['val_roc_auc']:.4f}")

        return df
    else:
        print("No results to analyze")
        return None


def answer_key_questions():
    """Answer the user's critical questions."""
    print("\n" + "="*80)
    print("ANSWERING KEY QUESTIONS")
    print("="*80)

    # Question 1: SMOTE undersampling?
    print("\n1. SMOTE UNDERSAMPLING?")
    print("-" * 80)
    print("CLARIFICATION: SMOTE is OVERSAMPLING, not undersampling!")
    print("\nSampling strategies tested:")
    print("  - Balanced (class_weight='balanced') - NO resampling")
    print("  - SMOTE (Synthetic Minority Over-sampling) - OVERSAMPLING minority class")
    print("  - Random Undersampling - UNDERSAMPLING majority class")
    print("\nSMOTE creates synthetic samples for the minority class (class 1)")
    print("to balance the dataset, NOT reduce the majority class.")

    # Question 2: Sampling strategies
    print("\n2. SAMPLING STRATEGIES TESTED:")
    print("-" * 80)

    feature_exp = client.get_experiment_by_name("credit_scoring_feature_engineering")
    if feature_exp:
        runs = client.search_runs([feature_exp.experiment_id])
        sampling_strategies = set()
        for run in runs:
            strategy = run.data.tags.get("sampling_strategy", "N/A")
            if strategy != "N/A":
                sampling_strategies.add(strategy)

        print(f"Strategies tested in feature engineering experiment:")
        for strategy in sorted(sampling_strategies):
            strategy_runs = [r for r in runs if r.data.tags.get("sampling_strategy") == strategy]
            best_roc = max([r.data.metrics.get("roc_auc", 0) for r in strategy_runs])
            print(f"  - {strategy:15s}: {len(strategy_runs)} runs | Best ROC-AUC: {best_roc:.4f}")

    # Question 3: Hyperparameter optimization
    print("\n3. HYPERPARAMETER OPTIMIZATION:")
    print("-" * 80)

    # Check baseline optimization
    baseline_opt = client.get_experiment_by_name("credit_scoring_baseline_models")
    hyperparam_opt = client.get_experiment_by_name("credit_scoring_hyperparameter_optimization")

    if hyperparam_opt:
        opt_runs = client.search_runs([hyperparam_opt.experiment_id])
        print(f"\nExisting hyperparameter optimization experiment found:")
        print(f"  Experiment: credit_scoring_hyperparameter_optimization")
        print(f"  Runs: {len(opt_runs)}")
        for run in opt_runs:
            run_name = run.data.tags.get("mlflow.runName", "Unknown")
            roc_auc = run.data.metrics.get("roc_auc", run.data.metrics.get("val_roc_auc", 0))
            feature_strategy = run.data.tags.get("feature_strategy", "baseline")
            sampling_strategy = run.data.tags.get("sampling_strategy", "balanced")
            print(f"    - {run_name}: ROC-AUC {roc_auc:.4f} | Features: {feature_strategy} | Sampling: {sampling_strategy}")

    # Check feature engineering best
    feature_exp = client.get_experiment_by_name("credit_scoring_feature_engineering")
    if feature_exp:
        runs = client.search_runs(
            [feature_exp.experiment_id],
            order_by=["metrics.roc_auc DESC"],
            max_results=1
        )
        if runs:
            best_run = runs[0]
            best_roc = best_run.data.metrics.get("roc_auc", 0)
            best_feature = best_run.data.tags.get("feature_strategy", "N/A")
            best_sampling = best_run.data.tags.get("sampling_strategy", "N/A")

            print(f"\nBest configuration from feature engineering experiment:")
            print(f"  Features: {best_feature}")
            print(f"  Sampling: {best_sampling}")
            print(f"  ROC-AUC: {best_roc:.4f}")

            if hyperparam_opt:
                opt_runs = client.search_runs([hyperparam_opt.experiment_id], order_by=["metrics.roc_auc DESC"], max_results=1)
                if opt_runs:
                    opt_roc = opt_runs[0].data.metrics.get("roc_auc", opt_runs[0].data.metrics.get("val_roc_auc", 0))
                    opt_feature = opt_runs[0].data.tags.get("feature_strategy", "baseline")

                    print(f"\nCOMPARISON:")
                    print(f"  Hyperparameter optimization ({opt_feature}): {opt_roc:.4f}")
                    print(f"  Best feature engineering ({best_feature}+{best_sampling}): {best_roc:.4f}")

                    if opt_feature != best_feature or opt_runs[0].data.tags.get("sampling_strategy", "balanced") != best_sampling:
                        print(f"\n  WARNING: Hyperparameter optimization was done on '{opt_feature}',")
                        print(f"           but best configuration is '{best_feature}+{best_sampling}'")
                        print(f"\n  RECOMMENDATION: Run hyperparameter optimization on '{best_feature}+{best_sampling}'")
                    else:
                        print(f"\n  Hyperparameter optimization was done on the best configuration!")


def main():
    """Run comprehensive overfitting analysis."""
    print("="*80)
    print("COMPREHENSIVE OVERFITTING/UNDERFITTING ANALYSIS")
    print("="*80)

    # List all experiments
    experiments = client.search_experiments()
    print(f"\nFound {len(experiments)} experiments:")
    for exp in experiments:
        print(f"  - {exp.name} (ID: {exp.experiment_id})")

    # Analyze each experiment
    all_results = {}
    for exp in experiments:
        if exp.name != "Default":
            result = analyze_experiment_for_overfitting(exp.name)
            if result is not None:
                all_results[exp.name] = result

    # Answer key questions
    answer_key_questions()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

    # Overall recommendation
    print("\nOVERALL RECOMMENDATIONS:")
    print("1. Review runs with OVERFITTING status and consider:")
    print("   - Reducing model complexity (lower max_depth, fewer n_estimators)")
    print("   - Adding regularization (higher reg_alpha, reg_lambda)")
    print("   - Increasing min_child_weight")
    print("\n2. If hyperparameter optimization wasn't done on best config:")
    print("   - Run optimization on domain+balanced configuration")
    print("\n3. SMOTE performed poorly - continue using balanced class weights")


if __name__ == "__main__":
    main()
