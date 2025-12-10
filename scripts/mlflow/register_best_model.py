"""
Re-register the best model to MLflow Model Registry.

This script finds the best run by ROC-AUC and registers it as the production model.

Usage:
    poetry run python scripts/mlflow/register_best_model.py
"""

import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
MLRUNS_DIR = PROJECT_ROOT / "mlruns"


def main():
    """Re-register the best model."""
    print("=" * 80)
    print("RE-REGISTERING BEST MODEL TO MLFLOW REGISTRY")
    print("=" * 80)
    print()

    # Set tracking URI
    mlflow.set_tracking_uri(f"sqlite:///{MLRUNS_DIR}/mlflow.db")
    client = MlflowClient()

    # Get all runs sorted by ROC-AUC
    print("Searching for best model...")
    experiments = client.search_experiments()
    all_runs = []

    for exp in experiments:
        try:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["metrics.roc_auc DESC"],
                max_results=100
            )
            all_runs.extend(runs)
        except Exception as e:
            print(f"WARNING: Error searching experiment {exp.name}: {e}")

    if not all_runs:
        print("ERROR: No runs found in MLflow!")
        print("\nYou may need to re-run model training:")
        print("   poetry run python scripts/pipeline/apply_best_model.py")
        return

    # Find best run
    best_run = None
    for run in all_runs:
        if 'roc_auc' in run.data.metrics:
            best_run = run
            break

    if not best_run:
        print("ERROR: No runs with roc_auc metric found!")
        return

    run_id = best_run.info.run_id
    roc_auc = best_run.data.metrics.get('roc_auc', 0)
    precision = best_run.data.metrics.get('precision', 0)
    recall = best_run.data.metrics.get('recall', 0)

    print(f"\nSUCCESS: Found best model:")
    print(f"   Run ID: {run_id}")
    print(f"   ROC-AUC: {roc_auc:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")

    # Check if model artifact exists
    try:
        artifacts = client.list_artifacts(run_id)
        model_artifact = None
        for artifact in artifacts:
            if artifact.path == "model" or "model" in artifact.path:
                model_artifact = artifact.path
                break

        if not model_artifact:
            print(f"\nWARNING: No model artifact found in run {run_id[:8]}...")
            print("   Looking for runs with model artifacts...")

            # Try to find another run with model artifact
            for run in all_runs:
                try:
                    artifacts = client.list_artifacts(run.info.run_id)
                    for artifact in artifacts:
                        if artifact.path == "model" or "model" in artifact.path:
                            run_id = run.info.run_id
                            roc_auc = run.data.metrics.get('roc_auc', 0)
                            model_artifact = artifact.path
                            print(f"\nSUCCESS: Found run with model artifact:")
                            print(f"   Run ID: {run_id}")
                            print(f"   ROC-AUC: {roc_auc:.4f}")
                            break
                    if model_artifact:
                        break
                except:
                    continue

        if not model_artifact:
            print("\nERROR: No runs with model artifacts found!")
            print("   You need to re-train the model with MLflow logging.")
            return

    except Exception as e:
        print(f"\nWARNING: Error checking artifacts: {e}")
        model_artifact = "model"  # Try default path

    # Register model
    print(f"\nRegistering model to registry...")
    model_name = "CreditScoringModel"

    try:
        # Try to get existing registered model
        try:
            registered_model = client.get_registered_model(model_name)
            print(f"   Found existing model: {model_name}")
        except:
            # Create new registered model
            client.create_registered_model(
                model_name,
                description="Credit scoring model for loan default prediction"
            )
            print(f"   Created new model: {model_name}")

        # Register this version
        model_uri = f"runs:/{run_id}/{model_artifact}"
        model_version = mlflow.register_model(model_uri, model_name)

        print(f"\nSUCCESS: Model registered successfully!")
        print(f"   Model Name: {model_name}")
        print(f"   Version: {model_version.version}")
        print(f"   Run ID: {run_id}")

        # Transition to production
        print(f"\nTransitioning to Production stage...")
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production",
            archive_existing_versions=True
        )

        print(f"SUCCESS: Model is now in Production stage!")

        # Add description
        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description=f"LightGBM model with ROC-AUC: {roc_auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"
        )

        print("\n" + "=" * 80)
        print("REGISTRATION COMPLETE")
        print("=" * 80)
        print(f"\nModel Details:")
        print(f"   Name: {model_name}")
        print(f"   Version: {model_version.version}")
        print(f"   Stage: Production")
        print(f"   ROC-AUC: {roc_auc:.4f}")
        print(f"\nSUCCESS: The API should now work correctly!")

    except Exception as e:
        print(f"\nERROR: Error registering model: {e}")
        print("\nTroubleshooting:")
        print("1. Check MLflow UI: http://localhost:5000")
        print("2. Verify runs exist with model artifacts")
        print("3. Try re-running model training")


if __name__ == "__main__":
    main()
