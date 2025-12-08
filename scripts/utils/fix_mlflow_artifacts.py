"""
Fix MLflow artifacts - properly log them to the database.

This script logs existing artifact files to MLflow so they appear in the UI.
"""
import warnings
warnings.filterwarnings('ignore')

import mlflow
from mlflow import MlflowClient
from pathlib import Path
import tempfile
import shutil

# Setup
PROJECT_ROOT = Path(__file__).parent.parent.parent
MLRUNS_DIR = PROJECT_ROOT / 'mlruns'
mlflow.set_tracking_uri(f"sqlite:///{MLRUNS_DIR}/mlflow.db")

def log_artifacts_for_run(run_id, run_name):
    """
    Log existing artifact files to MLflow database.

    Args:
        run_id: MLflow run ID
        run_name: Name of run for logging
    """
    print(f"\nProcessing: {run_name}")
    print(f"  Run ID: {run_id}")

    # Check if artifacts directory exists
    artifact_dir = MLRUNS_DIR / run_id[:2] / run_id / 'artifacts'

    if not artifact_dir.exists():
        print(f"  [SKIP] No artifacts directory found")
        return

    # Find artifact files
    artifact_files = list(artifact_dir.glob('*.png')) + list(artifact_dir.glob('*.csv'))

    if not artifact_files:
        print(f"  [SKIP] No artifact files found")
        return

    print(f"  Found {len(artifact_files)} artifact files")

    # Log artifacts to MLflow
    try:
        with mlflow.start_run(run_id=run_id):
            for artifact_file in artifact_files:
                mlflow.log_artifact(str(artifact_file))
                print(f"    [OK] Logged: {artifact_file.name}")

        print(f"  [SUCCESS] Logged {len(artifact_files)} artifacts to MLflow")

    except Exception as e:
        print(f"  [ERROR] Failed to log artifacts: {e}")


def main():
    """Main execution."""
    print("=" * 80)
    print("FIX MLFLOW ARTIFACTS")
    print("=" * 80)
    print("\nThis script will log existing artifact files to MLflow database")
    print("so they appear in the UI.\n")

    client = MlflowClient()

    # Get runs from feature engineering experiment
    exp = client.get_experiment_by_name('credit_scoring_feature_engineering_cv')
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.mean_roc_auc DESC", "metrics.cv_mean_roc_auc DESC"],
        max_results=20
    )

    print(f"Found {len(runs)} runs in experiment '{exp.name}'")

    # Process each run
    success_count = 0
    for run in runs:
        run_name = run.data.tags.get('mlflow.runName', 'Unnamed')

        # Check current artifact count
        current_artifacts = len(client.list_artifacts(run.info.run_id))

        if current_artifacts > 0:
            print(f"\n[SKIP] {run_name} - already has {current_artifacts} artifacts")
            continue

        log_artifacts_for_run(run.info.run_id, run_name)

        # Verify artifacts were logged
        new_count = len(client.list_artifacts(run.info.run_id))
        if new_count > 0:
            success_count += 1

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nSuccessfully logged artifacts for {success_count} runs")
    print("\nView them in MLflow UI: http://localhost:5000")


if __name__ == '__main__':
    main()
