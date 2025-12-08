"""Check MLflow status and artifact presence."""
from pathlib import Path
import mlflow
from mlflow import MlflowClient

# Check paths
print("=" * 80)
print("PATH CHECK")
print("=" * 80)
project_root = Path(__file__).parent.parent.parent
notebooks_mlruns = project_root / 'notebooks' / 'mlruns'
root_mlruns = project_root / 'mlruns'

print(f"\nProject root: {project_root}")
print(f"notebooks/mlruns exists: {notebooks_mlruns.exists()}")
if notebooks_mlruns.exists():
    print(f"  Size: {sum(f.stat().st_size for f in notebooks_mlruns.rglob('*') if f.is_file()) / 1024:.2f} KB")

print(f"root mlruns exists: {root_mlruns.exists()}")
if root_mlruns.exists():
    print(f"  Size: {sum(f.stat().st_size for f in root_mlruns.rglob('*') if f.is_file()) / 1024:.2f} KB")

# Check MLflow database
print("\n" + "=" * 80)
print("MLFLOW DATABASE CHECK")
print("=" * 80)

mlflow.set_tracking_uri(f"sqlite:///{root_mlruns}/mlflow.db")
client = MlflowClient()

# Get all experiments
experiments = client.search_experiments()
print(f"\nTotal experiments: {len(experiments)}")

for exp in experiments:
    if exp.lifecycle_stage == 'active':
        print(f"\nExperiment: {exp.name} (ID: {exp.experiment_id})")
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        print(f"  Total runs: {len(runs)}")

        runs_with_artifacts = []
        for run in runs:
            artifacts = client.list_artifacts(run.info.run_id)
            if len(artifacts) > 0:
                runs_with_artifacts.append((
                    run.data.tags.get('mlflow.runName', 'Unnamed'),
                    len(artifacts)
                ))

        print(f"  Runs with artifacts: {len(runs_with_artifacts)}")
        if runs_with_artifacts:
            print("  Sample runs with artifacts:")
            for name, count in runs_with_artifacts[:5]:
                print(f"    - {name}: {count} artifacts")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

if notebooks_mlruns.exists():
    print("\n[!] notebooks/mlruns still exists - consider removing it")
    print("    to avoid confusion about which location is canonical.")
