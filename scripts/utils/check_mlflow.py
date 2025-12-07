import mlflow
from pathlib import Path

# Set tracking URI to the one used in experiments
db_path = Path("notebooks/mlruns/mlflow.db")
tracking_uri = f"sqlite:///{db_path}"

print(f"Checking MLflow database at: {db_path.absolute()}")
print(f"Exists: {db_path.exists()}")

mlflow.set_tracking_uri(tracking_uri)

try:
    experiments = mlflow.search_experiments()
    print(f"\nFound {len(experiments)} experiments:")
    
    for exp in experiments:
        print(f"\nExperiment: {exp.name} (ID: {exp.experiment_id})")
        print(f"  Artifact Location: {exp.artifact_location}")
        
        # List runs
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        print(f"  Total Runs: {len(runs)}")
        
        if not runs.empty:
            # Show top runs by ROC-AUC if available
            if 'metrics.roc_auc' in runs.columns:
                best_run = runs.sort_values('metrics.roc_auc', ascending=False).iloc[0]
                print(f"  Best Run ROC-AUC: {best_run['metrics.roc_auc']:.4f}")
            elif 'metrics.mean_roc_auc' in runs.columns:
                 best_run = runs.sort_values('metrics.mean_roc_auc', ascending=False).iloc[0]
                 print(f"  Best Run Mean ROC-AUC: {best_run['metrics.mean_roc_auc']:.4f}")
            
            print(f"  Recent Run Names: {runs['tags.mlflow.runName'].head(3).tolist()}")

except Exception as e:
    print(f"\n[ERROR] Could not read from MLflow: {e}")
