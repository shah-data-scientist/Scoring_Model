
import sys
from pathlib import Path
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

# Mocking the setup from comparison_dashboard.py
sys.path.append(str(Path.cwd()))

try:
    from src.config import MLFLOW_TRACKING_URI
    print(f"MLFLOW_TRACKING_URI: {MLFLOW_TRACKING_URI}")
except ImportError as e:
    print(f"Error importing config: {e}")
    sys.exit(1)

def test_load_mlflow_data():
    print("Attempting to load MLflow data...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    try:
        experiments = client.search_experiments()
        print(f"Found {len(experiments)} experiments.")
        
        all_runs = []
        for exp in experiments:
            if exp.lifecycle_stage == 'deleted':
                continue
            
            print(f"Scanning experiment: {exp.name} (ID: {exp.experiment_id})")
            runs = client.search_runs([exp.experiment_id])
            print(f"  Found {len(runs)} runs.")
            
            for run in runs:
                # partial simulation of data extraction
                run_data = {
                    'experiment': exp.name,
                    'run_name': run.data.tags.get('mlflow.runName', 'Unnamed'),
                    'run_id': run.info.run_id,
                    'status': run.info.status,
                }
                # Check metric access
                for key, value in run.data.metrics.items():
                    pass 
                all_runs.append(run_data)
                
        print(f"Total runs processed: {len(all_runs)}")
        
    except Exception as e:
        print(f"Error during MLflow operations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_load_mlflow_data()
