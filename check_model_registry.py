import mlflow
from mlflow.tracking import MlflowClient
import os

mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
client = MlflowClient()

model_name = "credit_scoring_production_model"

try:
    print(f"Inspecting Registered Model: {model_name}")
    versions = client.get_latest_versions(model_name)
    
    for v in versions:
        print(f"\nVersion: {v.version}")
        print(f"  Run ID: {v.run_id}")
        print(f"  Source: {v.source}")
        print(f"  Status: {v.status}")
        
        # Verify run exists
        try:
            run = client.get_run(v.run_id)
            print(f"  Run Found: Yes")
            print(f"  Run Artifact URI: {run.info.artifact_uri}")
            print(f"  Run Metrics: {list(run.data.metrics.keys())}")
            
            # List artifacts
            artifacts = client.list_artifacts(v.run_id)
            print(f"  All Artifacts in Run:")
            for art in artifacts:
                print(f"    - {art.path}")
                
        except Exception as e:
            print(f"  Run Lookup Error: {e}")

except Exception as e:
    print(f"Error: {e}")
