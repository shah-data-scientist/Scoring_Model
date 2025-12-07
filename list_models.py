import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
client = MlflowClient()

print("Listing all registered models:")
try:
    models = client.search_registered_models()
    for m in models:
        print(f"- Name: {m.name}")
        for v in m.latest_versions:
            print(f"  - Version: {v.version}, Run ID: {v.run_id}, Status: {v.status}")
except Exception as e:
    print(f"Error listing models: {e}")
