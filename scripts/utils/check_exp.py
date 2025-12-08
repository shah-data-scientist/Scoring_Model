import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///../../mlruns/mlflow.db")
client = MlflowClient()

exp = client.get_experiment("2")
print(f"Experiment 2 Artifact Location: {exp.artifact_location}")
