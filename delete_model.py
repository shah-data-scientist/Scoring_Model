import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
client = MlflowClient()

model_name = "CreditScoringModel"

try:
    print(f"Attempting to delete registered model: {model_name}")
    client.delete_registered_model(name=model_name)
    print(f"Successfully deleted registered model: {model_name}")
except Exception as e:
    print(f"Error deleting registered model {model_name}: {e}")
