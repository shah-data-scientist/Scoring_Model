# Example MLflow Tracking Script

import mlflow

if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
    mlflow.set_experiment("example_experiment")
    with mlflow.start_run():
        mlflow.log_param("param1", 5)
        mlflow.log_metric("metric1", 0.85)
        print("MLflow tracking example completed.")
