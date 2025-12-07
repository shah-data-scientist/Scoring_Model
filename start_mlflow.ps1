# Script to launch MLflow UI with correct configuration

Write-Host "Starting MLflow UI..."
Write-Host "Database: mlruns/mlflow.db"
Write-Host "Artifacts: mlruns"

# Ensure poetry environment is used
poetry run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root mlruns --host 127.0.0.1 --port 5000
