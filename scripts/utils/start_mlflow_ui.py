"""
Start MLflow UI with correct backend URI.

This script starts the MLflow UI pointing to the root mlruns database.
"""
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
MLFLOW_DB = PROJECT_ROOT / 'mlruns' / 'mlflow.db'

print("=" * 80)
print("STARTING MLFLOW UI")
print("=" * 80)
print(f"\nBackend URI: sqlite:///{MLFLOW_DB}")
print(f"UI URL: http://localhost:5000")
print("\nPress Ctrl+C to stop the server")
print("=" * 80)

# Start MLflow UI
cmd = [
    'poetry', 'run', 'mlflow', 'ui',
    '--backend-store-uri', f'sqlite:///{MLFLOW_DB}',
    '--host', '0.0.0.0',
    '--port', '5000'
]

subprocess.run(cmd)
