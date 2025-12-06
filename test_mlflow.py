"""
Quick test to verify MLflow is working correctly.
Run this to create a test experiment and verify MLflow UI shows it.
"""

import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

print("="*80)
print("MLFLOW QUICK TEST")
print("="*80)

# Set experiment
experiment_name = "test_experiment"
mlflow.set_experiment(experiment_name)
print(f"\n[OK] Experiment set: {experiment_name}")

# Create dummy data
print("\n[OK] Creating dummy dataset...")
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run
print("\n[OK] Starting MLflow run...")
with mlflow.start_run(run_name="test_model_run"):

    # Log parameters
    n_estimators = 10
    max_depth = 5
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    print(f"  - Logged parameters: n_estimators={n_estimators}, max_depth={max_depth}")

    # Train simple model
    print("\n[OK] Training test model...")
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("roc_auc", roc_auc)
    print(f"  - Logged metrics: accuracy={accuracy:.4f}, roc_auc={roc_auc:.4f}")

    # Log model
    mlflow.sklearn.log_model(model, "model")
    print("  - Logged model")

    # Get run info
    run_id = mlflow.active_run().info.run_id
    print(f"\n[OK] Run ID: {run_id}")

print("\n" + "="*80)
print("SUCCESS! MLflow test complete!")
print("="*80)
print("\nNow check MLflow UI:")
print("1. Make sure MLflow UI is running: mlflow ui")
print("2. Open: http://localhost:5000")
print("3. You should see:")
print("   - Experiment: 'test_experiment'")
print("   - 1 run: 'test_model_run'")
print("   - Metrics: accuracy and roc_auc")
print("   - Parameters: n_estimators and max_depth")
print("\nIf you see this, MLflow is working correctly!")
print("="*80)
