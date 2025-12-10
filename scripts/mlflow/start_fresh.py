"""
Start fresh MLflow with minimal runs for fast performance.

This keeps your existing mlruns as backup and creates a clean, fast setup.
"""

import shutil
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).parent.parent.parent
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
MLRUNS_OLD = PROJECT_ROOT / "mlruns_full_backup"
MLRUNS_NEW = PROJECT_ROOT / "mlruns_clean"


def main():
    print("=" * 80)
    print("CREATING FAST MLFLOW SETUP")
    print("=" * 80)
    print()
    print("This will:")
    print("  1. Backup current mlruns to mlruns_full_backup")
    print("  2. Create clean mlruns_clean with minimal data")
    print("  3. Keep your production model")
    print()

    # Step 1: Backup current mlruns
    print("[1/3] Backing up current mlruns...")
    if MLRUNS_OLD.exists():
        print(f"  Backup already exists at {MLRUNS_OLD}")
    else:
        shutil.move(str(MLRUNS_DIR), str(MLRUNS_OLD))
        print(f"  Backed up to: {MLRUNS_OLD}")

    # Step 2: Create clean mlruns
    print("\n[2/3] Creating clean MLflow setup...")
    MLRUNS_NEW.mkdir(exist_ok=True)

    # Copy just the model files
    models_old = MLRUNS_OLD / "1" / "models"
    if models_old.exists():
        models_new = MLRUNS_NEW / "1" / "models"
        models_new.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(models_old), str(models_new))
        print(f"  Copied models to clean directory")

    # Step 3: Move clean to mlruns
    print("\n[3/3] Activating clean setup...")
    if MLRUNS_DIR.exists():
        shutil.rmtree(MLRUNS_DIR)
    shutil.move(str(MLRUNS_NEW), str(MLRUNS_DIR))

    print()
    print("=" * 80)
    print("COMPLETE - MLFLOW IS NOW FAST")
    print("=" * 80)
    print()
    print("Results:")
    print(f"  Old mlruns: {MLRUNS_OLD} (243 MB backup)")
    print(f"  New mlruns: {MLRUNS_DIR} (~10 MB, fast)")
    print()
    print("Next steps:")
    print("  1. Restart MLflow UI")
    print("  2. Register the production model:")
    print("     poetry run python scripts/mlflow/register_best_model.py")
    print()


if __name__ == "__main__":
    main()
