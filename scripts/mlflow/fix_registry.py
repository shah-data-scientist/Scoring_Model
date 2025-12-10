"""
Simple fix: Use the model.pkl file directly without MLflow registry.

This creates a symlink or copy of the best model to a known location
so the API can load it directly.
"""

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
MODELS_DIR = PROJECT_ROOT / "models"

def main():
    print("=" * 80)
    print("FIX: Using model file directly")
    print("=" * 80)

    # Find the best model.pkl in mlruns
    model_files = list(MLRUNS_DIR.glob("*/models/*/artifacts/model.pkl"))

    if not model_files:
        print("\nERROR: No model.pkl files found in mlruns!")
        return

    print(f"\nFound {len(model_files)} model files")

    # Use the first one (they should all be similar)
    source_model = model_files[0]
    print(f"Using: {source_model.parent.parent.parent.name}")

    # Ensure models directory exists
    MODELS_DIR.mkdir(exist_ok=True)

    # Copy to models/production_model.pkl
    dest_model = MODELS_DIR / "production_model.pkl"
    shutil.copy2(source_model, dest_model)

    print(f"\nSUCCESS: Model copied to:")
    print(f"  {dest_model}")
    print(f"  Size: {dest_model.stat().st_size / 1024:.1f} KB")

    print("\n" + "=" * 80)
    print("Now update the API to use this file directly")
    print("=" * 80)

if __name__ == "__main__":
    main()
