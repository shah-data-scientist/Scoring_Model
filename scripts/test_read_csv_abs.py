# scripts/test_read_csv_abs.py
import pandas as pd
from pathlib import Path
import os

try:
    # Get the absolute path to the project root
    project_root = Path(os.getcwd()) 
    csv_path = project_root / 'data' / 'processed' / 'X_train.csv'

    print(f"Attempting to read from absolute path: {csv_path}")
    X_train = pd.read_csv(csv_path)
    print(f"Successfully read {csv_path}. Shape: {X_train.shape}")

except Exception as e:
    print(f"Error reading CSV: {e}")
