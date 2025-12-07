"""
Configuration Loader

Loads project configuration from config.yaml.
"""
import yaml
from pathlib import Path
import os

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    # Handle path relative to root or src
    path = Path(config_path)
    if not path.exists():
        # Try checking root if we are in src/ or scripts/
        # Assuming src/config.py -> root is ../
        root_path = Path(__file__).parent.parent / config_path
        if root_path.exists():
            path = root_path
        else:
            # Try walking up
            current = Path.cwd()
            while current != current.parent:
                if (current / config_path).exists():
                    path = current / config_path
                    break
                current = current.parent
    
    if not path.exists():
        raise FileNotFoundError(f"Config file {config_path} not found.")

    with open(path, "r") as f:
        config = yaml.safe_load(f)
        
    return config

# Singleton config object
try:
    CONFIG = load_config()
except Exception as e:
    print(f"Warning: Could not load config: {e}")
    CONFIG = {}

# Helper accessors
def get_data_path():
    return Path(CONFIG.get('paths', {}).get('data', 'data/processed'))

def get_mlflow_uri():
    return CONFIG.get('mlflow', {}).get('tracking_uri', 'sqlite:///mlruns/mlflow.db')

def get_random_state():
    return CONFIG.get('project', {}).get('random_state', 42)