"""
Data Drift Monitoring Script

This script uses Evidently AI to detect data drift between:
1. Reference Data: The training dataset (X_train)
2. Current Data: Production logs (logs/production_logs.csv)

Run with:
    poetry run python scripts/monitor_drift.py
"""
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
except ImportError:
    print("Evidently not installed. Please run: poetry add evidently")
    exit(1)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports" / "monitoring"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load reference and current data."""
    print("Loading data...")
    
    # Reference data (Training set)
    # Loading only a sample to save memory if needed
    ref_path = DATA_DIR / "X_train.csv"
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference data not found at {ref_path}")
    
    reference_data = pd.read_csv(ref_path)
    
    # Current data (Production logs)
    curr_path = LOGS_DIR / "production_logs.csv"
    if not curr_path.exists():
        print(f"No production logs found at {curr_path}. Cannot calculate drift.")
        return reference_data, None
        
    current_data = pd.read_csv(curr_path)
    
    # Map generic feature names in logs (feature_0, feature_1...) to actual names if possible
    # In this simplified version, we just ensure column intersection
    # Ideally, the API logs should use the same column names as X_train
    
    # Check if logs use "feature_0" style or actual names
    # Our API implementation used "feature_i". 
    # We need to map them back to X_train columns to make comparison valid.
    
    if 'feature_0' in current_data.columns:
        print("Mapping log feature names to original feature names...")
        mapping = {f"feature_{i}": col for i, col in enumerate(reference_data.columns)}
        current_data = current_data.rename(columns=mapping)
    
    # Align columns
    common_cols = list(set(reference_data.columns) & set(current_data.columns))
    
    if not common_cols:
        print("No common columns found between reference and current data.")
        return reference_data, None
        
    return reference_data[common_cols], current_data[common_cols]

def generate_report(reference_data, current_data):
    """Generate drift report."""
    if current_data is None or len(current_data) < 5:
        print("Not enough current data to generate report (need at least 5 samples).")
        return

    print(f"Generating drift report for {len(current_data)} production samples...")
    
    # 1. HTML Report
    report = Report(metrics=[
        DataDriftPreset(), 
    ])
    
    report.run(reference_data=reference_data, current_data=current_data)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"drift_report_{timestamp}.html"
    report.save_html(str(report_path))
    
    print(f"Report saved to: {report_path}")
    
    # 2. JSON Summary for automated checks
    json_path = REPORTS_DIR / f"drift_summary_{timestamp}.json"
    report.save_json(str(json_path))
    
    # Simple check
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    drift_detected = data['metrics'][0]['result']['dataset_drift']
    if drift_detected:
        print("⚠️ DATA DRIFT DETECTED!")
    else:
        print("✅ No significant data drift detected.")

if __name__ == "__main__":
    try:
        ref_df, curr_df = load_data()
        generate_report(ref_df, curr_df)
    except Exception as e:
        print(f"Error monitoring drift: {e}")
