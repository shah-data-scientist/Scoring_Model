"""
Check notebooks/mlruns database contents
"""
import sqlite3
from pathlib import Path

db_path = Path('notebooks/mlruns/mlflow.db')
print('='*80)
print('NOTEBOOKS/MLRUNS DATABASE - DETAILED ANALYSIS')
print('='*80)
print(f'Database: {db_path}')
print(f'Size: {db_path.stat().st_size / 1024:.1f} KB\n')

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# Get experiments
cursor.execute('SELECT experiment_id, name, lifecycle_stage FROM experiments')
experiments = cursor.fetchall()
print(f'Total Experiments: {len(experiments)}\n')

for exp_id, exp_name, lifecycle in experiments:
    print(f'--- Experiment: {exp_name} (ID: {exp_id}) ---')
    print(f'Lifecycle: {lifecycle}')

    # Get runs for this experiment
    cursor.execute('SELECT run_uuid, name, status FROM runs WHERE experiment_id = ?', (exp_id,))
    runs = cursor.fetchall()
    print(f'Total Runs: {len(runs)}\n')

    if runs:
        for run_uuid, run_name, status in runs:
            print(f'  Run: {run_name or run_uuid[:8]}')
            print(f'    Status: {status}')
            print(f'    UUID: {run_uuid[:16]}...')

            # Get metrics
            cursor.execute('SELECT key, value FROM metrics WHERE run_uuid = ?', (run_uuid,))
            metrics = cursor.fetchall()
            if metrics:
                print('    Metrics:')
                for key, value in metrics:
                    print(f'      {key}: {value:.4f}')

            # Get tags
            cursor.execute('SELECT key, value FROM tags WHERE run_uuid = ?', (run_uuid,))
            tags = cursor.fetchall()
            if tags:
                print('    Tags:')
                for key, value in tags:
                    if key in ['feature_strategy', 'sampling_strategy', 'mlflow.runName', 'validation']:
                        print(f'      {key}: {value}')
            print()
    print()

conn.close()

# Now check root mlruns for comparison
print('\n' + '='*80)
print('ROOT MLRUNS DATABASE - COMPARISON')
print('='*80)

root_db_path = Path('mlruns/mlflow.db')
if root_db_path.exists():
    print(f'Database: {root_db_path}')
    print(f'Size: {root_db_path.stat().st_size / 1024:.1f} KB\n')

    conn = sqlite3.connect(str(root_db_path))
    cursor = conn.cursor()

    cursor.execute('SELECT experiment_id, name FROM experiments WHERE name != "Default"')
    experiments = cursor.fetchall()
    print(f'Total Experiments: {len(experiments)}\n')

    for exp_id, exp_name in experiments:
        print(f'--- Experiment: {exp_name} (ID: {exp_id}) ---')
        cursor.execute('SELECT COUNT(*) FROM runs WHERE experiment_id = ?', (exp_id,))
        run_count = cursor.fetchone()[0]
        print(f'Total Runs: {run_count}')
        print()

    conn.close()
else:
    print('Root mlruns database not found!')
