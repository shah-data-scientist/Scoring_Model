"""
Comprehensive analysis of all MLflow runs
Identifies what artifacts exist and what's missing
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import mlflow
from mlflow.tracking import MlflowClient
from src.config import MLFLOW_TRACKING_URI
import pandas as pd

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

print('='*80)
print('COMPREHENSIVE MLFLOW RUNS ANALYSIS')
print('='*80)
print(f'Tracking URI: {MLFLOW_TRACKING_URI}')
print()

experiments = client.search_experiments()

all_runs_data = []

for exp in experiments:
    if exp.lifecycle_stage == 'deleted':
        continue

    print(f'\n{"="*80}')
    print(f'EXPERIMENT: {exp.name} (ID: {exp.experiment_id})')
    print(f'{"="*80}')

    runs = client.search_runs([exp.experiment_id])
    print(f'Total Runs: {len(runs)}\n')

    for idx, run in enumerate(runs[:20], 1):  # Show first 20 runs per experiment
        run_name = run.data.tags.get('mlflow.runName', 'Unnamed')

        # Get metrics
        metrics = run.data.metrics
        roc_auc = metrics.get('roc_auc', metrics.get('mean_roc_auc', 0))

        # Get artifacts
        artifacts = client.list_artifacts(run.info.run_id)
        artifact_count = len(artifacts)
        artifact_names = [a.path for a in artifacts]

        # Get tags
        feature_strategy = run.data.tags.get('feature_strategy', 'N/A')
        sampling_strategy = run.data.tags.get('sampling_strategy', 'N/A')

        print(f'{idx}. {run_name}')
        print(f'   ROC-AUC: {roc_auc:.4f}')
        print(f'   Features: {feature_strategy}, Sampling: {sampling_strategy}')
        print(f'   Artifacts: {artifact_count} ({", ".join(artifact_names) if artifact_names else "NONE"})')
        print(f'   Metrics: {len(metrics)} logged')
        print(f'   Status: {run.info.status}')
        print()

        all_runs_data.append({
            'experiment': exp.name,
            'run_name': run_name,
            'run_id': run.info.run_id,
            'roc_auc': roc_auc,
            'feature_strategy': feature_strategy,
            'sampling_strategy': sampling_strategy,
            'artifact_count': artifact_count,
            'artifacts': ', '.join(artifact_names) if artifact_names else 'NONE',
            'metric_count': len(metrics),
            'status': run.info.status
        })

# Create summary
print('\n' + '='*80)
print('SUMMARY STATISTICS')
print('='*80)

df = pd.DataFrame(all_runs_data)

print(f'\nTotal Active Runs: {len(df)}')
print(f'Experiments: {df["experiment"].nunique()}')
print(f'\nRuns by Status:')
print(df['status'].value_counts())

print(f'\nRuns with Artifacts: {(df["artifact_count"] > 0).sum()}')
print(f'Runs without Artifacts: {(df["artifact_count"] == 0).sum()}')

print(f'\nTop 10 Runs by ROC-AUC:')
top_runs = df.nlargest(10, 'roc_auc')[['run_name', 'roc_auc', 'feature_strategy', 'sampling_strategy', 'artifacts']]
print(top_runs.to_string(index=False))

# Save to CSV
output_file = Path('results/all_runs_analysis.csv')
output_file.parent.mkdir(exist_ok=True)
df.to_csv(output_file, index=False)

print(f'\nFull analysis saved to: {output_file}')

# Recommendations
print('\n' + '='*80)
print('RECOMMENDATIONS')
print('='*80)

no_artifacts = df[df['artifact_count'] == 0]
if len(no_artifacts) > 0:
    print(f'\n1. ADD ARTIFACTS TO {len(no_artifacts)} RUNS:')
    print('   - Confusion matrices')
    print('   - ROC curves')
    print('   - Feature importance plots')
    print('   - Precision-Recall curves')

failed_runs = df[df['status'] == 'FAILED']
if len(failed_runs) > 0:
    print(f'\n2. FIX {len(failed_runs)} FAILED RUNS:')
    for _, run in failed_runs.iterrows():
        print(f'   - {run["run_name"]} ({run["experiment"]})')

running_runs = df[df['status'] == 'RUNNING']
if len(running_runs) > 0:
    print(f'\n3. COMPLETE {len(running_runs)} RUNNING RUNS:')
    for _, run in running_runs.iterrows():
        print(f'   - {run["run_name"]} ({run["experiment"]})')

print('\n' + '='*80)
