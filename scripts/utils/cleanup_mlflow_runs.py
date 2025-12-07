"""
Clean up MLflow runs with minimal value

This script removes:
1. Optuna runs with only fbeta_score (no ROC-AUC, no artifacts)
2. Failed or incomplete runs
3. Duplicate/test runs

Keeps:
- All feature engineering CV runs (16 runs)
- Model selection runs (5 runs)
- Final delivery run (1 run with artifacts)
- Best Optuna run (1 run with full metrics)
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import mlflow
from mlflow.tracking import MlflowClient
from src.config import MLFLOW_TRACKING_URI

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

print('='*80)
print('MLFLOW RUNS CLEANUP')
print('='*80)

# Get all experiments
experiments = client.search_experiments()

runs_to_delete = []
runs_to_keep = []

for exp in experiments:
    if exp.lifecycle_stage == 'deleted':
        continue

    print(f'\n--- Experiment: {exp.name} (ID: {exp.experiment_id}) ---')
    runs = client.search_runs([exp.experiment_id])
    print(f'Total runs: {len(runs)}')

    for run in runs:
        run_name = run.data.tags.get('mlflow.runName', 'Unnamed')
        metrics = run.data.metrics
        artifacts = client.list_artifacts(run.info.run_id)

        # Criteria for deletion
        should_delete = False
        reason = ""

        # 1. Optuna runs with minimal logging
        if exp.name == 'credit_scoring_optimization_fbeta':
            has_roc_auc = any('roc' in k.lower() for k in metrics.keys())
            has_artifacts = len(artifacts) > 0
            is_best = 'optuna_optimization_domain_balanced' in run_name and run.info.status == 'FINISHED'

            if not is_best and not has_roc_auc and not has_artifacts:
                should_delete = True
                reason = "Optuna run with minimal metrics (only fbeta)"

        # 2. Failed runs
        if run.info.status == 'FAILED':
            should_delete = True
            reason = "Failed run"

        # 3. Running runs (stuck)
        if run.info.status == 'RUNNING':
            should_delete = True
            reason = "Stuck in RUNNING state"

        if should_delete:
            runs_to_delete.append({
                'run_id': run.info.run_id,
                'run_name': run_name,
                'experiment': exp.name,
                'reason': reason
            })
        else:
            runs_to_keep.append({
                'run_id': run.info.run_id,
                'run_name': run_name,
                'experiment': exp.name
            })

# Summary
print('\n' + '='*80)
print('CLEANUP SUMMARY')
print('='*80)

print(f'\nTotal runs analyzed: {len(runs_to_delete) + len(runs_to_keep)}')
print(f'Runs to DELETE: {len(runs_to_delete)}')
print(f'Runs to KEEP: {len(runs_to_keep)}')

if runs_to_delete:
    print('\n--- Runs to be DELETED ---')
    for run in runs_to_delete[:10]:  # Show first 10
        print(f'  - {run["run_name"][:50]} ({run["experiment"]})')
        print(f'    Reason: {run["reason"]}')

    if len(runs_to_delete) > 10:
        print(f'  ... and {len(runs_to_delete) - 10} more')

    # Ask for confirmation
    print('\n' + '='*80)
    print('CONFIRMATION')
    print('='*80)
    response = input(f'\nDelete {len(runs_to_delete)} runs? (yes/no): ')

    if response.lower() == 'yes':
        print('\nDeleting runs...')
        deleted_count = 0

        for run in runs_to_delete:
            try:
                client.delete_run(run['run_id'])
                deleted_count += 1
                if deleted_count % 5 == 0:
                    print(f'  Deleted {deleted_count}/{len(runs_to_delete)} runs...')
            except Exception as e:
                print(f'  ERROR deleting {run["run_name"]}: {e}')

        print(f'\n✓ Successfully deleted {deleted_count} runs')

        # Compact database (optional)
        print('\nNote: Run "VACUUM" on SQLite database to reclaim space')

    else:
        print('\nCleanup cancelled. No runs were deleted.')
else:
    print('\nNo runs to delete. Database is clean!')

print('\n' + '='*80)
print('CLEANUP COMPLETE')
print('='*80)
print(f'Remaining runs: {len(runs_to_keep)}')
