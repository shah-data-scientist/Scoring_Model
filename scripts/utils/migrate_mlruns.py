"""
Migrate MLflow runs from notebooks/mlruns to root mlruns
Following industry best practice: mlruns should be at PROJECT_ROOT
"""
import shutil
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent

def migrate_mlruns():
    """Migrate mlruns from notebooks to root."""

    # Paths
    root_mlruns = PROJECT_ROOT / 'mlruns'
    notebooks_mlruns = PROJECT_ROOT / 'notebooks' / 'mlruns'
    backup_mlruns = PROJECT_ROOT / 'mlruns_backup_old'

    print('='*80)
    print('MLFLOW RUNS MIGRATION')
    print('='*80)
    print(f'Source: {notebooks_mlruns}')
    print(f'Destination: {root_mlruns}')
    print()

    # Step 1: Backup old root mlruns if it exists
    if root_mlruns.exists():
        print('[STEP 1] Backing up old root mlruns...')
        if backup_mlruns.exists():
            print(f'  Removing old backup: {backup_mlruns}')
            shutil.rmtree(backup_mlruns)
        print(f'  Moving {root_mlruns} -> {backup_mlruns}')
        shutil.move(str(root_mlruns), str(backup_mlruns))
        print('  SUCCESS: Old mlruns backed up')
    else:
        print('[STEP 1] No existing root mlruns to backup')

    # Step 2: Check source exists
    print()
    print('[STEP 2] Checking source location...')
    if not notebooks_mlruns.exists():
        print(f'  ERROR: Source not found: {notebooks_mlruns}')
        sys.exit(1)
    print(f'  SUCCESS: Source exists')

    # Get source size
    source_size = sum(f.stat().st_size for f in notebooks_mlruns.rglob('*') if f.is_file()) / (1024 * 1024)
    print(f'  Source size: {source_size:.2f} MB')

    # Step 3: Copy notebooks/mlruns to root
    print()
    print('[STEP 3] Copying notebooks/mlruns -> mlruns...')
    shutil.copytree(str(notebooks_mlruns), str(root_mlruns))
    print('  SUCCESS: All data copied to root location')

    # Verify
    dest_size = sum(f.stat().st_size for f in root_mlruns.rglob('*') if f.is_file()) / (1024 * 1024)
    print(f'  Destination size: {dest_size:.2f} MB')

    # Step 4: Verify database
    print()
    print('[STEP 4] Verifying database...')
    db_file = root_mlruns / 'mlflow.db'
    if db_file.exists():
        db_size_kb = db_file.stat().st_size / 1024
        print(f'  SUCCESS: Database found - {db_size_kb:.1f} KB')

        # Quick database check
        import sqlite3
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM experiments')
        exp_count = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM runs')
        run_count = cursor.fetchone()[0]

        conn.close()

        print(f'  Experiments: {exp_count}')
        print(f'  Runs: {run_count}')
    else:
        print(f'  ERROR: Database not found at {db_file}')
        sys.exit(1)

    # Step 5: Summary
    print()
    print('='*80)
    print('MIGRATION COMPLETE')
    print('='*80)
    print(f'All MLflow runs migrated to: {root_mlruns}')
    print(f'Database: {db_file}')
    print(f'Total experiments: {exp_count}')
    print(f'Total runs: {run_count}')
    print()
    print('NEXT STEPS:')
    print('1. Config already points to root location (no change needed)')
    print('2. Start MLflow UI: poetry run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db')
    print('3. Old root mlruns backed up to: mlruns_backup_old')
    print('4. notebooks/mlruns can be archived later (source data preserved)')
    print('='*80)

if __name__ == '__main__':
    try:
        migrate_mlruns()
    except Exception as e:
        print(f'\nERROR: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
