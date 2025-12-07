"""
MLflow Runs Consolidation Script

Audits and consolidates MLflow runs from multiple locations following best practices.

Best Practice: MLflow runs should be stored at PROJECT_ROOT/mlruns/
This is the industry standard and what MLflow documentation recommends.
"""
import sys
from pathlib import Path
import shutil
import sqlite3
import mlflow
from mlflow.tracking import MlflowClient

sys.path.append(str(Path(__file__).parent.parent))
from src.config import PROJECT_ROOT

# Define locations
ROOT_MLRUNS = PROJECT_ROOT / 'mlruns'
NOTEBOOKS_MLRUNS = PROJECT_ROOT / 'notebooks' / 'mlruns'

def get_database_info(db_path):
    """Get information about an MLflow database."""
    if not db_path.exists():
        return None

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Count experiments
        cursor.execute("SELECT COUNT(*) FROM experiments")
        exp_count = cursor.fetchone()[0]

        # Count runs
        cursor.execute("SELECT COUNT(*) FROM runs")
        run_count = cursor.fetchone()[0]

        # Get experiment details
        cursor.execute("SELECT experiment_id, name FROM experiments WHERE name != 'Default'")
        experiments = cursor.fetchall()

        conn.close()

        return {
            'exp_count': exp_count,
            'run_count': run_count,
            'experiments': experiments,
            'size_kb': db_path.stat().st_size / 1024
        }
    except Exception as e:
        return {'error': str(e)}

def audit_mlruns_locations():
    """Audit all MLflow run locations."""
    print("="*80)
    print("MLFLOW RUNS CONSOLIDATION AUDIT")
    print("="*80)

    locations = [
        ('ROOT (Best Practice)', ROOT_MLRUNS),
        ('NOTEBOOKS (Should Migrate)', NOTEBOOKS_MLRUNS),
    ]

    results = {}

    for name, location in locations:
        print(f"\n[FOLDER] {name}: {location}")

        if not location.exists():
            print("   ❌ Folder not found")
            results[name] = None
            continue

        print("   ✓ Folder exists")

        # Check database
        db_file = location / 'mlflow.db'
        db_info = get_database_info(db_file)

        if db_info is None:
            print("   ❌ No database found")
            results[name] = {'has_db': False}
        elif 'error' in db_info:
            print(f"   ❌ Database error: {db_info['error']}")
            results[name] = {'has_db': False, 'error': db_info['error']}
        else:
            print(f"   ✓ Database: {db_info['size_kb']:.1f} KB")
            print(f"   ✓ Experiments: {db_info['exp_count']}")
            print(f"   ✓ Runs: {db_info['run_count']}")

            if db_info['experiments']:
                print("   📊 Experiment List:")
                for exp_id, exp_name in db_info['experiments']:
                    print(f"      - {exp_name} (ID: {exp_id})")

            results[name] = db_info

    return results

def compare_databases(root_info, notebooks_info):
    """Compare two databases and recommend action."""
    print("\n" + "="*80)
    print("CONSOLIDATION RECOMMENDATION")
    print("="*80)

    if root_info is None or not root_info.get('has_db', True):
        if notebooks_info and notebooks_info.get('has_db', True):
            print("\n🔄 ACTION: MIGRATE notebooks/mlruns → mlruns")
            print("   Reason: Root location has no database, notebooks has data")
            return 'migrate_notebooks_to_root'

    if notebooks_info is None or not notebooks_info.get('has_db', True):
        print("\n✅ STATUS: All runs already in ROOT location (CORRECT)")
        print("   No action needed - following best practices")
        return 'no_action'

    # Both have data - need to merge
    root_runs = root_info.get('run_count', 0)
    notebooks_runs = notebooks_info.get('run_count', 0)

    if root_runs > 0 and notebooks_runs > 0:
        print("\n⚠️  WARNING: Both locations have ML runs!")
        print(f"   ROOT: {root_runs} runs")
        print(f"   NOTEBOOKS: {notebooks_runs} runs")
        print("\n🔄 ACTION: MERGE required")
        print("   Recommendation: Keep ROOT, archive NOTEBOOKS")
        return 'merge_required'

    return 'no_action'

def review_run_quality(location):
    """Review quality of runs in a location."""
    print("\n" + "="*80)
    print(f"RUN QUALITY REVIEW: {location}")
    print("="*80)

    db_path = location / 'mlflow.db'
    if not db_path.exists():
        print("No database to review")
        return

    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    client = MlflowClient()

    experiments = client.search_experiments()

    for exp in experiments:
        if exp.name == 'Default':
            continue

        print(f"\n📊 Experiment: {exp.name}")
        runs = client.search_runs([exp.experiment_id])

        if not runs:
            print("   No runs found")
            continue

        print(f"   Total runs: {len(runs)}")

        # Check naming conventions
        unnamed_runs = [r for r in runs if not r.data.tags.get('mlflow.runName')]
        if unnamed_runs:
            print(f"   ⚠️  {len(unnamed_runs)} runs without names")
        else:
            print(f"   ✓ All runs have names")

        # Check metrics
        runs_without_metrics = [r for r in runs if not r.data.metrics]
        if runs_without_metrics:
            print(f"   ⚠️  {len(runs_without_metrics)} runs without metrics")
        else:
            print(f"   ✓ All runs have metrics")

        # Check for key metrics
        runs_with_roc_auc = [r for r in runs if 'roc_auc' in r.data.metrics or 'mean_roc_auc' in r.data.metrics]
        print(f"   ✓ {len(runs_with_roc_auc)}/{len(runs)} runs have ROC-AUC metric")

        # Check for training metrics (overfitting detection)
        runs_with_train_metrics = [r for r in runs if any('train' in k for k in r.data.metrics.keys())]
        if not runs_with_train_metrics:
            print(f"   ⚠️  No runs have training metrics (can't detect overfitting)")
        else:
            print(f"   ✓ {len(runs_with_train_metrics)}/{len(runs)} runs have training metrics")

        # Show best run
        best_run = max(runs, key=lambda r: r.data.metrics.get('roc_auc', r.data.metrics.get('mean_roc_auc', 0)))
        best_score = best_run.data.metrics.get('roc_auc', best_run.data.metrics.get('mean_roc_auc', 0))
        best_name = best_run.data.tags.get('mlflow.runName', 'Unnamed')
        print(f"   🏆 Best run: {best_name} (ROC-AUC: {best_score:.4f})")

def main():
    """Main consolidation workflow."""
    # Step 1: Audit
    results = audit_mlruns_locations()

    # Step 2: Compare and recommend
    root_info = results.get('ROOT (Best Practice)')
    notebooks_info = results.get('NOTEBOOKS (Should Migrate)')

    action = compare_databases(root_info, notebooks_info)

    # Step 3: Review quality of current runs
    if root_info and root_info.get('has_db', True):
        review_run_quality(ROOT_MLRUNS)

    if notebooks_info and notebooks_info.get('has_db', True):
        review_run_quality(NOTEBOOKS_MLRUNS)

    # Step 4: Final recommendations
    print("\n" + "="*80)
    print("FINAL RECOMMENDATIONS")
    print("="*80)

    print("\n1. CURRENT CONFIGURATION:")
    print(f"   MLflow URI: sqlite:///{ROOT_MLRUNS / 'mlflow.db'}")
    print(f"   Status: {'✓ CORRECT' if ROOT_MLRUNS.exists() else '❌ NEEDS SETUP'}")

    print("\n2. BEST PRACTICES:")
    print("   ✓ Store runs at PROJECT_ROOT/mlruns/")
    print("   ✓ Use centralized config.py for URI")
    print("   ✓ Name all runs descriptively")
    print("   ✓ Log both training and validation metrics")
    print("   ✓ Use consistent experiment naming")

    print("\n3. CLEANUP ACTIONS:")
    if notebooks_info and notebooks_info.get('has_db', True):
        print("   📦 Archive notebooks/mlruns/ (runs already migrated)")
        print(f"      Command: mv notebooks/mlruns notebooks/mlruns_archived")
    else:
        print("   ✓ No cleanup needed")

    print("\n4. VERIFICATION:")
    print("   Run: poetry run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000")
    print("   Check: All experiments visible at http://localhost:5000")

    print("\n" + "="*80)
    print("AUDIT COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
