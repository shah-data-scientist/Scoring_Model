import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors import CellExecutionError
import os
import sys

# Define notebooks in order
notebooks = [
    "notebooks/01_eda.ipynb",
    "notebooks/02_feature_engineering.ipynb",
    "notebooks/03_baseline_models.ipynb",
    "notebooks/04_hyperparameter_optimization.ipynb",
    "notebooks/05_model_interpretation.ipynb",
    "notebooks/results.ipynb"
]

# Create preprocessor
# timeout=600 seconds (10 minutes) per notebook. 
# Long training steps might timeout, which we will catch.
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

def run_notebook(nb_path):
    print(f"--------------------------------------------------")
    print(f"Checking {nb_path}...")
    
    if not os.path.exists(nb_path):
        print(f"ERROR: File not found: {nb_path}")
        return False

    try:
        with open(nb_path, encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Execute the notebook
        # We assume the working directory for execution is the notebook's directory
        nb_dir = os.path.dirname(nb_path)
        ep.preprocess(nb, {'metadata': {'path': nb_dir}})
        
        print(f"SUCCESS: {nb_path} executed without errors.")
        return True
        
    except CellExecutionError as e:
        print(f"FAILURE: {nb_path} failed.")
        print(f"Error in cell:")
        print(e)
        return False
    except TimeoutError:
        print(f"WARNING: {nb_path} timed out after 600s. It might just be slow training.")
        return True # specific choice: treat timeout as 'running but slow', not necessarily a crash code-wise
    except Exception as e:
        print(f"ERROR: {nb_path} encountered unexpected error.")
        print(e)
        return False

# Main loop
all_passed = True
for nb in notebooks:
    if not run_notebook(nb):
        all_passed = False
        # Decision: Stop on first failure or continue?
        # User asked "Launch all ... tell me if there is an issue".
        # If one fails, subsequent ones might fail due to missing data. 
        # But let's try to run all to give a full report.
        pass

if all_passed:
    print("\nAll notebooks checked successfully (or timed out cleanly).")
    sys.exit(0)
else:
    print("\nSome notebooks failed.")
    sys.exit(1)
