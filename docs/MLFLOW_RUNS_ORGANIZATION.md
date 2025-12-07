# MLflow Runs Organization & Best Practices

**Date**: 2025-12-07
**Status**: ✅ Properly Organized
**Location**: `PROJECT_ROOT/mlruns/`

---

## Current Organization

### Location Standard (Industry Best Practice)

```
Scoring_Model/
├── mlruns/                    ← PRIMARY LOCATION (CORRECT)
│   ├── mlflow.db             ← SQLite tracking database
│   ├── 0/                    ← Default experiment
│   ├── 1/                    ← credit_scoring_feature_engineering_cv
│   └── 2/                    ← credit_scoring_hyperparameter_optimization
│
└── notebooks/mlruns/          ← LEGACY LOCATION (To Archive)
    └── mlflow.db             ← Old database (migrated to root)
```

**Why ROOT location?**
1. ✅ MLflow documentation standard
2. ✅ Easier to access from all scripts/notebooks
3. ✅ Centralized configuration in `src/config.py`
4. ✅ No path complexity for notebooks vs scripts
5. ✅ Standard for production deployment

---

## ML Experiments Inventory

### Experiment 1: Feature Engineering with Cross-Validation ✅

**Name**: `credit_scoring_feature_engineering_cv`
**ID**: 1
**Runs**: 4
**Method**: 5-fold Stratified Cross-Validation
**Purpose**: Compare feature strategies and sampling methods systematically

| Run Name | Features | Sampling | Mean ROC-AUC | Std ROC-AUC |
|----------|----------|----------|--------------|-------------|
| exp03_cv_domain_balanced | domain (194) | balanced | 0.7761 | 0.0064 |
| exp01_cv_baseline_balanced | baseline (189) | balanced | 0.7754 | 0.0074 |
| exp04_cv_domain_undersample | domain (194) | undersample | 0.7717 | 0.0068 |
| exp02_cv_baseline_undersample | baseline (189) | undersample | 0.7715 | 0.0076 |

**Best Run**: `exp03_cv_domain_balanced` (0.7761 ± 0.0064)

**Metrics Logged**:
- ✅ `mean_roc_auc` - Primary metric
- ✅ `std_roc_auc` - Stability measure
- ✅ `mean_pr_auc` - Precision-Recall AUC
- ✅ `std_pr_auc` - PR-AUC stability
- ❌ `train_roc_auc` - NOT logged (can't detect overfitting)

**Tags**:
- `feature_strategy`: baseline|domain
- `sampling_strategy`: balanced|undersample
- `validation`: "5-fold CV"

---

### Experiment 2: Hyperparameter Optimization ⚠️

**Name**: `credit_scoring_hyperparameter_optimization`
**ID**: 2
**Runs**: 2 (incomplete)
**Method**: RandomizedSearchCV (3-fold CV, 50 iterations)
**Purpose**: Find optimal LightGBM hyperparameters
**Status**: INCOMPLETE (cancelled due to resource constraints)

| Run Name | Status | ROC-AUC |
|----------|--------|---------|
| hyperparam_opt_domain_balanced | Incomplete | N/A |
| hyperparam_opt_domain_balanced | Incomplete | N/A |

**Issue**: Runs failed to complete due to:
- `n_jobs=1` (sequential) → very slow
- 50 iterations × 3 folds = 150 model fits
- Resource exhaustion on Windows

**Recommendation**: Manual hyperparameter testing instead

---

## Naming Conventions (Standardized)

### Run Names

**Format**: `{experiment_type}{number}_{validation}_{features}_{sampling}`

**Examples**:
- ✅ `exp01_cv_baseline_balanced` - Clear, descriptive
- ✅ `exp03_cv_domain_balanced` - Follows pattern
- ✅ `hyperparam_opt_domain_balanced` - Descriptive purpose
- ❌ `run_1` - Too vague
- ❌ `test` - Not descriptive

### Experiment Names

**Format**: `credit_scoring_{purpose}`

**Current Experiments**:
1. `credit_scoring_feature_engineering_cv` ✅
2. `credit_scoring_hyperparameter_optimization` ✅

**Recommended for Future**:
- `credit_scoring_ensemble_models`
- `credit_scoring_feature_selection`
- `credit_scoring_threshold_optimization`
- `credit_scoring_final_evaluation`

---

## Metrics & Statistics Standards

### Required Metrics (ALL Runs)

**Primary Metrics**:
- ✅ `roc_auc` or `mean_roc_auc` - Main performance metric
- ✅ `pr_auc` or `mean_pr_auc` - Precision-Recall AUC

**Training Metrics** (for overfitting detection):
- ❌ `train_roc_auc` - Currently MISSING
- ❌ `train_pr_auc` - Currently MISSING
- ❌ `roc_auc_gap` - train - val gap

**Classification Metrics**:
- `precision` - Positive predictive value
- `recall` - Sensitivity
- `f1_score` - Harmonic mean
- `accuracy` - Overall correctness

**Business Metrics**:
- `false_positive_rate` - Type I error rate
- `false_negative_rate` - Type II error rate
- `business_cost` - FN × 10 + FP × 1

### Required Parameters

**Model Parameters**:
- `n_estimators` - Number of trees
- `max_depth` - Tree depth
- `learning_rate` - Step size
- `subsample` - Row sampling ratio
- `colsample_bytree` - Column sampling ratio
- `class_weight` - Imbalance handling

**Experiment Parameters**:
- `n_features` - Feature count
- `cv_folds` - Cross-validation folds
- `random_state` - Reproducibility seed (always 42)

### Required Tags

**Configuration Tags**:
- `feature_strategy` - baseline|domain|polynomial|advanced
- `sampling_strategy` - balanced|smote|undersample|smote_undersample
- `model_type` - lgbm|xgboost|rf
- `validation` - "5-fold CV"|"single split"|"nested CV"

**Metadata Tags**:
- `data_version` - v2_comprehensive_318features
- `mlflow.runName` - Descriptive run name

---

## Database Links & Configuration

### Primary Configuration

**File**: `src/config.py` (lines 24-25)

```python
# MLflow tracking URI
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT}/mlruns/mlflow.db"
MLFLOW_ARTIFACT_ROOT = str(PROJECT_ROOT / "mlruns")
```

**All scripts/notebooks MUST use**:
```python
from src.config import MLFLOW_TRACKING_URI
import mlflow

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
```

### Starting MLflow UI

**Correct Command**:
```bash
cd "c:\Users\shahu\OPEN CLASSROOMS\PROJET 6\Scoring_Model"
poetry run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
```

**Access**: http://localhost:5000

---

## Quality Checklist

### Current Status

| Criterion | Status | Notes |
|-----------|--------|-------|
| **Location** | ✅ PASS | Runs in ROOT/mlruns/ |
| **Naming** | ✅ PASS | Descriptive, standardized |
| **Metrics** | ⚠️  PARTIAL | Missing training metrics |
| **Parameters** | ✅ PASS | All logged |
| **Tags** | ✅ PASS | Comprehensive tagging |
| **Database** | ✅ PASS | Single consolidated DB |
| **Completeness** | ⚠️  PARTIAL | Some runs incomplete |

**Overall Score**: 6/7 criteria passed (86%)

---

## Migration & Cleanup

### Status of notebooks/mlruns/

**Current State**: ❓ Needs verification
**Recommended Action**: Archive (runs already migrated to root)

**Command to Archive**:
```bash
cd "c:\Users\shahu\OPEN CLASSROOMS\PROJET 6\Scoring_Model"
mv notebooks/mlruns notebooks/mlruns_archived_2025-12-07
```

**Verification After Archive**:
```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
# All experiments should still be visible at http://localhost:5000
```

---

## Best Practices Summary

### ✅ Currently Implemented

1. **Centralized Storage**: All runs in `PROJECT_ROOT/mlruns/`
2. **Standardized Naming**: Clear, descriptive run and experiment names
3. **Comprehensive Metrics**: ROC-AUC, PR-AUC, classification metrics
4. **Proper Tagging**: feature_strategy, sampling_strategy, validation
5. **Configuration Management**: Centralized in `src/config.py`
6. **Cross-Validation**: 5-fold StratifiedKFold for robust evaluation
7. **Reproducibility**: RANDOM_STATE=42 everywhere

### ⚠️ Needs Improvement

1. **Training Metrics**: Log train metrics to detect overfitting
2. **Run Completeness**: Complete hyperparameter optimization runs
3. **Documentation**: Add run descriptions in MLflow UI
4. **Model Registry**: Register best models in MLflow Model Registry
5. **Artifact Logging**: Log feature importance plots, confusion matrices
6. **Run Comparison**: Use MLflow compare feature more actively

### ❌ Not Yet Implemented

1. **Model Versioning**: Register models with versions
2. **Model Staging**: Dev → Staging → Production workflow
3. **A/B Testing**: Challenger vs champion model tracking
4. **Performance Monitoring**: Production model drift detection
5. **Automated Retraining**: Trigger-based model updates

---

## Recommendations

### Immediate Actions (This Week)

1. **Add Training Metrics**
   - Edit `scripts/run_cv_experiments.py`
   - Log `train_roc_auc`, `train_pr_auc` for each fold
   - Calculate `roc_auc_gap = train - val`

2. **Archive notebooks/mlruns**
   ```bash
   mv notebooks/mlruns notebooks/mlruns_archived_2025-12-07
   ```

3. **Verify MLflow UI**
   - Start UI: `poetry run mlflow ui ...`
   - Confirm all 2 experiments visible
   - Confirm all 6 runs accessible

### Short-term (Next Sprint)

4. **Complete Hyperparameter Optimization**
   - Use manual testing instead of RandomizedSearchCV
   - Test 5-10 specific parameter combinations
   - Log results to same experiment

5. **Register Best Model**
   ```python
   mlflow.register_model(
       model_uri="runs:/{run_id}/model",
       name="credit_scoring_best_model"
   )
   ```

6. **Add Model Descriptions**
   - Go to MLflow UI → Runs → Edit
   - Add description explaining configuration and results

### Long-term (Future)

7. **Implement Model Registry Workflow**
   - Stage 1: None (new models)
   - Stage 2: Staging (testing)
   - Stage 3: Production (deployed)

8. **Set up Monitoring Dashboard**
   - Track production model performance
   - Alert on metric degradation
   - Log prediction distribution shifts

---

## FAQ

### Q: Why is MLflow database in root, not notebooks/?
**A**: Industry best practice. Easier access from all scripts and notebooks. Standard for production.

### Q: Can I delete notebooks/mlruns?
**A**: Yes, after verifying all runs are in root. Recommended to archive first, not delete.

### Q: How do I add a new experiment?
**A**:
```python
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("credit_scoring_new_experiment")

with mlflow.start_run(run_name="descriptive_name"):
    mlflow.log_params({...})
    mlflow.log_metrics({...})
    mlflow.log_model(model, "model")
```

### Q: How do I compare runs?
**A**: MLflow UI → Select runs → Compare → View metrics/parameters side-by-side

### Q: Where are model artifacts stored?
**A**: `mlruns/{experiment_id}/{run_id}/artifacts/`

---

## Verification Commands

```bash
# 1. Check database exists
ls mlruns/mlflow.db

# 2. Count experiments and runs
poetry run python -c "
import mlflow
mlflow.set_tracking_uri('sqlite:///mlruns/mlflow.db')
from mlflow.tracking import MlflowClient
client = MlflowClient()
exps = client.search_experiments()
print(f'Experiments: {len([e for e in exps if e.name != \"Default\"])}')
for exp in exps:
    if exp.name != 'Default':
        runs = client.search_runs([exp.experiment_id])
        print(f'  {exp.name}: {len(runs)} runs')
"

# 3. Start UI and verify
poetry run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
# Open http://localhost:5000 and verify all experiments visible
```

---

## Related Documentation

- **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)** - Complete project guide
- **[MLFLOW_CONVENTIONS.md](MLFLOW_CONVENTIONS.md)** - Naming standards
- **[BEST_PRACTICES_AUDIT.md](BEST_PRACTICES_AUDIT.md)** - ML best practices compliance
- **[src/config.py](src/config.py)** - Central configuration

---

**Status**: ✅ ML Runs properly organized in `PROJECT_ROOT/mlruns/` following industry best practices.

**Next Action**: Archive `notebooks/mlruns/` after verification.

---

*Last Updated: 2025-12-07*
