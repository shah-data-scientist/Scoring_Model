# MLflow Naming Conventions & Best Practices
## Credit Scoring Model Project

## 📋 Overview

This document establishes standardized naming conventions and best practices for MLflow experiment tracking in the Credit Scoring Model project.

---

## 🎯 Core Principles

1. **Consistency**: Use the same naming pattern across all experiments
2. **Descriptiveness**: Names should clearly indicate purpose and content
3. **Versioning**: Track versions explicitly
4. **Searchability**: Use prefixes and tags for easy filtering
5. **Hierarchy**: Organize experiments logically

---

## 1️⃣ Experiment Naming

### **Format**
```
{project_prefix}_{stage_number}_{stage_name}
```

### **Convention**
- **Lowercase** with **underscores** (snake_case)
- Include stage number for chronological ordering
- Keep names concise but descriptive

### **Examples**
```
✅ GOOD:
- credit_scoring_01_baseline
- credit_scoring_02_optimization
- credit_scoring_03_final_evaluation
- credit_scoring_04_production

❌ BAD:
- baseline_models (missing project prefix)
- credit-scoring-baseline-models (hyphens instead of underscores)
- CreditScoringBaseline (CamelCase)
- exp1 (not descriptive)
```

### **Current State → Recommended**
```
credit_scoring_baseline_models        → credit_scoring_01_baseline
credit_scoring_hyperparameter_optimization → credit_scoring_02_optimization
```

---

## 2️⃣ Run Naming

### **Format Options**

**Option A: Model + Key Parameters** (Recommended for tuning)
```
{model_name}_{key_param1}_{key_param2}_v{version}
```

**Option B: Model + Timestamp** (Recommended for baselines)
```
{model_name}_v{version}_{yyyymmdd}
```

**Option C: Model + Description** (Recommended for production)
```
{model_name}_{description}_v{version}
```

### **Convention**
- **Lowercase** with **underscores**
- Include version number
- Add key distinguishing information
- Keep under 50 characters if possible

### **Examples**
```
✅ GOOD (Baseline):
- lgbm_v1_balanced
- xgboost_v1_default
- logistic_regression_v1
- dummy_classifier_v1_reference

✅ GOOD (Optimization):
- lgbm_lr0.05_depth5_n250_v2
- xgboost_optimized_v3
- lgbm_best_cv_v4

✅ GOOD (Production):
- lgbm_production_v1_20231206
- xgboost_champion_v2

❌ BAD:
- LightGBM (mixed case, no version)
- model_1 (not descriptive)
- lgbm_n_estimators_250_max_depth_5_learning_rate_0.05 (too long)
```

### **Current State → Recommended**
```
Dummy_Classifier        → dummy_classifier_v1_reference
Logistic_Regression     → logistic_regression_v1_balanced
Random_Forest           → random_forest_v1_default
XGBoost                 → xgboost_v1_default
LightGBM                → lgbm_v1_default
```

---

## 3️⃣ Model Registry Naming

### **Format**
```
{project}_{model_type}
```

### **Convention**
- **Lowercase** with **underscores**
- Project prefix for multi-project environments
- Simple model type identifier
- **Use Stages**: None → Staging → Production → Archived

### **Examples**
```
✅ GOOD:
- credit_scoring_lgbm
- credit_scoring_xgboost
- credit_scoring_ensemble

❌ BAD:
- best_model (not specific)
- LightGBM_Model (mixed case)
- model_v1 (no project context)
```

### **Model Stages**
```
None       → Initial training, not yet validated
Staging    → Under review/testing, candidate for production
Production → Active in production serving predictions
Archived   → Retired, kept for historical reference
```

---

## 4️⃣ Tags Best Practices

### **Required Tags**
```python
mlflow.set_tag("project", "credit_scoring")
mlflow.set_tag("stage", "baseline")  # baseline, optimization, production
mlflow.set_tag("model_type", "lgbm")
mlflow.set_tag("data_version", "v2_comprehensive")  # Track data lineage
mlflow.set_tag("feature_count", "189")
```

### **Recommended Tags**
```python
mlflow.set_tag("author", "data_scientist_name")
mlflow.set_tag("purpose", "baseline_comparison")
mlflow.set_tag("framework", "sklearn")  # or xgboost, lightgbm
mlflow.set_tag("training_date", "2023-12-06")
mlflow.set_tag("notebook", "03_baseline_models.ipynb")
```

### **Conditional Tags**
```python
# For optimized models
mlflow.set_tag("optimization_method", "random_search")
mlflow.set_tag("cv_folds", "5")

# For production models
mlflow.set_tag("deployment_env", "production")
mlflow.set_tag("api_endpoint", "https://...")
```

---

## 5️⃣ Parameter Naming

### **Convention**
- Use **exact parameter names** from the model API
- For custom parameters, use **snake_case**
- Group related parameters with prefixes

### **Examples**
```python
# ✅ GOOD - Model parameters (exact API names)
mlflow.log_param("n_estimators", 100)
mlflow.log_param("max_depth", 6)
mlflow.log_param("learning_rate", 0.1)

# ✅ GOOD - Custom preprocessing parameters (snake_case)
mlflow.log_param("preprocessing_missing_threshold", 0.7)
mlflow.log_param("preprocessing_variance_threshold", 0.01)
mlflow.log_param("preprocessing_correlation_threshold", 0.95)

# ❌ BAD
mlflow.log_param("n_est", 100)  # Abbreviated
mlflow.log_param("MaxDepth", 6)  # CamelCase
mlflow.log_param("param1", 0.1)  # Not descriptive
```

---

## 6️⃣ Metric Naming

### **Convention**
- Use **lowercase** with **underscores**
- Include dataset split suffix: `_train`, `_val`, `_test`
- Use standard metric abbreviations

### **Standard Metrics**
```python
# Classification metrics
mlflow.log_metric("roc_auc", 0.78)
mlflow.log_metric("pr_auc", 0.27)
mlflow.log_metric("f1_score", 0.29)
mlflow.log_metric("precision", 0.18)
mlflow.log_metric("recall", 0.69)
mlflow.log_metric("accuracy", 0.73)

# With split suffix
mlflow.log_metric("roc_auc_train", 0.85)
mlflow.log_metric("roc_auc_val", 0.78)
mlflow.log_metric("roc_auc_test", 0.77)

# Business metrics
mlflow.log_metric("false_positive_rate", 0.27)
mlflow.log_metric("false_negative_rate", 0.31)

# Training metrics
mlflow.log_metric("training_time_seconds", 18.5)
mlflow.log_metric("prediction_time_ms", 2.3)
```

### **Avoid**
```python
# ❌ BAD
mlflow.log_metric("ROC-AUC", 0.78)  # Mixed case, hyphens
mlflow.log_metric("auc", 0.78)  # Ambiguous (ROC-AUC or PR-AUC?)
mlflow.log_metric("metric1", 0.78)  # Not descriptive
```

---

## 7️⃣ Artifact Naming

### **Convention**
- Use **descriptive names** with **underscores**
- Include **model name** in filename
- Use **standard extensions**
- Organize in **subdirectories** if many artifacts

### **Structure**
```
artifacts/
├── models/
│   ├── lgbm_v1_model.pkl
│   └── lgbm_v1_model.txt  # LightGBM text format
├── plots/
│   ├── lgbm_v1_roc_curve.png
│   ├── lgbm_v1_pr_curve.png
│   ├── lgbm_v1_confusion_matrix.png
│   └── lgbm_v1_feature_importance.png
├── reports/
│   ├── lgbm_v1_classification_report.txt
│   └── lgbm_v1_evaluation_summary.json
└── data/
    ├── lgbm_v1_predictions.csv
    └── lgbm_v1_feature_names.txt
```

### **Examples**
```python
# ✅ GOOD
mlflow.log_artifact("plots/lgbm_v1_roc_curve.png")
mlflow.log_artifact("models/lgbm_v1_model.pkl")
mlflow.log_artifact("reports/lgbm_v1_metrics.json")

# ❌ BAD
mlflow.log_artifact("plot.png")  # Not descriptive
mlflow.log_artifact("model_1.pkl")  # Generic numbering
mlflow.log_artifact("ROC-Curve.PNG")  # Mixed case, hyphens
```

---

## 8️⃣ Directory Structure

### **Recommended Structure**
```
Scoring_Model/
├── mlruns/                          # MLflow tracking data
│   ├── mlflow.db                    # SQLite backend
│   ├── 0/                           # Experiment 0 (Default)
│   ├── 1/                           # Experiment 1 (Baseline)
│   └── 2/                           # Experiment 2 (Optimization)
├── models/                          # Saved models (outside MLflow)
│   ├── baseline/
│   │   ├── lgbm_v1_default.pkl
│   │   └── xgboost_v1_default.pkl
│   ├── optimized/
│   │   └── lgbm_v2_optimized.pkl
│   └── production/
│       └── lgbm_v3_production.pkl
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_hyperparameter_optimization.ipynb
│   └── 05_model_interpretation.ipynb
└── src/
    ├── config.py                    # MLflow configuration
    └── mlflow_utils.py              # MLflow helper functions
```

---

## 9️⃣ Configuration Centralization

### **Create `src/config.py`**
```python
"""
Centralized MLflow configuration for the project.
"""
from pathlib import Path

# Project settings
PROJECT_NAME = "credit_scoring"
PROJECT_ROOT = Path(__file__).parent.parent

# MLflow settings
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT}/mlruns/mlflow.db"
MLFLOW_ARTIFACT_ROOT = str(PROJECT_ROOT / "mlruns")

# Experiment names (standardized)
EXPERIMENTS = {
    "baseline": f"{PROJECT_NAME}_01_baseline",
    "optimization": f"{PROJECT_NAME}_02_optimization",
    "final_evaluation": f"{PROJECT_NAME}_03_final_evaluation",
    "production": f"{PROJECT_NAME}_04_production"
}

# Model registry names
REGISTERED_MODELS = {
    "lgbm": f"{PROJECT_NAME}_lgbm",
    "xgboost": f"{PROJECT_NAME}_xgboost",
    "ensemble": f"{PROJECT_NAME}_ensemble"
}

# Data version tracking
DATA_VERSION = "v2_comprehensive_318features"
```

### **Usage in Notebooks**
```python
import sys
sys.path.append('../')
from src.config import MLFLOW_TRACKING_URI, EXPERIMENTS, DATA_VERSION

# Set tracking URI (centralized)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Use standardized experiment names
mlflow.set_experiment(EXPERIMENTS["baseline"])

# Tag data version consistently
mlflow.set_tag("data_version", DATA_VERSION)
```

---

## 🔟 Model Registry Workflow

### **Registration Process**
```python
# 1. Train and log model
with mlflow.start_run(run_name="lgbm_v1_optimized"):
    model.fit(X_train, y_train)
    mlflow.sklearn.log_model(model, "model")
    run_id = mlflow.active_run().info.run_id

# 2. Register model
from src.config import REGISTERED_MODELS

model_uri = f"runs:/{run_id}/model"
model_version = mlflow.register_model(
    model_uri=model_uri,
    name=REGISTERED_MODELS["lgbm"]
)

# 3. Add version description
client = mlflow.tracking.MlflowClient()
client.update_model_version(
    name=REGISTERED_MODELS["lgbm"],
    version=model_version.version,
    description="Optimized LightGBM with Random Search. ROC-AUC: 0.79"
)

# 4. Transition to staging
client.transition_model_version_stage(
    name=REGISTERED_MODELS["lgbm"],
    version=model_version.version,
    stage="Staging"
)

# 5. After validation, promote to production
client.transition_model_version_stage(
    name=REGISTERED_MODELS["lgbm"],
    version=model_version.version,
    stage="Production",
    archive_existing_versions=True  # Demote old production
)
```

---

## ✅ Implementation Checklist

### Phase 1: Cleanup & Standardization
- [ ] Rename experiments to follow convention
- [ ] Create `src/config.py` with centralized settings
- [ ] Create `src/mlflow_utils.py` with helper functions
- [ ] Update all notebooks to use centralized config
- [ ] Add standardized tags to all runs

### Phase 2: Model Registry
- [ ] Initialize model registry
- [ ] Register best baseline models
- [ ] Register optimized models
- [ ] Set up staging/production workflow

### Phase 3: Documentation
- [ ] Document current model versions
- [ ] Create model cards for registered models
- [ ] Add deployment instructions

---

## 📚 References

- [MLflow Best Practices](https://www.mlflow.org/docs/latest/tracking.html)
- [MLflow Model Registry](https://www.mlflow.org/docs/latest/model-registry.html)
- [Experiment Tracking Patterns](https://neptune.ai/blog/ml-experiment-tracking)

---

**Version**: 1.0
**Last Updated**: 2023-12-06
**Maintained By**: Data Science Team
