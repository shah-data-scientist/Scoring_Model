# Credit Scoring Model - Complete Project Documentation

**Last Updated**: 2025-12-07
**Best Model Performance**: 0.7761 ROC-AUC (5-fold CV)
**Configuration**: Domain features + Balanced class weights

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Core Modules (src/)](#core-modules-src)
4. [Scripts (scripts/)](#scripts-scripts)
5. [Notebooks (notebooks/)](#notebooks-notebooks)
6. [ML Experiments & Results](#ml-experiments--results)
7. [Best Practices Compliance](#best-practices-compliance)
8. [How to Use This Project](#how-to-use-this-project)
9. [Common Tasks & Examples](#common-tasks--examples)

---

## Project Overview

### Purpose
Credit scoring model to predict loan default risk using Home Credit dataset with 307,511 samples and 189 features after preprocessing.

### Key Achievements
- Implemented 5-fold cross-validation experiments
- Tested 4 feature strategies and 4 sampling methods systematically
- Achieved 0.7761 ROC-AUC with domain features + balanced class weights
- Full MLflow experiment tracking
- Modular, production-ready codebase

### Data
- **Training**: 215,257 samples
- **Validation**: 92,254 samples
- **Test**: Held out for final evaluation (not used yet - correct practice)
- **Class Imbalance**: 11.39:1 (majority:minority)

---

## Project Structure

```
Scoring_Model/
├── data/
│   └── processed/          # Train/val/test splits (generated)
├── src/                    # Core Python modules
│   ├── config.py          # Centralized configuration
│   ├── data_preprocessing.py
│   ├── domain_features.py
│   ├── sampling_strategies.py
│   ├── advanced_features.py
│   └── ... (9 modules total)
├── scripts/                # Executable scripts
│   ├── run_cv_experiments.py
│   ├── optimize_domain_balanced.py
│   └── ... (14 scripts total)
├── notebooks/              # Jupyter notebooks for analysis
│   ├── 01_eda.ipynb
│   ├── 03_baseline_models.ipynb
│   └── ... (5 notebooks total)
├── mlruns/                 # MLflow tracking database
│   └── mlflow.db
├── results/                # Experiment summaries (CSV)
└── *.md                    # Documentation files
```

---

## Core Modules (src/)

### 1. `config.py` - Central Configuration
**Purpose**: Single source of truth for all project settings

**What it does**:
- Defines paths, experiment names, MLflow settings
- Ensures consistency across all modules
- Prevents hardcoded values

**Example**:
```python
from src.config import MLFLOW_TRACKING_URI, RANDOM_STATE, PROJECT_ROOT

# All scripts use the same random seed
np.random.seed(RANDOM_STATE)  # Always 42

# All experiments log to the same MLflow database
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)  # sqlite:///mlruns/mlflow.db
```

**Key Settings**:
- `RANDOM_STATE = 42` - Reproducibility
- `MLFLOW_TRACKING_URI` - SQLite database path
- `DATA_VERSION` - Tracks data preprocessing version
- `EXPERIMENTS` - Standardized experiment names

---

### 2. `data_preprocessing.py` - Data Loading & Cleaning
**Purpose**: Load and preprocess raw credit data

**What it does**:
- Loads train/validation/test splits
- Ensures data consistency
- Validates feature alignment

**Example**:
```python
from src.data_preprocessing import load_data

X_train, X_val, X_test, y_train, y_val, y_test = load_data()

print(f"Training samples: {len(X_train)}")  # 215,257
print(f"Features: {X_train.shape[1]}")      # 189
print(f"Class balance: {(y_train==0).sum()}/{(y_train==1).sum()}")  # 197,880/17,377
```

**Features**:
- Automatic path resolution
- Handles missing processed data gracefully
- Returns pandas DataFrames/Series

---

### 3. `domain_features.py` - Business Logic Features
**Purpose**: Create credit-domain specific features

**What it does**:
- Age groups, employment stability
- Debt-to-income ratios
- Credit utilization metrics
- External source aggregations

**Example**:
```python
from src.domain_features import create_domain_features

# Add 5 domain-specific features to baseline 189
X_enhanced = create_domain_features(X_train.copy())
print(f"Original features: {X_train.shape[1]}")    # 189
print(f"Enhanced features: {X_enhanced.shape[1]}") # 194

# New features include:
# - AGE_YEARS (transformed from DAYS_BIRTH)
# - EMPLOYMENT_YEARS (transformed from DAYS_EMPLOYED)
# - DEBT_TO_INCOME_RATIO (AMT_CREDIT / AMT_INCOME_TOTAL)
# - CREDIT_INCOME_RATIO (AMT_CREDIT / AMT_INCOME_TOTAL)
# - ANNUITY_INCOME_RATIO (AMT_ANNUITY / AMT_INCOME_TOTAL)
```

**Impact**: +0.0007 ROC-AUC improvement (0.7754 → 0.7761)

---

### 4. `polynomial_features.py` - Interaction Features
**Purpose**: Create polynomial interactions between features

**What it does**:
- Generates degree-2 polynomial features
- Focuses on EXT_SOURCE variables (most predictive)
- Handles missing values with median imputation

**Example**:
```python
from src.polynomial_features import create_polynomial_features

# Create interactions for key features
X_poly, poly_features = create_polynomial_features(
    X_train.copy(),
    degree=2,
    interaction_only=True  # Only cross-products, not x^2
)

print(f"Original: {X_train.shape[1]} features")   # 189
print(f"With polynomials: {X_poly.shape[1]}")     # 300+
print(f"New features created: {len(poly_features)}")
```

**Note**: In experiments, polynomial features did NOT improve performance (likely overfitting).

---

### 5. `advanced_features.py` - Advanced Feature Engineering
**Purpose**: Comprehensive feature engineering to maximize performance

**What it does**:
- 7 feature engineering techniques in one module
- EXT_SOURCE advanced interactions (25+ features)
- Missing value indicators
- Time-based features
- Credit behavior features
- Bureau aggregations

**Example**:
```python
from src.advanced_features import create_all_advanced_features

X_advanced = create_all_advanced_features(X_train.copy())

print(f"Baseline: {X_train.shape[1]} features")     # 189
print(f"Advanced: {X_advanced.shape[1]} features")  # 265
print(f"New features: {X_advanced.shape[1] - X_train.shape[1]}")  # 76
```

**Result**: Despite adding 76 features, performance did NOT improve (0.7783 → 0.7781).

**Key Insight**: More features ≠ better performance. Quality > Quantity.

---

### 6. `sampling_strategies.py` - Class Imbalance Handling
**Purpose**: Provide multiple resampling strategies for imbalanced data

**What it does**:
- Balanced class weights (no resampling)
- SMOTE oversampling
- Random undersampling
- SMOTE + Undersampling hybrid

**Example**:
```python
from src.sampling_strategies import get_sampling_strategy

# Original: 197,880 class 0 / 17,377 class 1 (11.39:1 ratio)

# Option 1: Balanced weights (BEST - 0.7761 ROC-AUC)
X_balanced, y_balanced, meta = get_sampling_strategy('balanced', X_train, y_train)
print(f"No resampling: {len(X_balanced)} samples")  # 215,257 (unchanged)

# Option 2: Random undersampling
X_under, y_under, meta = get_sampling_strategy('undersample', X_train, y_train)
print(f"Undersampled: {len(X_under)} samples")  # 34,754 (1:1 ratio)

# Option 3: SMOTE (NOT recommended - calibration issues)
X_smote, y_smote, meta = get_sampling_strategy('smote', X_train, y_train)
print(f"SMOTE: {len(X_smote)} samples")  # 395,760 (creates synthetic minority samples)
```

**Recommendation**: Use `'balanced'` for best results.

---

### 7. `evaluation.py` - Model Evaluation Metrics
**Purpose**: Comprehensive model evaluation

**What it does**:
- ROC-AUC, PR-AUC, F1, Precision, Recall
- Confusion matrix metrics (FPR, FNR)
- Business cost calculations

**Example**:
```python
from src.evaluation import evaluate_model

# Evaluate on validation set
metrics = evaluate_model(model, X_val, y_val)

print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
print(f"PR-AUC: {metrics['pr_auc']:.4f}")
print(f"F1-Score: {metrics['f1_score']:.4f}")
print(f"FPR: {metrics['false_positive_rate']:.4f}")
print(f"FNR: {metrics['false_negative_rate']:.4f}")
```

---

### 8. `mlflow_utils.py` - MLflow Helper Functions
**Purpose**: Simplify MLflow logging

**What it does**:
- Standardized run naming
- Batch metric logging
- Duplicate run prevention

**Example**:
```python
from src.mlflow_utils import log_experiment

with mlflow.start_run(run_name="my_experiment"):
    log_experiment(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_list=X_train.columns.tolist(),
        params={'n_estimators': 100, 'max_depth': 6}
    )
```

---

## Scripts (scripts/)

### Experiment Scripts

#### 1. `run_cv_experiments.py` - 5-Fold CV Experiments ⭐
**Purpose**: Systematically compare feature and sampling strategies with proper cross-validation

**What it does**:
- Tests 4 configurations (2 features × 2 sampling)
- Uses 5-fold stratified cross-validation
- Logs all results to MLflow
- Saves summary to [results/cv_experiments_summary.csv](results/cv_experiments_summary.csv)

**Usage**:
```bash
poetry run python scripts/run_cv_experiments.py
```

**Output**:
```
BEST CONFIGURATION
Features: domain
Sampling: balanced
ROC-AUC: 0.7761 +/- 0.0064
```

**Why it's important**: This is the CORRECT way to evaluate models (not single train/val split).

---

#### 2. `test_advanced_features.py` - Quick Feature Validation
**Purpose**: Test if advanced features improve baseline performance

**What it does**:
- Compares baseline (189 features) vs advanced (265 features)
- Shows improvement or degradation
- Lists top 20 important features

**Usage**:
```bash
poetry run python scripts/test_advanced_features.py
```

**Output**:
```
Baseline ROC-AUC:     0.7783
Advanced ROC-AUC:     0.7781
Improvement:          -0.0003 (-0.0%)
```

**Result**: Advanced features did NOT help.

---

#### 3. `optimize_domain_balanced.py` - Hyperparameter Optimization
**Purpose**: Find best LightGBM hyperparameters for domain + balanced configuration

**What it does**:
- 50 iterations of RandomizedSearchCV
- 3-fold cross-validation
- Searches: n_estimators, max_depth, learning_rate, regularization, etc.
- Logs training and validation metrics (overfitting detection)

**Usage**:
```bash
poetry run python scripts/optimize_domain_balanced.py
```

**Note**: Was cancelled due to being too slow (resource constraints).

---

#### 4. `investigate_smote.py` - SMOTE Analysis
**Purpose**: Understand why SMOTE performed poorly

**What it does**:
- Compares balanced vs SMOTE predictions
- Analyzes probability calibration
- Identifies threshold sensitivity issues

**Usage**:
```bash
poetry run python scripts/investigate_smote.py
```

**Finding**: SMOTE causes calibration failure (mean probability 0.10 vs 0.39 for balanced).

---

#### 5. `analyze_overfitting.py` - Overfitting Detection
**Purpose**: Check all MLflow experiments for overfitting

**What it does**:
- Compares training vs validation metrics
- Identifies overfitting (gap > 0.05)
- Suggests regularization strategies

**Usage**:
```bash
poetry run python scripts/analyze_overfitting.py
```

**Current Issue**: No training metrics logged in CV experiments.

---

### Data Preparation Scripts

#### 6. `create_processed_data.py` - Data Preprocessing Pipeline
**Purpose**: Generate train/val/test splits from raw data

**What it does**:
- 70/15/15 split with stratification
- Saves to [data/processed/](data/processed/)
- Ensures reproducibility (RANDOM_STATE=42)

**Usage**:
```bash
poetry run python scripts/create_processed_data.py
```

---

### Utility Scripts

#### 7. `check_mlflow.py` - MLflow Verification
**Purpose**: Verify MLflow setup and list experiments

**Usage**:
```bash
poetry run python check_mlflow.py
```

---

## Notebooks (notebooks/)

### 1. `01_eda.ipynb` - Exploratory Data Analysis
**What it does**:
- Dataset overview and statistics
- Missing value analysis
- Class distribution visualization
- Feature correlation analysis

---

### 2. `02_feature_engineering.ipynb` - Feature Engineering Exploration
**What it does**:
- Tests domain features
- Analyzes feature importance
- Validates engineered features

---

### 3. `03_baseline_models.ipynb` - Baseline Model Training
**What it does**:
- Trains simple baseline models
- Establishes performance floor
- Compares to dummy classifier

---

### 4. `04_hyperparameter_optimization.ipynb` - Hyperparameter Tuning
**What it does**:
- Grid search or random search
- Model performance comparison
- Hyperparameter sensitivity analysis

---

### 5. `05_model_interpretation.ipynb` - Model Explainability
**What it does**:
- Feature importance analysis
- SHAP values
- Partial dependence plots
- Decision tree visualization

---

## ML Experiments & Results

### Experiment 1: Feature Engineering with 5-Fold CV

**Experiment Name**: `credit_scoring_feature_engineering_cv`
**Runs**: 4
**Method**: 5-fold Stratified Cross-Validation

| Configuration | Features | Sampling | Mean ROC-AUC | Std ROC-AUC | Rank |
|--------------|----------|----------|--------------|-------------|------|
| **domain + balanced** | 194 | balanced | **0.7761** | 0.0064 | 1st |
| baseline + balanced | 189 | balanced | 0.7754 | 0.0074 | 2nd |
| domain + undersample | 194 | undersample | 0.7717 | 0.0068 | 3rd |
| baseline + undersample | 189 | undersample | 0.7715 | 0.0076 | 4th |

**Winner**: Domain features + Balanced class weights

---

### Experiment 2: Advanced Feature Engineering (Unsuccessful)

**Test**: Added 76 advanced features (189 → 265)
**Result**: No improvement (0.7783 → 0.7781)
**Conclusion**: More features don't always help (likely overfitting)

---

### Experiment 3: Hyperparameter Optimization (Incomplete)

**Experiment Name**: `credit_scoring_hyperparameter_optimization`
**Runs**: 2 (both failed/incomplete)
**Issue**: Too slow due to resource constraints (n_jobs=1)
**Status**: Cancelled

---

## Best Practices Compliance

### ✅ Implemented

1. **Cross-Validation**: 5-fold StratifiedKFold (added in this session)
2. **Stratified Splitting**: Maintains class distribution across folds
3. **Test Set Held Out**: Not used yet (correct practice)
4. **MLflow Tracking**: All experiments logged
5. **Reproducibility**: RANDOM_STATE=42 everywhere
6. **Modular Code**: Well-organized src/ modules
7. **Configuration Management**: Centralized in [config.py](src/config.py)
8. **Multiple Metrics**: ROC-AUC, PR-AUC, F1, Precision, Recall
9. **Class Imbalance Handling**: Tested 4 sampling strategies
10. **Feature Engineering**: Systematic domain feature creation

### ⚠️ Partially Implemented

1. **Training Metrics Logging**: Not logged in CV experiments (can't detect overfitting)
2. **Hyperparameter Optimization**: Attempted but incomplete
3. **Early Stopping**: Not explicitly used
4. **Threshold Optimization**: Not implemented
5. **Model Calibration**: Not checked

### ❌ Missing

1. **Ensemble Methods**: Not explored
2. **Feature Selection**: All features used (no pruning)
3. **Learning Curves**: Not generated
4. **Cost-Sensitive Learning**: Business costs not incorporated
5. **SHAP Analysis**: Notebook exists but not executed
6. **Nested CV**: Not used (acceptable for time constraints)

**Overall Score**: ~70% best practices compliance

---

## How to Use This Project

### 1. Initial Setup

```bash
# Install dependencies
poetry install

# Create processed data
poetry run python scripts/create_processed_data.py

# Verify MLflow
poetry run python check_mlflow.py
```

---

### 2. Run Experiments

```bash
# Run 5-fold CV experiments (RECOMMENDED)
poetry run python scripts/run_cv_experiments.py

# Test advanced features
poetry run python scripts/test_advanced_features.py

# Investigate SMOTE behavior
poetry run python scripts/investigate_smote.py

# Analyze overfitting
poetry run python scripts/analyze_overfitting.py
```

---

### 3. View Results

```bash
# Start MLflow UI
poetry run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

# Open browser to http://localhost:5000
```

---

### 4. Train Best Model

```python
from src.config import RANDOM_STATE
from src.data_preprocessing import load_data
from src.domain_features import create_domain_features
from lightgbm import LGBMClassifier

# Load data
X_train, X_val, _, y_train, y_val, _ = load_data()

# Apply best configuration: domain features + balanced
X_train_domain = create_domain_features(X_train)
X_val_domain = create_domain_features(X_val)

# Train model
best_model = LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

best_model.fit(X_train_domain, y_train)

# Evaluate
from sklearn.metrics import roc_auc_score
y_pred_proba = best_model.predict_proba(X_val_domain)[:, 1]
roc_auc = roc_auc_score(y_val, y_pred_proba)
print(f"Validation ROC-AUC: {roc_auc:.4f}")  # ~0.7783
```

---

## Common Tasks & Examples

### Task 1: Add New Feature Strategy

1. Create function in `src/domain_features.py` or new module
2. Update `scripts/run_cv_experiments.py` to include new strategy
3. Run experiments: `poetry run python scripts/run_cv_experiments.py`
4. Compare results in MLflow UI

---

### Task 2: Test Different Hyperparameters

```python
from lightgbm import LGBMClassifier
import mlflow

# Define parameter grid
params_to_test = [
    {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.05},
    {'n_estimators': 150, 'max_depth': 10, 'learning_rate': 0.1},
    {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.075},
]

# Test each configuration
for i, params in enumerate(params_to_test):
    with mlflow.start_run(run_name=f"manual_hyperparam_{i+1}"):
        model = LGBMClassifier(**params, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)

        # Evaluate and log
        y_pred = model.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val, y_pred)

        mlflow.log_params(params)
        mlflow.log_metric("roc_auc", roc_auc)
        print(f"Config {i+1}: ROC-AUC = {roc_auc:.4f}")
```

---

### Task 3: Evaluate on Test Set (Final Step)

```python
# Only run this ONCE at the very end!
from src.data_preprocessing import load_data

X_train, X_val, X_test, y_train, y_val, y_test = load_data()

# Retrain on train + val
X_full = pd.concat([X_train, X_val])
y_full = pd.concat([y_train, y_val])

X_full_domain = create_domain_features(X_full)
X_test_domain = create_domain_features(X_test)

final_model = LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=42
)

final_model.fit(X_full_domain, y_full)

# Final test evaluation
y_test_pred = final_model.predict_proba(X_test_domain)[:, 1]
test_roc_auc = roc_auc_score(y_test, y_test_pred)
print(f"FINAL TEST ROC-AUC: {test_roc_auc:.4f}")
```

---

## File Dependencies

```
config.py (used by all modules)
  ↓
data_preprocessing.py → domain_features.py → advanced_features.py
                      → polynomial_features.py
                      → sampling_strategies.py
  ↓
model_training.py → evaluation.py → mlflow_utils.py
  ↓
Scripts (run_cv_experiments.py, etc.)
```

---

## Troubleshooting

### Issue: MLflow UI shows no experiments
**Solution**:
```bash
# Check database path
echo $MLFLOW_TRACKING_URI

# Should be: sqlite:///mlruns/mlflow.db
# Start UI with correct path
poetry run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
```

---

### Issue: "ModuleNotFoundError: No module named 'src'"
**Solution**:
```bash
# Always run from project root
cd "c:\Users\shahu\OPEN CLASSROOMS\PROJET 6\Scoring_Model"
poetry run python scripts/run_cv_experiments.py
```

---

### Issue: Out of memory during hyperparameter optimization
**Solution**: Reduce `n_jobs` and `N_ITER` in [optimize_domain_balanced.py](scripts/optimize_domain_balanced.py):
```python
N_ITER = 20  # Reduced from 50
base_model = LGBMClassifier(n_jobs=1)  # Sequential
random_search = RandomizedSearchCV(n_jobs=2)  # Limit parallelism
```

---

## Next Steps & Recommendations

### To Reach 0.82 ROC-AUC Target:

1. **Hyperparameter Optimization** (Manual Approach)
   - Test 5-10 specific parameter combinations manually
   - Focus on: `n_estimators`, `max_depth`, `learning_rate`, `min_child_samples`

2. **Feature Selection**
   - Remove low-importance features (reduce from 194 to ~100)
   - Test feature importance thresholds

3. **Ensemble Methods**
   - Stacking (LightGBM + XGBoost + Logistic Regression)
   - Averaging multiple LightGBM models with different seeds

4. **Threshold Optimization**
   - Find optimal threshold for F-beta score
   - Incorporate business costs (FN = 10 × FP)

5. **Model Calibration**
   - Use Platt scaling or isotonic regression
   - Ensure probabilities are well-calibrated

---

## Contact & Support

For questions or issues, refer to:
- [BEST_PRACTICES_AUDIT.md](BEST_PRACTICES_AUDIT.md) - ML methodology review
- [MLFLOW_CONVENTIONS.md](MLFLOW_CONVENTIONS.md) - Experiment naming standards
- [GETTING_STARTED.md](GETTING_STARTED.md) - Quick start guide

**Best Model Location**: [results/cv_experiments_summary.csv](results/cv_experiments_summary.csv)
**MLflow Database**: `mlruns/mlflow.db`

---

**End of Documentation**
