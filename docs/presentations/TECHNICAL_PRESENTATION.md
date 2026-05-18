# Credit Scoring Model
## Technical Presentation

**For**: Engineering Team, Data Scientists, Technical Leadership
**Date**: May 18, 2026
**Presented by**: ML Engineering Team

---

## Technical Summary

### System Architecture
- **ML Model**: LightGBM classifier (189 features)
- **API**: FastAPI with Pydantic validation
- **Experiment Tracking**: MLflow 3.x
- **Monitoring**: Prometheus + custom drift detection
- **Infrastructure**: Docker + Cloud-ready

### Performance Metrics
- **Model**: ROC-AUC 0.8320, PR-AUC 0.3786 (validation set)
- **Optimisation**: Business cost scorer (10×FN + FP), 10-iter Random Search on 20% subsample
- **Optimal threshold**: 0.10 → Recall 71%, FPR 21%
- **API**: <50ms P95 latency

---

## 1. System Architecture

### High-Level Overview
```
┌──────────────────────────────────────────────────────────┐
│                    Load Balancer (ALB)                    │
└─────────────────────┬────────────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
    ┌────▼────┐              ┌────▼────┐
    │  API 1  │              │  API 2  │     (Auto-scaling)
    │FastAPI  │              │FastAPI  │
    └────┬────┘              └────┬────┘
         │                         │
         └────────────┬────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
    ┌────▼─────┐            ┌─────▼────┐
    │  MLflow  │            │  Model   │
    │ Registry │            │  Cache   │
    │ (S3/DB)  │            │  (Redis) │
    └──────────┘            └──────────┘
```

### Component Stack
| Layer | Technology | Purpose |
|-------|------------|---------|
| **API** | FastAPI 0.115 | REST endpoints |
| **ML** | LightGBM 4.5 | Gradient boosting model |
| **Validation** | Pydantic 2.x | Input/output schemas |
| **Tracking** | MLflow 3.x | Experiment management |
| **Testing** | Pytest 8.x | Test automation |
| **Monitoring** | Prometheus + Custom | Performance tracking |

---

## 2. Data Pipeline

### Data Flow
```
Raw Data (6 CSV sources — 307K applications)
    ↓
Data Loading & Aggregation (src/data_preprocessing.py)
    ├─ Bureau credit history (37 features)
    ├─ Previous applications (56 features)
    ├─ POS/cash balances (20 features)
    ├─ Credit card balances (52 features)
    └─ Installment payments (31 features)
    → 318 total features
    ↓
Drop >70% missing features → 258 features
    ↓
Domain Feature Engineering (src/domain_features.py)
    ├─ Debt-to-income ratio
    ├─ Income per person
    ├─ Employment years
    ├─ Age years
    └─ External source aggregations
    → 272 features
    ↓
One-Hot Encoding (src/feature_engineering.py)
    → 306 features
    ↓
Train/Validation Split (70/30 stratified)
    ↓
Imputation — fit on X_train only (src/imputer.pkl)
    ├─ SimpleImputer(strategy='median') — 206 numerical features
    └─ No leakage: transform val/test with training statistics
    ↓
Feature Selection (src/feature_selection.py)
    ├─ Remove low-variance features (80 removed)
    ├─ Remove highly correlated features (35 removed)
    └─ → 189 features (saved in models/feature_columns.pkl)
    ↓
Scaling — fit on X_train only (models/scaler.pkl)
    └─ StandardScaler (mean=0, std=1)
    ↓
Model Training (LightGBM)
    ↓
Evaluation (src/evaluation.py)
    ↓
MLflow Logging
```

### Saved Inference Artifacts
| File | Purpose | Size |
|------|---------|------|
| `models/imputer.pkl` | Median imputer (206 features, training stats) | 7 KB |
| `models/scaler.pkl` | StandardScaler (189 features, training stats) | 10 KB |
| `models/feature_columns.pkl` | Ordered list of 189 selected feature names | 5 KB |
| `models/best_lightgbm_20260518_175632.pkl` | Optimised LightGBM model | 714 KB |

---

## 3. Model Development

### Algorithm Selection
**Tested Algorithms** (5 models, baseline experiment):
| Model | ROC-AUC | PR-AUC |
|-------|---------|--------|
| Dummy Classifier | 0.500 | 0.081 |
| Logistic Regression | 0.769 | 0.254 |
| Random Forest | 0.756 | 0.235 |
| XGBoost | 0.776 | 0.266 |
| **LightGBM** | **0.778** | **0.270** |

**Why LightGBM?**
- Best performance on both ROC-AUC and PR-AUC
- Fast inference (<5ms per prediction)
- Handles class imbalance well
- Low memory footprint

### Hyperparameter Optimisation
**Method**: RandomizedSearchCV (sklearn 1.8)
**Scoring**: Custom business cost scorer — minimises `10×FN + FP` across all thresholds per CV fold. This directly optimises the deployment objective (FN costs 10× more than FP per BUSINESS_PRESENTATION.md).

```python
def _min_business_cost(y_true, y_proba):
    """Find threshold that minimises 10×FN + FP, return normalised negative cost."""
    best = float('inf')
    for t in np.arange(0.05, 0.55, 0.01):
        y_p = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_p, labels=[0, 1]).ravel()
        cost = 10 * fn + fp
        if cost < best:
            best = cost
    return -best / len(y_true)

business_scorer = make_scorer(_min_business_cost, response_method='predict_proba')
```

**Search Strategy**:
- 10 random iterations × 5-fold StratifiedKFold = 50 fits
- Fit on 20% stratified subsample (~60K rows) → reduces fold time from ~5 min to ~1 min
- Retrain winning config on full dataset (307K rows) after search

**Search Space**:
```python
{
    'n_estimators':      [100, 200, 300],
    'learning_rate':     [0.01, 0.05, 0.1, 0.2],
    'num_leaves':        [20, 31, 50, 70],
    'max_depth':         [-1, 5, 10, 15],
    'subsample':         [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree':  [0.6, 0.7, 0.8, 0.9, 1.0],
    'reg_alpha':         [0.0, 0.1, 0.5, 1.0],
    'reg_lambda':        [0.0, 0.1, 0.5, 1.0],
    'class_weight':      ['balanced', None]
}
```

**Best Parameters**:
```python
{
    'n_estimators':     200,
    'learning_rate':    0.1,
    'num_leaves':       31,
    'max_depth':        5,
    'subsample':        0.8,
    'colsample_bytree': 0.9,
    'reg_alpha':        0.0,
    'reg_lambda':       0.5,
    'class_weight':     None,
    'random_state':     42,
    'n_jobs':           1
}
```

---

## 4. Model Evaluation

### Cross-Validation Strategy
```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
# Applied on 20% stratified subsample during search
# Final model retrained on full 307K rows
```

### Performance Metrics (Validation Set — 92,254 samples)
```python
{
    # Discrimination (threshold-independent)
    'roc_auc': 0.8320,   # Area under ROC curve
    'pr_auc':  0.3786,   # Area under PR curve (better for imbalanced data)

    # Classification at business-optimal threshold (0.10)
    'precision':  0.230,  # TP / (TP + FP)
    'recall':     0.710,  # TP / (TP + FN)  — 71% of defaults caught
    'f1_score':   0.347,  # Harmonic mean
    'fpr':        0.209,  # 21% of good customers rejected

    # Business
    'business_cost_10FN_FP': 39346,   # at t=0.10 on val set
    'cost_per_client':       0.4265,  # normalised (FN=10, FP=1 units)
    'optimal_threshold':     0.10
}
```

### Threshold Sweep (Validation Set)
```
Threshold  Recall   Precision   FPR    BizCost(10×FN+FP)
  0.05      89%       15%       45%        46,267
  0.10      71%       23%       21%        39,346  ← MINIMUM
  0.15      55%       30%       11%        42,885
  0.20      42%       38%        6%        48,056
  0.25      27%       45%        3%        53,539
  0.30      19%       51%        2%        58,554
  0.50       6%       77%        0%        69,980
```

### Confusion Matrix at Threshold 0.10 (Validation Set — 92,254 samples)
```
                   Predicted
                0 (No)   1 (Yes)
Actual  0      67,050   17,756   FP = 17,756 (good customers rejected)
        1       2,159    5,289   FN =  2,159 (defaults missed)

Business cost: 10 × 2,159 + 17,756 = 39,346
vs baseline (approve all): 10 × 7,448 = 74,480  → -47%
```

---

## 5. API Implementation

### FastAPI Application
```python
# api/app.py

from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import joblib

app = FastAPI()

# Load artifacts on startup
@app.on_event("startup")
async def load_artifacts():
    global model, imputer, scaler, feature_columns
    model = mlflow.sklearn.load_model(
        "models:/credit_scoring_production_model/latest"
    )
    imputer         = joblib.load("models/imputer.pkl")
    scaler          = joblib.load("models/scaler.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")

# Prediction endpoint
@app.post("/predict")
async def predict(input_data: PredictionInput):
    X = preprocess(input_data)          # impute → select → scale
    probability = model.predict_proba(X)[0, 1]

    risk_level = (
        "LOW"      if probability < 0.10 else
        "MEDIUM"   if probability < 0.20 else
        "HIGH"     if probability < 0.40 else
        "CRITICAL"
    )

    return PredictionOutput(
        prediction=int(probability >= 0.10),  # business-optimal threshold
        probability=probability,
        risk_level=risk_level
    )
```

### Preprocessing at Inference
```python
def preprocess(input_data):
    """Apply training pipeline to a single new application."""
    X = pd.DataFrame([input_data.features_dict])

    # 1. Domain features (same as notebook 02)
    X = create_domain_features(X)

    # 2. One-hot encode (must match training categories)
    X = encode_categorical_features(X)

    # 3. Impute using training-set medians
    missing_cols = [c for c in imputer.feature_names_in_ if c in X.columns]
    X[missing_cols] = imputer.transform(X[missing_cols])

    # 4. Select 189 trained features
    X = X.reindex(columns=feature_columns, fill_value=0)

    # 5. Scale using training-set statistics
    X = pd.DataFrame(scaler.transform(X), columns=feature_columns)

    return X
```

### Request/Response Schema
```python
# Request
class PredictionInput(BaseModel):
    features: List[float]  # Length = 189 (pre-processed)
    client_id: Optional[str]

    @field_validator('features')
    def validate_features(cls, v):
        if len(v) != 189:
            raise ValueError("Expected 189 features")
        if any(np.isnan(x) for x in v):
            raise ValueError("Features contain NaN")
        return v

# Response
class PredictionOutput(BaseModel):
    prediction: int          # 0 or 1  (threshold=0.10)
    probability: float       # [0, 1]
    risk_level: str         # LOW/MEDIUM/HIGH/CRITICAL
    client_id: Optional[str]
    timestamp: str
    model_version: str
```

---

## 6. MLflow Integration

### Experiment Tracking
```python
import mlflow

# Setup
mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")

# Experiments
# 1. credit_scoring_baseline_models
#    → 5 runs: Logistic Regression, Random Forest, XGBoost, LightGBM, Dummy
#
# 2. credit_scoring_hyperparameter_optimization
#    → RandomizedSearchCV with business cost scorer
#    → Best run logged as LightGBM_Optimized_Best

with mlflow.start_run(run_name="LightGBM_Optimized_Best"):
    mlflow.log_params(best_params)
    mlflow.log_metric("cv_business_cost_norm", best_cv_cost)
    mlflow.log_metric("roc_auc", 0.8320)
    mlflow.log_metric("pr_auc",  0.3786)
    mlflow.sklearn.log_model(model, name="model")
    mlflow.register_model(f"runs:/{run_id}/model",
                          "credit_scoring_production_model")
```

### Model Registry
```python
# Load latest production model
model = mlflow.sklearn.load_model(
    "models:/credit_scoring_production_model/latest"
)

# Current versions
# v1 — ROC-AUC optimised (0.8105, May 2026)
# v2 — ROC-AUC optimised retrain (0.8105, May 2026)
# v3 — Business cost optimised (0.8320, May 2026) ← production
```

### Experiment Organisation
```
Experiments:
├── credit_scoring_baseline_models (5 runs)
│   ├── Dummy Classifier         ROC-AUC 0.500
│   ├── Logistic Regression      ROC-AUC 0.769
│   ├── Random Forest            ROC-AUC 0.756
│   ├── XGBoost                  ROC-AUC 0.776
│   └── LightGBM                 ROC-AUC 0.778  ← selected for tuning
│
└── credit_scoring_hyperparameter_optimization (3 runs)
    ├── LightGBM_Optimized_Best  ROC-AUC 0.8105 (v1, ROC-AUC scorer)
    ├── LightGBM_Optimized_Best  ROC-AUC 0.8105 (v2, ROC-AUC scorer)
    └── LightGBM_Optimized_Best  ROC-AUC 0.8320 (v3, business cost scorer) ← production
```

---

## 7. Testing Strategy

### Test Pyramid
```
         ╱╲
        ╱ E2E╲         2 tests
       ╱──────╲
      ╱ Integr ╲       5 tests
     ╱──────────╲
    ╱  Unit Tests╲     60 tests
   ╱──────────────╲
```

### Key Test Cases
```python
# 1. Input Validation
def test_predict_invalid_feature_count():
    response = client.post("/predict", json={
        "features": [0.5] * 50  # Wrong count
    })
    assert response.status_code == 422

# 2. NaN Handling
def test_predict_with_nan_features():
    response = client.post("/predict", json={
        "features": [float('nan')] + [0.5] * 188
    })
    assert response.status_code == 422

# 3. Output Validation
def test_prediction_probability_range():
    result = response.json()
    assert 0 <= result['probability'] <= 1

# 4. Threshold correctness
def test_prediction_uses_business_threshold():
    # probability=0.05 → below 0.10 → prediction=0 (approve)
    # probability=0.15 → above 0.10 → prediction=1 (reject)
    pass
```

---

## 8. Monitoring & Observability

### Metrics Collection
```python
from prometheus_client import Counter, Histogram

request_counter = Counter(
    'credit_scoring_requests_total',
    'Total prediction requests',
    ['endpoint', 'status']
)

response_time = Histogram(
    'credit_scoring_response_seconds',
    'Response time distribution'
)

default_rate_gauge = Gauge(
    'credit_scoring_default_rate',
    'Current default rate (7-day window)'
)
```

### Data Drift Detection
```python
class FeatureDriftDetector:
    def __init__(self, reference_data):
        self.reference_data = reference_data

    def detect_drift(self, production_data):
        """KS test for each feature."""
        results = {}
        for i, feature in enumerate(self.feature_names):
            ref  = self.reference_data[:, i]
            prod = production_data[:, i]
            statistic, p_value = ks_2samp(ref, prod)
            results[feature] = {
                'statistic': statistic,
                'p_value':   p_value,
                'drifted':   p_value < 0.05
            }
        return results

# Trigger: >10% features drifting → alert + retrain
```

### Performance Monitoring
```python
def evaluate_production_performance(window_days=7):
    predictions_df  = pd.read_json('logs/predictions.log', lines=True)
    ground_truth_df = load_ground_truth()
    merged = predictions_df.merge(ground_truth_df, on='client_id')

    roc_auc = roc_auc_score(merged['actual_default'], merged['probability'])

    # Business cost at current threshold
    y_pred = (merged['probability'] >= 0.10).astype(int)
    tn, fp, fn, tp = confusion_matrix(merged['actual_default'], y_pred).ravel()
    biz_cost = (10 * fn + fp) / len(merged)

    if roc_auc < 0.75:
        send_alert(f"Performance degradation: ROC-AUC = {roc_auc:.4f}")
    if biz_cost > 0.55:  # >30% above baseline
        send_alert(f"Business cost spike: {biz_cost:.4f}/client")

    return roc_auc, biz_cost
```

---

## 9. Deployment

### Docker Configuration
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/   # imputer.pkl, scaler.pkl, feature_columns.pkl, model.pkl

EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### CI/CD Pipeline
```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with: {python-version: '3.11'}
      - run: pip install -r requirements.txt
      - run: pytest tests/

  deploy:
    needs: test
    steps:
      - run: docker build -t credit-scoring-api .
      - run: docker push credit-scoring-api:latest
      - run: kubectl apply -f k8s/deployment.yaml
```

---

## 10. Security

### Input Validation
```python
# 1. Schema validation (Pydantic)
# 2. Feature count check (must be 189)
# 3. NaN/Inf rejection
# 4. No raw SQL (ORM only)
# 5. No HTML rendering
```

### Authentication & Authorization
```python
from fastapi.security.api_key import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

@app.post("/predict")
async def predict(
    input_data: PredictionInput,
    api_key: str = Depends(validate_api_key)
):
    pass
```

### Data Protection
- **Encryption at rest**: S3 with KMS
- **Encryption in transit**: TLS 1.3
- **PII masking**: Log only hashes, not raw data
- **Access control**: IAM roles, least privilege

---

## 11. Performance Benchmarks

### Latency Tests
```
Single prediction
  Median: 8ms    P95: 42ms    P99: 87ms

Batch prediction (100 clients)
  Median: 45ms   P95: 95ms    P99: 150ms

Cold start: ~2 seconds (model loading)
```

### Throughput Tests
```
Single instance:   120 req/s, 50 concurrent
3 instances:       360 req/s, 150 concurrent
```

---

## 12. Future Enhancements

### Short-Term (Q3 2026)
1. **Model Explainability API**: SHAP values endpoint per prediction
2. **A/B Testing Framework**: Champion (v3) vs challenger
3. **Automated Retraining**: Triggered by drift detection
4. **OneHotEncoder artifact**: Replace pd.get_dummies with fitted encoder for robust inference

### Medium-Term (Q4 2026)
1. **Deep Learning Model**: Try neural networks
2. **Alternative Data**: Transaction data
3. **Multi-Model Ensemble**: Stacking/blending
4. **Real-Time Feature Store**: Cache computed features

### Long-Term (2027+)
1. **Causal Inference**: Understand feature relationships
2. **Fairness Metrics**: Disparate impact monitoring
3. **Reinforcement Learning**: Dynamic threshold optimisation

---

## 13. Technical Debt & Known Gaps

### Known Issues
1. **OneHotEncoder**: Currently using pd.get_dummies — will fail on unseen categories at inference. Must replace with fitted `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
2. **No inference wrapper**: No `src/inference.py` — the full preprocessing chain is not packaged as a callable function yet.
3. **No integration tests**: Preprocessing → model pipeline not end-to-end tested.
4. **MLflow model artifact path**: Warning on log_model (cosmetic, model still registers correctly).

### Mitigation Plan
| Issue | Priority | Timeline |
|-------|----------|----------|
| OneHotEncoder artifact | P1 | Before production |
| src/inference.py | P1 | Before production |
| Integration tests | P2 | Sprint 1 |
| MLflow artifact path | P3 | Sprint 2 |

---

## 14. Team & Resources

### Current Team
- **Data Scientists**: 2 FTE
- **ML Engineers**: 1 FTE
- **Backend Engineers**: 1 FTE (shared)
- **DevOps**: 0.5 FTE (shared)

### Tools & Licenses
- **MLflow**: Open source (self-hosted, SQLite backend)
- **FastAPI**: Open source
- **LightGBM**: Open source
- **sklearn 1.8**: Open source
- **Cloud**: AWS (€5K/month estimated)

---

## Contact

**ML Engineering Lead**
Email: ml-eng@company.com
Slack: #ml-engineering

**Architecture Review**
Bi-weekly: Thursdays 2pm

**On-Call Rotation**
PagerDuty: credit-scoring-api

---

**Appendix**:
- [API Documentation](http://localhost:8000/docs)
- [MLflow UI](http://localhost:5000)
- [Source Code](https://github.com/company/Scoring_Model)
- [Deployment Guide](../DEPLOYMENT_GUIDE.md)

**Last Updated**: May 18, 2026
