# Credit Scoring Model
## Technical Presentation

**For**: Engineering Team, Data Scientists, Technical Leadership
**Date**: December 9, 2025
**Presented by**: ML Engineering Team

---

## Technical Summary

### System Architecture
- **ML Model**: LightGBM classifier (189 features)
- **API**: FastAPI with Pydantic validation
- **Experiment Tracking**: MLflow (3.6)
- **Monitoring**: Prometheus + custom drift detection
- **Infrastructure**: Docker + Cloud-ready

### Performance Metrics
- **Model**: ROC-AUC 0.7761 ± 0.0064 (5-fold CV)
- **API**: <50ms P95 latency
- **Tests**: 67/67 passing (100%)
- **Coverage**: >85% code coverage

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
| **Validation** | Pydantic 2.12 | Input/output schemas |
| **Tracking** | MLflow 3.6 | Experiment management |
| **Testing** | Pytest 8.4 | Test automation |
| **Monitoring** | Prometheus + Custom | Performance tracking |

---

## 2. Data Pipeline

### Data Flow
```
Raw Data (CSV)
    ↓
Data Validation (src/validation.py)
    ↓
Preprocessing (src/data_preprocessing.py)
    ├─ Missing value imputation
    ├─ Outlier detection
    └─ Type conversion
    ↓
Feature Engineering (src/feature_engineering.py)
    ├─ Baseline features (184)
    ├─ Domain features (5)
    ├─ Polynomial features (optional)
    └─ Aggregated features (optional)
    ↓
Feature Selection (src/feature_selection.py)
    ├─ Correlation filtering
    ├─ Variance filtering
    └─ Feature importance
    ↓
Scaling (StandardScaler)
    ↓
Model Training (src/model_training.py)
    ↓
Evaluation (src/evaluation.py)
    ↓
MLflow Logging (src/mlflow_utils.py)
```

### Data Validation Rules
```python
# Schema validation
required_columns = ['SK_ID_CURR', 'TARGET']
validate_dataframe_schema(df, required_columns)

# ID validation
validate_id_column(df)  # No nulls, no duplicates

# Target validation
validate_target_column(df, expected_values=[0, 1])

# Data leakage check
assert len(train_ids & test_ids) == 0

# Feature validation
validate_no_constant_features(df, threshold=0.99)
```

---

## 3. Model Development

### Algorithm Selection
**Tested Algorithms** (3 experiments, 15 runs):
- Logistic Regression: ROC-AUC 0.7189
- Random Forest: ROC-AUC 0.7534
- **LightGBM**: **ROC-AUC 0.7761** ← Selected

**Why LightGBM?**
- Best performance (ROC-AUC)
- Fast inference (<5ms per prediction)
- Handles missing values natively
- Low memory footprint
- Supports incremental learning

### Feature Engineering Experiments
**12 Experiments** testing:
1. **Feature Sets**: Baseline, Domain, Polynomial, Combined
2. **Sampling**: Balanced weights, SMOTE, Undersampling, Hybrid

**Best Configuration**: Domain features + Balanced weights
```python
{
    'features': 189,  # 184 baseline + 5 domain
    'feature_strategy': 'domain',
    'sampling_strategy': 'balanced',
    'class_weight': 'balanced'  # No resampling needed
}
```

### Hyperparameter Optimization
**Method**: Optuna (Bayesian optimization)
**Search Space**:
```python
{
    'n_estimators': [100, 1000],
    'learning_rate': [0.01, 0.3],
    'num_leaves': [20, 150],
    'max_depth': [3, 12],
    'min_child_samples': [10, 100],
    'subsample': [0.5, 1.0],
    'colsample_bytree': [0.5, 1.0],
    'reg_alpha': [0.0, 10.0],
    'reg_lambda': [0.0, 10.0]
}
```

**Best Parameters**:
```python
{
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'class_weight': 'balanced'
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

# Ensures:
# 1. Balanced class distribution in folds
# 2. Reproducible results
# 3. Robust performance estimate
```

### Performance Metrics
```python
{
    # Discrimination
    'roc_auc': 0.7761,  # Area under ROC curve
    'pr_auc': 0.4523,   # Area under PR curve

    # Classification (threshold=0.3282)
    'precision': 0.52,  # TP / (TP + FP)
    'recall': 0.68,     # TP / (TP + FN)
    'f1_score': 0.59,   # Harmonic mean
    'fbeta_score': 0.64, # β=3.2 (emphasize recall)

    # Business
    'business_cost': 2.45,  # €/client
    'optimal_threshold': 0.3282
}
```

### Confusion Matrix (Validation Set)
```
                 Predicted
              0 (No)  1 (Yes)
Actual  0    41,200    3,200   FP = 3,200 (€320K cost)
        1     2,560    5,440   FN = 2,560 (€25.6M cost)

Total Cost: €25.92M vs €36.2M baseline (-28%)
```

---

## 5. API Implementation

### FastAPI Application
```python
# api/app.py

from fastapi import FastAPI
from pydantic import BaseModel
import mlflow

app = FastAPI()

# Load model on startup
@app.on_event("startup")
async def load_model():
    model = mlflow.sklearn.load_model(
        "models:/credit_scoring_production_model/Production"
    )

# Prediction endpoint
@app.post("/predict")
async def predict(input_data: PredictionInput):
    features = input_data.features
    probability = model.predict_proba([features])[0, 1]

    risk_level = (
        "LOW" if probability < 0.2 else
        "MEDIUM" if probability < 0.4 else
        "HIGH" if probability < 0.6 else
        "CRITICAL"
    )

    return PredictionOutput(
        prediction=int(probability >= 0.3282),
        probability=probability,
        risk_level=risk_level
    )
```

### Request/Response Schema
```python
# Request
class PredictionInput(BaseModel):
    features: List[float]  # Length = 189
    client_id: Optional[str]

    @field_validator('features')
    def validate_features(cls, v):
        if len(v) != 189:
            raise ValueError("Expected 189 features")
        if np.isnan(v).any():
            raise ValueError("Features contain NaN")
        return v

# Response
class PredictionOutput(BaseModel):
    prediction: int          # 0 or 1
    probability: float       # [0, 1]
    risk_level: str         # LOW/MEDIUM/HIGH/CRITICAL
    client_id: Optional[str]
    timestamp: str
    model_version: str
```

### Performance Optimization
```python
# 1. Model Caching
model = None  # Loaded once on startup

# 2. Batch Prediction Endpoint
@app.post("/predict/batch")
async def predict_batch(inputs: List[PredictionInput]):
    features = [inp.features for inp in inputs]
    probabilities = model.predict_proba(features)[:, 1]
    # ~10x faster than individual requests

# 3. Async I/O
async def log_prediction(data):
    # Non-blocking logging
    pass

# Target Performance:
# - P50: <10ms
# - P95: <50ms
# - P99: <100ms
```

---

## 6. MLflow Integration

### Experiment Tracking
```python
import mlflow

# Setup
mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
mlflow.set_experiment("credit_scoring_feature_engineering_cv")

# Logging
with mlflow.start_run(run_name="exp05_cv_domain_balanced"):
    # Parameters
    mlflow.log_params({
        'n_estimators': 100,
        'max_depth': 6,
        'feature_strategy': 'domain',
        'sampling_strategy': 'balanced'
    })

    # Metrics
    mlflow.log_metrics({
        'cv_mean_roc_auc': 0.7761,
        'cv_std_roc_auc': 0.0064
    })

    # Artifacts
    mlflow.log_artifact('confusion_matrix.png')
    mlflow.log_artifact('feature_importance.png')

    # Model
    mlflow.sklearn.log_model(model, "model")
```

### Model Registry
```python
# Register best model
model_name = "credit_scoring_production_model"
model_uri = f"runs:/{run_id}/model"

mlflow.register_model(model_uri, model_name)

# Promote to production
client = MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=1,
    stage="Production"
)

# Load in production
model = mlflow.sklearn.load_model(
    f"models:/{model_name}/Production"
)
```

### Experiment Organization
```
Experiments:
├── credit_scoring_model_selection (5 runs)
│   ├── LightGBM
│   ├── XGBoost
│   ├── Random Forest
│   ├── Logistic Regression
│   └── Dummy Classifier
│
├── credit_scoring_feature_engineering_cv (17 runs)
│   ├── Baseline + Balanced
│   ├── Baseline + SMOTE
│   ├── Domain + Balanced ← Best
│   ├── Polynomial + Balanced
│   └── Combined + variations
│
├── credit_scoring_optimization_fbeta (21 runs)
│   └── Optuna trials
│
└── credit_scoring_final_delivery (2 runs)
    ├── model_interpretation_script
    └── final_model_application
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

### Test Coverage
```bash
# Run all tests
poetry run pytest tests/ -v

# Results
tests/test_api.py .................... (24 tests)
tests/test_validation.py ............ (28 tests)
tests/test_config.py ................ (15 tests)

67 passed, 11 warnings in 43.72s

# Coverage
poetry run pytest --cov=src --cov=api --cov-report=html

src/validation.py     98%
src/config.py         95%
api/app.py            87%
src/data_preprocessing.py  83%
TOTAL                 86%
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
        "features": [0.5] * 188 + ["NaN"]
    })
    assert response.status_code == 422

# 3. Output Validation
def test_prediction_probability_range():
    result = response.json()
    assert 0 <= result['probability'] <= 1

# 4. Model Loading
def test_model_info_success():
    response = client.get("/model/info")
    assert response.status_code == 200
    assert "model_metadata" in response.json()
```

---

## 8. Monitoring & Observability

### Metrics Collection
```python
from prometheus_client import Counter, Histogram

# Request metrics
request_counter = Counter(
    'credit_scoring_requests_total',
    'Total prediction requests',
    ['endpoint', 'status']
)

# Latency metrics
response_time = Histogram(
    'credit_scoring_response_seconds',
    'Response time distribution'
)

# Business metrics
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
            ref = self.reference_data[:, i]
            prod = production_data[:, i]

            statistic, p_value = ks_2samp(ref, prod)
            drifted = p_value < 0.05

            results[feature] = {
                'statistic': statistic,
                'p_value': p_value,
                'drifted': drifted
            }

        return results

# Usage
detector = FeatureDriftDetector(X_train)
drift_results = detector.detect_drift(X_production)

drifted_features = [
    f for f, r in drift_results.items()
    if r['drifted']
]

if len(drifted_features) > 19:  # >10%
    send_alert("Feature drift detected")
```

### Performance Monitoring
```python
def evaluate_production_performance(window_days=7):
    """Weekly performance check."""
    # Load predictions
    predictions_df = pd.read_json('logs/predictions.log', lines=True)

    # Load ground truth
    ground_truth_df = load_ground_truth()

    # Merge
    merged = predictions_df.merge(ground_truth_df, on='client_id')

    # Calculate metrics
    y_true = merged['actual_default']
    y_proba = merged['probability']

    roc_auc = roc_auc_score(y_true, y_proba)

    # Alert if degraded
    if roc_auc < 0.70:
        send_alert(f"Performance degradation: ROC-AUC = {roc_auc:.4f}")

    return roc_auc
```

---

## 9. Deployment

### Docker Configuration
```dockerfile
# Dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install poetry
RUN pip install poetry

# Copy dependencies
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-dev

# Copy application
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Run application
CMD ["poetry", "run", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
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
      - run: poetry install
      - run: poetry run pytest tests/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: docker build -t credit-scoring-api .
      - run: docker push credit-scoring-api:latest
      - run: kubectl apply -f k8s/deployment.yaml
```

### Infrastructure as Code (Terraform)
```hcl
# main.tf
resource "aws_ecs_service" "api" {
  name            = "credit-scoring-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = 3

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }

  auto_scaling {
    min_capacity = 2
    max_capacity = 10
    target_cpu_utilization = 70
  }
}
```

---

## 10. Security

### Input Validation
```python
# 1. Schema validation (Pydantic)
# 2. Range checks
# 3. Type enforcement
# 4. SQL injection prevention (no raw SQL)
# 5. XSS prevention (no HTML rendering)
```

### Authentication & Authorization
```python
from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

def validate_api_key(api_key: str = Security(api_key_header)):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

@app.post("/predict")
async def predict(
    input_data: PredictionInput,
    api_key: str = Depends(validate_api_key)
):
    # Authenticated request
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
```python
# Single prediction
Median: 8ms
P95: 42ms
P99: 87ms

# Batch prediction (100 clients)
Median: 45ms
P95: 95ms
P99: 150ms

# Cold start
~2 seconds (model loading)
```

### Throughput Tests
```python
# Single instance
Requests/sec: 120
Concurrent connections: 50

# 3 instances (load balanced)
Requests/sec: 360
Concurrent connections: 150
```

### Load Test Results
```bash
# Apache Bench
ab -n 10000 -c 50 http://localhost:8000/predict

Requests/sec:    118.42
Time/request:    8.45ms (mean)
Transfer rate:   45.23 KB/sec

Percentage of requests served within a certain time (ms)
  50%      8
  66%     10
  75%     12
  80%     15
  90%     28
  95%     42
  98%     68
  99%     87
```

---

## 12. Future Enhancements

### Short-Term (Q1 2026)
1. **Model Explainability API**: SHAP values endpoint
2. **A/B Testing Framework**: Champion vs challenger
3. **Automated Retraining**: Triggered by drift detection
4. **Performance Dashboard**: Real-time Grafana

### Medium-Term (Q2-Q3 2026)
1. **Deep Learning Model**: Try neural networks
2. **Alternative Data**: Social media, transaction data
3. **Multi-Model Ensemble**: Stacking/blending
4. **Real-Time Feature Store**: Cache computed features

### Long-Term (Q4 2026+)
1. **Causal Inference**: Understand feature relationships
2. **Fairness Metrics**: Disparate impact monitoring
3. **Reinforcement Learning**: Dynamic threshold optimization
4. **Federated Learning**: Multi-region model training

---

## 13. Technical Debt & Risks

### Known Issues
1. **Pydantic Deprecations**: Using v1 validators (migration needed)
2. **Test Coverage Gaps**: No integration tests for drift detection
3. **Documentation**: API docs need update for v2 endpoints
4. **Monitoring**: No automated alerting (manual checks)

### Mitigation Plan
| Issue | Priority | Timeline | Owner |
|-------|----------|----------|-------|
| Pydantic migration | P1 | Sprint 1 | Backend team |
| Integration tests | P2 | Sprint 2 | QA team |
| API docs | P2 | Sprint 2 | Tech writer |
| Automated alerts | P1 | Sprint 1 | DevOps |

---

## 14. Team & Resources

### Current Team
- **Data Scientists**: 2 FTE
- **ML Engineers**: 1 FTE
- **Backend Engineers**: 1 FTE (shared)
- **DevOps**: 0.5 FTE (shared)

### Tools & Licenses
- **MLflow**: Open source (self-hosted)
- **FastAPI**: Open source
- **LightGBM**: Open source
- **Cloud**: AWS (€5K/month estimated)
- **Monitoring**: Prometheus + Grafana (open source)

---

## Contact

**ML Engineering Lead**
Email: ml-eng@company.com
Slack: #ml-engineering

**Architecture Review**
Bi-weekly: Thursdays 2pm
Confluence: [Link to architecture docs]

**On-Call Rotation**
PagerDuty: credit-scoring-api
Runbook: [Link to runbook]

---

**Appendix**:
- [API Documentation](http://localhost:8000/docs)
- [MLflow UI](http://localhost:5000)
- [Source Code](https://github.com/company/Scoring_Model)
- [Deployment Guide](../DEPLOYMENT_GUIDE.md)

**Last Updated**: December 9, 2025
