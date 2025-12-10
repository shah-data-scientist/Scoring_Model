# Technical Guide - Credit Scoring Model

## Overview

This document provides technical details about the credit scoring model, including architecture, model performance, features, and implementation details.

---

## Model Summary

### Best Model: LightGBM Classifier

**Performance Metrics**:
- **ROC-AUC**: 0.7761 ± 0.0064 (5-fold CV)
- **Precision**: 0.52
- **Recall**: 0.68
- **Optimal Threshold**: 0.3282
- **Business Cost**: €2.45/client (32% reduction from baseline)

**Training Details**:
- **Algorithm**: LightGBM (Gradient Boosting)
- **Cross-Validation**: 5-fold StratifiedKFold
- **Class Handling**: Balanced class weights
- **Random State**: 42 (reproducible)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     USER INTERFACES                      │
├──────────────┬──────────────────┬──────────────────────┤
│  MLflow UI   │   Dashboard      │    API Docs         │
│  Port 5000   │   Port 8501      │    Port 8000        │
└──────┬───────┴────────┬─────────┴──────────┬───────────┘
       │                │                    │
       ▼                ▼                    ▼
┌──────────────┐ ┌──────────┐ ┌─────────────────────┐
│   MLflow     │ │ Streamlit│ │     FastAPI         │
│   Tracking   │ │ Dashboard│ │     REST API        │
└──────┬───────┘ └─────┬────┘ └──────────┬──────────┘
       │               │                  │
       └───────────────┴──────────────────┘
                       │
                       ▼
       ┌───────────────────────────────┐
       │      Model Pipeline           │
       │  - Preprocessing              │
       │  - Feature Engineering        │
       │  - Model Prediction           │
       └───────────────┬───────────────┘
                       │
           ┌───────────┴──────────┐
           │                      │
           ▼                      ▼
    ┌──────────┐          ┌─────────────┐
    │  Model   │          │   MLflow    │
    │  Files   │          │  Registry   │
    └──────────┘          └─────────────┘
```

---

## Features (189 Total)

### Feature Categories

1. **Baseline Features (184)**
   - Original application data
   - Demographic information
   - Credit history
   - Financial indicators

2. **Domain-Engineered Features (5)**

| Feature | Description | Formula | Business Logic |
|---------|-------------|---------|----------------|
| **DEBT_TO_INCOME_RATIO** | Debt burden | `AMT_CREDIT / AMT_INCOME_TOTAL` | Higher = more risky |
| **EMPLOYMENT_YEARS** | Years employed | `DAYS_EMPLOYED / -365` | Longer = less risky |
| **INCOME_PER_PERSON** | Per-capita income | `AMT_INCOME_TOTAL / CNT_FAM_MEMBERS` | Higher = less risky |
| **AGE_YEARS** | Age in years | `DAYS_BIRTH / -365` | Older = less risky |
| **CREDIT_UTILIZATION** | Credit usage | `AMT_CREDIT / AMT_GOODS_PRICE` | Higher = more risky |

### Feature Importance (Top 10)

1. EXT_SOURCE_3 (0.084)
2. EXT_SOURCE_2 (0.079)
3. EXT_SOURCE_1 (0.068)
4. DAYS_BIRTH (0.043)
5. AMT_CREDIT (0.037)
6. AMT_GOODS_PRICE (0.035)
7. **DEBT_TO_INCOME_RATIO** (0.032) - Domain feature
8. DAYS_EMPLOYED (0.028)
9. AMT_ANNUITY (0.026)
10. REGION_RATING_CLIENT (0.024)

---

## Model Performance Details

### Cross-Validation Results

```
5-Fold Stratified Cross-Validation:
  Fold 1: ROC-AUC = 0.7823
  Fold 2: ROC-AUC = 0.7745
  Fold 3: ROC-AUC = 0.7701
  Fold 4: ROC-AUC = 0.7758
  Fold 5: ROC-AUC = 0.7779
  ──────────────────────────
  Mean:   ROC-AUC = 0.7761 ± 0.0064
```

### Confusion Matrix (Validation Set)

At optimal threshold (0.3282):

```
                  Predicted
                 No Default   Default
Actual
No Default       84.3%       15.7%
                (92.8%)     (33.1%)

Default          32.1%       67.9%
                 (7.2%)     (66.9%)

Values shown: Row % (Column %)
```

**Interpretation**:
- **True Negatives**: 84.3% of non-defaulters correctly identified
- **True Positives**: 67.9% of defaulters correctly caught (Recall)
- **False Positives**: 15.7% of non-defaulters incorrectly flagged
- **False Negatives**: 32.1% of defaulters missed

### Business Metrics

| Metric | Before (Baseline) | After (Optimized) | Improvement |
|--------|-------------------|-------------------|-------------|
| **Cost per Client** | €3.60 | €2.45 | **-32%** |
| **Default Detection Rate** | 45% | 68% | **+51%** |
| **False Alarm Rate** | 10% | 15.7% | +57% (trade-off) |
| **Annual Savings** | - | €25.5M | - |

---

## Data Pipeline

### 1. Data Loading
```python
# Load training data
df = pd.read_csv('data/application_train.csv')

# Shape: (307,511 rows, 122 columns)
# Target distribution: 8% default, 92% no default
```

### 2. Feature Engineering
```python
from src.domain_features import create_domain_features

# Add 5 domain-engineered features
df = create_domain_features(df)

# Total features: 122 + 5 = 127 input features
# After preprocessing: 189 features (one-hot encoding)
```

### 3. Preprocessing
- Handle missing values (imputation)
- One-hot encode categorical variables
- Scale numerical features (StandardScaler)
- Balance classes (class weights)

### 4. Model Training
```python
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

model = LGBMClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    class_weight='balanced',
    random_state=42
)

# 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
```

### 5. Threshold Optimization
```python
# Find threshold that minimizes business cost
# Cost = €10 * FN + €1 * FP

optimal_threshold = 0.3282  # Found via grid search
```

### 6. Model Registration
```python
import mlflow

with mlflow.start_run():
    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Log metrics
    mlflow.log_metric("roc_auc", 0.7761)
    mlflow.log_metric("optimal_threshold", 0.3282)

    # Register for production
    mlflow.register_model("runs:/{run_id}/model", "credit_scoring_model")
```

---

## API Implementation

### Technology Stack
- **Framework**: FastAPI 0.115+
- **Server**: Uvicorn (ASGI)
- **Validation**: Pydantic v2
- **Model Loading**: MLflow client

### Endpoints

#### GET /health
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": "1",
        "timestamp": datetime.now()
    }
```

#### POST /predict
```python
@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    # 1. Validate input (189 features)
    # 2. Load model from MLflow
    # 3. Generate prediction
    # 4. Apply threshold
    # 5. Return risk level

    probability = model.predict_proba([features])[0, 1]
    risk_level = get_risk_level(probability)

    return PredictionOutput(
        client_id=input_data.client_id,
        probability=probability,
        risk_level=risk_level,
        threshold=0.3282,
        model_version="1"
    )
```

### Input Validation

```python
from pydantic import BaseModel, Field, field_validator

class PredictionInput(BaseModel):
    features: List[float] = Field(..., min_length=189, max_length=189)
    client_id: str = Field(..., min_length=1)

    @field_validator('features')
    def validate_features(cls, v):
        # Check for NaN/Inf
        if any(not math.isfinite(x) for x in v):
            raise ValueError("Features contain NaN or Inf")
        return v
```

### Performance Optimization

- **Model Caching**: Load model once at startup
- **Async Endpoints**: Non-blocking I/O
- **Pydantic Validation**: Fast input validation
- **No Database**: Stateless predictions

**Results**:
- P95 Latency: <50ms
- Throughput: 120 req/sec
- Memory: ~200MB

---

## Testing Strategy

### Test Pyramid

```
        ┌─────────┐
        │   E2E   │  (6 tests)
        │  Tests  │
        └─────────┘
       ┌───────────┐
       │Integration│  (24 tests)
       │   Tests   │
       └───────────┘
      ┌─────────────┐
      │    Unit     │  (37 tests)
      │   Tests     │
      └─────────────┘
```

### Test Coverage

| Component | Tests | Coverage |
|-----------|-------|----------|
| **API** | 24 tests | 92% |
| **Validation** | 28 tests | 88% |
| **Config** | 15 tests | 81% |
| **Overall** | 67 tests | **86%** |

### Key Test Cases

**API Tests** (`test_api.py`):
- Health endpoint returns 200
- Predict with valid input returns 200
- Predict with invalid input returns 422
- Features length validation
- NaN/Inf rejection
- Client ID validation

**Validation Tests** (`test_validation.py`):
- Feature validation logic
- Input schema validation
- Boundary conditions
- Error handling

**Config Tests** (`test_config.py`):
- Configuration loading
- Path validation
- Parameter validation

---

## MLflow Integration

### Experiment Tracking

**Logged Information**:
- **Parameters**: Model hyperparameters, preprocessing steps
- **Metrics**: ROC-AUC, precision, recall, F-beta, business cost
- **Artifacts**: Plots (ROC, PR, confusion matrix, feature importance)
- **Tags**: Experiment name, feature set, class handling

### Model Registry

**Registered Model**: `credit_scoring_model`
- **Version 1**: Best LightGBM model (ROC-AUC: 0.7761)
- **Stage**: Production
- **Description**: Domain-engineered features, balanced classes

### Accessing MLflow

```python
from mlflow import MlflowClient

# Get production model
client = MlflowClient()
model_version = client.get_latest_versions("credit_scoring_model", stages=["Production"])[0]

# Load model
model = mlflow.sklearn.load_model(f"models:/credit_scoring_model/{model_version.version}")

# Get run metrics
run = client.get_run(model_version.run_id)
metrics = run.data.metrics
```

---

## Deployment

### Local Deployment (Development)

```bash
# Start all services
./launch_services.bat  # Windows
./launch_services.sh   # Linux/Mac
```

### Docker Deployment (Production)

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . /app

RUN pip install poetry
RUN poetry install --no-dev

EXPOSE 8000
CMD ["poetry", "run", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build
docker build -t credit-scoring-api .

# Run
docker run -p 8000:8000 credit-scoring-api
```

### Cloud Deployment Options

**AWS**:
- ECS/Fargate for containers
- Lambda for serverless API
- S3 for model artifacts
- RDS for MLflow database

**Azure**:
- AKS for containers
- Functions for serverless
- Blob Storage for artifacts
- Azure ML for MLflow

**GCP**:
- Cloud Run for containers
- Cloud Functions for serverless
- Cloud Storage for artifacts
- Vertex AI for MLflow

---

## Repository Structure

```
Scoring_Model/
├── src/                           # Core source code
│   ├── config.py                  # Configuration
│   ├── domain_features.py         # Feature engineering
│   ├── data_preprocessing.py      # Data cleaning
│   ├── model_training.py          # Training utilities
│   └── evaluation.py              # Metrics
│
├── api/                           # REST API
│   └── app.py                     # FastAPI application
│
├── scripts/                       # Utility scripts
│   ├── deployment/                # Service launchers
│   │   ├── start_all.py          # Launch all services
│   │   ├── start_api.py          # API only
│   │   └── dashboard.py          # Streamlit dashboard
│   ├── mlflow/                    # MLflow management
│   ├── experiments/               # ML experiments
│   └── data/                      # Data utilities
│
├── tests/                         # Test suite
│   ├── test_api.py               # API tests (24)
│   ├── test_validation.py        # Validation tests (28)
│   └── test_config.py            # Config tests (15)
│
├── docs/                          # Documentation
│   ├── USER_GUIDE.md             # User guide
│   ├── TECHNICAL_GUIDE.md        # This file
│   ├── MODEL_MONITORING.md       # Monitoring guide
│   ├── INDEX.md                  # Documentation index
│   └── presentations/            # Presentations
│
├── notebooks/                     # Jupyter notebooks
├── data/                          # Data files
├── mlruns/                        # MLflow tracking
├── models/                        # Saved models
└── results/                       # Outputs
```

---

## Dependencies

### Core Libraries

```toml
[tool.poetry.dependencies]
python = "^3.12"
pandas = "^2.2.0"
numpy = "^1.26.3"
scikit-learn = "^1.4.0"
lightgbm = "^4.3.0"
mlflow = "^3.6.0"
fastapi = "^0.115.6"
uvicorn = "^0.34.0"
pydantic = "^2.10.4"
streamlit = "^1.41.1"
```

### Development Libraries

```toml
[tool.poetry.group.dev.dependencies]
pytest = "^8.3.4"
pytest-cov = "^6.0.0"
black = "^24.8.0"
ruff = "^0.8.4"
httpx = "^0.28.1"
```

---

## Performance Benchmarks

### Model Training
- **Training Time**: ~45 seconds (5-fold CV)
- **Memory Usage**: ~2GB RAM
- **Model Size**: 6.8MB

### API Performance
- **Startup Time**: ~3 seconds (model loading)
- **P50 Latency**: 28ms
- **P95 Latency**: 42ms
- **P99 Latency**: 68ms
- **Throughput**: 120 requests/second
- **Memory**: ~200MB

### MLflow UI
- **Startup Time**: ~2 seconds
- **Database Size**: 840KB
- **Artifacts Size**: 243MB (can be optimized)

---

## Future Enhancements

### Short-Term (1-3 months)
- [ ] Add model explainability (SHAP values in API)
- [ ] Implement A/B testing framework
- [ ] Add monitoring dashboard
- [ ] Set up automated retraining pipeline

### Medium-Term (3-6 months)
- [ ] Integrate with additional data sources
- [ ] Implement ensemble models
- [ ] Add real-time drift detection
- [ ] Build prediction explanation UI

### Long-Term (6-12 months)
- [ ] Deep learning model exploration
- [ ] Multi-model serving
- [ ] Advanced feature engineering (AutoML)
- [ ] Real-time scoring infrastructure

---

## Technical Debt & Known Issues

### Current Limitations

1. **Single Model**: Only LightGBM implemented
   - **Impact**: No model fallback
   - **Mitigation**: Add XGBoost or Random Forest backup

2. **Manual Threshold**: Fixed at 0.3282
   - **Impact**: Not adaptive to changing conditions
   - **Mitigation**: Implement dynamic threshold adjustment

3. **Large MLflow Artifacts**: 243MB
   - **Impact**: Slow MLflow UI loading
   - **Mitigation**: Clean up old runs, compress artifacts

4. **No Authentication**: API is open
   - **Impact**: Security risk in production
   - **Mitigation**: Add JWT authentication

---

## References

### Papers & Resources
- LightGBM: https://lightgbm.readthedocs.io/
- MLflow: https://mlflow.org/docs/latest/
- FastAPI: https://fastapi.tiangolo.com/
- Pydantic: https://docs.pydantic.dev/

### Internal Documentation
- [User Guide](USER_GUIDE.md)
- [Monitoring Guide](MODEL_MONITORING.md)
- [Business Presentation](presentations/BUSINESS_PRESENTATION.md)
- [Technical Presentation](presentations/TECHNICAL_PRESENTATION.md)

---

**Last Updated**: December 10, 2025
**Version**: 1.0.0
**Maintainer**: Data Science Team
