# User Guide - Credit Scoring Model

## Quick Start

### Launch All Services

**Windows:**
```bash
launch_services.bat
```

**Linux/Mac:**
```bash
./launch_services.sh
```

This will automatically start and open:
- **MLflow UI**: http://localhost:5000 - View experiments and model registry
- **Dashboard**: http://localhost:8501 - Interactive threshold adjustment
- **API Docs**: http://localhost:8000/docs - Test API endpoints

---

## Installation

### Prerequisites
- Python 3.12+
- Poetry (dependency manager)

### Setup Steps

```bash
# 1. Navigate to project
cd Scoring_Model

# 2. Install dependencies
poetry install

# 3. Verify installation
poetry run pytest tests/ -v

# 4. Launch services
./launch_services.bat  # Windows
./launch_services.sh   # Linux/Mac
```

---

## Using the Services

### MLflow UI (http://localhost:5000)

**Purpose**: View model experiments and training history

**Key Features**:
- Compare different model runs
- View metrics (ROC-AUC, precision, recall)
- Download artifacts (plots, feature importance)
- Access model registry

**How to Use**:
1. Open http://localhost:5000
2. Click "Experiments" to see all runs
3. Click any run name to view details
4. Click "Artifacts" tab to download plots

### Dashboard (http://localhost:8501)

**Purpose**: Interactively adjust decision threshold

**Key Features**:
- Adjust threshold slider (0.0 to 1.0)
- See real-time impact on:
  - Business cost
  - Recall (how many defaults we catch)
  - Precision (accuracy of predictions)
- View confusion matrix
- Analyze probability distribution

**How to Use**:
1. Open http://localhost:8501
2. Use slider to adjust threshold
3. Watch metrics update in real-time
4. Find optimal balance for your business needs

**Recommended Threshold**: 0.3282 (pre-calculated optimal)

### API Server (http://localhost:8000)

**Purpose**: REST API for production predictions

**Key Features**:
- `/health` - Check API status
- `/predict` - Get credit score prediction
- `/docs` - Interactive API documentation

---

## API Testing Guide

### Method 1: Interactive Swagger UI (Easiest)

1. Open http://localhost:8000/docs
2. Click "POST /predict"
3. Click "Try it out"
4. Edit the JSON example
5. Click "Execute"
6. View response below

### Method 2: cURL (Command Line)

```bash
# Health check
curl http://localhost:8000/health

# Prediction example
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.5, 0.3, 0.2, ...],
    "client_id": "100002"
  }'
```

### Method 3: Python Requests

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())  # {"status": "healthy", ...}

# Prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "features": [0.5, 0.3, 0.2, ...],  # 189 features
        "client_id": "100002"
    }
)

result = response.json()
print(f"Risk Level: {result['risk_level']}")  # LOW, MEDIUM, HIGH, CRITICAL
print(f"Probability: {result['probability']:.4f}")
print(f"Threshold: {result['threshold']}")
```

### Method 4: Automated Test Scripts

Run the provided test scripts:

```bash
# Test health endpoint
poetry run python scripts/experiments/test_api_health.py

# Test prediction endpoint
poetry run python scripts/experiments/test_api_predict.py
```

---

## Understanding the Predictions

### Risk Levels

| Probability | Risk Level | Interpretation | Recommendation |
|-------------|-----------|----------------|----------------|
| 0.0 - 0.2 | **LOW** | Very low default risk | Approve automatically |
| 0.2 - 0.4 | **MEDIUM** | Moderate risk | Manual review |
| 0.4 - 0.6 | **HIGH** | High default risk | Reject or require collateral |
| 0.6+ | **CRITICAL** | Very high risk | Reject |

### Response Format

```json
{
  "client_id": "100002",
  "probability": 0.1234,
  "risk_level": "LOW",
  "threshold": 0.3282,
  "model_version": "1",
  "timestamp": "2025-12-10T10:30:45"
}
```

**Fields**:
- `probability`: Predicted default probability (0.0 to 1.0)
- `risk_level`: Risk category (LOW/MEDIUM/HIGH/CRITICAL)
- `threshold`: Decision threshold used (default: 0.3282)
- `model_version`: Model version from registry
- `timestamp`: Prediction time (ISO format)

---

## Running Tests

### All Tests
```bash
poetry run pytest tests/ -v
```

### Specific Test Categories
```bash
# API tests only
poetry run pytest tests/test_api.py -v

# Validation tests only
poetry run pytest tests/test_validation.py -v

# With coverage report
poetry run pytest tests/ --cov=src --cov=api --cov-report=html
```

**Expected Results**: 67/67 tests passing

---

## Troubleshooting

### MLflow UI Not Starting

**Symptom**: MLflow UI won't load or shows errors

**Solution**:
```bash
# Check MLflow database
poetry run python -c "import mlflow; print(mlflow.get_tracking_uri())"

# Restart MLflow
poetry run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

### Dashboard Shows Error

**Symptom**: Dashboard displays error message

**Common Causes**:
1. Predictions file missing - Run training pipeline first
2. Large file - Dashboard will automatically sample to 100k rows
3. Memory issue - Close other applications

**Solution**:
```bash
# Generate predictions if missing
poetry run python scripts/pipeline/apply_best_model.py
```

### API Returns 503 Error

**Symptom**: API responds with 503 Service Unavailable

**Cause**: Model not loaded from MLflow registry

**Solution**:
```bash
# Check model registry
poetry run python -c "from mlflow import MlflowClient; client = MlflowClient(); print(client.search_registered_models())"

# Restart API
poetry run python scripts/deployment/start_api.py
```

### Tests Failing

**Symptom**: Some tests fail when running pytest

**Solution**:
```bash
# Reinstall dependencies
poetry install --no-cache

# Clear pytest cache
poetry run pytest --cache-clear tests/ -v
```

### Services Running Slowly

**Symptoms**:
- MLflow UI takes minutes to load
- Dashboard is laggy
- API responses are slow

**Solutions**:
1. **MLflow Performance**: Clean up old experiment runs
   ```bash
   # Backup first
   cp -r mlruns mlruns_backup

   # Use MLflow UI to delete old experiments
   # Or use cleanup script
   poetry run python scripts/mlflow/cleanup_old_runs.py
   ```

2. **Dashboard Performance**: Already optimized to sample large files automatically

3. **API Performance**: Check model loading time
   ```bash
   poetry run python scripts/deployment/start_api.py --reload
   ```

---

## File Locations

### Important Files
- **Configuration**: `src/config.py`
- **Predictions**: `results/train_predictions.csv`
- **Models**: `mlruns/` and `models/`
- **Logs**: Check console output
- **Tests**: `tests/`

### Data Files
- **Training Data**: `data/application_train.csv`
- **Test Data**: `data/application_test.csv`
- **Processed Features**: `data/processed/`

---

## Performance Metrics

### Expected Performance

| Metric | Value | Status |
|--------|-------|--------|
| **API Latency (P95)** | <50ms | ✅ Excellent |
| **Throughput** | 120 req/sec | ✅ High |
| **Model ROC-AUC** | 0.7761 | ✅ Good |
| **Test Coverage** | 86% | ✅ Strong |

---

## Getting Help

### Documentation
- **This Guide**: Quick reference for users
- **Technical Guide**: [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md) - Model details
- **Monitoring Guide**: [MODEL_MONITORING.md](MODEL_MONITORING.md) - Production monitoring
- **Business Presentation**: [presentations/BUSINESS_PRESENTATION.md](presentations/BUSINESS_PRESENTATION.md)
- **Technical Presentation**: [presentations/TECHNICAL_PRESENTATION.md](presentations/TECHNICAL_PRESENTATION.md)

### External Resources
- **MLflow Docs**: https://mlflow.org/docs/latest/
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Streamlit Docs**: https://docs.streamlit.io/

---

**Last Updated**: December 10, 2025
**Version**: 1.0.0
