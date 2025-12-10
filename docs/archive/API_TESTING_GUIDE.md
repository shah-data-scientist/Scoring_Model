# API Testing Guide

## Overview
This guide shows you how to test the Credit Scoring API using the interactive documentation, command line tools, and Python scripts.

---

## Method 1: Interactive API Documentation (Easiest)

### Start the API Server
```bash
cd "c:\Users\shahu\OPEN CLASSROOMS\PROJET 6\Scoring_Model"
poetry run uvicorn api.app:app --reload --port 8000
```

### Access Interactive Docs
Open your browser and visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Using Swagger UI

#### 1. Test Health Check
1. Click on `GET /health` endpoint
2. Click "Try it out"
3. Click "Execute"
4. You should see:
   ```json
   {
     "status": "healthy",
     "model_loaded": true,
     "model_name": "credit_scoring_production_model",
     "model_version": "Staging",
     "timestamp": "2025-12-08T18:00:00"
   }
   ```

#### 2. Test Single Prediction
1. Click on `POST /predict` endpoint
2. Click "Try it out"
3. Replace the example JSON with your data:
   ```json
   {
     "features": [
       0.5, 0.3, 0.7, 0.2, 0.8, 0.1, 0.4, 0.6, 0.9, 0.3,
       0.5, 0.3, 0.7, 0.2, 0.8, 0.1, 0.4, 0.6, 0.9, 0.3,
       ... (189 features total)
     ],
     "client_id": "100002"
   }
   ```
4. Click "Execute"
5. You should see:
   ```json
   {
     "prediction": 0,
     "probability": 0.234,
     "risk_level": "MEDIUM",
     "client_id": "100002",
     "timestamp": "2025-12-08T18:00:00",
     "model_version": "Staging"
   }
   ```

#### 3. Test Batch Prediction
1. Click on `POST /predict/batch` endpoint
2. Click "Try it out"
3. Replace with batch data:
   ```json
   {
     "features": [
       [0.5, 0.3, ...],  // Client 1 (189 features)
       [0.6, 0.4, ...],  // Client 2 (189 features)
       [0.4, 0.2, ...]   // Client 3 (189 features)
     ],
     "client_ids": ["100002", "100003", "100004"]
   }
   ```
4. Click "Execute"

#### 4. Test Model Info
1. Click on `GET /model/info` endpoint
2. Click "Try it out"
3. Click "Execute"

---

## Method 2: Command Line (curl)

### Health Check
```bash
curl http://localhost:8000/health
```

### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.5, 0.3, 0.7, 0.2, 0.8, 0.1, 0.4, 0.6, 0.9, 0.3, 0.5, 0.3, 0.7, 0.2, 0.8, 0.1, 0.4, 0.6, 0.9, 0.3, 0.5, 0.3, 0.7, 0.2, 0.8, 0.1, 0.4, 0.6, 0.9, 0.3, 0.5, 0.3, 0.7, 0.2, 0.8, 0.1, 0.4, 0.6, 0.9, 0.3, 0.5, 0.3, 0.7, 0.2, 0.8, 0.1, 0.4, 0.6, 0.9, 0.3, 0.5, 0.3, 0.7, 0.2, 0.8, 0.1, 0.4, 0.6, 0.9, 0.3, 0.5, 0.3, 0.7, 0.2, 0.8, 0.1, 0.4, 0.6, 0.9, 0.3, 0.5, 0.3, 0.7, 0.2, 0.8, 0.1, 0.4, 0.6, 0.9, 0.3, 0.5, 0.3, 0.7, 0.2, 0.8, 0.1, 0.4, 0.6, 0.9, 0.3, 0.5, 0.3, 0.7, 0.2, 0.8, 0.1, 0.4, 0.6, 0.9, 0.3, 0.5, 0.3, 0.7, 0.2, 0.8, 0.1, 0.4, 0.6, 0.9, 0.3, 0.5, 0.3, 0.7, 0.2, 0.8, 0.1, 0.4, 0.6, 0.9, 0.3, 0.5, 0.3, 0.7, 0.2, 0.8, 0.1, 0.4, 0.6, 0.9, 0.3, 0.5, 0.3, 0.7, 0.2, 0.8, 0.1, 0.4, 0.6, 0.9, 0.3, 0.5, 0.3, 0.7, 0.2, 0.8, 0.1, 0.4, 0.6, 0.9, 0.3, 0.5, 0.3, 0.7, 0.2, 0.8, 0.1, 0.4, 0.6, 0.9, 0.3, 0.5, 0.3, 0.7, 0.2, 0.8, 0.1, 0.4, 0.6, 0.9, 0.3, 0.5, 0.3, 0.7, 0.2, 0.8, 0.1, 0.4, 0.6, 0.9, 0.3, 0.5, 0.3, 0.7, 0.2, 0.8, 0.1, 0.4, 0.6, 0.9],
    "client_id": "100002"
  }'
```

### Batch Prediction
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      [0.5, 0.3, ...],
      [0.6, 0.4, ...]
    ],
    "client_ids": ["100002", "100003"]
  }'
```

---

## Method 3: Python Script

### Simple Test Script
Create `test_api.py`:

```python
import requests
import numpy as np

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("Testing /health...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")
    return response.status_code == 200

def test_predict():
    """Test single prediction."""
    print("Testing /predict...")

    # Generate random features (189 total)
    features = np.random.random(189).tolist()

    data = {
        "features": features,
        "client_id": "TEST_001"
    }

    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result['prediction']}")
        print(f"Probability: {result['probability']:.4f}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Client ID: {result['client_id']}\n")
        return True
    else:
        print(f"Error: {response.json()}\n")
        return False

def test_batch_predict():
    """Test batch prediction."""
    print("Testing /predict/batch...")

    # Generate 3 random feature vectors
    features = [np.random.random(189).tolist() for _ in range(3)]
    client_ids = ["TEST_001", "TEST_002", "TEST_003"]

    data = {
        "features": features,
        "client_ids": client_ids
    }

    response = requests.post(f"{BASE_URL}/predict/batch", json=data)
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Processed: {result['count']} predictions")

        for i, pred in enumerate(result['predictions'][:3], 1):
            print(f"\n  Client {i} ({pred['client_id']}):")
            print(f"    Prediction: {pred['prediction']}")
            print(f"    Probability: {pred['probability']:.4f}")
            print(f"    Risk: {pred['risk_level']}")

        return True
    else:
        print(f"Error: {response.json()}\n")
        return False

def test_model_info():
    """Test model info endpoint."""
    print("\nTesting /model/info...")
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        info = response.json()
        print(f"Model: {info['model_metadata']['name']}")
        print(f"Stage: {info['model_metadata']['stage']}")
        print(f"Expected features: {info['expected_features']}")
        return True
    else:
        print(f"Error: {response.json()}\n")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Credit Scoring API Test Suite")
    print("=" * 60)
    print()

    results = {
        "health": test_health(),
        "predict": test_predict(),
        "batch": test_batch_predict(),
        "model_info": test_model_info()
    }

    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:15} {status}")

    all_passed = all(results.values())
    print("\n" + ("All tests passed! ✅" if all_passed else "Some tests failed ❌"))
```

### Run the Test Script
```bash
poetry run python test_api.py
```

---

## Method 4: Using Real Data

### Load Actual Features from CSV
```python
import requests
import pandas as pd

# Load test data
X_test = pd.read_csv('data/processed/X_test.csv')

# Get first record (remove ID column if present)
if 'SK_ID_CURR' in X_test.columns:
    X_test = X_test.drop('SK_ID_CURR', axis=1)

features = X_test.iloc[0].tolist()

# Make prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"features": features, "client_id": "REAL_CLIENT_001"}
)

print(response.json())
```

---

## Error Handling Examples

### Invalid Feature Count
```python
import requests

# Only 50 features (should be 189)
data = {
    "features": [0.5] * 50,
    "client_id": "TEST"
}

response = requests.post("http://localhost:8000/predict", json=data)
print(f"Status: {response.status_code}")
print(f"Error: {response.json()}")

# Output:
# Status: 422
# Error: {
#   "detail": "Features must have exactly 189 elements"
# }
```

### NaN in Features
```python
import requests
import math

data = {
    "features": [0.5] * 188 + [math.nan],  # NaN in last position
    "client_id": "TEST"
}

response = requests.post("http://localhost:8000/predict", json=data)
print(f"Status: {response.status_code}")
print(f"Error: {response.json()}")

# Output:
# Status: 422
# Error: {
#   "detail": "Features contain NaN values"
# }
```

---

## Performance Testing

### Test Response Time
```python
import requests
import time
import numpy as np

BASE_URL = "http://localhost:8000"
features = np.random.random(189).tolist()

# Warm-up request
requests.post(f"{BASE_URL}/predict", json={"features": features})

# Measure response time
start = time.time()
response = requests.post(f"{BASE_URL}/predict", json={"features": features})
elapsed = time.time() - start

print(f"Response time: {elapsed*1000:.2f}ms")
print(f"Status: {response.status_code}")

# Target: < 50ms for single prediction
if elapsed < 0.05:
    print("✅ Performance target met")
else:
    print("⚠️ Performance slower than target")
```

### Load Testing (Multiple Requests)
```python
import requests
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

BASE_URL = "http://localhost:8000"

def make_prediction():
    """Make a single prediction request."""
    features = np.random.random(189).tolist()
    response = requests.post(f"{BASE_URL}/predict", json={"features": features})
    return response.elapsed.total_seconds()

# Test with 100 concurrent requests
print("Running load test with 100 concurrent requests...")
start = time.time()

with ThreadPoolExecutor(max_workers=10) as executor:
    times = list(executor.map(lambda _: make_prediction(), range(100)))

elapsed = time.time() - start

print(f"\nTotal time: {elapsed:.2f}s")
print(f"Requests/second: {100/elapsed:.2f}")
print(f"Average response time: {np.mean(times)*1000:.2f}ms")
print(f"95th percentile: {np.percentile(times, 95)*1000:.2f}ms")
```

---

## Troubleshooting

### API Not Starting
```bash
# Check if port 8000 is already in use
netstat -ano | findstr :8000

# Kill process if needed (replace <PID> with actual process ID)
taskkill /F /PID <PID>

# Try different port
poetry run uvicorn api.app:app --reload --port 8001
```

### Model Not Loading
Check the API logs for errors:
```
ERROR: Failed to load model: ...
API will start but predictions will fail until model is loaded.
```

**Solution**:
1. Ensure MLflow database exists: `mlruns/mlflow.db`
2. Check model is registered:
   ```bash
   poetry run python -c "import mlflow; from mlflow import MlflowClient; mlflow.set_tracking_uri('sqlite:///mlruns/mlflow.db'); client = MlflowClient(); print(client.search_registered_models())"
   ```
3. Register model if needed:
   ```bash
   poetry run python register_best_model.py
   ```

### Connection Refused
```python
requests.exceptions.ConnectionError: Connection refused
```

**Solution**: Make sure API server is running:
```bash
poetry run uvicorn api.app:app --reload --port 8000
```

---

## Next Steps

1. **Automated Tests**: See `tests/test_api.py` for pytest-based tests
2. **Monitoring**: See `docs/MODEL_MONITORING.md` for production monitoring setup
3. **Deployment**: See `PRODUCTION_DEPLOYMENT_GUIDE.md` for deployment instructions

---

**Last Updated**: December 8, 2025
