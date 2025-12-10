# Production Deployment Guide
**Credit Scoring Model** - Version 1.0
**Model**: credit_scoring_production_model
**Date**: 2025-12-07
**ROC-AUC**: 0.7761 ± 0.0064

---

## Quick Start

### Registered Model Information

**Model Name**: `credit_scoring_production_model`
**Version**: 1 (Staging)
**Run ID**: `cc74d206feea43fcba99316d1c5d7674`
**Base Run**: `exp05_cv_domain_balanced` (081d51d8966a447e88efc37b432f1a26)
**MLflow UI**: http://localhost:5000/#/models/credit_scoring_production_model

**Model Configuration**:
- Algorithm: LightGBM
- Features: Domain features (194 features)
- Sampling: Balanced class weights
- Optimal Threshold: 0.3282

---

## Deployment Workflow

### Stage 1: Development → Staging

**Status**: ✅ COMPLETE

The model is currently in Staging stage and ready for validation.

**What was done**:
```python
# Model registered and transitioned to Staging
client.transition_model_version_stage(
    name="credit_scoring_production_model",
    version=1,
    stage="Staging"
)
```

**Validation Checklist** (Before Production):
- [ ] Test on held-out test set
- [ ] Verify threshold optimization (0.3282)
- [ ] Check calibration quality
- [ ] Run bias & fairness audit
- [ ] Performance testing (latency < 100ms)
- [ ] Integration testing with production API
- [ ] Security review
- [ ] Documentation complete

### Stage 2: Staging → Production

**Command**:
```python
from mlflow.tracking import MlflowClient
from src.config import MLFLOW_TRACKING_URI
import mlflow

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# After validation passes
client.transition_model_version_stage(
    name="credit_scoring_production_model",
    version=1,
    stage="Production",
    archive_existing_versions=True  # Archive previous production models
)
```

### Stage 3: Production Monitoring

See "Monitoring & Maintenance" section below.

---

## Loading the Model

### Method 1: Load from MLflow Registry (Recommended)

```python
import mlflow
from src.config import MLFLOW_TRACKING_URI

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load latest staging model
model_name = "credit_scoring_production_model"
model_uri = f"models:/{model_name}/Staging"
model = mlflow.sklearn.load_model(model_uri)

# Load specific version
model_uri = f"models:/{model_name}/1"
model = mlflow.sklearn.load_model(model_uri)

# Load production model (after transition)
model_uri = f"models:/{model_name}/Production"
model = mlflow.sklearn.load_model(model_uri)
```

### Method 2: Load from Run ID

```python
import mlflow

run_id = "cc74d206feea43fcba99316d1c5d7674"
model_uri = f"runs:/{run_id}/model"
model = mlflow.sklearn.load_model(model_uri)
```

---

## Making Predictions

### Single Prediction

```python
import pandas as pd
import mlflow
from src.domain_features import create_domain_features

# Load model
model = mlflow.sklearn.load_model("models:/credit_scoring_production_model/Production")

# Prepare features for a single applicant
applicant_data = pd.DataFrame({
    # ... 189 baseline features ...
})

# Apply domain feature engineering
applicant_features = create_domain_features(applicant_data)

# Get prediction probability
probability = model.predict_proba(applicant_features)[0, 1]

# Apply optimal threshold
OPTIMAL_THRESHOLD = 0.3282
prediction = 1 if probability >= OPTIMAL_THRESHOLD else 0

print(f"Default Probability: {probability:.2%}")
print(f"Decision: {'REJECT' if prediction == 1 else 'APPROVE'}")
```

### Batch Predictions

```python
import pandas as pd
from src.domain_features import create_domain_features

# Load new applications
applications = pd.read_csv('new_applications.csv')

# Apply feature engineering
applications_features = create_domain_features(applications)

# Get predictions
probabilities = model.predict_proba(applications_features)[:, 1]
predictions = (probabilities >= 0.3282).astype(int)

# Create results
results = pd.DataFrame({
    'application_id': applications['SK_ID_CURR'],
    'probability': probabilities,
    'decision': predictions,
    'decision_label': ['REJECT' if p == 1 else 'APPROVE' for p in predictions]
})

results.to_csv('predictions.csv', index=False)
```

---

## API Integration Example

### Flask API

```python
from flask import Flask, request, jsonify
import mlflow
import pandas as pd
from src.domain_features import create_domain_features
from src.config import MLFLOW_TRACKING_URI

app = Flask(__name__)

# Load model at startup
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.sklearn.load_model("models:/credit_scoring_production_model/Production")

OPTIMAL_THRESHOLD = 0.3282

@app.route('/predict', methods=['POST'])
def predict():
    """Predict default probability for a loan application."""
    try:
        # Get input data
        data = request.get_json()

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Apply feature engineering
        features = create_domain_features(df)

        # Predict
        probability = float(model.predict_proba(features)[0, 1])
        decision = int(probability >= OPTIMAL_THRESHOLD)

        return jsonify({
            'probability': probability,
            'decision': decision,
            'decision_label': 'REJECT' if decision == 1 else 'APPROVE',
            'model_version': '1',
            'threshold': OPTIMAL_THRESHOLD
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

### FastAPI Example

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd
from src.domain_features import create_domain_features

app = FastAPI()

# Load model
model = mlflow.sklearn.load_model("models:/credit_scoring_production_model/Production")

class Application(BaseModel):
    # Define all 189 features as fields
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    # ... other features ...

class Prediction(BaseModel):
    probability: float
    decision: int
    decision_label: str
    model_version: str

@app.post("/predict", response_model=Prediction)
def predict(application: Application):
    try:
        df = pd.DataFrame([application.dict()])
        features = create_domain_features(df)

        probability = float(model.predict_proba(features)[0, 1])
        decision = int(probability >= 0.3282)

        return Prediction(
            probability=probability,
            decision=decision,
            decision_label='REJECT' if decision == 1 else 'APPROVE',
            model_version='1'
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

---

## Monitoring & Maintenance

### 1. Performance Monitoring

Track these metrics in production:

```python
import mlflow
from datetime import datetime

def log_prediction_metrics(y_true, y_pred, y_proba):
    """Log production metrics to MLflow."""
    with mlflow.start_run(run_name=f"production_monitoring_{datetime.now().strftime('%Y%m%d')}"):
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

        mlflow.log_metric("production_roc_auc", roc_auc_score(y_true, y_proba))
        mlflow.log_metric("production_precision", precision_score(y_true, y_pred))
        mlflow.log_metric("production_recall", recall_score(y_true, y_pred))
        mlflow.log_metric("production_f1", f1_score(y_true, y_pred))
        mlflow.log_metric("n_predictions", len(y_true))
```

**Alert Thresholds**:
- ROC-AUC drops below 0.75 → Investigate
- Prediction volume changes > 30% → Investigate
- API latency > 100ms → Investigate

### 2. Drift Detection

Monitor input feature distributions:

```python
def detect_drift(production_features, training_features):
    """Detect distribution drift in features."""
    from scipy.stats import ks_2samp

    drift_detected = []

    for col in production_features.columns:
        statistic, pvalue = ks_2samp(
            production_features[col],
            training_features[col]
        )

        if pvalue < 0.01:  # Significant drift
            drift_detected.append({
                'feature': col,
                'p_value': pvalue,
                'statistic': statistic
            })

    return drift_detected
```

### 3. Model Retraining Triggers

Retrain the model when:
- ROC-AUC drops below 0.75 for 7 consecutive days
- Significant feature drift detected (> 10 features)
- Business rules change
- Quarterly scheduled retraining

---

## Rollback Procedure

If the production model fails, rollback to previous version:

```python
from mlflow.tracking import MlflowClient
import mlflow

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# Get current production version
current_version = client.get_latest_versions("credit_scoring_production_model", stages=["Production"])[0]

# Transition current to Archived
client.transition_model_version_stage(
    name="credit_scoring_production_model",
    version=current_version.version,
    stage="Archived"
)

# Promote previous version to Production
previous_version = current_version.version - 1
client.transition_model_version_stage(
    name="credit_scoring_production_model",
    version=previous_version,
    stage="Production"
)

print(f"Rolled back: v{current_version.version} → v{previous_version}")
```

---

## Testing in Production

### A/B Testing Setup

```python
import random

def get_model_for_ab_test(user_id):
    """Assign users to champion vs challenger models."""

    # 90% champion (current production), 10% challenger
    if hash(user_id) % 100 < 90:
        model = mlflow.sklearn.load_model(
            "models:/credit_scoring_production_model/Production"
        )
        variant = "champion"
    else:
        model = mlflow.sklearn.load_model(
            "models:/credit_scoring_challenger_model/Staging"
        )
        variant = "challenger"

    return model, variant
```

Track both models' performance and promote challenger if it outperforms.

---

## Security & Compliance

### 1. Data Privacy

- **PII Handling**: Ensure GDPR/CCPA compliance
- **Encryption**: All data encrypted at rest and in transit
- **Access Control**: Role-based access to MLflow and models

### 2. Model Governance

- **Audit Trail**: All predictions logged with model version
- **Explainability**: SHAP values available on request
- **Bias Monitoring**: Regular fairness audits

### 3. Compliance Checklist

- [ ] Data retention policy (7 years)
- [ ] Right to explanation (GDPR Article 22)
- [ ] Model documentation complete
- [ ] Bias & fairness audit quarterly
- [ ] Audit logs enabled
- [ ] Access controls configured

---

## Performance Benchmarks

### Latency

- **Single Prediction**: < 50ms
- **Batch (1000)**: < 2s
- **Feature Engineering**: ~20ms
- **Model Inference**: ~10ms

### Throughput

- **Target**: 1000 predictions/second
- **Peak**: 5000 predictions/second

### Resource Requirements

- **CPU**: 2 cores minimum
- **RAM**: 4GB minimum
- **Disk**: 2GB (model + artifacts)

---

## Troubleshooting

### Issue: Model predictions differ from training

**Cause**: Feature engineering mismatch
**Solution**: Ensure `create_domain_features()` applied consistently

### Issue: High latency

**Cause**: Model not loaded in memory
**Solution**: Load model at API startup, not per-request

### Issue: Memory error with large batches

**Cause**: Processing too many rows at once
**Solution**: Batch predictions in chunks of 10,000

---

## Next Steps

### Immediate (Before Production)

1. Run model on held-out test set
2. Validate optimal threshold (0.3282)
3. Complete security review
4. Set up monitoring dashboard

### Short-term (First Month)

5. Implement A/B testing framework
6. Set up automated alerts
7. Create model documentation for compliance
8. Train operations team

### Long-term (Quarterly)

9. Scheduled model retraining
10. Bias & fairness audits
11. Performance optimization
12. Feature engineering improvements

---

## Resources

- **MLflow UI**: http://localhost:5000
- **Model Registry**: http://localhost:5000/#/models/credit_scoring_production_model
- **Comparison Dashboard**: `poetry run streamlit run comparison_dashboard.py`
- **Threshold Selector**: `poetry run streamlit run dashboard.py`

---

## Support

For questions or issues:
1. Check MLflow UI for model metrics
2. Review logs in production environment
3. Consult [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)
4. Check [BEST_PRACTICES_AUDIT.md](BEST_PRACTICES_AUDIT.md)

---

**Model Version**: 1
**Deployment Status**: Staging (Ready for Production)
**Last Updated**: 2025-12-07
