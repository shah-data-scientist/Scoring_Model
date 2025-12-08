# Model Monitoring Guide

## Overview
This guide explains how to set up comprehensive monitoring for the credit scoring model in production, including data drift detection, performance tracking, and automated alerting.

---

## Table of Contents
1. [Monitoring Architecture](#monitoring-architecture)
2. [Key Metrics to Monitor](#key-metrics-to-monitor)
3. [Data Drift Detection](#data-drift-detection)
4. [Performance Monitoring](#performance-monitoring)
5. [Alerting Setup](#alerting-setup)
6. [Dashboard Implementation](#dashboard-implementation)
7. [Automated Retraining](#automated-retraining)

---

## Monitoring Architecture

### Components

```
┌─────────────────┐
│  API Requests   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│  Logging Layer  │────▶│  Metrics Store   │
│  (predictions,  │     │  (Prometheus)    │
│   features,     │     └──────────────────┘
│   timestamps)   │              │
└─────────────────┘              │
         │                       ▼
         │              ┌──────────────────┐
         │              │  Visualization   │
         │              │  (Grafana)       │
         │              └──────────────────┘
         ▼
┌─────────────────┐     ┌──────────────────┐
│  Drift          │────▶│  Alert Manager   │
│  Detection      │     │  (Email, Slack)  │
└─────────────────┘     └──────────────────┘
         │
         ▼
┌─────────────────┐
│  Auto Retrain   │
│  Trigger        │
└─────────────────┘
```

---

## Key Metrics to Monitor

### 1. Business Metrics

#### Default Rate
- **Metric**: Percentage of predictions that are "default" (1)
- **Target**: Should match training distribution (~8%)
- **Alert**: If > 12% or < 4% (50% deviation)

```python
# Implementation
from datetime import datetime, timedelta

def calculate_default_rate(predictions, window_days=7):
    """Calculate rolling default rate."""
    cutoff = datetime.now() - timedelta(days=window_days)
    recent = predictions[predictions['timestamp'] >= cutoff]

    default_rate = recent['prediction'].mean()
    return default_rate

# Example
default_rate = calculate_default_rate(predictions)
print(f"7-day default rate: {default_rate:.2%}")

# Alert if outside expected range
if default_rate > 0.12 or default_rate < 0.04:
    send_alert(f"Default rate anomaly: {default_rate:.2%}")
```

#### Average Probability
- **Metric**: Mean predicted probability
- **Target**: ~0.08 (matches training target rate)
- **Alert**: If > 0.12 or < 0.04

```python
def calculate_avg_probability(predictions, window_days=7):
    """Calculate average prediction probability."""
    cutoff = datetime.now() - timedelta(days=window_days)
    recent = predictions[predictions['timestamp'] >= cutoff]

    avg_prob = recent['probability'].mean()
    return avg_prob
```

#### Business Cost
- **Metric**: Total cost based on FN=10, FP=1
- **Target**: Minimize total cost
- **Alert**: If cost increases > 20% week-over-week

```python
def calculate_business_cost(y_true, y_pred, cost_fn=10, cost_fp=1):
    """Calculate business cost."""
    from sklearn.metrics import confusion_matrix

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    total_cost = (fn * cost_fn) + (fp * cost_fp)
    avg_cost = total_cost / len(y_true)

    return {
        'total_cost': total_cost,
        'avg_cost': avg_cost,
        'false_negatives': fn,
        'false_positives': fp
    }
```

### 2. Model Performance Metrics

#### ROC-AUC Score
- **Metric**: Area under ROC curve
- **Target**: > 0.75 (close to training: 0.7761)
- **Alert**: If < 0.70 (>10% degradation)

```python
from sklearn.metrics import roc_auc_score

def monitor_roc_auc(y_true, y_proba, threshold=0.70):
    """Monitor ROC-AUC score."""
    roc_auc = roc_auc_score(y_true, y_proba)

    if roc_auc < threshold:
        send_alert(f"ROC-AUC degradation: {roc_auc:.4f}")

    return roc_auc
```

#### Precision & Recall
- **Metrics**: Precision and recall at optimal threshold (0.3282)
- **Target**: Precision > 0.50, Recall > 0.60
- **Alert**: If either drops > 10%

```python
from sklearn.metrics import precision_score, recall_score

def monitor_precision_recall(y_true, y_pred, thresholds=(0.50, 0.60)):
    """Monitor precision and recall."""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    precision_threshold, recall_threshold = thresholds

    if precision < precision_threshold:
        send_alert(f"Precision drop: {precision:.4f}")

    if recall < recall_threshold:
        send_alert(f"Recall drop: {recall:.4f}")

    return precision, recall
```

### 3. System Metrics

#### Request Volume
- **Metric**: Requests per second
- **Target**: Track baseline, alert on anomalies
- **Alert**: If drops > 50% or spikes > 200%

```python
def monitor_request_volume():
    """Monitor API request volume."""
    # Using Prometheus counter
    from prometheus_client import Counter

    request_counter = Counter(
        'credit_scoring_requests_total',
        'Total prediction requests',
        ['endpoint']
    )

    # Increment on each request
    request_counter.labels(endpoint='predict').inc()
```

#### Response Time
- **Metric**: P50, P95, P99 latencies
- **Target**: P95 < 50ms
- **Alert**: If P95 > 100ms

```python
from prometheus_client import Histogram
import time

# Define histogram
response_time = Histogram(
    'credit_scoring_response_seconds',
    'Response time in seconds',
    ['endpoint']
)

# Measure in API endpoint
@response_time.labels(endpoint='predict').time()
def predict(input_data):
    """Prediction with timing."""
    # ... prediction logic ...
    pass
```

#### Error Rate
- **Metric**: 4xx and 5xx errors per total requests
- **Target**: < 1%
- **Alert**: If > 5%

```python
from prometheus_client import Counter

error_counter = Counter(
    'credit_scoring_errors_total',
    'Total errors',
    ['error_type']
)

# Track errors
try:
    prediction = model.predict(features)
except ValueError as e:
    error_counter.labels(error_type='validation').inc()
except Exception as e:
    error_counter.labels(error_type='prediction').inc()
```

---

## Data Drift Detection

### Feature Drift

**Definition**: Change in feature distributions between training and production

#### Implementation

```python
import numpy as np
from scipy.stats import ks_2samp
import pandas as pd

class FeatureDriftDetector:
    """Detect feature distribution drift."""

    def __init__(self, reference_data, feature_names, threshold=0.05):
        """
        Args:
            reference_data: Training data features
            feature_names: List of feature names
            threshold: P-value threshold for KS test
        """
        self.reference_data = reference_data
        self.feature_names = feature_names
        self.threshold = threshold

    def detect_drift(self, production_data):
        """
        Detect drift in production data.

        Returns:
            dict: Drift results per feature
        """
        results = {}

        for i, feature in enumerate(self.feature_names):
            ref_values = self.reference_data[:, i]
            prod_values = production_data[:, i]

            # Kolmogorov-Smirnov test
            statistic, p_value = ks_2samp(ref_values, prod_values)

            drifted = p_value < self.threshold

            results[feature] = {
                'statistic': statistic,
                'p_value': p_value,
                'drifted': drifted
            }

        return results

    def get_drifted_features(self, production_data):
        """Get list of features with detected drift."""
        results = self.detect_drift(production_data)

        drifted = [
            feature for feature, result in results.items()
            if result['drifted']
        ]

        return drifted


# Usage
detector = FeatureDriftDetector(
    reference_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    threshold=0.05
)

# Check production data weekly
drifted_features = detector.get_drifted_features(X_production.values)

if len(drifted_features) > 10:  # >5% of features
    send_alert(f"Significant feature drift detected: {len(drifted_features)} features")
```

### Prediction Drift

**Definition**: Change in model output distribution

#### Implementation

```python
def detect_prediction_drift(train_probabilities, prod_probabilities, threshold=0.05):
    """
    Detect drift in prediction probabilities.

    Args:
        train_probabilities: Training set probabilities
        prod_probabilities: Production probabilities
        threshold: P-value threshold

    Returns:
        dict: Drift detection results
    """
    from scipy.stats import ks_2samp

    # KS test
    statistic, p_value = ks_2samp(train_probabilities, prod_probabilities)

    # Calculate mean shift
    train_mean = np.mean(train_probabilities)
    prod_mean = np.mean(prod_probabilities)
    mean_shift = prod_mean - train_mean
    pct_shift = (mean_shift / train_mean) * 100

    results = {
        'statistic': statistic,
        'p_value': p_value,
        'drifted': p_value < threshold,
        'train_mean': train_mean,
        'prod_mean': prod_mean,
        'mean_shift': mean_shift,
        'pct_shift': pct_shift
    }

    return results


# Usage
drift_results = detect_prediction_drift(train_proba, prod_proba)

if drift_results['drifted']:
    send_alert(
        f"Prediction drift detected!\n"
        f"Mean shift: {drift_results['pct_shift']:.2f}%\n"
        f"P-value: {drift_results['p_value']:.4f}"
    )
```

### Target Drift

**Definition**: Change in actual outcomes (requires ground truth)

```python
def detect_target_drift(train_targets, prod_targets, threshold=0.05):
    """
    Detect drift in target distribution.

    Args:
        train_targets: Training targets
        prod_targets: Production targets (requires labels)
        threshold: P-value threshold
    """
    from scipy.stats import chi2_contingency

    # Contingency table
    train_counts = pd.Series(train_targets).value_counts()
    prod_counts = pd.Series(prod_targets).value_counts()

    contingency = pd.DataFrame({
        'train': train_counts,
        'prod': prod_counts
    }).fillna(0)

    # Chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency)

    drifted = p_value < threshold

    return {
        'chi2': chi2,
        'p_value': p_value,
        'drifted': drifted,
        'train_positive_rate': train_counts.get(1, 0) / len(train_targets),
        'prod_positive_rate': prod_counts.get(1, 0) / len(prod_targets)
    }
```

---

## Performance Monitoring

### Logging Predictions

```python
import logging
import json
from datetime import datetime

# Setup logging
logging.basicConfig(
    filename='logs/predictions.log',
    level=logging.INFO,
    format='%(message)s'
)

def log_prediction(client_id, features, prediction, probability, timestamp=None):
    """Log prediction for monitoring."""
    if timestamp is None:
        timestamp = datetime.now().isoformat()

    log_entry = {
        'timestamp': timestamp,
        'client_id': client_id,
        'prediction': int(prediction),
        'probability': float(probability),
        'features': features  # Consider hashing for privacy
    }

    logging.info(json.dumps(log_entry))
```

### Collecting Ground Truth

```python
def collect_ground_truth(client_id, actual_default, days_to_default=None):
    """
    Collect actual outcomes for performance evaluation.

    Args:
        client_id: Client identifier
        actual_default: Whether client actually defaulted (0/1)
        days_to_default: Days until default (if applicable)
    """
    import sqlite3

    conn = sqlite3.connect('monitoring/ground_truth.db')
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ground_truth (
            client_id TEXT,
            timestamp TEXT,
            actual_default INTEGER,
            days_to_default INTEGER
        )
    """)

    cursor.execute("""
        INSERT INTO ground_truth VALUES (?, ?, ?, ?)
    """, (client_id, datetime.now().isoformat(), actual_default, days_to_default))

    conn.commit()
    conn.close()
```

### Performance Evaluation

```python
def evaluate_production_performance():
    """
    Evaluate model performance using collected ground truth.

    Runs weekly to assess real-world performance.
    """
    import sqlite3
    import pandas as pd
    from sklearn.metrics import roc_auc_score, classification_report

    # Load predictions
    predictions_df = pd.read_json('logs/predictions.log', lines=True)

    # Load ground truth
    conn = sqlite3.connect('monitoring/ground_truth.db')
    ground_truth_df = pd.read_sql('SELECT * FROM ground_truth', conn)
    conn.close()

    # Merge
    merged = predictions_df.merge(
        ground_truth_df,
        on='client_id',
        how='inner'
    )

    if len(merged) < 100:
        print("Not enough ground truth data for evaluation")
        return

    # Calculate metrics
    y_true = merged['actual_default']
    y_proba = merged['probability']
    y_pred = (y_proba >= 0.3282).astype(int)

    roc_auc = roc_auc_score(y_true, y_proba)

    print(f"Production ROC-AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # Check for degradation
    if roc_auc < 0.70:
        send_alert(f"Production performance degradation: ROC-AUC = {roc_auc:.4f}")

    return roc_auc
```

---

## Alerting Setup

### Alert Configuration

```python
class AlertManager:
    """Manage alerts for model monitoring."""

    def __init__(self, email=None, slack_webhook=None):
        """
        Args:
            email: Email address for alerts
            slack_webhook: Slack webhook URL
        """
        self.email = email
        self.slack_webhook = slack_webhook

    def send_alert(self, title, message, severity='WARNING'):
        """
        Send alert via configured channels.

        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity (INFO, WARNING, CRITICAL)
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        alert = f"""
[{severity}] {title}
Time: {timestamp}

{message}
        """

        # Email alert
        if self.email:
            self._send_email(alert)

        # Slack alert
        if self.slack_webhook:
            self._send_slack(alert, severity)

        # Log alert
        logging.warning(alert)

    def _send_email(self, message):
        """Send email alert."""
        import smtplib
        from email.mime.text import MIMEText

        # Configure SMTP
        # ... implementation ...
        pass

    def _send_slack(self, message, severity):
        """Send Slack alert."""
        import requests

        color = {
            'INFO': 'good',
            'WARNING': 'warning',
            'CRITICAL': 'danger'
        }.get(severity, 'warning')

        payload = {
            'attachments': [{
                'color': color,
                'text': message
            }]
        }

        requests.post(self.slack_webhook, json=payload)


# Usage
alert_manager = AlertManager(
    email='ml-team@company.com',
    slack_webhook='https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
)

# Send alert
alert_manager.send_alert(
    title='Feature Drift Detected',
    message='10 features showing significant drift',
    severity='WARNING'
)
```

### Alert Rules

```yaml
# alert_rules.yaml
alerts:
  - name: high_default_rate
    metric: default_rate
    condition: "> 0.12 or < 0.04"
    severity: WARNING
    action: notify_team

  - name: performance_degradation
    metric: roc_auc
    condition: "< 0.70"
    severity: CRITICAL
    action: notify_team, trigger_retraining

  - name: feature_drift
    metric: num_drifted_features
    condition: "> 10"
    severity: WARNING
    action: notify_team, log_investigation

  - name: high_error_rate
    metric: error_rate
    condition: "> 0.05"
    severity: CRITICAL
    action: notify_team, rollback_model
```

---

## Dashboard Implementation

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "Credit Scoring Model Monitoring",
    "panels": [
      {
        "title": "Prediction Volume",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(credit_scoring_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "Default Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "credit_scoring_default_rate",
            "legendFormat": "Default Rate"
          }
        ]
      },
      {
        "title": "Response Time (P95)",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, credit_scoring_response_seconds_bucket)",
            "legendFormat": "P95 Latency"
          }
        ]
      },
      {
        "title": "Model Performance",
        "type": "stat",
        "targets": [
          {
            "expr": "credit_scoring_roc_auc",
            "legendFormat": "ROC-AUC"
          }
        ]
      }
    ]
  }
}
```

### Custom Streamlit Dashboard

```python
# monitoring_dashboard.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Model Monitoring", layout="wide")

st.title("Credit Scoring Model Monitoring Dashboard")

# Time range selector
time_range = st.sidebar.selectbox(
    "Time Range",
    ["Last 24 Hours", "Last 7 Days", "Last 30 Days"]
)

# Load prediction logs
@st.cache_data
def load_predictions(days=7):
    predictions = pd.read_json('logs/predictions.log', lines=True)
    predictions['timestamp'] = pd.to_datetime(predictions['timestamp'])

    cutoff = datetime.now() - timedelta(days=days)
    recent = predictions[predictions['timestamp'] >= cutoff]

    return recent

predictions = load_predictions(days=7)

# Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Predictions",
        f"{len(predictions):,}",
        f"+{len(predictions) - len(predictions[:-100]):,} (vs previous period)"
    )

with col2:
    default_rate = predictions['prediction'].mean()
    st.metric(
        "Default Rate",
        f"{default_rate:.2%}",
        f"{(default_rate - 0.08):.2%} vs target"
    )

with col3:
    avg_prob = predictions['probability'].mean()
    st.metric(
        "Avg Probability",
        f"{avg_prob:.4f}",
        f"{(avg_prob - 0.08):.4f} vs target"
    )

with col4:
    high_risk_pct = (predictions['probability'] > 0.6).mean()
    st.metric(
        "High Risk %",
        f"{high_risk_pct:.2%}"
    )

# Prediction volume over time
st.subheader("Prediction Volume Over Time")
volume_by_day = predictions.groupby(predictions['timestamp'].dt.date).size()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=volume_by_day.index,
    y=volume_by_day.values,
    mode='lines+markers',
    name='Predictions'
))
fig.update_layout(xaxis_title="Date", yaxis_title="Count")
st.plotly_chart(fig, use_container_width=True)

# Probability distribution
st.subheader("Probability Distribution")
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=predictions['probability'],
    nbinsx=50,
    name='Production'
))
fig.update_layout(xaxis_title="Probability", yaxis_title="Count")
st.plotly_chart(fig, use_container_width=True)

# Drift detection
st.subheader("Feature Drift Detection")
if st.button("Run Drift Detection"):
    with st.spinner("Detecting drift..."):
        # Load training data
        # Run drift detector
        # Display results
        st.success("Drift detection complete")
```

---

## Automated Retraining

### Retraining Triggers

```python
class RetrainingManager:
    """Manage automated model retraining."""

    def __init__(self, performance_threshold=0.70, drift_threshold=0.10):
        """
        Args:
            performance_threshold: Min ROC-AUC before retraining
            drift_threshold: Max feature drift % before retraining
        """
        self.performance_threshold = performance_threshold
        self.drift_threshold = drift_threshold

    def should_retrain(self, current_roc_auc, drifted_features_pct):
        """
        Determine if model should be retrained.

        Args:
            current_roc_auc: Current production ROC-AUC
            drifted_features_pct: % of features with drift

        Returns:
            tuple: (should_retrain, reasons)
        """
        reasons = []

        # Performance degradation
        if current_roc_auc < self.performance_threshold:
            reasons.append(
                f"Performance degradation: ROC-AUC = {current_roc_auc:.4f}"
            )

        # Feature drift
        if drifted_features_pct > self.drift_threshold:
            reasons.append(
                f"Significant feature drift: {drifted_features_pct:.1%} of features"
            )

        return len(reasons) > 0, reasons

    def trigger_retraining(self):
        """Trigger automated retraining pipeline."""
        import subprocess

        # Log retraining trigger
        logging.info("Triggering automated model retraining")

        # Run retraining pipeline
        result = subprocess.run(
            ['python', 'scripts/pipeline/retrain_model.py'],
            capture_output=True
        )

        if result.returncode == 0:
            logging.info("Retraining completed successfully")
            send_alert("Model retrained successfully", "New model ready for validation")
        else:
            logging.error(f"Retraining failed: {result.stderr}")
            send_alert("Model retraining failed", result.stderr, severity='CRITICAL')


# Usage
manager = RetrainingManager(
    performance_threshold=0.70,
    drift_threshold=0.10
)

# Check weekly
should_retrain, reasons = manager.should_retrain(
    current_roc_auc=0.68,
    drifted_features_pct=0.12
)

if should_retrain:
    print(f"Retraining triggered. Reasons:")
    for reason in reasons:
        print(f"  - {reason}")

    manager.trigger_retraining()
```

---

## Complete Monitoring Script

```python
# run_monitoring.py
"""
Complete monitoring pipeline.

Run this script weekly to monitor model health.
"""

def main():
    """Main monitoring pipeline."""
    print("=" * 60)
    print("CREDIT SCORING MODEL MONITORING")
    print("=" * 60)

    # 1. Feature drift detection
    print("\n1. Checking for feature drift...")
    detector = FeatureDriftDetector(X_train.values, X_train.columns.tolist())
    drifted = detector.get_drifted_features(X_production.values)
    drift_pct = len(drifted) / len(X_train.columns)

    print(f"   Drifted features: {len(drifted)}/{len(X_train.columns)} ({drift_pct:.1%})")

    if drift_pct > 0.10:
        alert_manager.send_alert(
            "Feature Drift Detected",
            f"{len(drifted)} features drifting",
            severity='WARNING'
        )

    # 2. Performance evaluation
    print("\n2. Evaluating production performance...")
    roc_auc = evaluate_production_performance()

    if roc_auc < 0.70:
        alert_manager.send_alert(
            "Performance Degradation",
            f"ROC-AUC = {roc_auc:.4f}",
            severity='CRITICAL'
        )

    # 3. Check retraining triggers
    print("\n3. Checking retraining triggers...")
    retrain_manager = RetrainingManager()
    should_retrain, reasons = retrain_manager.should_retrain(roc_auc, drift_pct)

    if should_retrain:
        print("   RETRAINING TRIGGERED:")
        for reason in reasons:
            print(f"     - {reason}")

        retrain_manager.trigger_retraining()

    print("\n" + "=" * 60)
    print("Monitoring complete")
    print("=" * 60)


if __name__ == '__main__':
    main()
```

---

## Summary

**Key Monitoring Components**:
1. ✅ Business metrics (default rate, business cost)
2. ✅ Model performance (ROC-AUC, precision, recall)
3. ✅ System metrics (latency, throughput, errors)
4. ✅ Data drift detection (features, predictions, targets)
5. ✅ Alerting system (email, Slack)
6. ✅ Dashboards (Grafana, Streamlit)
7. ✅ Automated retraining triggers

**Run Schedule**:
- **Real-time**: Request volume, latency, errors
- **Daily**: Prediction distribution, default rate
- **Weekly**: Feature drift, performance evaluation, retraining check
- **Monthly**: Comprehensive model review, business impact analysis

---

**Last Updated**: December 8, 2025
