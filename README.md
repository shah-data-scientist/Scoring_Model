# Credit Scoring Model - Production ML System

[![Tests](https://img.shields.io/badge/tests-67%2F67%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.13-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)]()
[![MLflow](https://img.shields.io/badge/MLflow-3.6-orange)]()

A production-ready machine learning system for credit default prediction, featuring automated monitoring, comprehensive testing, and REST API serving.

---

## 🎯 Quick Start

### Launch All Services (Easiest)
```bash
# Windows
launch_services.bat

# Linux/Mac
./launch_services.sh
```

This opens:
- **MLflow UI**: http://localhost:5000 (Experiment tracking)
- **Dashboard**: http://localhost:8501 (Threshold optimization)
- **API Docs**: http://localhost:8000/docs (Interactive API)

### Individual Services
```bash
# MLflow UI only
poetry run python scripts/deployment/start_mlflow_ui.py

# Dashboard only
poetry run streamlit run scripts/deployment/dashboard.py

# API Server only
poetry run python scripts/deployment/start_api.py
```

---

## 📊 Project Overview

### Business Problem
Predict credit default risk for loan applications to minimize financial losses while maintaining customer approval rates.

### Solution
Machine learning system that:
- **Predicts** default probability for each application
- **Optimizes** decision threshold for business cost (FN=€10, FP=€1)
- **Achieves** 0.7761 ROC-AUC with domain-engineered features
- **Serves** predictions via REST API (<50ms latency)

### Key Metrics
| Metric | Value | Target |
|--------|-------|--------|
| **ROC-AUC** | 0.7761 | > 0.75 |
| **Precision** | 0.52 | > 0.50 |
| **Recall** | 0.68 | > 0.60 |
| **Business Cost** | €2.45/client | Minimize |
| **API Latency** | <50ms | <100ms |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACES                          │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  Web Apps    │  Mobile Apps │  Dashboards  │  Notebooks     │
└──────┬───────┴──────┬───────┴──────┬───────┴────────┬───────┘
       │              │              │                │
       └──────────────┴──────────────┴────────────────┘
                          │
                  ┌───────▼────────┐
                  │   REST API     │
                  │  (FastAPI)     │
                  │  Port 8000     │
                  └───────┬────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
  ┌───────▼────────┐ ┌───▼────┐ ┌────────▼────────┐
  │   ML Model     │ │MLflow  │ │   Monitoring    │
  │   (LightGBM)   │ │Registry│ │   (Drift, Perf) │
  │   189 Features │ │Port    │ │                 │
  └────────────────┘ │5000    │ └─────────────────┘
                     └────────┘
```

---

## 📁 Repository Structure

```
Scoring_Model/
├── README.md                          # This file
├── launch_services.bat/sh             # Quick service launcher
├── pyproject.toml                     # Poetry dependencies
│
├── src/                               # Production source code
│   ├── config.py                      # Configuration management
│   ├── validation.py                  # Data validation utilities
│   ├── data_preprocessing.py          # Data loading & preprocessing
│   ├── feature_engineering.py         # Feature creation
│   ├── domain_features.py             # Business domain features
│   ├── model_training.py              # Model training utilities
│   ├── evaluation.py                  # Evaluation metrics
│   └── mlflow_utils.py                # MLflow integration
│
├── api/                               # REST API
│   └── app.py                         # FastAPI application
│
├── scripts/                           # Utility scripts
│   ├── deployment/                    # Service launchers
│   ├── mlflow/                        # MLflow management
│   ├── experiments/                   # ML experiments
│   └── data/                          # Data utilities
│
├── tests/                             # Test suite (67 tests)
│   ├── test_api.py                   # API endpoint tests
│   ├── test_validation.py            # Validation tests
│   └── test_config.py                # Configuration tests
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   └── 04_hyperparameter_optimization.ipynb
│
├── docs/                              # Documentation
│   ├── API_TESTING_GUIDE.md          # How to test API
│   ├── MODEL_MONITORING.md           # Monitoring guide
│   ├── DEPLOYMENT_GUIDE.md           # Deployment instructions
│   └── presentations/                # Presentations
│
├── data/processed/                    # Processed features
├── mlruns/                            # MLflow tracking
├── models/                            # Saved models
└── results/                           # Generated outputs
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.13+
- Poetry (dependency management)
- 8GB RAM minimum
- 10GB disk space

### Installation

```bash
# 1. Clone repository
git clone <repository-url>
cd Scoring_Model

# 2. Install dependencies
poetry install

# 3. Run tests to verify installation
poetry run pytest tests/ -v

# 4. Launch services
./launch_services.bat  # Windows
# or
./launch_services.sh   # Linux/Mac
```

---

## 🧪 Testing

```bash
# Run all tests
poetry run pytest tests/ -v

# Run with coverage
poetry run pytest tests/ --cov=src --cov=api --cov-report=html
```

**Results**: 67/67 tests passing ✅

---

## 📡 API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Single Prediction
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "features": [0.5, 0.3, ...],  # 189 features
        "client_id": "100002"
    }
)

result = response.json()
print(f"Risk: {result['risk_level']}")  # LOW, MEDIUM, HIGH, CRITICAL
print(f"Probability: {result['probability']:.4f}")
```

**Full API Guide**: [docs/API_TESTING_GUIDE.md](docs/API_TESTING_GUIDE.md)

---

## 📊 Model Details

### Best Model
- **Algorithm**: LightGBM Classifier
- **Features**: 189 (baseline + domain-engineered)
- **Cross-Validation**: 5-fold StratifiedKFold
- **ROC-AUC**: 0.7761 ± 0.0064
- **Optimal Threshold**: 0.3282

### Feature Categories
1. **Baseline** (184): Original application data
2. **Domain** (5): Business logic features
   - DEBT_TO_INCOME_RATIO
   - EMPLOYMENT_YEARS
   - INCOME_PER_PERSON
   - AGE_YEARS
   - CREDIT_UTILIZATION

---

## 🔍 Monitoring

### Metrics Tracked
- Business: Default rate, business cost
- Performance: ROC-AUC, precision, recall
- System: API latency, throughput, errors
- Data: Feature drift, prediction drift

**Monitoring Guide**: [docs/MODEL_MONITORING.md](docs/MODEL_MONITORING.md)

---

## 📚 Documentation

### Guides
- [Getting Started](docs/GETTING_STARTED.md)
- [API Testing](docs/API_TESTING_GUIDE.md)
- [Model Monitoring](docs/MODEL_MONITORING.md)
- [Deployment](docs/DEPLOYMENT_GUIDE.md)

### Presentations
- [Business Overview](docs/presentations/BUSINESS_PRESENTATION.md)
- [Technical Deep Dive](docs/presentations/TECHNICAL_PRESENTATION.md)

---

**Last Updated**: December 9, 2025
**Version**: 1.0.0
**Status**: Production Ready ✅
