# Documentation Index

## 📚 Complete Documentation Guide

This index provides quick access to all project documentation.

---

## 🚀 Quick Start

### Essential Guides (Read First)
1. **[README.md](../README.md)** - Project overview and quick start
2. **[USER_GUIDE.md](USER_GUIDE.md)** - Complete user guide (installation, services, API, troubleshooting)
3. **[TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md)** - Technical details (model, architecture, deployment)
4. **[MODEL_MONITORING.md](MODEL_MONITORING.md)** - Production monitoring guide

### Presentations (For Stakeholders)
5. **[Business Presentation](presentations/BUSINESS_PRESENTATION.md)** - Executive summary, ROI, implementation plan (14 sections)
6. **[Technical Presentation](presentations/TECHNICAL_PRESENTATION.md)** - Architecture, implementation details (14 sections)

---

## 📖 Documentation by Role

### 👤 End Users / Analysts
**Start here**: [USER_GUIDE.md](USER_GUIDE.md)

**What you'll learn**:
- How to launch services (one-click)
- Using MLflow UI to view experiments
- Using Dashboard to adjust thresholds
- Testing API with different methods
- Understanding predictions and risk levels
- Troubleshooting common issues

**Estimated reading time**: 15 minutes

### 👨‍💻 Developers / Data Scientists
**Start here**: [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md)

**What you'll learn**:
- Model architecture and performance
- Feature engineering details
- Data pipeline implementation
- API implementation details
- Testing strategy and coverage
- MLflow integration
- Deployment options

**Estimated reading time**: 30 minutes

### 👔 Business Stakeholders
**Start here**: [Business Presentation](presentations/BUSINESS_PRESENTATION.md)

**What you'll learn**:
- Business problem and solution
- Financial impact (€25.5M savings)
- ROI analysis (11,200%)
- Implementation roadmap
- Risk management
- Success metrics

**Estimated reading time**: 20 minutes

### 🔧 DevOps / SRE
**Start here**: [MODEL_MONITORING.md](MODEL_MONITORING.md)

**What you'll learn**:
- Monitoring setup
- Drift detection
- Performance tracking
- Alert configuration
- Production best practices

**Estimated reading time**: 20 minutes

---

## 📊 Quick Reference

### Services
| Service | URL | Purpose |
|---------|-----|---------|
| **MLflow UI** | http://localhost:5000 | Experiment tracking |
| **Dashboard** | http://localhost:8501 | Threshold optimization |
| **API Docs** | http://localhost:8000/docs | Interactive API testing |

### Launch Commands
```bash
# All services
./launch_services.bat  # Windows
./launch_services.sh   # Linux/Mac

# Individual services
poetry run python scripts/deployment/start_mlflow_ui.py
poetry run streamlit run scripts/deployment/dashboard.py
poetry run python scripts/deployment/start_api.py
```

### Testing
```bash
# Run all tests
poetry run pytest tests/ -v

# Run with coverage
poetry run pytest tests/ --cov=src --cov=api --cov-report=html
```

---

## 📁 Documentation by Audience

### 👔 Business Stakeholders
**Start here**: [Business Presentation](presentations/BUSINESS_PRESENTATION.md)

Key documents:
- ROI analysis and business impact
- Implementation phases
- Risk management
- Success metrics

### 👨‍💻 Technical Team
**Start here**: [Technical Presentation](presentations/TECHNICAL_PRESENTATION.md)

Key documents:
- System architecture
- API implementation
- Monitoring setup
- Deployment guide

### 🔬 Data Scientists
**Start here**: [MLflow UI](http://localhost:5000)

Key documents:
- [Feature Engineering Guide](FEATURE_ENGINEERING_EXPERIMENT_DESIGN.md)
- [MLflow Organization](MLFLOW_RUNS_ORGANIZATION.md)
- [Project Summary](PROJECT_SUMMARY.md)

### 🧪 QA Engineers
**Start here**: [API Testing Guide](API_TESTING_GUIDE.md)

Key documents:
- Test procedures
- Expected behaviors
- Edge cases

### 🚢 DevOps Engineers
**Start here**: [Deployment Guide](DEPLOYMENT_GUIDE.md)

Key documents:
- Infrastructure requirements
- Monitoring setup
- Scaling considerations

---

## 🆘 Troubleshooting

All troubleshooting information is in the [USER_GUIDE.md](USER_GUIDE.md#troubleshooting) - see section "Troubleshooting"

**Common issues covered**:
- MLflow UI not starting
- Dashboard errors
- API 503 errors
- Tests failing
- Services running slowly (NEW - performance fixes)

---

## 🔧 Performance Optimization

If services are slow:

1. **Dashboard**: Already optimized (auto-samples large files)
2. **MLflow UI**: Clean up old runs
   ```bash
   # Dry run first (see what would be deleted)
   poetry run python scripts/mlflow/cleanup_old_runs.py --dry-run

   # Actually clean up (keeps top 10 runs)
   poetry run python scripts/mlflow/cleanup_old_runs.py
   ```
3. **API**: Restart with `--reload` flag

---

## 📞 Support Resources

### Documentation
- **User Guide**: Quick start and troubleshooting
- **Technical Guide**: Deep technical details
- **Monitoring Guide**: Production operations

### External Resources
- **MLflow Docs**: https://mlflow.org/docs/latest/
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **LightGBM Docs**: https://lightgbm.readthedocs.io/

---

## 📋 Quick Checklist

### New Users
- [ ] Read [README.md](../README.md)
- [ ] Read [USER_GUIDE.md](USER_GUIDE.md)
- [ ] Launch services: `launch_services.bat`
- [ ] Test all three UIs

### Developers
- [ ] Read [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md)
- [ ] Run tests: `poetry run pytest tests/ -v`
- [ ] Review presentations

### Stakeholders
- [ ] Read [Business Presentation](presentations/BUSINESS_PRESENTATION.md)
- [ ] Read [Technical Presentation](presentations/TECHNICAL_PRESENTATION.md)

---

**Last Updated**: December 10, 2025
**Version**: 2.0 (Consolidated)
**Maintained by**: Data Science Team
