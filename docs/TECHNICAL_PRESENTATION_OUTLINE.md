# Technical Presentation Outline: MLOps & Architecture

## 1. Title Slide
- **Project:** Credit Scoring Model - Technical Deep Dive
- **Focus:** Architecture, Methodology, and MLOps
- **Date:** December 2025

## 2. Architecture Overview
- **Stack:** 
    - **Language:** Python 3.12+
    - **Experiment Tracking:** MLflow
    - **API:** FastAPI
    - **Frontend:** Streamlit
    - **Testing:** Pytest
    - **Dependency Management:** Poetry
- **Flow:** Data -> Notebooks (Experiments) -> MLflow (Registry) -> API -> Dashboard.

## 3. Data Pipeline & Feature Engineering
- **Dataset:** Home Credit Default Risk (Imbalanced: ~8% positive).
- **Preprocessing:** 
    - Handling missing values (Imputation).
    - Encoding categorical variables (One-Hot/Label).
    - Handling imbalance (SMOTE/Class Weights).
- **Feature Engineering:**
    - Domain features (Credit-to-Income, Annuity-to-Income).
    - Aggregations from bureau/prev_app data.
    - Feature Selection (LGBM Importance).

## 4. Model Development & Optimization
- **Baseline Models:** Logistic Regression, Random Forest, LGBM.
- **Experiment Tracking:** Used MLflow to log params, metrics (AUC, F1), and artifacts.
- **Hyperparameter Tuning:** Bayesian Optimization / Grid Search via Hyperopt or Scikit-learn.
- **Champion Model:** LightGBM (Selected for speed and performance).
    - Best AUC: 0.7x
    - Optimized threshold for business cost.

## 5. Interpretability (SHAP)
- **Why SHAP?** Consistent and theoretically sound feature attribution.
- **Implementation:** `TreeExplainer` for LightGBM.
- **Optimization:** Optimized calculation for real-time inference (using background datasets or approximations if needed).

## 6. API Design & Implementation
- **Framework:** FastAPI (Async, Type Validation).
- **Endpoints:**
    - `/health`: System status.
    - `/predict`: Single client scoring.
    - `/predict/batch`: Batch processing.
- **Schema:** Pydantic models for strict input validation.
- **Error Handling:** Robust handling of NaN/Inf and missing fields.

## 7. MLOps & Quality Assurance
- **Versioning:** Git + MLflow Model Registry.
- **Testing:** 
    - Unit tests for API endpoints (`pytest`).
    - Data validation tests.
- **CI/CD:** (Mention if set up, e.g., GitHub Actions).
- **Drift Monitoring:** Strategy for monitoring feature distribution changes over time (Evidently).

## 8. Dashboard Development
- **Framework:** Streamlit (Rapid prototyping).
- **Features:**
    - API integration (decoupled frontend/backend).
    - Visualizations using Matplotlib/Plotly.
    - SHAP waterfall plots for individual explanation.

## 9. Challenges & Solutions
- **Imbalance:** Solved via class weights and specific metrics (PR-AUC).
- **Data Volume:** Optimized memory usage (parquet/optimization).
- **Deployment:** Containerization (Docker) readiness.

## 10. Conclusion
- Robust, scalable, and explainable AI solution ready for deployment.
