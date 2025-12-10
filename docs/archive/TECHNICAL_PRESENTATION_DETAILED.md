# Technical Presentation: MLOps & Architecture for Credit Scoring

## 1. Title Slide
- **Project:** Credit Scoring Model - Technical Deep Dive
- **Focus:** Architecture, Methodology, and MLOps Best Practices
- **Date:** December 2025
- **Objective:** Showcase the robust engineering behind the credit scoring solution.

## 2. Architecture Overview: End-to-End MLOps Pipeline
- **Core Technology Stack:**
    - **Language:** Python 3.12+ for all development.
    - **Dependency Management:** Poetry for reproducible environments and package management.
    - **API Framework:** FastAPI for high-performance, asynchronous model serving.
    - **Interactive Frontend:** Streamlit for rapid dashboard development and user interaction.
    - **Experiment Tracking & Model Management:** MLflow for comprehensive MLOps lifecycle management.
    - **Testing:** Pytest for robust unit and integration testing.
- **Data & Model Flow Diagram:**
    1.  **Raw Data:** CSVs from Home Credit Default Risk dataset.
    2.  **Data Preprocessing & Feature Engineering:** Notebooks and `src/` modules.
    3.  **Experimentation:** Models trained and evaluated, logged to MLflow.
    4.  **Model Registry:** Best model registered and versioned in MLflow.
    5.  **API:** FastAPI serves predictions from the registered model.
    6.  **Dashboard:** Streamlit application consumes API for interactive insights.

## 3. Data Pipeline & Feature Engineering
- **Dataset Context:** Home Credit Default Risk dataset, featuring an extreme class imbalance (~8% positive class - defaults).
- **Data Preprocessing Steps:**
    - **Missing Value Handling:** Imputation strategies (mean, median, mode) and creation of missing indicators to preserve information.
    - **Categorical Encoding:** One-Hot Encoding for nominal features, Label Encoding for ordinal features where appropriate.
    - **Class Imbalance Solutions:** Employed `class_weight='balanced'` in models and focused on metrics suitable for imbalance (PR-AUC, F-beta score).
- **Feature Engineering Philosophy:**
    - **Domain-Specific Features:** Creation of expert-derived features (e.g., `DEBT_TO_INCOME_RATIO`, `ANNUITY_TO_INCOME_RATIO`, `EMPLOYMENT_YEARS`).
    - **Aggregated Features:** Extracted summary statistics (min, max, mean, std, count) from auxiliary tables (`bureau`, `previous_application`, `credit_card_balance`, etc.) to capture behavioral patterns and historical context.
    - **Feature Expansion:** Successfully expanded the feature set from the original **122 to 318 features**, significantly enriching the model's predictive power.
- **Data Validation:** Implemented schema validation at various stages to ensure data quality and consistency.

## 4. Model Development & Optimization
- **Baseline Models:**
    - Evaluated Logistic Regression, Random Forest, and LightGBM as initial candidates.
    - MLflow was instrumental in comparing these baseline experiments.
- **Champion Model Selection:**
    - **LightGBM** emerged as the top performer due to its superior predictive accuracy, efficiency with large datasets, and built-in handling of categorical features.
- **MLflow for Experiment Tracking:**
    - **Parameter Logging:** Tracked all hyperparameters (e.g., `n_estimators`, `max_depth`, `learning_rate`, `class_weight`).
    - **Metric Logging:** Recorded key performance indicators such as ROC-AUC, Precision, Recall, F1-Score, and critically, the F-beta score (`beta=3.2`) for business cost sensitivity.
    - **Artifact Storage:** Stored trained models, feature importance plots, confusion matrices, and ROC/PR curves for each run.
- **Hyperparameter Optimization:**
    - Employed a systematic approach (e.g., `RandomizedSearchCV` or `Optuna`) combined with **Stratified K-Fold Cross-Validation** (N_FOLDS=5) to ensure robust and unbiased evaluation.
    - Optimization targeted the F-beta score to align with business objectives, identifying optimal model configurations.
    - **Achieved Performance:** Consistently maintained **ROC-AUC > 0.77** and optimized F3.2 score on cross-validation folds.
- **Business Cost-Sensitive Thresholding:** Identified and applied an optimal decision threshold (e.g., 0.328) based on a business cost function to balance False Positives and False Negatives.

## 5. Model Interpretability (SHapley Additive exPlanations - SHAP)
- **Rationale for SHAP:** Chosen for its theoretical soundness and consistency in attributing feature contributions to individual predictions, crucial for transparency and regulatory compliance.
- **Implementation:**
    - Utilized `shap.TreeExplainer` specifically designed for tree-based models like LightGBM, providing efficient and accurate SHAP value computation.
    - **Global Interpretability:** Generated SHAP summary plots to visualize overall feature importance, indicating features that have the largest average impact on model output magnitude.
    - **Local Interpretability:** Produced SHAP force plots and waterfall plots for individual predictions within the Streamlit dashboard, explaining *why* a specific applicant received their predicted score.

## 6. API Design & Implementation
- **Framework:** FastAPI
    - **Asynchronous Processing:** Enables high concurrency and low latency for prediction requests.
    - **Pydantic Models:** Enforced strict schema validation for both request payloads and response structures, minimizing data-related errors.
    - **Interactive Documentation:** Automatic generation of OpenAPI (Swagger UI) documentation at `/docs`, facilitating easy API testing and integration.
- **Key Endpoints:**
    - `/health`: Provides real-time status of the API and model loading (e.g., `healthy`, `unhealthy`, `model_loaded`).
    - `/predict`: Accepts a single applicant's features and `client_id`, returning prediction, probability, and risk level.
    - `/predict/batch`: Optimized for multiple applicant predictions, processing arrays of features and `client_ids`.
- **Robust Error Handling:** Implemented comprehensive exception handling for:
    - **Validation Errors:** Automatically handles incorrect feature counts, malformed JSON, and invalid data types (e.g., strings where floats are expected).
    - **Data Integrity:** Explicitly handles non-numeric values like `NaN` and `Infinity` (which are not JSON-compliant for numbers), returning a 422 Unprocessable Entity.
    - **Model Availability:** Returns 503 Service Unavailable if the model is not yet loaded or encounters an internal error.

## 7. MLOps & Quality Assurance
- **Version Control:** All code managed with Git, ensuring history, collaboration, and rollback capabilities.
- **MLflow Model Registry:**
    - Centralized repository for managing model versions, stages (Staging, Production, Archived), and metadata.
    - Provides a clear lineage from experiment run to deployable model.
- **Comprehensive Testing Suite (Pytest):**
    - **Unit Tests:** Verified individual functions and modules (`src/`).
    - **API Integration Tests:** Ensured all FastAPI endpoints function correctly, handling valid and invalid inputs, edge cases (e.g., missing features, malformed JSON).
    - **Data Validation Tests:** Checked schema conformity and data ranges.
- **CI/CD Strategy (Planned/Conceptual):**
    - Integration with GitHub Actions (or similar) to automate testing, linting, and potential model deployment on code pushes.
- **Model Monitoring & Maintenance:**
    - **Data Drift Detection:** Plan to use tools (e.g., Evidently AI) or custom scripts to monitor changes in input feature distributions over time.
    - **Concept Drift Detection:** Monitor model performance metrics on live data to identify drops in accuracy, triggering retraining or model review.
    - **Feedback Loops:** Establish mechanisms to incorporate production data feedback for continuous model improvement.

## 8. Dashboard Development (Streamlit)
- **Rapid Prototyping & Interactivity:** Streamlit's framework allowed for quick development of a highly interactive and intuitive user interface.
- **Backend Integration:** The dashboard directly calls the FastAPI for real-time predictions, ensuring the model served is the same one used by other applications.
- **Key Visualizations:**
    - **Confusion Matrix:** Interactive display that updates with threshold changes, showing row and column percentages.
    - **Probability Distribution:** Histogram of predicted probabilities, segmented by actual target.
    - **Metrics Display:** Real-time update of Precision, Recall, F-beta, and Business Cost as the decision threshold is adjusted.
    - **SHAP Waterfall Plots:** Integrated to provide individual prediction explanations.
    - **Client Profile Comparison:** Visual comparisons of applicant features against population statistics.

## 9. Challenges & Solutions
- **Challenge: Class Imbalance:**
    - **Solution:** Employed `StratifiedKFold` for robust cross-validation, used `class_weight='balanced'` in LGBM, and optimized for F-beta/PR-AUC.
- **Challenge: Large Data Volume & Memory Constraints:**
    - **Solution:** Utilized memory-optimized data loading (chunking, efficient dtypes), and provided clear recommendations for cloud-based processing (Kaggle Kernels) for comprehensive feature engineering.
- **Challenge: Ensuring Reproducibility:**
    - **Solution:** Poetry for dependency management, MLflow for experiment tracking (random seeds, code versions, parameters), and a structured project layout.
- **Challenge: Model Explainability for Business Users:**
    - **Solution:** Integrated SHAP values into the dashboard, presenting complex explanations in an accessible, visual format.
- **Challenge: Robust API Design:**
    - **Solution:** FastAPI with Pydantic for strong type hinting and automatic validation, comprehensive error handling for edge cases.

## 10. Conclusion
- **Outcome:** Successfully developed a robust, explainable, and production-ready credit scoring model, supported by a comprehensive MLOps pipeline.
- **Readiness:** The system is prepared for pilot deployment, offering significant enhancements in predictive accuracy, operational efficiency, and transparency for lending decisions.
- **Future:** Positions the organization for continuous improvement and responsible AI deployment in credit risk management.
