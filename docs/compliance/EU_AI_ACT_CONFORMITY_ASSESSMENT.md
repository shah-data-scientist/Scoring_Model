# EU AI Act Conformity Assessment
## Technical Documentation under Regulation (EU) 2024/1689, Annex IV

| Field | Value |
|-------|-------|
| **System identifier** | Credit Scoring System — Home Credit Default Risk |
| **System version** | Production v3 (MLflow: `credit_scoring_production_model`) |
| **Assessment date** | 2026-05-19 |
| **Auditor** | Shahul SHAIK, CISA-certified |
| **Audit type** | Internal self-assessment |
| **Risk classification** | **HIGH-RISK** — Annex III, Point 5(b) |
| **Document status** | Final |

---

## Risk Classification Rationale

Article 6(2) of Regulation (EU) 2024/1689 designates AI systems listed in Annex III as high-risk. **Annex III, Point 5(b)** explicitly covers:

> *"AI systems intended to be used to evaluate the creditworthiness of natural persons or establish their credit score, with the exception of AI systems used for the purpose of detecting financial fraud."*

This system predicts the probability of loan default for individual applicants and outputs an approve/reject recommendation. It falls unambiguously within this definition. All obligations under Chapter III, Section 2 (Articles 9–15) therefore apply.

---

## §1 — General Description of the AI System

### 1.1 Intended Purpose

The system calculates a default probability score (0–1) for consumer credit applications. A score at or above the decision threshold (0.10) flags the application for enhanced human review or rejection. The model is intended to **assist** credit analysts — it is not a sole automated decision-maker (see §9, Art 14).

### 1.2 Intended Users and Deployers

- **Direct users**: Credit risk analysts and loan officers at a consumer credit institution
- **Affected persons**: Individual loan applicants (natural persons)
- **Deployer obligations**: Users must complete model-explainability training and understand SHAP-based explanations before operating the system

### 1.3 System Outputs

| Output | Type | Use |
|--------|------|-----|
| Default probability score | Float in [0, 1] | Risk quantification |
| Binary flag @ threshold=0.10 | 0 / 1 | Approve / escalate recommendation |
| SHAP feature contributions | Vector | Individual decision explanation |

### 1.4 Lifecycle Stage

Development and validation complete. Production deployment not yet active. This assessment covers the development artefact as registered in MLflow (`credit_scoring_production_model` v3). A full third-party conformity assessment is required prior to any EU-regulated production deployment.

### 1.5 Geographic and Demographic Scope

Training data originates from the Home Credit Default Risk dataset (Kaggle), covering multi-country consumer credit portfolios in Home Credit's operating regions. The validation population comprises 92,254 applications with a 8.07% observed default rate.

---

## §2 — Detailed Description of Elements and Development Process

### 2.1 Data Sources

| Dataset | Rows | Description |
|---------|------|-------------|
| `application_train.csv` | 307,511 | Primary loan application features (target variable: `TARGET`) |
| `bureau.csv` | 1,716,428 | Credit bureau history (aggregated per applicant) |
| `bureau_balance.csv` | 27,299,925 | Monthly bureau balance status |
| `previous_application.csv` | 1,670,214 | Prior loan applications |
| `POS_CASH_balance.csv` | 10,001,358 | POS and cash loan monthly balances |
| `installments_payments.csv` | 13,605,401 | Instalment payment history |
| `credit_card_balance.csv` | 3,840,312 | Credit card monthly balances |

All secondary tables were aggregated per `SK_ID_CURR` and joined to the application table before modelling. Full pipeline documented in `notebooks/02_feature_engineering.ipynb`.

### 2.2 Feature Engineering

| Stage | Feature count | Action |
|-------|---------------|--------|
| Raw application features | 122 | Loaded as-is |
| After bureau/transaction aggregation | ~316 | Aggregation stats (mean, sum, max, min, nunique) |
| After missing-value filter (>70% missing dropped) | ~258 | 58 sparse features removed |
| After domain feature creation | ~272 | 14 engineered features added |
| After one-hot encoding (14 low-cardinality columns) | ~306 | Categorical features encoded |
| **Final feature set (post importance filter)** | **189** | Selected by LightGBM feature importance threshold |

Key engineered features: `DEBT_TO_INCOME_RATIO`, `EXT_SOURCE_MEAN`, `CREDIT_TO_GOODS_RATIO`, `AGE_YEARS`, `CREDIT_INCOME_PERCENT`, `ANNUITY_INCOME_PERCENT`.

Full feature list: `results/artifacts/feature_names.csv` and `models/feature_columns.pkl`.

### 2.3 Sensitive Attributes in the Feature Set

The following demographic and socioeconomic proxy features are **present** in the final 189-feature set:

| Attribute | Features | EU AI Act relevance |
|-----------|----------|---------------------|
| Gender | `CODE_GENDER_M` | Protected — Art 10 monitoring required |
| Age | `AGE_YEARS`, `DAYS_BIRTH` | Protected — Art 10 monitoring required |
| Marital status | `NAME_FAMILY_STATUS_*` (4 dummies) | Social characteristic |
| Education level | `NAME_EDUCATION_TYPE_*` (4 dummies) | Social characteristic |
| Income type | `NAME_INCOME_TYPE_*` (3 dummies) | Socioeconomic proxy |
| Housing type | `NAME_HOUSING_TYPE_*` (4 dummies) | Socioeconomic proxy |
| Family composition | `CNT_CHILDREN`, `HAS_CHILDREN` | Social characteristic |

Use of these features is legally permissible under EU credit regulation subject to demonstrated non-discrimination. Bias testing results are documented in §6 and Annex A.

### 2.4 Model Architecture

**Algorithm**: LightGBM gradient boosting classifier  
**Hyperparameter optimisation**: RandomizedSearchCV, 10 iterations × 5-fold stratified CV on 20% subsample; objective metric: business cost scorer (minimises 10×FN + FP per fold)  
**Full training**: Best configuration retrained on full training set (n=215,257)  
**MLflow experiment**: `credit_scoring_optimization_fbeta` (notebook NB04)  
**Registered model**: `credit_scoring_production_model` v3

Key hyperparameters (from MLflow):

| Parameter | Value |
|-----------|-------|
| `n_estimators` | See MLflow run `credit_scoring_optimization_fbeta` |
| `learning_rate` | See MLflow run |
| `num_leaves` | See MLflow run |
| `objective` | `binary` |
| `class_weight` | `balanced` |
| `random_state` | 42 |

### 2.5 Training Methodology

- **Split**: Stratified 70/30 split (`RANDOM_STATE=42`) producing training (n=215,257) and validation (n=92,254) sets — notebook NB02
- **No data leakage**: `SimpleImputer` (median strategy) and `StandardScaler` fit exclusively on the training split, then applied to validation. Artefacts: `models/imputer.pkl`, `models/scaler.pkl`
- **Target encoding**: Raw binary target `TARGET` (1 = defaulted within 12 months)
- **Class imbalance**: Addressed via `class_weight='balanced'` in LightGBM; no synthetic oversampling

### 2.6 Decision Threshold Justification

The default threshold of 0.50 is inappropriate for a population with an 8.07% default rate. The threshold was selected by minimising a business cost function that reflects the institution's actual loss asymmetry:

```
Business Cost = 10 × FN + FP
```

Where:
- **FN** (missed default) costs 10× more than a **FP** (wrongful rejection)
- The cost ratio of 10:1 was derived from the business presentation: loan loss ~€10,000 vs opportunity cost ~€1,000

A sweep over thresholds 0.05–0.50 identified **0.10** as the cost-minimising point, producing normalised cost of **€0.43/client** versus €0.81/client baseline (−47%).

---

## §3 — Monitoring, Functioning, and Control

### 3.1 Experiment and Run Logging

All training runs are logged to MLflow at `sqlite:///mlruns/mlflow.db`. For each run the following are recorded: all hyperparameters, CV metrics, validation metrics, confusion matrix values, model artefacts, and timestamp. The audit trail is non-destructive — runs are soft-deleted, not purged.

### 3.2 Post-Deployment Monitoring Plan

See `docs/MODEL_MONITORING.md` for the full monitoring specification. Key signals:

| Signal | Alert threshold | Action |
|--------|----------------|--------|
| ROC-AUC (weekly) | < 0.75 | Trigger retraining |
| Score distribution drift (PSI) | > 0.20 | Investigate + potential retrain |
| Gender DIR monthly | < 0.50 | Mandatory model review |
| Positive prediction rate shift | > 5pp vs baseline | Investigate |

### 3.3 Log Retention

MLflow artefacts are stored in `mlruns/` with SQLite backend. Logs must be retained for a minimum of 10 years post-deployment per Art 12(1) requirements.

---

## §4 — Description of Changes Made During Lifecycle

| Notebook | Stage | Key outputs |
|----------|-------|-------------|
| NB01 — EDA | Exploratory analysis | Data quality report, distribution analysis |
| NB02 — Feature Engineering | Data preparation | 189 features, imputer.pkl, scaler.pkl, feature_columns.pkl, train/val splits |
| NB03 — Baseline Models | Model selection | 5 baseline models compared; LightGBM selected (ROC-AUC=0.7778) |
| NB04 — Hyperparameter Optimisation | Model training | Business-cost-optimised LightGBM v3; registered in MLflow |
| NB05 — Model Interpretation | Explainability | SHAP TreeExplainer, force plots, dependence plots, threshold analysis |
| NB06 — Compliance Audit | This document | Bias testing, fairness CSVs, 7 compliance plots |

All changes are tracked in MLflow and git history. The git repository is at `github.com/shah-data-scientist/Scoring_Model`, commit `portfolio V1`.

---

## §5 — Risk Management System (Art 9)

### 5.1 Identified Risks

| Risk ID | Risk | Probability | Magnitude |
|---------|------|-------------|-----------|
| R1 | Discriminatory credit outcomes due to demographic disparities | MEDIUM | HIGH |
| R2 | Model drift — performance degradation over time | MEDIUM | HIGH |
| R3 | Data drift — applicant population shift | MEDIUM | MEDIUM |
| R4 | Privacy — PII exposure in training data | LOW | HIGH |
| R5 | Adversarial manipulation of input features | LOW | MEDIUM |
| R6 | System downtime blocking credit decisions | LOW | MEDIUM |

### 5.2 Risk Assessment

**R1 — Discriminatory Outcomes**: Gender DIR=0.612 fails the EEOC 80% rule (see Annex A). However, calibration analysis demonstrates the disparity is base-rate-driven (male true default rate 10.14% vs female 7.01%) and the model does not amplify this gap. The primary risk is that historical base-rate differences may themselves reflect structural inequality — this is a societal risk, not a model error risk. Residual risk is MEDIUM after mitigation.

**R2/R3 — Drift**: Not yet monitored (pre-deployment). Monitoring plan in `docs/MODEL_MONITORING.md` specifies weekly ROC-AUC checks and monthly PSI analysis. Risk reduces to LOW post-deployment.

### 5.3 Mitigation Measures

| Risk | Mitigation |
|------|-----------|
| R1 | Bias testing every release cycle (NB06); human oversight for all flagged applications (Art 14); SHAP explanations for customer disputes |
| R2 | Weekly ROC-AUC monitoring; retrain trigger at AUC < 0.75 |
| R3 | Monthly Population Stability Index (PSI) on feature distributions |
| R4 | No PII (names, addresses, account numbers) in the 189-feature set; data processed within secure environment |
| R5 | FastAPI input validation (Pydantic schema) rejects malformed requests |
| R6 | 99.9% SLA target; manual review fallback during outages |

### 5.4 Residual Risk Statement

**Accepted residual risk**: Gender Disparate Impact Ratio = 0.612, below the 80% rule threshold.

**Rationale for acceptance**: The disparity is proportional to the difference in true default rates between male (10.14%) and female (7.01%) applicants. Model calibration is excellent (mean bias < 0.002). A model constrained to achieve DIR ≥ 0.80 would necessarily worsen calibration, increasing false negatives (missed defaults) and transferring financial risk to the institution without improving individual fairness for equal-risk applicants.

**Escalation trigger**: If gender DIR falls below 0.50 at any post-deployment monitoring check, the model must be immediately reviewed and a new training cycle initiated.

**Human oversight**: All applications with scores in the range 0.08–0.15 (borderline zone) are subject to mandatory human review before final credit decision.

---

## §6 — Performance Metrics and Testing (Art 15)

### 6.1 Overall Performance (Validation Set, n=92,254)

| Metric | Value | Notes |
|--------|-------|-------|
| ROC-AUC | **0.8320** | Monitoring threshold: ≥ 0.75 |
| Recall (TPR) @ t=0.10 | **71.0%** | Fraction of defaults detected |
| False Positive Rate @ t=0.10 | **21.3%** | Fraction of good applicants rejected |
| Brier Score | **0.0617** | Lower is better; 0 = perfect |
| Business cost (normalised) | **€0.43/client** | 10×FN + FP / n |
| Baseline cost (approve-all) | €0.81/client | No model comparison |
| Cost reduction | **−47%** | |
| TP | 5,289 | Defaults correctly flagged |
| FP | 17,756 | Good applicants incorrectly flagged |
| FN | 2,159 | Missed defaults |
| TN | 67,050 | Good applicants correctly approved |

### 6.2 Per-Demographic-Group Performance

Full table: `results/compliance/fairness_metrics_by_group.csv`. Summary:

| Attribute | DIR | 80% Rule | TPR range | Max cal bias |
|-----------|-----|----------|-----------|-------------|
| Gender | 0.612 | **FAIL** | 0.123 | 0.00111 |
| Age group | 0.164 | **FAIL** | 0.495 | 0.00174 |
| Education | 0.405 | **FAIL** | 0.224 | 0.01158 |
| Income type | 0.509 | **FAIL** | 0.117 | 0.00554 |

**Interpretation**: All attributes fail the 80% rule because the model's predicted positive rate tracks true default rates across groups. All groups show excellent calibration (bias < 0.012). The DIR failures are base-rate-driven, not model-error-driven — confirmed by the base-rate-vs-PPR scatter plot (Annex A, Figure 4).

---

## §7 — Data Governance (Art 10)

### 7.1 Data Provenance

Training data: Home Credit Default Risk competition dataset (Kaggle, 2018). This is an openly published research dataset derived from a real consumer credit portfolio. The exact originating countries, collection period, and any post-processing applied by Home Credit prior to publication are not disclosed in the competition documentation.

**Implication**: This model must not be deployed on a population materially different from the original Home Credit applicant pool without temporal and distributional validation.

### 7.2 Data Quality Measures

- **Missing values**: Analysed in NB01; features with >70% missing removed (58 features). Remaining missing values imputed with median strategy (fit on training set only).
- **Outliers**: `DAYS_EMPLOYED` anomaly (value 365,243 = employed since birth) identified in EDA and treated.
- **Target leakage check**: All features are application-time variables; no post-approval information included.
- **Class balance**: 8.07% positive (default). Not artificially balanced in training data; `class_weight='balanced'` handles imbalance algorithmically.

### 7.3 Representativeness

The validation set was created via stratified random split (30% of training data), preserving the overall default rate (8.07%). Demographic distributions in the validation set mirror the training set.

### 7.4 Personal Data Handling

The 189 features in the final model **do not include direct PII** (no names, national identity numbers, full addresses, or account numbers). Demographic proxies (gender, age, family status) are present and are used as predictive features. Their use is consistent with EU credit regulation provided:

1. Discrimination complaints can be investigated via SHAP explanations
2. Human override is always available
3. The basis for decision can be communicated to the applicant (Art 13)

---

## §8 — Transparency and Provision of Information (Art 13)

### 8.1 Model Card

The primary Art 13 transparency document is `docs/compliance/MODEL_CARD.md`. It contains:
- System description and intended use
- Performance metrics (overall and per demographic group)
- Known limitations and caveats
- Contact information for the responsible party

### 8.2 Individual Decision Explanation

For any flagged application, a SHAP force plot can be generated identifying the top contributing features and their direction of impact (see NB05). This enables communication of the form:

> *"Your application was flagged due to: high debt-to-income ratio (your value: 0.85, population median: 0.40), low external credit score (EXT_SOURCE_2: 0.32), and short employment history (1.2 years)."*

This capability satisfies the right-to-explanation obligations under GDPR Art 22 and EU AI Act Art 13(3)(b).

### 8.3 Threshold Transparency

The decision threshold (0.10) and its business justification (10:1 FN/FP cost ratio) are documented in NB04, NB05, and `docs/presentations/TECHNICAL_PRESENTATION.md`. The threshold is adjustable by authorised personnel based on evolving business risk appetite.

---

## §9 — Human Oversight Measures (Art 14)

### 9.1 Oversight Architecture

This system is designed exclusively for **human-in-the-loop** credit decisions. The model output is an **advisory score**, not a binding approval or rejection.

```
Applicant → Application system → ML Score
                                     ↓
                          Score < 0.08  → Recommended approval
                          Score 0.08–0.15 → MANDATORY human review
                          Score > 0.15  → Recommended rejection (human confirms)
```

### 9.2 Override Mechanism

Loan officers are explicitly authorised to:
- Approve applications the model recommends rejecting
- Reject applications the model recommends approving
- Request additional documentation before deciding

All overrides must be logged with a reason code. Accumulated override data is used to identify model weaknesses and inform retraining.

### 9.3 Operator Training Requirements

Before operating the system, loan officers must:
1. Complete training on SHAP-based explanation interpretation
2. Understand the 10:1 FN/FP cost asymmetry and its implication for threshold selection
3. Be aware of the demographic findings documented in this assessment
4. Know the escalation procedure for suspected discriminatory outcomes

### 9.4 Incapacitation Override

In the event of system unavailability, the institution must have a documented manual review fallback capable of handling the full application volume. The AI system must not become a single point of failure for credit operations.

---

## §10 — Accuracy, Robustness, and Cybersecurity (Art 15)

### 10.1 Accuracy

Validated at ROC-AUC = 0.8320 on a held-out validation set of 92,254 applications. This exceeds the internal monitoring minimum of 0.75. Performance was validated against the Dummy Classifier baseline (ROC-AUC = 0.50) and four alternative algorithms (LightGBM, XGBoost, Logistic Regression, Random Forest — see `results/model_comparison.csv`).

### 10.2 Robustness

- The model was retrained from scratch on the full training set (not just cross-validation folds) to maximise robustness
- No temporal hold-out validation has been performed. **Before production deployment, time-based validation is required.**
- LightGBM is robust to moderate feature collinearity and handles missing values natively (though imputation is applied pre-model for consistency)

### 10.3 Cybersecurity

- Model artefacts are stored in MLflow registry with SQLite backend; access should be restricted to authorised data scientists in production
- API inference endpoint (if deployed via `api/`) uses FastAPI with Pydantic schema validation to reject malformed or out-of-distribution inputs
- No model parameters are exposed in API responses

---

## Auditor Declaration

I, **Shahul SHAIK**, holder of the Certified Information Systems Auditor (CISA) certification issued by ISACA, declare that:

1. I conducted this conformity assessment independently and objectively as internal auditor.

2. The technical documentation contained herein accurately represents the AI system as developed, tested, and registered in the MLflow Model Registry as `credit_scoring_production_model` v3 on 2026-05-19.

3. The fairness testing results reported in §6 and Annex A were produced by executing `notebooks/06_eu_ai_act_compliance.ipynb` against model v3 on the held-out validation set (`data/processed/X_val.csv`, n=92,254). The notebook is reproducible and version-controlled.

4. To the best of my knowledge and professional judgement, this system meets the requirements of Articles 9, 10, 13, 14, and 15 of Regulation (EU) 2024/1689, **with the exception of the gender Disparate Impact Ratio finding** documented in §5.4, which I assess as a residual risk arising from historical base-rate differences rather than from model design error, and which is addressed through mandatory human oversight.

5. I acknowledge that this is a development and educational implementation. **A full third-party conformity assessment by an accredited notified body is required before any deployment in an EU-regulated credit granting context.**

6. This assessment will require renewal upon any material change to the model, training data, feature set, or deployment configuration.

---

**Signed**: Shahul SHAIK  
**Date**: 2026-05-19  
**CISA Certification**: ISACA CISA  
**Contact**: [github.com/shah-data-scientist](https://github.com/shah-data-scientist)

---

## Annex A — Fairness Test Results

*Generated by `notebooks/06_eu_ai_act_compliance.ipynb`, model v3, 2026-05-19*

### Table A1 — Per-Group Fairness Metrics (threshold = 0.10)

| Attribute | Group | n | True DR | Pred Pos Rate | TPR | FPR | Cal Bias | Brier | ROC-AUC |
|-----------|-------|---|---------|---------------|-----|-----|----------|-------|---------|
| gender | Male | 31,426 | 10.14% | 33.57% | 78.03% | 28.55% | −0.00111 | 0.0746 | 0.8313 |
| gender | Female | 60,828 | 7.01% | 20.54% | 65.77% | 17.14% | +0.00051 | 0.0550 | 0.8279 |
| age_group | <35 | 25,386 | 10.79% | 38.04% | 80.09% | 32.95% | +0.00099 | 0.0793 | 0.8191 |
| age_group | 35–50 | 35,776 | 8.15% | 24.51% | 70.93% | 20.39% | −0.00091 | 0.0621 | 0.8358 |
| age_group | 50–65 | 28,660 | 5.96% | 15.59% | 58.61% | 12.86% | −0.00003 | 0.0480 | 0.8216 |
| age_group | 65+ | 2,432 | 3.50% | 6.25% | 30.59% | 5.37% | +0.00174 | 0.0317 | 0.8331 |
| education | Higher education | 22,609 | 5.28% | 13.75% | 57.59% | 11.30% | +0.00146 | 0.0427 | 0.8434 |
| education | Incomplete higher | 3,026 | 7.67% | 28.88% | 72.41% | 25.27% | +0.01158 | 0.0606 | 0.8204 |
| education | Lower secondary | 1,161 | 10.77% | 33.94% | 80.00% | 28.38% | −0.00933 | 0.0762 | 0.8376 |
| education | Secondary/special | 65,398 | 9.02% | 28.53% | 73.49% | 24.07% | −0.00096 | 0.0680 | 0.8248 |
| income_type | Commercial assoc. | 21,511 | 7.25% | 22.70% | 67.18% | 19.22% | +0.00180 | 0.0566 | 0.8308 |
| income_type | State servant | 6,321 | 5.43% | 15.96% | 64.72% | 13.16% | +0.00554 | 0.0425 | 0.8472 |
| income_type | Working | 47,790 | 9.73% | 31.38% | 76.46% | 26.52% | −0.00158 | 0.0724 | 0.8276 |

### Table A2 — Fairness Summary

| Attribute | DIR | 80% Rule | DPD | TPR max diff | FPR max diff | Mean Cal Bias |
|-----------|-----|----------|-----|--------------|--------------|---------------|
| gender | 0.612 | **FAIL** | 0.130 | 0.123 | 0.114 | 0.00081 |
| age_group | 0.164 | **FAIL** | 0.318 | 0.495 | 0.276 | 0.00092 |
| education | 0.405 | **FAIL** | 0.202 | 0.224 | 0.171 | 0.00583 |
| income_type | 0.509 | **FAIL** | 0.154 | 0.117 | 0.134 | 0.00297 |

### Table A3 — Statistical Significance (Gender)

| Test | Comparison | Statistic | p-value | Significant (α=0.01) |
|------|-----------|-----------|---------|----------------------|
| Mann-Whitney U | Score distribution: Male vs Female | — | < 0.001 | Yes |
| Chi-squared | Approval rate: Male vs Female | — | < 0.001 | Yes |

*Score and approval rate differences between genders are statistically significant. Significance confirms the effect is real, not sampling noise. Root-cause analysis (§5.4 and Figure 4) establishes the cause as differential base rates, not model error.*

### Figures

| Figure | File | Description |
|--------|------|-------------|
| Figure 1 | `results/compliance/plots/equalized_odds_gender_age.png` | TPR and FPR by gender and age group |
| Figure 2 | `results/compliance/plots/disparate_impact_ratio.png` | DIR by attribute (PASS/FAIL vs 80% rule) |
| Figure 3 | `results/compliance/plots/calibration_by_gender.png` | Reliability diagram by gender |
| Figure 4 | `results/compliance/plots/base_rate_vs_predicted_rate.png` | Base rate vs PPR — disparity diagnosis |
| Figure 5 | `results/compliance/plots/score_distribution_by_gender.png` | Score distribution by gender |
| Figure 6 | `results/compliance/plots/score_distribution_by_age.png` | Score distribution by age group |
| Figure 7 | `results/compliance/plots/fairness_heatmap_all_groups.png` | All-group metrics heatmap |

---

*Last updated: 2026-05-19 | Document owner: Shahul SHAIK | Next review: upon model change or annually*
