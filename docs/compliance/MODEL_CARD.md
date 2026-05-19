# Model Card
## Credit Scoring Model — Home Credit Default Risk

*Transparency documentation under EU AI Act Art 13 and Google Model Card format*

**Model card version**: 1.0  
**Date**: 2026-05-19  
**Author**: Shahul SHAIK, CISA-certified

---

## 1. Model Details

| Field | Value |
|-------|-------|
| **Name** | `credit_scoring_production_model` |
| **Version** | v3 (MLflow registry) |
| **Type** | LightGBM gradient boosting classifier |
| **Framework** | LightGBM 4.x, scikit-learn 1.8, Python 3.11 |
| **Training date** | 2026-05-18 |
| **Owner** | Shahul SHAIK |
| **Contact** | [github.com/shah-data-scientist](https://github.com/shah-data-scientist) |
| **License** | Educational / research — not for commercial deployment without review |
| **Repository** | `github.com/shah-data-scientist/Scoring_Model` |
| **Conformity assessment** | `docs/compliance/EU_AI_ACT_CONFORMITY_ASSESSMENT.md` |
| **EU AI Act risk class** | HIGH-RISK — Annex III, Point 5(b) |

---

## 2. Intended Use

### Primary Use
Predict the probability that a consumer credit applicant will default on a loan within 12 months. The score supports **human credit analysts** in approve/reject decisions.

### Intended Users
Credit risk analysts and loan officers who have completed model-explainability training (SHAP interpretation).

### Out-of-Scope Uses

> These uses are **prohibited** without a separate impact assessment:

- **Sole automated decision-making** without human review — violates EU AI Act Art 14 and GDPR Art 22
- Medical, employment, or insurance screening
- Scoring populations materially different from the Home Credit applicant pool (geography, income level, cultural context) without revalidation
- Real-time decisions at volumes that prevent human review of borderline cases

---

## 3. System Architecture and Input/Output Specification

### Inputs
189 numerical and binary features derived from:
- Loan application form (age, income, employment, family status, housing)
- Credit bureau history (aggregated repayment behaviour)
- Previous loan applications (prior defaults, approval rates)
- Instalment and credit card payment histories

Full feature list: `results/artifacts/feature_names.csv`  
Feature ordering: `models/feature_columns.pkl`

### Preprocessing
1. `SimpleImputer(strategy='median')` — `models/imputer.pkl`
2. `StandardScaler()` — `models/scaler.pkl`

Both were fit exclusively on the training split. Must be applied identically at inference.

### Outputs

| Output | Type | Range | Use |
|--------|------|-------|-----|
| Default probability | Float | [0, 1] | Risk quantification |
| Binary recommendation | {0, 1} | — | Flag at threshold=0.10 |

### Decision Threshold
**0.10** — selected by minimising business cost function `10×FN + FP`. Scores ≥ 0.10 are flagged for human review. Threshold is configurable by authorised risk management personnel.

---

## 4. Training Data

| Property | Value |
|----------|-------|
| Source | Home Credit Default Risk (Kaggle, 2018) |
| Training set size | 307,511 applications |
| Training split | 215,257 rows (70% stratified) |
| Target variable | `TARGET`: 1 = defaulted within 12 months |
| Positive class rate | 8.07% |
| Feature count | 189 (after engineering and selection) |
| Supplementary tables | Bureau, previous applications, POS/cash, credit card, instalments |

### Demographic Distribution (Training Set, approximate)

| Attribute | Distribution |
|-----------|-------------|
| Gender | ~34% Male, ~66% Female |
| Age | Range 21–69 years; median ~43 |
| Education | ~25% Higher education, ~71% Secondary/special, ~4% other |
| Income type | ~52% Working, ~23% Commercial associate, ~7% State servant, ~18% other |
| Default rate by gender | Male ~10.1%, Female ~7.0% |

### Known Data Limitations
- Historical data may embed past lending practices and structural inequalities
- The exact collection period and originating countries are not disclosed in the public dataset
- Occupation type was dropped (18 categories, high cardinality) — an important proxy variable is absent from the model

---

## 5. Evaluation Data

| Property | Value |
|----------|-------|
| Source | Same as training — held-out split |
| Validation set size | 92,254 rows (30% stratified) |
| Default rate | 8.07% |
| Split method | Stratified random (`RANDOM_STATE=42`) |
| Leakage prevention | Imputer and scaler fit on train split only, applied to validation |

The validation set was **never used** during any training or hyperparameter selection step.

---

## 6. Overall Performance Metrics

*Evaluated on validation set (n=92,254) at decision threshold=0.10*

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.8320** |
| Recall (TPR) | **71.0%** — 71% of true defaults are flagged |
| False Positive Rate | **21.3%** — 21% of good applicants are flagged |
| Brier Score | **0.0617** — well-calibrated probability outputs |
| Business cost (10×FN+FP) | **€0.43/client** |
| Baseline cost (approve-all) | €0.81/client |
| Cost reduction | **−47%** |
| Confusion matrix | TP=5,289 · FP=17,756 · FN=2,159 · TN=67,050 |

*Monitoring alert threshold: ROC-AUC < 0.75 triggers retraining.*

---

## 7. Performance by Demographic Group

*Source: `results/compliance/fairness_metrics_by_group.csv` — generated by NB06, 2026-05-19*

### 7.1 Gender

| Group | n | True DR | TPR | FPR | Pred Pos Rate | Cal Bias | Brier |
|-------|---|---------|-----|-----|---------------|----------|-------|
| Male | 31,426 | 10.14% | **78.0%** | **28.6%** | 33.6% | −0.00111 | 0.0746 |
| Female | 60,828 | 7.01% | **65.8%** | **17.1%** | 20.5% | +0.00051 | 0.0550 |

**Disparate Impact Ratio**: 0.612 — fails EEOC 80% rule  
**Assessment**: Disparity is proportional to base-rate difference. Calibration bias < 0.002.

### 7.2 Age Group

| Group | n | True DR | TPR | FPR | Pred Pos Rate |
|-------|---|---------|-----|-----|---------------|
| <35 | 25,386 | 10.79% | 80.1% | 33.0% | 38.0% |
| 35–50 | 35,776 | 8.15% | 70.9% | 20.4% | 24.5% |
| 50–65 | 28,660 | 5.96% | 58.6% | 12.9% | 15.6% |
| 65+ | 2,432 | 3.50% | 30.6% | 5.4% | 6.3% |

**Disparate Impact Ratio**: 0.164 — fails 80% rule  
**Assessment**: 3× base-rate difference across age groups drives the TPR range (0.495). All groups show ROC-AUC > 0.82, indicating consistent discriminative power.

### 7.3 Education Level

| Group | n | True DR | TPR | FPR | Pred Pos Rate |
|-------|---|---------|-----|-----|---------------|
| Higher education | 22,609 | 5.28% | 57.6% | 11.3% | 13.7% |
| Incomplete higher | 3,026 | 7.67% | 72.4% | 25.3% | 28.9% |
| Lower secondary | 1,161 | 10.77% | 80.0% | 28.4% | 33.9% |
| Secondary/special | 65,398 | 9.02% | 73.5% | 24.1% | 28.5% |

**Disparate Impact Ratio**: 0.405 — fails 80% rule  
**Assessment**: DR ranges from 5.28% (Higher education) to 10.77% (Lower secondary). Model reflects differential risk, not differential treatment.

### 7.4 Income Type

| Group | n | True DR | TPR | FPR | Pred Pos Rate |
|-------|---|---------|-----|-----|---------------|
| Commercial assoc. | 21,511 | 7.25% | 67.2% | 19.2% | 22.7% |
| State servant | 6,321 | 5.43% | 64.7% | 13.2% | 16.0% |
| Working | 47,790 | 9.73% | 76.5% | 26.5% | 31.4% |

**Disparate Impact Ratio**: 0.509 — fails 80% rule  
**Assessment**: Working-class applicants have ~4.3pp higher true default rate than State servants. Model tracks this difference accurately (calibration bias < 0.006).

---

## 8. Fairness Analysis

### 8.1 Core Finding

All four sensitive attributes fail the EEOC Disparate Impact Ratio (DIR ≥ 0.80). This is a consistent finding, explained by a single mechanism: **groups with higher true default rates receive higher predicted positive rates**.

This is confirmed by Figure 4 (`base_rate_vs_predicted_rate.png`): all demographic group points cluster near the y=x diagonal, meaning the model's predicted positive rate closely tracks the group's actual default rate. A model that achieved DIR ≥ 0.80 across all groups would necessarily diverge from the diagonal — it would either over-flag good-risk groups or under-flag high-risk groups, worsening calibration.

### 8.2 Calibration as the Primary Fairness Criterion

In credit scoring, **calibration parity** is the most actionable fairness criterion: a well-calibrated model assigns equal scores to equal-risk applicants regardless of demographic group. Calibration bias across all groups is < 0.012 (mean < 0.003), confirming the model meets this criterion.

The alternative criterion — **demographic parity** (equal approval rates regardless of risk) — would require approving proportionally more high-risk applicants from disadvantaged groups, increasing expected losses and not improving individual fairness.

### 8.3 Residual Concern: Historical Base-Rate Bias

The model accurately reflects **historical** default rates. If those rates themselves embed structural inequalities (e.g., lower-income groups historically faced discriminatory lending that damaged creditworthiness), the model will perpetuate those inequalities. This is a policy question beyond the model's scope — it requires institutional intervention, not model adjustment.

---

## 9. Ethical Considerations

### Gender and Age as Input Features
Gender (`CODE_GENDER_M`) and age (`AGE_YEARS`, `DAYS_BIRTH`) are present as model inputs. Their inclusion is:
- Legally permissible under EU credit regulation provided human oversight prevents discriminatory individual outcomes
- Technically justified by their predictive value (EXT_SOURCE features dominate importance, but age/gender contribute)
- Monitored through this model card and the quarterly DIR monitoring alert (trigger: DIR < 0.50)

### Calibration vs Demographic Parity Trade-off
The auditor's position, consistent with current academic consensus (Chouldechova 2017; Corbett-Davies et al. 2018), is that simultaneous satisfaction of calibration and demographic parity is mathematically impossible when base rates differ across groups. **Calibration is prioritised** because it ensures equal treatment of equal-risk individuals — the closest approximation to individual fairness achievable.

### Feedback Loop Risk
If the model is deployed and its rejections prevent certain groups from building credit history, future models trained on this data will observe even lower approval rates for those groups. This feedback loop must be monitored by tracking approval rates by demographic group over time.

---

## 10. Caveats and Recommendations

| Caveat | Recommendation |
|--------|---------------|
| No temporal validation | Validate on a time-held-out sample before production deployment |
| 21% FPR at t=0.10 | Size review team to handle ~21K flagged per 100K applications |
| Historical data only | Retrain quarterly or on material population shift |
| No OHE pipeline in inference | Implement proper sklearn OHE pipeline (pd.get_dummies fails on unseen categories) |
| `OCCUPATION_TYPE` excluded | High-cardinality; consider target-encoding for future versions |
| Kaggle dataset only | Revalidate on institution's own application data before live use |

---

## 11. Contact and Responsible Party

| Role | Detail |
|------|--------|
| **Model owner** | Shahul SHAIK |
| **Contact** | [github.com/shah-data-scientist](https://github.com/shah-data-scientist) |
| **CISA certification** | ISACA CISA |
| **Conformity assessment** | `docs/compliance/EU_AI_ACT_CONFORMITY_ASSESSMENT.md` |
| **Audit notebook** | `notebooks/06_eu_ai_act_compliance.ipynb` |
| **Audit artefacts** | `results/compliance/` |
| **Monitoring plan** | `docs/MODEL_MONITORING.md` |
| **Next review** | On model change, or annually by 2027-05-19 |

---

*This model card was produced as part of the EU AI Act conformity assessment for the Home Credit credit scoring system. It satisfies the transparency and information provision requirements of Art 13 of Regulation (EU) 2024/1689.*
