# Feature Engineering & Sampling Strategy Experiment Design

## 🎯 Objective

Systematically compare the impact of different feature engineering techniques and sampling strategies on model performance to identify the optimal configuration for credit scoring.

---

## 📊 Experimental Matrix

### **Dimensions**

1. **Feature Engineering Strategies** (4 configurations)
2. **Sampling Strategies** (3 methods)
3. **Total Experiments**: 4 × 3 = **12 configurations**

---

## 🔬 Feature Engineering Strategies

### **1. Baseline (No Added Features)**
- **Description**: Use only the original 189 features from comprehensive data integration
- **Features**: Raw aggregated features from all 8 data sources
- **Purpose**: Establish baseline performance
- **Tag**: `feature_strategy=baseline`

### **2. Domain Features Only**
- **Description**: Add domain-knowledge-driven engineered features
- **Added Features**:
  - `AGE_YEARS` = -DAYS_BIRTH / 365
  - `EMPLOYMENT_YEARS` = -DAYS_EMPLOYED / 365
  - `INCOME_PER_PERSON` = AMT_INCOME_TOTAL / CNT_FAM_MEMBERS
  - `DEBT_TO_INCOME_RATIO` = AMT_CREDIT / AMT_INCOME_TOTAL
  - `CREDIT_TO_GOODS_RATIO` = AMT_CREDIT / AMT_GOODS_PRICE
  - `ANNUITY_TO_INCOME_RATIO` = AMT_ANNUITY / AMT_INCOME_TOTAL
  - `CREDIT_UTILIZATION` = AMT_CREDIT / AMT_GOODS_PRICE
  - `HAS_CHILDREN` = (CNT_CHILDREN > 0)
  - `CHILDREN_RATIO` = CNT_CHILDREN / CNT_FAM_MEMBERS
  - `TOTAL_DOCUMENTS_PROVIDED` = sum of FLAG_DOCUMENT_*
  - `EXT_SOURCE_MEAN/MAX/MIN` = aggregations of external scores
  - `REGION_RATING_COMBINED` = average of region ratings
- **Expected Features**: ~189 + 15 = **~204 features**
- **Purpose**: Test impact of business logic features
- **Tag**: `feature_strategy=domain`

### **3. Polynomial Features Only**
- **Description**: Add polynomial (interaction & squared) features for key numeric variables
- **Selected Features for Polynomial Expansion** (degree=2):
  - Financial: `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE`
  - External Scores: `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`
  - Time: `DAYS_BIRTH`, `DAYS_EMPLOYED`, `DAYS_ID_PUBLISH`
  - Credit Bureau: Top 5 bureau features by importance
- **Polynomial Degree**: 2 (includes interactions and squares)
- **Expected New Features**: ~15 original × (15+1)/2 = **~120 polynomial features**
- **Total Features**: ~189 + 120 = **~309 features**
- **Purpose**: Capture non-linear relationships and interactions
- **Tag**: `feature_strategy=polynomial`

### **4. Domain + Polynomial Features**
- **Description**: Combine both domain and polynomial feature engineering
- **Expected Features**: ~189 + 15 (domain) + 120 (poly) = **~324 features**
- **Purpose**: Test if combining strategies yields best results
- **Tag**: `feature_strategy=combined`

---

## 🎲 Sampling Strategies

### **1. Balanced (Class Weight)**
- **Method**: Use `class_weight='balanced'` parameter
- **Pros**: Fast, no data modification, sklearn native
- **Cons**: Doesn't create new samples
- **Tag**: `sampling_strategy=balanced`

### **2. SMOTE (Oversampling)**
- **Method**: Synthetic Minority Over-sampling Technique
- **Implementation**: `imblearn.over_sampling.SMOTE(random_state=42)`
- **Pros**: Creates synthetic minority samples, proven effective
- **Cons**: Computationally expensive, risk of overfitting
- **Tag**: `sampling_strategy=smote`

### **3. Random Under-sampling**
- **Method**: Randomly remove majority class samples
- **Implementation**: `imblearn.under_sampling.RandomUnderSampler(random_state=42)`
- **Pros**: Fast, reduces dataset size
- **Cons**: Loses potentially useful data
- **Tag**: `sampling_strategy=undersample`

---

## 📋 Experiment Matrix

| Exp # | Feature Strategy | Sampling Strategy | Expected Features | MLflow Run Name |
|-------|-----------------|-------------------|-------------------|-----------------|
| 1     | Baseline        | Balanced          | ~189              | lgbm_v1_baseline_balanced |
| 2     | Baseline        | SMOTE             | ~189              | lgbm_v1_baseline_smote |
| 3     | Baseline        | Undersample       | ~189              | lgbm_v1_baseline_undersample |
| 4     | Domain          | Balanced          | ~204              | lgbm_v1_domain_balanced |
| 5     | Domain          | SMOTE             | ~204              | lgbm_v1_domain_smote |
| 6     | Domain          | Undersample       | ~204              | lgbm_v1_domain_undersample |
| 7     | Polynomial      | Balanced          | ~309              | lgbm_v1_poly_balanced |
| 8     | Polynomial      | SMOTE             | ~309              | lgbm_v1_poly_smote |
| 9     | Polynomial      | Undersample       | ~309              | lgbm_v1_poly_undersample |
| 10    | Combined        | Balanced          | ~324              | lgbm_v1_combined_balanced |
| 11    | Combined        | SMOTE             | ~324              | lgbm_v1_combined_smote |
| 12    | Combined        | Undersample       | ~324              | lgbm_v1_combined_undersample |

---

## 🎯 Model Configuration

### **Base Model**: LightGBM (best performer from baseline)

**Parameters**:
```python
{
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
    # class_weight='balanced' only for balanced strategy
}
```

### **Evaluation Metrics** (tracked in MLflow):
- `roc_auc` - Primary metric
- `pr_auc` - Important for imbalanced data
- `f1_score`
- `precision`
- `recall`
- `accuracy`
- `false_positive_rate`
- `false_negative_rate`
- `training_time_seconds`

### **Additional Tracking**:
- `feature_count` - Number of features used
- `feature_list` - CSV file of all feature names
- `training_samples` - Number of samples after sampling
- `feature_strategy` - Tag
- `sampling_strategy` - Tag

---

## 🚀 Execution Plan

### **Phase 1: Implementation** (1-2 hours)
1. Create polynomial feature engineering module
2. Create sampling strategy module
3. Create experiment execution script
4. Update MLflow tracking to include feature lists

### **Phase 2: Execution** (2-3 hours)
1. Run all 12 experiments sequentially
2. Track all metrics and artifacts in MLflow
3. Save feature lists for each configuration
4. Log training times and sample counts

### **Phase 3: Analysis** (30 min)
1. Compare ROC-AUC across all configurations
2. Analyze feature importance for best model
3. Create comparison visualizations
4. Identify best configuration

### **Phase 4: Optimization** (1 hour)
1. Hyperparameter tuning for best configuration
2. Final model training
3. Performance report

---

## 📊 Expected Outcomes

### **Hypotheses**:

**H1: Feature Engineering Impact**
- Domain features will improve performance (+1-3% ROC-AUC)
- Polynomial features will capture non-linearities (+2-4% ROC-AUC)
- Combined features may have diminishing returns due to redundancy

**H2: Sampling Strategy Impact**
- SMOTE will improve recall but may reduce precision
- Undersampling will be fastest but may lose information
- Balanced class weight will provide good baseline

**H3: Optimal Configuration**
- Expected best: Domain + Polynomial features with SMOTE or Balanced
- Expected ROC-AUC improvement: +3-7% over current baseline

---

## 🎓 Success Criteria

**Minimum Success**:
- Complete all 12 experiments
- Identify clear winner (>2% improvement)
- Understand feature/sampling trade-offs

**Ideal Success**:
- ROC-AUC > 0.80 (current: 0.7900)
- Clear insights on feature importance
- Actionable recommendations for production

---

## 📁 Deliverables

1. **Code**:
   - `src/polynomial_features.py` - Polynomial feature engineering
   - `src/sampling_strategies.py` - Sampling methods
   - `scripts/run_experiments.py` - Experiment execution

2. **MLflow Experiments**:
   - New experiment: `credit_scoring_feature_engineering`
   - 12 runs with complete tracking

3. **Analysis**:
   - Comparison DataFrame (CSV)
   - Visualization plots
   - Best model identification

4. **Documentation**:
   - Experiment results summary
   - Feature importance analysis
   - Recommendations

---

## ⚠️ Considerations

### **Computational Cost**:
- 12 experiments × ~5 min each = **~60 minutes**
- SMOTE experiments will be slower
- Polynomial features increase memory usage

### **Memory Management**:
- Monitor memory during polynomial feature creation
- May need to reduce polynomial feature selection
- Consider using sparse matrices if needed

### **Validation**:
- Use same train/validation split for all experiments
- Ensure reproducibility with random seeds
- Cross-validation for final best model only

---

**Status**: Ready for implementation
**Priority**: High
**Timeline**: 4-6 hours total
