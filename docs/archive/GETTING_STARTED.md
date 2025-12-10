# 🚀 Getting Started Guide

## Welcome, Data Science Learner!

This guide will help you get started with the Credit Scoring Model project. Follow these steps in order to complete the project successfully.

---

## 📚 Prerequisites

Before starting, make sure you understand:

1. **Basic Python Programming**
   - Variables, functions, loops
   - Lists, dictionaries, NumPy arrays
   - Reading and understanding code

2. **Basic Statistics & ML Concepts**
   - Mean, median, standard deviation
   - What is classification?
   - Training vs testing data
   - *Don't worry if you're not an expert - the notebooks explain everything!*

3. **Basic Command Line**
   - How to run commands in terminal
   - How to navigate directories

---

## ⚙️ Environment Setup

### Step 1: Install Dependencies

```bash
# Make sure you're in the project directory
cd Scoring_Model

# Install all dependencies using Poetry
poetry install

# Activate the virtual environment
poetry shell
```

### Step 2: Verify Installation

```bash
# Check Python version (should be 3.12+)
python --version

# Check if Jupyter is installed
jupyter --version

# Check if MLflow is installed
mlflow --version
```

---

## 📂 Repository Overview

Here's what you have:

```
Scoring_Model/
│
├── README.md                    # Comprehensive project documentation (READ THIS FIRST!)
├── GETTING_STARTED.md           # This file
│
├── data/                        # Your datasets
│   ├── application_train.csv   # Training data (307K samples)
│   ├── application_test.csv    # Test data (48K samples)
│   └── *.csv                   # Additional tables
│
├── notebooks/                   # Jupyter notebooks (YOUR MAIN WORK)
│   ├── 01_eda.ipynb            # Start here! Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb  # Create new features
│   ├── 03_baseline_models.ipynb      # Train first models
│   ├── 04_hyperparameter_optimization.ipynb  # Tune models
│   └── 05_model_interpretation.ipynb  # Explain with SHAP
│
├── src/                         # Reusable Python modules
│   ├── __init__.py
│   ├── data_preprocessing.py    # Data loading and cleaning utilities
│   ├── evaluation.py            # Model evaluation functions
│   └── ... (more modules)
│
├── scripts/                     # Automation scripts
│   ├── create_notebooks.py      # Generate notebooks programmatically
│   └── mlflow_example.py        # MLflow basic example
│
├── models/                      # Saved models will go here
├── mlruns/                      # MLflow experiment tracking data
│
└── pyproject.toml               # Dependencies configuration
```

---

## 🎯 Step-by-Step Workflow

### Phase 1: Data Exploration (Week 1)

**Goal:** Understand your data thoroughly

**Tasks:**
1. ✅ **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```
   This opens a browser window with your notebooks.

2. ✅ **Open and Run: `01_eda.ipynb`**
   - Read all the markdown cells (explanations)
   - Run each code cell in order (Shift + Enter)
   - Take notes on your observations
   - Complete all exercises

**What You'll Learn:**
- How to load and inspect data
- Identifying missing values
- Understanding distributions
- Analyzing class imbalance
- Correlation analysis

**Deliverable:**
- Completed EDA notebook with all cells executed
- Notes on key findings
- List of data quality issues to address

---

### Phase 2: Feature Engineering (Week 1-2)

**Goal:** Create useful features for modeling

**Tasks:**
1. ✅ **Open: `02_feature_engineering.ipynb`**
   - Handle missing values systematically
   - Create domain-based features (debt ratios, etc.)
   - Encode categorical variables
   - Scale numerical features

**What You'll Learn:**
- Different imputation strategies
- Creating ratio and aggregation features
- One-hot encoding vs label encoding
- Standardization vs normalization

**Deliverable:**
- Clean, processed dataset ready for modeling
- At least 5 new engineered features
- Documentation of your feature engineering choices

---

### Phase 3: Baseline Models (Week 2)

**Goal:** Train your first models with MLflow tracking

**Tasks:**
1. ✅ **Start MLflow UI** (in a separate terminal):
   ```bash
   mlflow ui
   ```
   Open browser to: `http://localhost:5000`

2. ✅ **Open: `03_baseline_models.ipynb`**
   - Train Logistic Regression (baseline)
   - Train Random Forest
   - Train XGBoost
   - Train LightGBM
   - Compare all models in MLflow UI

**What You'll Learn:**
- Setting up MLflow experiments
- Training different model types
- Logging parameters and metrics
- Comparing models visually
- Understanding trade-offs between models

**Deliverable:**
- At least 3 different models trained
- All runs logged in MLflow
- Comparison of models using ROC-AUC and PR-AUC
- Initial model selection

---

### Phase 4: Hyperparameter Optimization (Week 3)

**Goal:** Improve your best model through systematic tuning

**Tasks:**
1. ✅ **Open: `04_hyperparameter_optimization.ipynb`**
   - Choose your best baseline model
   - Define hyperparameter search space
   - Run GridSearchCV or RandomizedSearchCV
   - Use Stratified K-Fold cross-validation
   - Log all experiments to MLflow

**What You'll Learn:**
- What hyperparameters are
- Grid search vs random search
- Cross-validation strategies
- Avoiding overfitting
- Systematic optimization process

**Deliverable:**
- Optimized model with best hyperparameters
- Multiple optimization runs logged
- Performance improvement documentation
- Final model saved

---

### Phase 5: Model Interpretation (Week 3-4)

**Goal:** Understand WHY your model makes predictions

**Tasks:**
1. ✅ **Open: `05_model_interpretation.ipynb`**
   - Analyze global feature importance
   - Generate SHAP summary plots
   - Explain individual predictions
   - Identify potential biases
   - Create business insights

**What You'll Learn:**
- Feature importance interpretation
- SHAP values and theory
- Global vs local explanations
- Making ML models transparent
- Communicating results to non-technical stakeholders

**Deliverable:**
- Feature importance analysis
- SHAP visualizations
- Interpretation of top features
- Business recommendations
- Complete project documentation

---

## 🔧 Using the Utility Modules

The `src/` folder contains reusable functions. Here's how to use them:

### In Any Notebook:

```python
# Import utilities
from src.data_preprocessing import load_data, analyze_missing_values
from src.evaluation import evaluate_model, plot_roc_curve

# Load data
train_df, test_df = load_data()

# Analyze missing values
missing_summary = analyze_missing_values(train_df)

# Evaluate a trained model
metrics = evaluate_model(y_true, y_pred, y_pred_proba, "My Model")

# Plot ROC curve
plot_roc_curve(y_true, y_pred_proba, "My Model")
```

**Benefits:**
- ✅ Less code duplication
- ✅ Consistent evaluation across notebooks
- ✅ Professional code organization
- ✅ Easy to test and debug

---

## 💡 Key Concepts Explained

### 1. Class Imbalance

**Problem:** Only ~8% of loans default. If you always predict "no default", you get 92% accuracy!

**Why It's Bad:** The minority class (defaults) is what we care about most!

**Solutions:**
- ✅ Use ROC-AUC, PR-AUC, F1-Score (NOT accuracy!)
- ✅ Use stratified sampling
- ✅ Set `class_weight='balanced'` in models
- ✅ Consider SMOTE or undersampling

### 2. MLflow Experiment Tracking

**What It Does:**
- Logs all your experiments automatically
- Compares model performance visually
- Saves models and artifacts
- Makes your work reproducible

**How to Use:**
```python
import mlflow

# Set experiment name
mlflow.set_experiment("my_experiment")

# Start a run
with mlflow.start_run(run_name="random_forest_v1"):
    # Log parameters
    mlflow.log_param("n_estimators", 100)

    # Train model
    model.fit(X_train, y_train)

    # Log metrics
    mlflow.log_metric("roc_auc", roc_auc)

    # Log model
    mlflow.sklearn.log_model(model, "model")
```

### 3. Train-Validation-Test Split

**Why Three Sets?**

1. **Training (60%):** Teach the model
2. **Validation (20%):** Tune hyperparameters, select models
3. **Test (20%):** Final evaluation only (unbiased)

**Golden Rule:** NEVER touch test set until the very end!

### 4. Feature Engineering

**What:** Creating new features from existing data

**Examples:**
- `credit_to_income_ratio = credit / income`
- `age_years = -days_birth / 365`
- `is_employed = days_employed > 0`

**Why:** Better features → Better models!

---

## 🚨 Common Pitfalls to Avoid

### ❌ DON'T:
1. **Use accuracy on imbalanced data** → Use ROC-AUC or F1-Score
2. **Tune on test set** → Use validation set for tuning
3. **Forget stratified sampling** → Always use `stratify=y`
4. **Ignore missing value patterns** → Create "is_missing" indicators
5. **Remove outliers blindly** → Investigate first!
6. **Over-complicate features** → Start simple
7. **Skip cross-validation** → Use StratifiedKFold
8. **Forget to set random_state** → Set to 42 for reproducibility

### ✅ DO:
1. **Start with simple baseline** → Logistic Regression first
2. **Use appropriate metrics** → ROC-AUC, PR-AUC, F1
3. **Log all experiments** → Use MLflow consistently
4. **Document your decisions** → Add markdown cells explaining choices
5. **Validate assumptions** → Check for data leakage
6. **Consider business context** → FP vs FN costs
7. **Interpret your models** → Use SHAP for explanations
8. **Keep code clean** → Use functions from src/

---

## 🐛 Troubleshooting

### Issue: Jupyter won't start
```bash
# Make sure virtual environment is activated
poetry shell

# Reinstall Jupyter
poetry add jupyter --group dev

# Try again
jupyter notebook
```

### Issue: MLflow UI won't show experiments
```bash
# Make sure you're in the project directory
cd Scoring_Model

# Start MLflow UI
mlflow ui
```

### Issue: Import errors in notebooks
```python
# Make sure src is in Python path
import sys
sys.path.append('../')

# Then import
from src.data_preprocessing import load_data
```

### Issue: Out of memory
```python
# Use sampling for development
train_df = train_df.sample(n=10000, random_state=42)
```

---

## 📊 Evaluation Checklist

Before submitting, ensure you have:

- [ ] ✅ Complete EDA with visualizations and insights
- [ ] ✅ At least 5 engineered features with justification
- [ ] ✅ Minimum 3 different models trained
- [ ] ✅ All experiments logged in MLflow (visible in UI)
- [ ] ✅ Hyperparameter optimization performed
- [ ] ✅ Best model identified with clear justification
- [ ] ✅ SHAP analysis completed
- [ ] ✅ Confusion matrix and ROC/PR curves for best model
- [ ] ✅ Documentation of findings and recommendations
- [ ] ✅ Reproducible code (someone else can run it)
- [ ] ✅ Clear conclusions and business insights

---

## 🎓 Learning Resources

### When You're Stuck:

1. **Read the README.md** - Most questions are answered there
2. **Check notebook markdown cells** - Explanations are inline
3. **Look at utility module docstrings** - Examples included
4. **Search Kaggle notebooks** - Home Credit competition
5. **Read official docs:**
   - [Scikit-learn](https://scikit-learn.org/stable/user_guide.html)
   - [MLflow](https://mlflow.org/docs/latest/index.html)
   - [SHAP](https://shap.readthedocs.io/)

### Recommended Learning Path:

1. **Week 1-2:** Focus on understanding the data (EDA + Feature Engineering)
2. **Week 2-3:** Model training and MLflow (Baseline + Optimization)
3. **Week 3-4:** Interpretation and documentation (SHAP + Final Report)

---

## 🏆 Success Tips

1. **Work Incrementally:** Complete one notebook before moving to next
2. **Take Notes:** Document your observations and decisions
3. **Experiment:** Try different approaches and compare
4. **Ask Why:** Don't just run code, understand what it does
5. **Stay Organized:** Use MLflow to track everything
6. **Be Patient:** Machine learning takes time to learn!
7. **Have Fun:** This is real-world data science!

---

## 📞 Need Help?

If you have questions:

1. Review the documentation files:
   - [README.md](README.md) - Comprehensive project guide
   - [GETTING_STARTED.md](GETTING_STARTED.md) - This file

2. Check the notebook markdown cells for explanations

3. Look at the docstrings in `src/` modules

4. Review MLflow UI for experiment comparisons

---

## 🎉 Final Words

**Remember:** The goal is not just to build a model, but to:
- ✅ Understand the entire ML workflow
- ✅ Make informed, justified decisions
- ✅ Communicate findings effectively
- ✅ Build reproducible, professional work

**You've got this! Happy learning! 🚀**

---

**Author:** Shahul SHAIK
**Project:** Credit Scoring Model - MLOps Educational Project
**Date:** December 2025
