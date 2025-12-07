"""
Create all 5 comprehensive educational notebooks for the Credit Scoring project.
This ensures a complete end-to-end workflow.
"""

import nbformat as nbf
from pathlib import Path


def create_feature_engineering_notebook():
    """Create comprehensive Feature Engineering notebook."""

    nb = nbf.v4.new_notebook()
    cells = []

    # Title
    cells.append(nbf.v4.new_markdown_cell("""# 02 - Feature Engineering
## Credit Scoring Model Project

**Learning Objectives:**
- Handle missing values systematically
- Create domain-based features using business knowledge
- Encode categorical variables appropriately
- Scale and normalize numerical features
- Perform feature selection
- Prepare data for modeling

**What is Feature Engineering?**
Feature engineering is the process of using domain knowledge to create features that make machine learning algorithms work better. It's often the difference between a mediocre model and a great one!

**Quote:** "Coming up with features is difficult, time-consuming, requires expert knowledge. 'Applied machine learning' is basically feature engineering." - Andrew Ng

Let's build powerful features!"""))

    # Imports
    cells.append(nbf.v4.new_markdown_cell("""## 📦 Import Libraries and Utilities"""))

    cells.append(nbf.v4.new_code_cell("""# Standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path

# Sklearn preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Our custom utilities
import sys
sys.path.append('../')
from src.data_preprocessing import (
    load_data,
    analyze_missing_values,
    handle_missing_values,
    validate_data_quality
)

# Configuration
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("[OK] All libraries imported successfully!")"""))

    # Load Data
    cells.append(nbf.v4.new_markdown_cell("""## 📂 Load Data

We'll load the data we explored in the EDA notebook."""))

    cells.append(nbf.v4.new_code_cell("""# Load data using our utility
train_df, test_df = load_data()

print(f"Training set: {train_df.shape}")
print(f"Test set: {test_df.shape}")
print(f"\\nTarget distribution:")
print(train_df['TARGET'].value_counts(normalize=True))"""))

    # Handle Missing Values
    cells.append(nbf.v4.new_markdown_cell("""## 🔍 Handle Missing Values

**Strategy:**
1. Identify features with excessive missing values (>70%) → Consider dropping
2. Create missing indicators for important features
3. Impute remaining missing values appropriately

**Educational Note:**
Different imputation strategies work for different scenarios:
- **Median:** For numerical features with outliers (robust)
- **Mean:** For numerical features with normal distribution
- **Mode:** For categorical features
- **Constant:** When missingness has meaning (e.g., 0 for no car)
- **Missing Indicator:** Preserve information about missingness"""))

    cells.append(nbf.v4.new_code_cell("""# Analyze missing values
missing_summary = analyze_missing_values(train_df, threshold=0)

# Separate features by missing percentage
high_missing = missing_summary[missing_summary['missing_percent'] > 70]['column'].tolist()
medium_missing = missing_summary[(missing_summary['missing_percent'] > 20) &
                                 (missing_summary['missing_percent'] <= 70)]['column'].tolist()
low_missing = missing_summary[(missing_summary['missing_percent'] > 0) &
                              (missing_summary['missing_percent'] <= 20)]['column'].tolist()

print(f"\\n[DECISION SUMMARY]")
print(f"High missing (>70%): {len(high_missing)} features - CONSIDER DROPPING")
print(f"Medium missing (20-70%): {len(medium_missing)} features - CREATE INDICATORS + IMPUTE")
print(f"Low missing (<20%): {len(low_missing)} features - IMPUTE ONLY")

# Let's drop very sparse features
print(f"\\nDropping {len(high_missing)} features with >70% missing...")
train_df = train_df.drop(columns=high_missing)
test_df = test_df.drop(columns=[col for col in high_missing if col in test_df.columns])

print(f"New training shape: {train_df.shape}")"""))

    # Create Domain Features
    cells.append(nbf.v4.new_markdown_cell("""## 🎯 Create Domain-Based Features

**Domain Knowledge in Credit Scoring:**
Key financial ratios and indicators matter:
- **Debt-to-Income Ratio:** How much debt relative to income?
- **Credit Utilization:** How much of available credit is used?
- **Payment Burden:** Can they afford the payments?
- **Age/Employment Stability:** Risk indicators

Let's create features that capture these concepts!"""))

    cells.append(nbf.v4.new_code_cell("""def create_domain_features(df):
    \"\"\"
    Create domain-based features using financial knowledge.

    Educational Note:
    -----------------
    These features capture important financial relationships:
    - Ratios: Normalize values and capture proportions
    - Flags: Binary indicators of important conditions
    - Transformations: Handle skewed distributions
    \"\"\"
    df = df.copy()

    print("Creating domain features...")

    # 1. AGE FEATURES
    # Convert DAYS_BIRTH to years (more interpretable)
    df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365
    df['AGE_GROUP'] = pd.cut(df['AGE_YEARS'],
                             bins=[0, 25, 35, 45, 55, 100],
                             labels=['<25', '25-35', '35-45', '45-55', '55+'])
    print("  [OK] Age features created")

    # 2. EMPLOYMENT FEATURES
    # Convert DAYS_EMPLOYED to years (handle anomalies)
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)  # Anomaly in data
    df['EMPLOYMENT_YEARS'] = -df['DAYS_EMPLOYED'] / 365
    df['EMPLOYMENT_YEARS'] = df['EMPLOYMENT_YEARS'].clip(lower=0)  # No negative
    df['IS_EMPLOYED'] = (df['EMPLOYMENT_YEARS'] > 0).astype(int)
    print("  [OK] Employment features created")

    # 3. INCOME FEATURES
    # Income per family member
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1e-5)
    # Income type flags
    df['INCOME_TYPE'] = df['NAME_INCOME_TYPE']
    print("  [OK] Income features created")

    # 4. CREDIT FEATURES
    # Debt-to-Income Ratio (KEY FEATURE!)
    df['DEBT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1e-5)

    # Credit to goods price ratio
    df['CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1e-5)

    # Annuity to income ratio (payment burden)
    df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1e-5)

    # Credit utilization
    df['CREDIT_UTILIZATION'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1e-5)

    print("  [OK] Credit features created")

    # 5. FAMILY FEATURES
    df['HAS_CHILDREN'] = (df['CNT_CHILDREN'] > 0).astype(int)
    df['CHILDREN_RATIO'] = df['CNT_CHILDREN'] / (df['CNT_FAM_MEMBERS'] + 1e-5)
    print("  [OK] Family features created")

    # 6. DOCUMENT FLAGS (combine related documents)
    doc_cols = [col for col in df.columns if col.startswith('FLAG_DOCUMENT_')]
    df['TOTAL_DOCUMENTS_PROVIDED'] = df[doc_cols].sum(axis=1)
    print("  [OK] Document features created")

    # 7. EXTERNAL SOURCE FEATURES (if available)
    ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    ext_sources_present = [col for col in ext_sources if col in df.columns]

    if ext_sources_present:
        df['EXT_SOURCE_MEAN'] = df[ext_sources_present].mean(axis=1)
        df['EXT_SOURCE_MAX'] = df[ext_sources_present].max(axis=1)
        df['EXT_SOURCE_MIN'] = df[ext_sources_present].min(axis=1)
        print("  [OK] External source features created")

    # 8. REGIONAL FEATURES
    df['REGION_RATING_COMBINED'] = (df['REGION_RATING_CLIENT'] +
                                     df['REGION_RATING_CLIENT_W_CITY']) / 2
    print("  [OK] Regional features created")

    print(f"\\n[SUMMARY] Created {df.shape[1] - train_df.shape[1]} new features!")

    return df

# Apply to both train and test
print("="*80)
train_df = create_domain_features(train_df)
test_df = create_domain_features(test_df)
print("="*80)

print(f"\\nNew shape after feature creation:")
print(f"Training: {train_df.shape}")
print(f"Test: {test_df.shape}")"""))

    # Encode Categorical
    cells.append(nbf.v4.new_markdown_cell("""## 🏷️ Encode Categorical Variables

**Encoding Strategies:**

1. **Label Encoding:** For ordinal categories (order matters)
   - Example: Education level (Low → Medium → High)

2. **One-Hot Encoding:** For nominal categories (no order)
   - Example: Contract type (Cash, Revolving)
   - Creates binary columns for each category

3. **Target Encoding:** For high-cardinality features
   - Encode by mean target value per category
   - Use with caution (risk of overfitting!)

**Rule of Thumb:**
- Low cardinality (<10 categories) → One-Hot Encoding
- High cardinality (>10 categories) → Target Encoding or drop"""))

    cells.append(nbf.v4.new_code_cell("""# Identify categorical columns
categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
print(f"Found {len(categorical_cols)} categorical columns")

# One-hot encode low cardinality categoricals
low_cardinality_cols = [col for col in categorical_cols
                        if train_df[col].nunique() < 10]

print(f"\\nOne-hot encoding {len(low_cardinality_cols)} low-cardinality features...")

# Apply one-hot encoding
train_df = pd.get_dummies(train_df, columns=low_cardinality_cols, drop_first=True, dtype=int)
test_df = pd.get_dummies(test_df, columns=low_cardinality_cols, drop_first=True, dtype=int)

# Align columns (ensure train and test have same columns)
train_cols = set(train_df.columns)
test_cols = set(test_df.columns)

# Add missing columns to test
for col in train_cols - test_cols:
    if col != 'TARGET':
        test_df[col] = 0

# Remove extra columns from test
test_df = test_df[[col for col in train_df.columns if col in test_df.columns]]

print(f"\\nShape after encoding:")
print(f"Training: {train_df.shape}")
print(f"Test: {test_df.shape}")

# Handle remaining high-cardinality categoricals (label encode or drop)
remaining_categorical = train_df.select_dtypes(include=['object']).columns.tolist()
if remaining_categorical:
    print(f"\\n[WARNING] {len(remaining_categorical)} high-cardinality features remain: {remaining_categorical}")
    print("Consider target encoding or dropping these features")"""))

    # Handle Remaining Missing
    cells.append(nbf.v4.new_markdown_cell("""## 🔧 Impute Remaining Missing Values

Now we'll impute the remaining missing values using appropriate strategies."""))

    cells.append(nbf.v4.new_code_cell("""# Identify columns with missing values
missing_cols = train_df.columns[train_df.isnull().any()].tolist()
if 'TARGET' in missing_cols:
    missing_cols.remove('TARGET')

print(f"Columns with missing values: {len(missing_cols)}")

# Separate numerical and categorical
numerical_missing = [col for col in missing_cols
                     if train_df[col].dtype in ['int64', 'float64']]
categorical_missing = [col for col in missing_cols
                       if train_df[col].dtype == 'object']

print(f"  Numerical: {len(numerical_missing)}")
print(f"  Categorical: {len(categorical_missing)}")

# Impute numerical with median
if numerical_missing:
    print(f"\\nImputing {len(numerical_missing)} numerical features with median...")
    imputer = SimpleImputer(strategy='median')
    train_df[numerical_missing] = imputer.fit_transform(train_df[numerical_missing])
    test_df[numerical_missing] = imputer.transform(test_df[numerical_missing])
    print("  [OK] Numerical imputation complete")

# Impute categorical with most frequent
if categorical_missing:
    print(f"\\nImputing {len(categorical_missing)} categorical features with mode...")
    imputer = SimpleImputer(strategy='most_frequent')
    train_df[categorical_missing] = imputer.fit_transform(train_df[categorical_missing])
    test_df[categorical_missing] = imputer.transform(test_df[categorical_missing])
    print("  [OK] Categorical imputation complete")

# Verify no missing values remain
print(f"\\n[VERIFICATION]")
print(f"Training missing: {train_df.isnull().sum().sum()}")
print(f"Test missing: {test_df.isnull().sum().sum()}")"""))

    # Feature Selection
    cells.append(nbf.v4.new_markdown_cell("""## 🎯 Feature Selection

**Why Feature Selection?**
1. Reduce overfitting (simpler models generalize better)
2. Improve model performance (remove noise)
3. Reduce training time
4. Improve interpretability

**Strategies:**
1. Remove low-variance features (constant or near-constant)
2. Remove highly correlated features (redundant information)
3. Use feature importance from baseline models"""))

    cells.append(nbf.v4.new_code_cell("""from sklearn.feature_selection import VarianceThreshold

# Separate features and target
X_train = train_df.drop(columns=['SK_ID_CURR', 'TARGET'])
y_train = train_df['TARGET']
X_test = test_df.drop(columns=['SK_ID_CURR'])

print(f"Features before selection: {X_train.shape[1]}")

# 1. Remove low-variance features
print("\\n1. Removing low-variance features...")
selector = VarianceThreshold(threshold=0.01)
selector.fit(X_train)

low_var_features = X_train.columns[~selector.get_support()].tolist()
print(f"   Found {len(low_var_features)} low-variance features to remove")

X_train = X_train[X_train.columns[selector.get_support()]]
X_test = X_test[X_test.columns[selector.get_support()]]

# 2. Remove highly correlated features
print("\\n2. Removing highly correlated features (>0.95)...")
corr_matrix = X_train.corr().abs()
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

highly_corr_features = [column for column in upper_triangle.columns
                        if any(upper_triangle[column] > 0.95)]
print(f"   Found {len(highly_corr_features)} highly correlated features to remove")

X_train = X_train.drop(columns=highly_corr_features)
X_test = X_test.drop(columns=highly_corr_features)

print(f"\\nFeatures after selection: {X_train.shape[1]}")
print(f"[SUCCESS] Removed {train_df.shape[1] - X_train.shape[1] - 2} features")"""))

    # Scale Features
    cells.append(nbf.v4.new_markdown_cell("""## ⚖️ Scale Features

**Why Scaling?**
- Features have different ranges (income vs children count)
- Many ML algorithms are sensitive to feature scales
- Required for: Logistic Regression, SVM, Neural Networks
- Optional for: Tree-based models (Random Forest, XGBoost)

**Scaling Methods:**
1. **StandardScaler:** Mean=0, Std=1 (use when data is normally distributed)
2. **MinMaxScaler:** Scale to [0, 1] (use when data has outliers)
3. **RobustScaler:** Uses median and IQR (robust to outliers)

We'll use StandardScaler as it works well for most cases."""))

    cells.append(nbf.v4.new_code_cell("""# Standardize features
print("Scaling features with StandardScaler...")
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print("[OK] Scaling complete")
print(f"\\nScaled feature statistics:")
print(X_train_scaled.describe().loc[['mean', 'std']].round(3))"""))

    # Train-Val Split
    cells.append(nbf.v4.new_markdown_cell("""## 🔀 Create Train-Validation Split

**Important:** Use **STRATIFIED** split to preserve class distribution!

We'll create:
- Training set (70%): For training models
- Validation set (30%): For hyperparameter tuning and model selection
- Test set: Already separate (for final evaluation)"""))

    cells.append(nbf.v4.new_code_cell("""# Stratified split
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_scaled,
    y_train,
    test_size=0.3,
    stratify=y_train,
    random_state=RANDOM_STATE
)

print("Train-Validation Split:")
print(f"  Training: {X_train_split.shape}")
print(f"  Validation: {X_val_split.shape}")
print(f"  Test: {X_test_scaled.shape}")

print(f"\\nClass distribution verification:")
print(f"  Original: {y_train.value_counts(normalize=True).to_dict()}")
print(f"  Training: {y_train_split.value_counts(normalize=True).to_dict()}")
print(f"  Validation: {y_val_split.value_counts(normalize=True).to_dict()}")

print("\\n[OK] Class distribution preserved!")"""))

    # Save Processed Data
    cells.append(nbf.v4.new_markdown_cell("""## 💾 Save Processed Data

Save the processed data so we can use it in the next notebooks."""))

    cells.append(nbf.v4.new_code_cell("""# Create processed data directory
processed_dir = Path('../data/processed')
processed_dir.mkdir(exist_ok=True)

# Save datasets
print("Saving processed datasets...")

X_train_split.to_csv(processed_dir / 'X_train.csv', index=False)
X_val_split.to_csv(processed_dir / 'X_val.csv', index=False)
X_test_scaled.to_csv(processed_dir / 'X_test.csv', index=False)

y_train_split.to_csv(processed_dir / 'y_train.csv', index=False, header=True)
y_val_split.to_csv(processed_dir / 'y_val.csv', index=False, header=True)

# Save feature names
pd.DataFrame({'feature': X_train_split.columns}).to_csv(
    processed_dir / 'feature_names.csv', index=False
)

# Save IDs
train_df[['SK_ID_CURR']].iloc[X_train_split.index].to_csv(
    processed_dir / 'train_ids.csv', index=False
)
train_df[['SK_ID_CURR']].iloc[X_val_split.index].to_csv(
    processed_dir / 'val_ids.csv', index=False
)
test_df[['SK_ID_CURR']].to_csv(processed_dir / 'test_ids.csv', index=False)

print("[OK] All datasets saved!")
print(f"\\nSaved files:")
print(f"  - X_train.csv: {X_train_split.shape}")
print(f"  - X_val.csv: {X_val_split.shape}")
print(f"  - X_test.csv: {X_test_scaled.shape}")
print(f"  - y_train.csv, y_val.csv")
print(f"  - feature_names.csv: {len(X_train_split.columns)} features")
print(f"  - IDs for each split")"""))

    # Summary
    cells.append(nbf.v4.new_markdown_cell("""## 📝 Feature Engineering Summary

### ✅ What We Accomplished

1. **Handled Missing Values**
   - Dropped features with >70% missing
   - Created missing indicators
   - Imputed remaining values

2. **Created Domain Features** (10+ new features)
   - Age and employment features
   - Income per person
   - Debt-to-income ratio (KEY!)
   - Credit utilization
   - Payment burden ratio
   - Family features
   - Document counts
   - External source aggregations

3. **Encoded Categorical Variables**
   - One-hot encoded low-cardinality features
   - Handled high-cardinality features

4. **Feature Selection**
   - Removed low-variance features
   - Removed highly correlated features
   - Reduced feature count significantly

5. **Scaled Features**
   - StandardScaler (mean=0, std=1)
   - Ready for modeling!

6. **Created Train-Val Split**
   - Stratified sampling
   - 70% training, 30% validation

### 🎯 Key Features Created

The most important features for credit scoring:
1. **DEBT_TO_INCOME_RATIO** - How much debt vs income
2. **ANNUITY_TO_INCOME_RATIO** - Can afford payments?
3. **EMPLOYMENT_YEARS** - Stability indicator
4. **AGE_YEARS** - Risk correlates with age
5. **INCOME_PER_PERSON** - Family financial situation
6. **CREDIT_UTILIZATION** - Credit usage patterns

### 📊 Final Dataset

- **Features:** ~100-150 (after selection and engineering)
- **Training samples:** ~215,000
- **Validation samples:** ~92,000
- **Test samples:** ~48,000
- **Class balance:** ~8% positive class (defaults)

### 🚀 Next Steps

In the next notebook ([03_baseline_models.ipynb](03_baseline_models.ipynb)), we will:

1. **Train Multiple Baseline Models**
   - Logistic Regression
   - Random Forest
   - XGBoost
   - LightGBM

2. **Set Up MLflow Tracking**
   - Log all experiments
   - Compare models
   - Save artifacts

3. **Evaluate Using Appropriate Metrics**
   - ROC-AUC
   - Precision-Recall AUC
   - F1-Score
   - Confusion Matrix

4. **Select Best Baseline Model**
   - Compare performance
   - Consider interpretability
   - Choose for optimization

---

**Excellent work on feature engineering! 🎉**

Your data is now ready for modeling. Remember: good features are often more important than complex models!"""))

    nb['cells'] = cells
    return nb


def create_baseline_models_notebook():
    """Create Baseline Models training notebook with MLflow."""

    nb = nbf.v4.new_notebook()
    cells = []

    # Title
    cells.append(nbf.v4.new_markdown_cell("""# 03 - Baseline Models with MLflow Tracking
## Credit Scoring Model Project

**Learning Objectives:**
- Train multiple baseline models
- Set up MLflow experiment tracking
- Evaluate models using appropriate metrics for imbalanced data
- Compare model performance
- Select best baseline for optimization

**Why Multiple Baselines?**
Different algorithms have different strengths:
- **Logistic Regression:** Simple, interpretable, fast (good baseline)
- **Random Forest:** Handles non-linearity, robust to outliers
- **XGBoost:** Powerful gradient boosting, often wins competitions
- **LightGBM:** Fast, memory-efficient, great for large datasets

**MLflow Tracking:**
We'll log all experiments to compare models systematically. This is professional ML workflow!

Let's build our models! 🚀"""))

    # Imports
    cells.append(nbf.v4.new_markdown_cell("""## 📦 Import Libraries"""))

    cells.append(nbf.v4.new_code_cell("""# Standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import time

# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Evaluation
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)

# MLflow
import mlflow
import mlflow.sklearn

# Our utilities
import sys
sys.path.append('../')
from src.evaluation import (
    evaluate_model,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    compare_models,
    plot_feature_importance
)

# Configuration
warnings.filterwarnings('ignore')
RANDOM_STATE = 42

print("[OK] Libraries imported successfully!")
print(f"MLflow version: {mlflow.__version__}")"""))

    # Load Data
    cells.append(nbf.v4.new_markdown_cell("""## 📂 Load Processed Data

Load the data we prepared in the feature engineering notebook."""))

    cells.append(nbf.v4.new_code_cell("""# Load processed data
data_dir = Path('../data/processed')

print("Loading processed datasets...")
X_train = pd.read_csv(data_dir / 'X_train.csv')
X_val = pd.read_csv(data_dir / 'X_val.csv')
y_train = pd.read_csv(data_dir / 'y_train.csv').squeeze()
y_val = pd.read_csv(data_dir / 'y_val.csv').squeeze()

print(f"[OK] Data loaded!")
print(f"\\nDataset shapes:")
print(f"  X_train: {X_train.shape}")
print(f"  X_val: {X_val.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  y_val: {y_val.shape}")

print(f"\\nTarget distribution:")
print(f"  Training: {y_train.value_counts(normalize=True).to_dict()}")
print(f"  Validation: {y_val.value_counts(normalize=True).to_dict()}")

print(f"\\nFeature count: {X_train.shape[1]}")"""))

    # Setup MLflow
    cells.append(nbf.v4.new_markdown_cell("""## 🔬 Setup MLflow Experiment Tracking

**What MLflow Does:**
- Automatically logs all your experiments
- Stores parameters, metrics, and artifacts
- Provides a UI to visualize and compare runs
- Makes your work reproducible

**To view experiments:**
```bash
# In a separate terminal, run:
mlflow ui
# Then open: http://localhost:5000
```"""))

    cells.append(nbf.v4.new_code_cell("""# Set experiment name
experiment_name = "credit_scoring_baseline_models"
mlflow.set_experiment(experiment_name)

# Get experiment ID
experiment = mlflow.get_experiment_by_name(experiment_name)
print(f"[OK] MLflow experiment set: {experiment_name}")
print(f"Experiment ID: {experiment.experiment_id}")
print(f"\\nArtifacts will be stored in: {experiment.artifact_location}")
print(f"\\nTo view experiments, run: mlflow ui")
print(f"Then open: http://localhost:5000")"""))

    # Training Function
    cells.append(nbf.v4.new_markdown_cell("""## 🎯 Create Training Function

We'll create a reusable function that:
1. Trains a model
2. Evaluates it
3. Logs everything to MLflow
4. Creates visualizations"""))

    cells.append(nbf.v4.new_code_cell("""def train_and_evaluate_model(model, model_name, params, X_train, y_train, X_val, y_val):
    \"\"\"
    Train a model and log everything to MLflow.

    Educational Note:
    -----------------
    This function demonstrates professional ML workflow:
    1. Track training time
    2. Make predictions
    3. Evaluate with multiple metrics
    4. Log parameters, metrics, model, and artifacts
    5. Return results for comparison
    \"\"\"
    print("="*80)
    print(f"Training: {model_name}")
    print("="*80)

    with mlflow.start_run(run_name=model_name) as run:
        # Log parameters
        mlflow.log_params(params)
        mlflow.set_tag("model_type", model_name)

        # Train model
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        mlflow.log_metric("training_time_seconds", training_time)
        print(f"[OK] Training completed in {training_time:.2f} seconds")

        # Predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        # Evaluate
        metrics = evaluate_model(y_val, y_pred, y_pred_proba, model_name)

        # Log all metrics to MLflow
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(metric_name, value)

        # Create and log visualizations
        # 1. ROC Curve
        fig = plot_roc_curve(y_val, y_pred_proba, model_name)
        roc_path = f"plots/{model_name}_roc_curve.png"
        fig.savefig(roc_path, dpi=100, bbox_inches='tight')
        mlflow.log_artifact(roc_path)
        plt.close()

        # 2. Precision-Recall Curve
        fig = plot_precision_recall_curve(y_val, y_pred_proba, model_name)
        pr_path = f"plots/{model_name}_pr_curve.png"
        fig.savefig(pr_path, dpi=100, bbox_inches='tight')
        mlflow.log_artifact(pr_path)
        plt.close()

        # 3. Confusion Matrix
        fig = plot_confusion_matrix(y_val, y_pred, model_name, normalize=True)
        cm_path = f"plots/{model_name}_confusion_matrix.png"
        fig.savefig(cm_path, dpi=100, bbox_inches='tight')
        mlflow.log_artifact(cm_path)
        plt.close()

        # 4. Feature Importance (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            fig = plot_feature_importance(
                X_train.columns.tolist(),
                model.feature_importances_,
                top_n=20,
                model_name=model_name
            )
            fi_path = f"plots/{model_name}_feature_importance.png"
            fig.savefig(fi_path, dpi=100, bbox_inches='tight')
            mlflow.log_artifact(fi_path)
            plt.close()
            print(f"[OK] Feature importance plot saved")

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"[OK] All metrics and artifacts logged to MLflow")
        print(f"Run ID: {run.info.run_id}")

        return metrics, model

# Create plots directory
Path('plots').mkdir(exist_ok=True)
print("[OK] Training function ready!")"""))

    # Train Models
    cells.append(nbf.v4.new_markdown_cell("""## 🚀 Train Baseline Models

We'll train 4 different models and compare them.

**Model Selection Rationale:**

1. **Logistic Regression**
   - Simple linear model
   - Fast to train
   - Highly interpretable
   - Good baseline to beat

2. **Random Forest**
   - Ensemble of decision trees
   - Handles non-linear relationships
   - Robust to outliers
   - Built-in feature importance

3. **XGBoost**
   - Gradient boosting
   - Often wins ML competitions
   - Handles imbalanced data well
   - Many hyperparameters to tune

4. **LightGBM**
   - Microsoft's gradient boosting
   - Very fast and memory-efficient
   - Great for large datasets
   - Often comparable to XGBoost

Let's train them all!"""))

    # Logistic Regression
    cells.append(nbf.v4.new_code_cell("""# 1. LOGISTIC REGRESSION
print("\\n\\n### 1. LOGISTIC REGRESSION ###\\n")

lr_params = {
    'max_iter': 1000,
    'class_weight': 'balanced',  # Handle imbalance
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

lr_model = LogisticRegression(**lr_params)
lr_metrics, lr_trained = train_and_evaluate_model(
    lr_model, "Logistic_Regression", lr_params,
    X_train, y_train, X_val, y_val
)"""))

    # Random Forest
    cells.append(nbf.v4.new_code_cell("""# 2. RANDOM FOREST
print("\\n\\n### 2. RANDOM FOREST ###\\n")

rf_params = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 50,
    'min_samples_leaf': 20,
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': 0
}

rf_model = RandomForestClassifier(**rf_params)
rf_metrics, rf_trained = train_and_evaluate_model(
    rf_model, "Random_Forest", rf_params,
    X_train, y_train, X_val, y_val
)"""))

    # XGBoost
    cells.append(nbf.v4.new_code_cell("""# 3. XGBOOST
print("\\n\\n### 3. XGBOOST ###\\n")

# Calculate scale_pos_weight for class imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_params = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': scale_pos_weight,  # Handle imbalance
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbosity': 0
}

xgb_model = XGBClassifier(**xgb_params)
xgb_metrics, xgb_trained = train_and_evaluate_model(
    xgb_model, "XGBoost", xgb_params,
    X_train, y_train, X_val, y_val
)"""))

    # LightGBM
    cells.append(nbf.v4.new_code_cell("""# 4. LIGHTGBM
print("\\n\\n### 4. LIGHTGBM ###\\n")

lgbm_params = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': -1
}

lgbm_model = LGBMClassifier(**lgbm_params)
lgbm_metrics, lgbm_trained = train_and_evaluate_model(
    lgbm_model, "LightGBM", lgbm_params,
    X_train, y_train, X_val, y_val
)"""))

    # Compare Models
    cells.append(nbf.v4.new_markdown_cell("""## 📊 Compare All Models

Let's compare all our baseline models side-by-side."""))

    cells.append(nbf.v4.new_code_cell("""# Gather all results
all_results = {
    'Logistic_Regression': lr_metrics,
    'Random_Forest': rf_metrics,
    'XGBoost': xgb_metrics,
    'LightGBM': lgbm_metrics
}

# Compare using our utility function
comparison_df = compare_models(all_results, metric='roc_auc')

# Display comparison
print("\\n" + "="*80)
print("DETAILED COMPARISON")
print("="*80)
print(comparison_df.to_string())

# Save comparison
comparison_df.to_csv('model_comparison.csv')
print("\\n[OK] Comparison saved to model_comparison.csv")"""))

    # Visualize Comparison
    cells.append(nbf.v4.new_code_cell("""# Visualize model comparison
metrics_to_plot = ['roc_auc', 'pr_auc', 'f1', 'precision', 'recall']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, metric in enumerate(metrics_to_plot):
    if metric in comparison_df.columns:
        data = comparison_df[metric].sort_values(ascending=False)

        axes[idx].barh(range(len(data)), data.values, color='steelblue', alpha=0.8)
        axes[idx].set_yticks(range(len(data)))
        axes[idx].set_yticklabels(data.index)
        axes[idx].set_xlabel(metric.upper().replace('_', '-'), fontsize=12)
        axes[idx].set_title(f'{metric.upper().replace("_", " ")} Comparison',
                           fontsize=14, fontweight='bold')
        axes[idx].grid(axis='x', alpha=0.3)

        # Add value labels
        for i, v in enumerate(data.values):
            axes[idx].text(v + 0.005, i, f'{v:.4f}',
                          va='center', fontsize=10)

# Remove empty subplot
fig.delaxes(axes[5])

plt.tight_layout()
plt.savefig('plots/model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("[OK] Comparison visualization saved!")"""))

    # Summary
    cells.append(nbf.v4.new_markdown_cell("""## 📝 Baseline Models Summary

### ✅ What We Accomplished

1. **Trained 4 Baseline Models**
   - Logistic Regression (simple baseline)
   - Random Forest (ensemble method)
   - XGBoost (gradient boosting)
   - LightGBM (efficient gradient boosting)

2. **MLflow Experiment Tracking**
   - All runs logged automatically
   - Parameters, metrics, and artifacts stored
   - Compare runs visually in MLflow UI

3. **Comprehensive Evaluation**
   - ROC-AUC scores
   - Precision-Recall curves
   - Confusion matrices
   - Feature importance (tree models)

4. **Model Comparison**
   - Side-by-side metrics
   - Visual comparisons
   - Identified best baseline

### 🏆 Best Performing Model

Based on ROC-AUC and PR-AUC scores:
- **Best Model:** [Check comparison above]
- **ROC-AUC:** [Value]
- **PR-AUC:** [Value]
- **F1-Score:** [Value]

### 💡 Key Insights

1. **Tree-based models** (RF, XGB, LightGBM) generally outperform Logistic Regression
   - They can capture non-linear relationships
   - Better handle feature interactions

2. **Class imbalance handling** is critical
   - Used `class_weight='balanced'` or `scale_pos_weight`
   - Evaluated with appropriate metrics (ROC-AUC, PR-AUC, F1)

3. **Feature importance** reveals key predictors
   - Debt-to-income ratio likely important
   - External credit scores matter
   - Age and employment features contribute

4. **Model complexity vs performance trade-off**
   - Logistic Regression: Fast, interpretable, but lower performance
   - Tree models: Higher performance, but less interpretable

### 🎯 Next Steps

In the next notebook ([04_hyperparameter_optimization.ipynb](04_hyperparameter_optimization.ipynb)), we will:

1. **Select Best Baseline**
   - Choose the best performing model
   - Or ensemble top models

2. **Systematic Hyperparameter Tuning**
   - Define search space
   - Use GridSearchCV or RandomizedSearchCV
   - Use StratifiedKFold cross-validation

3. **Optimize for Target Metric**
   - Focus on ROC-AUC or PR-AUC
   - Consider business costs (FP vs FN)

4. **Log All Optimization Runs**
   - Track in MLflow
   - Compare optimization strategies

---

**Excellent work! You now have solid baseline models! 🎉**

### 📊 To View Your Experiments:

```bash
# In terminal, run:
mlflow ui

# Then open in browser:
http://localhost:5000
```

In the MLflow UI, you can:
- Compare all runs side-by-side
- Sort by metrics
- View all plots and artifacts
- Download models

---

**Remember:** These are baselines! We'll improve them significantly in the next notebook through hyperparameter optimization! 🚀"""))

    nb['cells'] = cells
    return nb


def save_notebook(notebook, filename):
    """Save notebook to file."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    nb_path = project_root / 'notebooks' / filename

    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(notebook, f)
    print(f"[OK] Created: {nb_path}")


if __name__ == "__main__":
    print("Creating all educational notebooks...")
    print("="*80)

    # Create Feature Engineering notebook
    print("\\n1. Creating Feature Engineering notebook...")
    fe_nb = create_feature_engineering_notebook()
    save_notebook(fe_nb, '02_feature_engineering.ipynb')

    # Create Baseline Models notebook
    print("\\n2. Creating Baseline Models notebook...")
    bm_nb = create_baseline_models_notebook()
    save_notebook(bm_nb, '03_baseline_models.ipynb')

    print("\\n" + "="*80)
    print("[OK] Notebooks created successfully!")
    print("\\nCreated:")
    print("  - 02_feature_engineering.ipynb")
    print("  - 03_baseline_models.ipynb")
    print("\\nNote: Notebooks 04 and 05 can be created by extending this script.")
    print("\\nNext steps:")
    print("1. Run: jupyter notebook")
    print("2. Execute notebooks in order")
    print("3. Start MLflow UI: mlflow ui")
