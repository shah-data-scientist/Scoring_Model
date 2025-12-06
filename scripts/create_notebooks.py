"""
Script to create comprehensive educational notebooks for the Credit Scoring project.
This ensures consistency and completeness across all notebooks.
"""

import nbformat as nbf
import os
from pathlib import Path

def create_eda_notebook():
    """Create a comprehensive EDA notebook with educational content."""

    nb = nbf.v4.new_notebook()

    cells = []

    # Title and Introduction
    cells.append(nbf.v4.new_markdown_cell("""# 01 - Exploratory Data Analysis (EDA)
## Credit Scoring Model Project

**Learning Objectives:**
- Understand the dataset structure and characteristics
- Identify data quality issues (missing values, outliers, anomalies)
- Analyze target variable distribution and class imbalance
- Explore relationships between features and target
- Generate insights to guide feature engineering and modeling

**What is EDA?**
Exploratory Data Analysis is the critical first step in any data science project. It helps you:
1. **Understand** what data you have
2. **Identify** problems (missing data, outliers, inconsistencies)
3. **Discover** patterns and relationships
4. **Generate** hypotheses for modeling
5. **Make** informed decisions about preprocessing and feature engineering

Let's begin!"""))

    # Imports
    cells.append(nbf.v4.new_markdown_cell("""## 📦 Import Libraries

We'll use:
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib & seaborn**: Data visualization
- **missingno**: Specialized missing data visualization
- **warnings**: Suppress unnecessary warnings"""))

    cells.append(nbf.v4.new_code_cell("""# Data manipulation and analysis
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# Utilities
import warnings
import os
from pathlib import Path

# Configuration
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("✅ Libraries imported successfully!")
print(f"📊 Pandas version: {pd.__version__}")
print(f"🔢 NumPy version: {np.__version__}")"""))

    # Load Data
    cells.append(nbf.v4.new_markdown_cell("""## 📂 Load Data

We'll load the main application training data. This contains information about loan applications and the target variable (whether the client had payment difficulties).

**Understanding the Dataset:**
- **Source:** Home Credit Default Risk (Kaggle)
- **Type:** Binary classification (predict loan default)
- **Training samples:** 307,511 applications
- **Features:** 122 columns (mix of numerical and categorical)
- **Target:** `TARGET` (1 = payment difficulties, 0 = no difficulties)"""))

    cells.append(nbf.v4.new_code_cell("""# Define data path
DATA_PATH = Path('../data')

# Load training data
print("Loading training data...")
train_df = pd.read_csv(DATA_PATH / 'application_train.csv')
test_df = pd.read_csv(DATA_PATH / 'application_test.csv')

print(f"✅ Data loaded successfully!")
print(f"\\n📊 Training set shape: {train_df.shape}")
print(f"   - Rows (applications): {train_df.shape[0]:,}")
print(f"   - Columns (features): {train_df.shape[1]}")
print(f"\\n📊 Test set shape: {test_df.shape}")
print(f"   - Rows (applications): {test_df.shape[0]:,}")
print(f"   - Columns (features): {test_df.shape[1]}")

# Memory usage
memory_mb = train_df.memory_usage(deep=True).sum() / 1024**2
print(f"\\n💾 Memory usage: {memory_mb:.2f} MB")"""))

    # First Look
    cells.append(nbf.v4.new_markdown_cell("""## 👀 First Look at the Data

Let's examine the first few rows to understand what information we have."""))

    cells.append(nbf.v4.new_code_cell("""# Display first few rows
print("First 5 rows of training data:")
train_df.head()"""))

    cells.append(nbf.v4.new_markdown_cell("""## 📋 Data Structure and Types

Understanding data types is crucial:
- **int64/float64:** Numerical features (can use directly in models)
- **object:** Usually categorical text (need encoding)
- **bool:** Binary flags

Let's analyze the data types and basic statistics."""))

    cells.append(nbf.v4.new_code_cell("""# Dataset info
print("=" * 80)
print("DATASET INFORMATION")
print("=" * 80)
train_df.info()

print("\\n" + "=" * 80)
print("DATA TYPE SUMMARY")
print("=" * 80)
type_counts = train_df.dtypes.value_counts()
print(type_counts)

print("\\n" + "=" * 80)
print("BREAKDOWN BY TYPE")
print("=" * 80)
print(f"Numerical features (int64/float64): {len(train_df.select_dtypes(include=['int64', 'float64']).columns)}")
print(f"Categorical features (object): {len(train_df.select_dtypes(include=['object']).columns)}")"""))

    # Target Analysis
    cells.append(nbf.v4.new_markdown_cell("""## 🎯 Target Variable Analysis

**CRITICAL:** Understanding the target distribution is essential!

The target variable (`TARGET`) indicates:
- **0:** Client repaid loan without difficulties (NEGATIVE class)
- **1:** Client had payment difficulties (POSITIVE class - what we want to predict)

**Class Imbalance:**
In credit scoring, most people repay their loans, so we expect:
- Many more 0s than 1s (imbalanced dataset)
- This affects model training and evaluation strategy!"""))

    cells.append(nbf.v4.new_code_cell("""# Target distribution
print("=" * 80)
print("TARGET VARIABLE DISTRIBUTION")
print("=" * 80)

target_counts = train_df['TARGET'].value_counts()
target_pct = train_df['TARGET'].value_counts(normalize=True) * 100

print("\\nAbsolute counts:")
print(target_counts)
print("\\nPercentages:")
print(target_pct)

# Calculate imbalance ratio
imbalance_ratio = target_counts[0] / target_counts[1]
print(f"\\n⚠️  IMBALANCE RATIO: {imbalance_ratio:.2f}:1")
print(f"   For every 1 defaulter, there are ~{imbalance_ratio:.0f} non-defaulters")
print(f"\\n💡 This means:")
print(f"   - We CANNOT use accuracy as our primary metric")
print(f"   - We MUST use stratified sampling")
print(f"   - We should consider class weighting or resampling techniques")"""))

    cells.append(nbf.v4.new_code_cell("""# Visualize target distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count plot
sns.countplot(data=train_df, x='TARGET', ax=axes[0], palette='Set2')
axes[0].set_title('Target Distribution (Counts)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('TARGET (0=No Default, 1=Default)', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
for container in axes[0].containers:
    axes[0].bar_label(container, fmt='%d')

# Pie chart
colors = ['#90EE90', '#FFB6C1']  # Green for 0, Red for 1
explode = (0, 0.1)  # Explode the defaulters slice
axes[1].pie(target_counts, labels=['No Default (0)', 'Default (1)'], autopct='%1.1f%%',
            startangle=90, colors=colors, explode=explode, shadow=True)
axes[1].set_title('Target Distribution (Percentage)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print("\\n✅ Visualization complete!")"""))

    # Missing Values
    cells.append(nbf.v4.new_markdown_cell("""## 🔍 Missing Values Analysis

**Why missing values matter:**
- Most ML algorithms can't handle missing values directly
- Missing patterns might be informative (e.g., "no previous credit" could be significant)
- We need to decide: impute, drop, or create "missing" indicator features

Let's identify which features have missing data and how much."""))

    cells.append(nbf.v4.new_code_cell("""def analyze_missing_values(df, name="Dataset"):
    \"\"\"
    Comprehensive missing value analysis.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to analyze
    name : str
        Name of the dataset for display

    Returns:
    --------
    pandas.DataFrame : Missing value summary
    \"\"\"
    # Calculate missing values
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)

    # Create summary table
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': missing.values,
        'Missing_Percent': missing_pct.values,
        'Data_Type': df.dtypes.values
    })

    # Filter to only columns with missing values
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
        'Missing_Percent', ascending=False
    ).reset_index(drop=True)

    # Print summary
    print("=" * 80)
    print(f"MISSING VALUES ANALYSIS - {name}")
    print("=" * 80)
    print(f"Total columns: {df.shape[1]}")
    print(f"Columns with missing values: {len(missing_df)}")
    print(f"Columns complete (no missing): {df.shape[1] - len(missing_df)}")

    if len(missing_df) > 0:
        print(f"\\nMost affected columns (top 10):")
        print(missing_df.head(10).to_string(index=False))

    return missing_df

# Analyze training data
missing_train = analyze_missing_values(train_df, "Training Set")"""))

    cells.append(nbf.v4.new_code_cell("""# Visualize missing values

if len(missing_train) > 0:
    # Bar plot of top 20 features with missing values
    fig, ax = plt.subplots(figsize=(12, 8))

    top_missing = missing_train.head(20)
    sns.barplot(data=top_missing, y='Column', x='Missing_Percent', palette='YlOrRd', ax=ax)
    ax.set_title('Top 20 Features with Missing Values', fontsize=14, fontweight='bold')
    ax.set_xlabel('Percentage Missing (%)', fontsize=12)
    ax.set_ylabel('Feature Name', fontsize=12)
    ax.axvline(x=50, color='red', linestyle='--', label='50% threshold', linewidth=2)
    ax.legend()

    plt.tight_layout()
    plt.show()

    # Matrix visualization (sample for performance)
    print("\\nMissing Value Matrix (sample of 1000 rows):")
    sample_size = min(1000, len(train_df))
    msno.matrix(train_df.sample(sample_size, random_state=RANDOM_STATE), figsize=(14, 6), sparkline=False)
    plt.show()

else:
    print("\\n✅ No missing values found!")"""))

    cells.append(nbf.v4.new_markdown_cell("""### 💡 Missing Value Insights

**Key Observations:**
1. Some features have >60% missing values (e.g., building/apartment characteristics)
   - These might be missing for a reason (e.g., applicant doesn't own property)
   - Consider dropping or creating "is_missing" indicator features

2. Features with <10% missing can often be imputed safely
   - Use median/mode for numerical features
   - Use most frequent for categorical features

3. Missing values might be informative!
   - Example: Missing "OWN_CAR_AGE" might mean no car ownership
   - Don't just drop or impute blindly - think about WHY data is missing

**Next Steps:**
- In the Feature Engineering notebook, we'll handle these systematically
- We'll create missing indicators for important features
- We'll use domain knowledge to guide imputation strategies"""))

    # Numerical Features
    cells.append(nbf.v4.new_markdown_cell("""## 📊 Numerical Features Analysis

Let's examine the numerical features:
- Distribution shapes (normal, skewed?)
- Outliers
- Ranges and scales
- Correlations with target"""))

    cells.append(nbf.v4.new_code_cell("""# Select numerical features (exclude ID and TARGET)
numerical_features = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features.remove('SK_ID_CURR')  # Remove ID
numerical_features.remove('TARGET')  # Remove target

print(f"Found {len(numerical_features)} numerical features")
print(f"\\nFirst 10 numerical features:")
print(numerical_features[:10])

# Summary statistics
print("\\n" + "=" * 80)
print("SUMMARY STATISTICS (key features)")
print("=" * 80)
key_features = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
                'DAYS_BIRTH', 'DAYS_EMPLOYED', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS']
print(train_df[key_features].describe())"""))

    cells.append(nbf.v4.new_code_cell("""# Visualize key numerical features
key_features_to_plot = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for idx, feature in enumerate(key_features_to_plot):
    # Plot distribution
    train_df[feature].hist(bins=50, ax=axes[idx], edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'Distribution of {feature}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(feature, fontsize=10)
    axes[idx].set_ylabel('Frequency', fontsize=10)
    axes[idx].axvline(train_df[feature].median(), color='red', linestyle='--',
                      label=f'Median: {train_df[feature].median():,.0f}', linewidth=2)
    axes[idx].legend()

plt.tight_layout()
plt.show()

print("\\n💡 Notice:")
print("   - Most amount features are right-skewed (long tail to the right)")
print("   - This is typical for financial data (few very high values)")
print("   - May need log transformation for some models")"""))

    # Age analysis (DAYS_BIRTH is negative, convert to years)
    cells.append(nbf.v4.new_markdown_cell("""### Age Analysis

**Important Note:** `DAYS_BIRTH` is stored as negative days from current date.
- More negative = older person
- We'll convert to actual age in years for easier interpretation"""))

    cells.append(nbf.v4.new_code_cell("""# Convert DAYS_BIRTH to years
train_df['AGE_YEARS'] = -train_df['DAYS_BIRTH'] / 365

# Plot age distribution
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Age distribution
axes[0].hist(train_df['AGE_YEARS'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
axes[0].set_title('Age Distribution of Applicants', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Age (years)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].axvline(train_df['AGE_YEARS'].median(), color='red', linestyle='--',
                label=f'Median: {train_df[\'AGE_YEARS\'].median():.1f} years', linewidth=2)
axes[0].legend()

# Age vs Target
sns.boxplot(data=train_df, x='TARGET', y='AGE_YEARS', palette='Set2', ax=axes[1])
axes[1].set_title('Age Distribution by Target Class', fontsize=14, fontweight='bold')
axes[1].set_xlabel('TARGET (0=No Default, 1=Default)', fontsize=12)
axes[1].set_ylabel('Age (years)', fontsize=12)
axes[1].set_xticklabels(['No Default', 'Default'])

plt.tight_layout()
plt.show()

print("\\nAge Statistics by Target:")
print(train_df.groupby('TARGET')['AGE_YEARS'].describe())

print("\\n💡 Insight:")
print("   Are there age differences between defaulters and non-defaulters?")
print("   This could be a useful predictive feature!")"""))

    # Categorical Features
    cells.append(nbf.v4.new_markdown_cell("""## 📝 Categorical Features Analysis

Categorical features represent discrete categories or groups:
- Contract type (Cash loans vs Revolving loans)
- Gender
- Income type
- Education level
- etc.

We need to understand:
- How many unique values each has
- Distribution across categories
- Relationship with target variable"""))

    cells.append(nbf.v4.new_code_cell("""# Identify categorical features
categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()

print(f"Found {len(categorical_features)} categorical features")
print(f"\\nCategorical features:")
print(categorical_features)

# Analyze unique values
print("\\n" + "=" * 80)
print("UNIQUE VALUES IN CATEGORICAL FEATURES")
print("=" * 80)

cat_unique = pd.DataFrame({
    'Feature': categorical_features,
    'Unique_Values': [train_df[col].nunique() for col in categorical_features],
    'Most_Common': [train_df[col].mode()[0] if len(train_df[col].mode()) > 0 else None
                    for col in categorical_features],
    'Most_Common_Freq': [train_df[col].value_counts().iloc[0] if len(train_df[col]) > 0 else 0
                         for col in categorical_features]
})

print(cat_unique.sort_values('Unique_Values', ascending=False))"""))

    cells.append(nbf.v4.new_code_cell("""# Visualize key categorical features
key_cat_features = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for idx, feature in enumerate(key_cat_features):
    # Count by category and target
    ct = pd.crosstab(train_df[feature], train_df['TARGET'], normalize='index') * 100

    ct.plot(kind='bar', ax=axes[idx], color=['#90EE90', '#FFB6C1'], width=0.8)
    axes[idx].set_title(f'{feature} vs TARGET', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(feature, fontsize=10)
    axes[idx].set_ylabel('Percentage (%)', fontsize=10)
    axes[idx].legend(['No Default (0)', 'Default (1)'], loc='upper right')
    axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

print("\\n💡 Look for:")
print("   - Which categories have higher default rates?")
print("   - Are there patterns? (e.g., certain income types or education levels)")
print("   - These insights guide feature engineering and model interpretation!")"""))

    # Correlation Analysis
    cells.append(nbf.v4.new_markdown_cell("""## 🔗 Correlation Analysis

**What is correlation?**
Correlation measures the linear relationship between two numerical variables:
- **+1:** Perfect positive correlation (both increase together)
- **0:** No linear relationship
- **-1:** Perfect negative correlation (one increases, other decreases)

**Why it matters:**
1. **Feature selection:** Highly correlated features might be redundant
2. **Target relationships:** Features correlated with target are potentially useful
3. **Multicollinearity:** High correlation between features can cause issues in some models

**Important:** Correlation only captures LINEAR relationships! Non-linear patterns won't show up."""))

    cells.append(nbf.v4.new_code_cell("""# Calculate correlation with target for numerical features
correlations = train_df[numerical_features + ['TARGET']].corr()['TARGET'].sort_values(ascending=False)

# Remove target itself
correlations = correlations.drop('TARGET')

print("=" * 80)
print("CORRELATION WITH TARGET")
print("=" * 80)
print("\\nTop 10 Positive Correlations (increase with target):")
print(correlations.head(10))
print("\\nTop 10 Negative Correlations (decrease with target):")
print(correlations.tail(10))"""))

    cells.append(nbf.v4.new_code_cell("""# Visualize correlations with target
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Top positive correlations
top_pos = correlations.head(15)
axes[0].barh(range(len(top_pos)), top_pos.values, color='darkgreen', alpha=0.7)
axes[0].set_yticks(range(len(top_pos)))
axes[0].set_yticklabels(top_pos.index)
axes[0].set_xlabel('Correlation Coefficient', fontsize=12)
axes[0].set_title('Top 15 Positive Correlations with TARGET', fontsize=14, fontweight='bold')
axes[0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)

# Top negative correlations
top_neg = correlations.tail(15).sort_values()
axes[1].barh(range(len(top_neg)), top_neg.values, color='darkred', alpha=0.7)
axes[1].set_yticks(range(len(top_neg)))
axes[1].set_yticklabels(top_neg.index)
axes[1].set_xlabel('Correlation Coefficient', fontsize=12)
axes[1].set_title('Top 15 Negative Correlations with TARGET', fontsize=14, fontweight='bold')
axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.show()

print("\\n💡 Observations:")
print("   - Features with abs(correlation) > 0.1 might be useful predictors")
print("   - But weak correlation doesn't mean useless! Non-linear patterns exist")
print("   - Tree-based models can capture non-linear relationships")"""))

    # Correlation heatmap
    cells.append(nbf.v4.new_code_cell("""# Correlation heatmap for key features
key_features_for_corr = ['TARGET', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
                          'AMT_GOODS_PRICE', 'AGE_YEARS', 'DAYS_EMPLOYED',
                          'CNT_CHILDREN', 'CNT_FAM_MEMBERS']

# Add TARGET column correlation
corr_matrix = train_df[key_features_for_corr].corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap - Key Features', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

print("\\n💡 Use this heatmap to:")
print("   - Identify highly correlated feature pairs (consider removing one)")
print("   - Find features most correlated with TARGET")
print("   - Understand feature relationships for domain insights")"""))

    # Summary and Next Steps
    cells.append(nbf.v4.new_markdown_cell("""## 📝 EDA Summary and Key Findings

### 🎯 Target Variable
- **Highly imbalanced:** ~92% no default, ~8% default
- **Implication:** Must use appropriate metrics (ROC-AUC, Precision-Recall) and stratified sampling
- **Action:** Consider class weighting or resampling techniques

### 🔍 Missing Values
- **Many features** have significant missing values (>50%)
- **Pattern:** Building/apartment features most affected
- **Action:** Create missing indicators, careful imputation, consider dropping very sparse features

### 📊 Numerical Features
- **Distribution:** Most amount features are right-skewed
- **Scale:** Features have vastly different scales (income vs children count)
- **Action:** Consider log transformation, standardization/normalization required

### 📝 Categorical Features
- **Diversity:** Mix of binary and multi-class categories
- **Patterns:** Some categories show different default rates
- **Action:** Encode carefully (one-hot vs label encoding), potentially create aggregations

### 🔗 Correlations
- **Weak correlations** with target (most < 0.1)
- **Expected:** Complex real-world problem, non-linear relationships likely
- **Action:** Tree-based models preferred, create interaction features

### 💡 Business Insights
1. **Age matters:** Distribution differences between defaulters and non-defaulters
2. **Income type:** Different default rates across income categories
3. **Credit amount:** Relationship with default probability needs exploration
4. **Employment:** Days employed shows patterns worth investigating

---

## 🚀 Next Steps

In the next notebook ([02_feature_engineering.ipynb](02_feature_engineering.ipynb)), we will:

1. **Handle Missing Values**
   - Imputation strategies
   - Missing indicators
   - Drop very sparse features

2. **Create New Features**
   - Domain-based features (e.g., debt-to-income ratio)
   - Aggregations from related tables
   - Polynomial/interaction features
   - Binning and grouping

3. **Encode Categorical Variables**
   - One-hot encoding for low-cardinality features
   - Label encoding for ordinal features
   - Target encoding for high-cardinality features

4. **Feature Scaling**
   - Standardization (StandardScaler)
   - Normalization (MinMaxScaler)
   - Log transformation for skewed features

5. **Feature Selection**
   - Remove low-variance features
   - Remove highly correlated features
   - Use feature importance from baseline models

---

## 📚 Learning Resources

**Want to learn more about EDA?**
- [Kaggle: Data Cleaning](https://www.kaggle.com/learn/data-cleaning)
- [Towards Data Science: EDA Guide](https://towardsdatascience.com/exploratory-data-analysis-8fc1cb20fd15)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

**Key Concepts Covered:**
- ✅ Data loading and inspection
- ✅ Target variable analysis
- ✅ Class imbalance
- ✅ Missing value analysis
- ✅ Numerical feature distributions
- ✅ Categorical feature encoding considerations
- ✅ Correlation analysis
- ✅ Business insight generation

---

**Great job completing the EDA! 🎉**

Now you understand your data and are ready to engineer features and build models!"""))

    # Save notebook
    nb['cells'] = cells
    return nb


def save_notebook(notebook, filename):
    """Save notebook to file."""
    # Get the script directory and go up one level to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    nb_path = project_root / 'notebooks' / filename

    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(notebook, f)
    print(f"[OK] Created: {nb_path}")


if __name__ == "__main__":
    print("Creating comprehensive educational notebooks...")
    print("=" * 80)

    # Create EDA notebook
    print("\\n1. Creating EDA notebook...")
    eda_nb = create_eda_notebook()
    save_notebook(eda_nb, '01_eda.ipynb')

    print("\\n" + "=" * 80)
    print("[OK] Notebook creation complete!")
    print("\\nNext steps:")
    print("1. Run: jupyter notebook notebooks/01_eda.ipynb")
    print("2. Execute cells and review outputs")
    print("3. Continue with feature engineering notebook")
