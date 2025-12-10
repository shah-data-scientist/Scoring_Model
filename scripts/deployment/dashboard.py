"""
Interactive Model Dashboard

Run with: poetry run streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, fbeta_score, accuracy_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import sys

# Add project root to path (go up 2 levels from scripts/deployment/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.domain_features import create_domain_features
from src.config import CONFIG

# Configuration
st.set_page_config(page_title="Credit Scoring Model Dashboard", layout="wide")
RANDOM_STATE = CONFIG['project']['random_state']
BETA = CONFIG['business']['f_beta']
N_FOLDS = CONFIG['project']['n_folds']
RESULTS_DIR = PROJECT_ROOT / CONFIG['paths']['results']

@st.cache_data
def load_predictions():
    """Load and validate cached predictions."""
    pred_path = RESULTS_DIR / 'train_predictions.csv'

    # Check file exists
    if not pred_path.exists():
        st.error(f"Predictions file not found: {pred_path}")
        st.info("Please run the model training pipeline first to generate predictions.")
        st.code("poetry run python scripts/pipeline/apply_best_model.py", language="bash")
        st.stop()

    # Load data with optimization for large files
    try:
        # Only load required columns to reduce memory usage
        df = pd.read_csv(pred_path, usecols=['TARGET', 'PROBABILITY'])

        # Sample if file is very large (>100k rows) for faster dashboard performance
        if len(df) > 100000:
            st.sidebar.warning(f"Large dataset detected ({len(df):,} rows). Using 100k sample for dashboard.")
            df = df.sample(n=100000, random_state=42)

    except Exception as e:
        st.error(f"Error loading predictions file: {e}")
        st.stop()

    # Validate schema
    required_cols = ['TARGET', 'PROBABILITY']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        st.error(f"Predictions file is missing required columns: {missing_cols}")
        st.info(f"Available columns: {list(df.columns)}")
        st.stop()

    # Validate data types and ranges
    try:
        # Check TARGET is binary
        if not set(df['TARGET'].unique()).issubset({0, 1}):
            st.warning(f"TARGET has unexpected values: {df['TARGET'].unique()}")

        # Check PROBABILITY range
        if (df['PROBABILITY'] < 0).any() or (df['PROBABILITY'] > 1).any():
            invalid_count = ((df['PROBABILITY'] < 0) | (df['PROBABILITY'] > 1)).sum()
            st.error(f"PROBABILITY has {invalid_count} values outside [0,1] range")
            st.stop()

        # Check for NaN
        if df[required_cols].isna().any().any():
            nan_counts = df[required_cols].isna().sum()
            st.error(f"Predictions contain NaN values:\n{nan_counts}")
            st.stop()

        st.sidebar.success(f"Loaded {len(df):,} predictions")

    except Exception as e:
        st.error(f"Data validation failed: {e}")
        st.stop()

    return df['TARGET'], df['PROBABILITY']

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix with row and column percentages."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Row percentages (Recall for positive class)
    # Avoid division by zero
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_row_pct = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums!=0)
    
    # Column percentages (Precision for positive class)
    col_sums = cm.sum(axis=0)[np.newaxis, :]
    cm_col_pct = np.divide(cm.astype('float'), col_sums, out=np.zeros_like(cm, dtype=float), where=col_sums!=0)
    
    # Create annotation labels: "Row% \n (Col%)"
    annot_labels = np.empty_like(cm, dtype=object)
    rows, cols = cm.shape
    for i in range(rows):
        for j in range(cols):
            annot_labels[i, j] = f"{cm_row_pct[i, j]:.1%}\n({cm_col_pct[i, j]:.1%})"
    
    fig, ax = plt.subplots(figsize=(8, 6))
    # We assume row percentages for the color map intensity
    sns.heatmap(cm_row_pct, annot=annot_labels, fmt='', cmap='Greens', ax=ax,
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix\nRow % (Col %)')
    return fig, cm

# --- APP ---

st.title("🏦 Credit Scoring Model Dashboard")
st.markdown("Interactive threshold adjustment for the optimized LightGBM model.")

# Load Data
with st.spinner("Loading cached predictions..."):
    try:
        y, y_proba = load_predictions()
        st.success(f"Loaded predictions for {len(y):,} samples.")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Sidebar
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.328, 0.01, help="Probability cutoff for classifying as Default.")

st.sidebar.markdown("---")
# We hardcode this here as a reference, but ideally it should be loaded from MLflow metrics
# In a real app, we'd fetch the 'optimal_threshold' metric from the last run
st.sidebar.metric("Optimal Threshold (Ref)", "0.3282", help="Threshold found during training")

# Metrics Calculation
y_pred = (y_proba >= threshold).astype(int)

# Calculate Metrics
cm = confusion_matrix(y, y_pred)
tn, fp, fn, tp = cm.ravel()
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
f1 = f1_score(y, y_pred)
accuracy = accuracy_score(y, y_pred)
cost = CONFIG['business']['cost_fn'] * fn + CONFIG['business']['cost_fp'] * fp

# Display Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Business Cost", f"{cost:,}", help=f"{CONFIG['business']['cost_fn']} * FN + {CONFIG['business']['cost_fp']} * FP")
col2.metric("Recall (TPR)", f"{recall:.4f}")
col3.metric("Precision", f"{precision:.4f}")
col4.metric("F3.2 Score", f"{fbeta_score(y, y_pred, beta=BETA):.4f}")

# Main Layout
st.subheader(f"Confusion Matrix at Threshold {threshold:.2f}")

col_left, col_right = st.columns([2, 1])

with col_left:
    fig, _ = plot_confusion_matrix(y, y_pred)
    st.pyplot(fig)
    
    st.info("""
    **Interpreting the Percentages:**
    - **Top value (Row %):**  
      *"Of all clients who actually [Row Label], what % did we predict as [Col Label]?"*
      - For **Actual Default -> Pred Default**, this is **Recall**.
    - **Bottom value in brackets (Col %):**  
      *"Of all clients we predicted as [Col Label], what % actually [Row Label]?"*
      - For **Pred Default -> Actual Default**, this is **Precision**.
    """)

with col_right:
    st.write("### Raw Counts")
    st.dataframe(pd.DataFrame(cm, 
                 columns=['Pred No Default', 'Pred Default'], 
                 index=['Actual No Default', 'Actual Default']))
    
    st.write("### Details")
    st.write(f"**False Negatives (Missed Defaults):** {fn:,}")
    st.write(f"**False Positives (False Alarms):** {fp:,}")
    st.write(f"**True Positives (Caught Defaults):** {tp:,}")
    st.write(f"**True Negatives (Correct Approvals):** {tn:,}")

# Histogram of Probabilities
st.subheader("Predicted Probability Distribution")
fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
sns.histplot(x=y_proba, hue=y, bins=50, kde=True, element="step", stat="density", common_norm=False, ax=ax_hist, palette=["green", "red"])
ax_hist.axvline(threshold, color='black', linestyle='--', label=f'Threshold {threshold}')
ax_hist.set_xlabel("Predicted Probability of Default")
ax_hist.legend(title="Actual Target", labels=['Default (1)', 'No Default (0)'])
st.pyplot(fig_hist)