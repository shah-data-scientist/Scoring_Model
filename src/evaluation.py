"""
Model Evaluation Utilities

Functions for comprehensive model evaluation on imbalanced datasets.
Includes metrics, visualizations, and comparison tools.

Educational focus on understanding WHY certain metrics matter for credit scoring!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score, accuracy_score
)
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


def evaluate_model(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  y_pred_proba: Optional[np.ndarray] = None,
                  model_name: str = "Model") -> Dict[str, float]:
    """
    Comprehensive model evaluation with focus on imbalanced classification metrics.

    Educational Note:
    -----------------
    For imbalanced datasets (like credit scoring), DON'T rely on accuracy!

    **Key Metrics:**

    1. **ROC-AUC (0.5-1.0):** Ability to rank predictions
       - 0.5 = Random guessing
       - 1.0 = Perfect classifier
       - Good for comparing models overall

    2. **Precision-Recall AUC:** Better for imbalanced data
       - Focuses on positive class (defaulters)
       - More informative than ROC-AUC when classes are skewed

    3. **Precision:** Of predicted defaults, how many are correct?
       - High precision = Low false alarms
       - Important if rejecting good customers is costly

    4. **Recall (Sensitivity):** Of actual defaults, how many did we catch?
       - High recall = Catching most defaulters
       - Important if missing defaulters is costly

    5. **F1-Score:** Harmonic mean of precision and recall
       - Balances both concerns
       - Good single metric for imbalanced data

    **Business Context:**
    - False Positive = Reject good customer (lost business)
    - False Negative = Approve bad customer (financial loss)
    - Choose metric based on which error is more costly!

    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels (binary)
    y_pred_proba : np.ndarray, optional
        Predicted probabilities for positive class
    model_name : str
        Name of the model for display

    Returns:
    --------
    Dict[str, float]
        Dictionary with all evaluation metrics

    Example:
    --------
    >>> metrics = evaluate_model(y_val, y_pred, y_pred_proba, "Random Forest")
    >>> print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    """
    metrics = {}

    print("=" * 80)
    print(f"EVALUATION RESULTS - {model_name}")
    print("=" * 80)

    # Basic metrics (work with binary predictions)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

    print(f"\\n📊 Classification Metrics:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f} ⚠️  (Can be misleading for imbalanced data!)")
    print(f"   Precision: {metrics['precision']:.4f} (Of predicted defaults, % correct)")
    print(f"   Recall:    {metrics['recall']:.4f} (Of actual defaults, % caught)")
    print(f"   F1-Score:  {metrics['f1']:.4f} (Balance of precision & recall)")

    # Probability-based metrics (need predict_proba)
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)

        print(f"\\n📈 Probability-Based Metrics:")
        print(f"   ROC-AUC:    {metrics['roc_auc']:.4f} (Overall ranking ability)")
        print(f"   PR-AUC:     {metrics['pr_auc']:.4f} (Better for imbalanced data)")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics['true_negatives'] = tn
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn
    metrics['true_positives'] = tp

    print(f"\\n📋 Confusion Matrix:")
    print(f"   True Negatives (TN):  {tn:,} (Correctly predicted non-defaults)")
    print(f"   False Positives (FP): {fp:,} (Good customers rejected)")
    print(f"   False Negatives (FN): {fn:,} (Bad customers approved)")
    print(f"   True Positives (TP):  {tp:,} (Correctly predicted defaults)")

    # Business metrics
    total = len(y_true)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    metrics['false_positive_rate'] = fpr
    metrics['false_negative_rate'] = fnr

    print(f"\\n💼 Business Impact:")
    print(f"   False Positive Rate: {fpr:.2%} (% of good customers rejected)")
    print(f"   False Negative Rate: {fnr:.2%} (% of bad customers approved)")

    print("=" * 80)

    return metrics


def plot_roc_curve(y_true: np.ndarray,
                  y_pred_proba: np.ndarray,
                  model_name: str = "Model",
                  ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Plot ROC (Receiver Operating Characteristic) curve.

    Educational Note:
    -----------------
    **What is ROC Curve?**
    Shows trade-off between:
    - True Positive Rate (Recall): How many defaults we catch
    - False Positive Rate: How many non-defaults we falsely flag

    **How to Read:**
    - Diagonal line = Random guessing (AUC = 0.5)
    - Closer to top-left corner = Better model
    - AUC (Area Under Curve) = Overall performance metric

    **When to Use:**
    - Comparing multiple models
    - Understanding threshold trade-offs
    - General performance visualization

    **Limitation:**
    - Can be optimistic for imbalanced data
    - Use Precision-Recall curve as complement

    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities for positive class
    model_name : str
        Model name for title
    ax : plt.Axes, optional
        Matplotlib axis to plot on

    Returns:
    --------
    plt.Figure
        Figure object

    Example:
    --------
    >>> plot_roc_curve(y_val, model.predict_proba(X_val)[:, 1], "Random Forest")
    >>> plt.show()
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    # Plot ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'{model_name} (AUC = {roc_auc:.4f})')

    # Plot random baseline
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Random Classifier (AUC = 0.50)')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
    ax.set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    return fig


def plot_precision_recall_curve(y_true: np.ndarray,
                                 y_pred_proba: np.ndarray,
                                 model_name: str = "Model",
                                 ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Plot Precision-Recall curve.

    Educational Note:
    -----------------
    **What is Precision-Recall Curve?**
    Shows trade-off between:
    - Precision: Of predicted defaults, how many are correct?
    - Recall: Of actual defaults, how many did we catch?

    **Why It's Better for Imbalanced Data:**
    - Focuses on positive class (defaults) performance
    - Not affected by large number of true negatives
    - More informative than ROC when classes are skewed

    **How to Read:**
    - Higher curve = Better model
    - Perfect model = curve at top-right corner
    - Baseline = % of positive class in dataset

    **Use This When:**
    - Dealing with imbalanced data (like credit scoring!)
    - Positive class is more important
    - Want realistic performance assessment

    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities for positive class
    model_name : str
        Model name for title
    ax : plt.Axes, optional
        Matplotlib axis to plot on

    Returns:
    --------
    plt.Figure
        Figure object

    Example:
    --------
    >>> plot_precision_recall_curve(y_val, model.predict_proba(X_val)[:, 1])
    >>> plt.show()
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    baseline = y_true.sum() / len(y_true)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    # Plot PR curve
    ax.plot(recall, precision, color='darkorange', lw=2,
            label=f'{model_name} (AUC = {pr_auc:.4f})')

    # Plot baseline
    ax.axhline(y=baseline, color='navy', linestyle='--', lw=2,
               label=f'Baseline (No Skill = {baseline:.4f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    return fig


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          model_name: str = "Model",
                          normalize: bool = False,
                          ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Plot confusion matrix heatmap.

    Educational Note:
    -----------------
    **Confusion Matrix Structure:**
    ```
                    Predicted
                    0 (No Default)  1 (Default)
    Actual  0 (No)  TN              FP
            1 (Yes) FN              TP
    ```

    **Interpretation:**
    - TN (True Negative): Correctly predicted non-defaults ✅
    - FP (False Positive): Falsely predicted default (rejected good customer) ❌
    - FN (False Negative): Falsely predicted non-default (approved bad customer) ❌❌
    - TP (True Positive): Correctly predicted defaults ✅

    **Business Impact:**
    - FP = Lost business opportunity
    - FN = Financial loss from bad loans
    - Usually FN is more costly in credit scoring!

    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    model_name : str
        Model name for title
    normalize : bool
        Whether to normalize values
    ax : plt.Axes, optional
        Matplotlib axis to plot on

    Returns:
    --------
    plt.Figure
        Figure object

    Example:
    --------
    >>> plot_confusion_matrix(y_val, y_pred, "Random Forest", normalize=True)
    >>> plt.show()
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title_suffix = "(Normalized)"
    else:
        fmt = 'd'
        title_suffix = "(Counts)"

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})

    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_name} {title_suffix}',
                 fontsize=14, fontweight='bold')
    ax.set_yticklabels(['No Default (0)', 'Default (1)'], rotation=0)
    ax.set_xticklabels(['No Default (0)', 'Default (1)'], rotation=0)

    return fig


def compare_models(results: Dict[str, Dict[str, float]],
                  metric: str = 'roc_auc') -> pd.DataFrame:
    """
    Compare multiple models side-by-side.

    Educational Note:
    -----------------
    When comparing models, consider:
    1. **Primary metric:** ROC-AUC or PR-AUC for imbalanced data
    2. **Business metric:** F1-score or custom cost function
    3. **Interpretability:** Simpler models might be preferred if performance is close
    4. **Training time:** For production, efficiency matters
    5. **Robustness:** How does it perform on different data splits?

    **Model Selection Strategy:**
    - Start with simple baseline (Logistic Regression)
    - Try tree-based models (Random Forest, XGBoost)
    - Compare using cross-validation
    - Consider ensemble of best models

    Parameters:
    -----------
    results : Dict[str, Dict[str, float]]
        Dictionary mapping model names to their metrics
        Example: {'LogisticRegression': {'roc_auc': 0.75, ...}, ...}
    metric : str
        Primary metric to sort by

    Returns:
    --------
    pd.DataFrame
        Comparison table sorted by primary metric

    Example:
    --------
    >>> results = {
    ...     'Logistic Regression': metrics_lr,
    ...     'Random Forest': metrics_rf,
    ...     'XGBoost': metrics_xgb
    ... }
    >>> comparison = compare_models(results, metric='roc_auc')
    >>> print(comparison)
    """
    # Convert to DataFrame
    df = pd.DataFrame(results).T

    # Sort by specified metric (descending)
    if metric in df.columns:
        df = df.sort_values(metric, ascending=False)

    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(df.to_string())
    print("=" * 80)

    # Highlight best model
    if metric in df.columns:
        best_model = df.index[0]
        best_score = df.loc[best_model, metric]
        print(f"\\n🏆 Best Model: {best_model}")
        print(f"   {metric.upper()}: {best_score:.4f}")

    return df


def plot_feature_importance(feature_names: list,
                           importances: np.ndarray,
                           top_n: int = 20,
                           model_name: str = "Model") -> plt.Figure:
    """
    Plot feature importance from tree-based models.

    Educational Note:
    -----------------
    **Feature Importance** tells us which features the model relies on most.

    **For Tree-Based Models:**
    - Measures how much each feature reduces impurity (Gini/entropy)
    - Higher importance = More useful for making predictions

    **Interpretation:**
    - High importance doesn't mean causation!
    - Correlated features share importance
    - Consider domain knowledge when interpreting

    **Use Cases:**
    - Feature selection (remove low-importance features)
    - Model interpretation (explain to stakeholders)
    - Feature engineering (create more features like important ones)
    - Business insights (what drives defaults?)

    Parameters:
    -----------
    feature_names : list
        List of feature names
    importances : np.ndarray
        Feature importance values
    top_n : int
        Number of top features to display
    model_name : str
        Model name for title

    Returns:
    --------
    plt.Figure
        Figure object

    Example:
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> model = RandomForestClassifier().fit(X_train, y_train)
    >>> plot_feature_importance(X_train.columns, model.feature_importances_)
    >>> plt.show()
    """
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)

    # Plot
    fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.3)))
    sns.barplot(data=importance_df, y='feature', x='importance', palette='viridis', ax=ax)

    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances - {model_name}',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    return fig


# Example usage
if __name__ == "__main__":
    print("Model Evaluation Utilities")
    print("=" * 80)
    print("\\nThis module provides functions for:")
    print("  ✅ Comprehensive model evaluation")
    print("  ✅ ROC and Precision-Recall curves")
    print("  ✅ Confusion matrix visualization")
    print("  ✅ Model comparison")
    print("  ✅ Feature importance plotting")
    print("\\nImport in notebooks:")
    print("  from src.evaluation import evaluate_model, plot_roc_curve, compare_models")
