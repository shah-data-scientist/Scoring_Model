"""
Investigate why SMOTE performed poorly in experiments.

SMOTE experiments showed:
- Very low recall (0.016-0.026)
- High precision (0.50-0.53)
- Poor ROC-AUC (0.754-0.760)

Compared to Balanced:
- High recall (0.66-0.69)
- Lower precision (0.18)
- Better ROC-AUC (0.771-0.778)

This script investigates potential causes.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.sampling_strategies import get_sampling_strategy

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Model parameters (same as experiments)
BASE_MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': -1
}


def analyze_smote_distribution(X_train, y_train):
    """Analyze the distribution of SMOTE synthetic samples."""
    print("="*80)
    print("SMOTE DISTRIBUTION ANALYSIS")
    print("="*80)

    # Apply SMOTE
    print("\nApplying SMOTE...")
    X_smote, y_smote, metadata = get_sampling_strategy('smote', X_train, y_train)

    print(f"\nOriginal minority class samples: {(y_train == 1).sum():,}")
    print(f"After SMOTE minority class samples: {(y_smote == 1).sum():,}")
    print(f"Synthetic samples created: {metadata['synthetic_samples_created']:,}")

    # Check if synthetic samples are different from original
    original_minority_indices = y_train[y_train == 1].index
    n_original = len(original_minority_indices)

    # Compare statistics of original vs synthetic minority samples
    original_minority = X_train.loc[y_train == 1]
    all_smote_minority = X_smote.loc[y_smote == 1]
    synthetic_minority = all_smote_minority.iloc[n_original:]  # After original samples

    print("\n" + "="*80)
    print("FEATURE STATISTICS COMPARISON")
    print("="*80)

    # Select a few key features to analyze
    key_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AMT_CREDIT', 'AMT_INCOME_TOTAL']
    available_features = [f for f in key_features if f in X_train.columns]

    if available_features:
        comparison = pd.DataFrame({
            'Feature': available_features,
            'Original_Mean': [original_minority[f].mean() for f in available_features],
            'Synthetic_Mean': [synthetic_minority[f].mean() for f in available_features],
            'Original_Std': [original_minority[f].std() for f in available_features],
            'Synthetic_Std': [synthetic_minority[f].std() for f in available_features],
        })

        comparison['Mean_Diff_%'] = ((comparison['Synthetic_Mean'] - comparison['Original_Mean']) /
                                       comparison['Original_Mean'].abs() * 100)

        print("\n", comparison.to_string(index=False))

    return X_smote, y_smote, original_minority, synthetic_minority


def compare_model_predictions(X_train, y_train, X_val, y_val):
    """Compare model predictions between Balanced and SMOTE."""
    print("\n" + "="*80)
    print("MODEL PREDICTION COMPARISON")
    print("="*80)

    results = {}

    for strategy in ['balanced', 'smote']:
        print(f"\n{'='*80}")
        print(f"Training with {strategy.upper()}")
        print(f"{'='*80}")

        # Apply sampling
        X_train_sampled, y_train_sampled, metadata = get_sampling_strategy(
            strategy, X_train, y_train, random_state=RANDOM_STATE
        )

        # Configure model
        model_params = BASE_MODEL_PARAMS.copy()
        if strategy == 'balanced':
            model_params['class_weight'] = 'balanced'

        # Train
        model = LGBMClassifier(**model_params)
        model.fit(X_train_sampled, y_train_sampled)

        # Predict on validation
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        # Calculate metrics
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        pr_auc = average_precision_score(y_val, y_pred_proba)

        print(f"\nValidation Performance:")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  PR-AUC:  {pr_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, target_names=['Class 0', 'Class 1']))

        # Analyze prediction distribution
        print(f"\nPrediction Distribution:")
        print(f"  Predicted Class 0: {(y_pred == 0).sum():,} ({(y_pred == 0).sum()/len(y_pred)*100:.2f}%)")
        print(f"  Predicted Class 1: {(y_pred == 1).sum():,} ({(y_pred == 1).sum()/len(y_pred)*100:.2f}%)")

        # Analyze probability distribution
        print(f"\nProbability Statistics:")
        print(f"  Mean probability for class 1: {y_pred_proba.mean():.4f}")
        print(f"  Median probability for class 1: {np.median(y_pred_proba):.4f}")
        print(f"  Std probability for class 1: {y_pred_proba.std():.4f}")
        print(f"  Max probability for class 1: {y_pred_proba.max():.4f}")
        print(f"  Min probability for class 1: {y_pred_proba.min():.4f}")

        # Check default threshold
        threshold_05 = (y_pred_proba >= 0.5).sum()
        print(f"  Samples with P(class=1) >= 0.5: {threshold_05:,} ({threshold_05/len(y_pred_proba)*100:.2f}%)")

        results[strategy] = {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'model': model,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }

    return results


def investigate_threshold_issue(results, y_val):
    """Investigate if SMOTE issue is threshold-related."""
    print("\n" + "="*80)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("="*80)

    for strategy, data in results.items():
        print(f"\n{strategy.upper()} Strategy:")
        y_pred_proba = data['y_pred_proba']

        # Test different thresholds
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        print("\nThreshold | Pred Class 1 | Recall | Precision | F1-Score")
        print("-" * 70)

        from sklearn.metrics import recall_score, precision_score, f1_score

        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)

            if (y_pred_thresh == 1).sum() > 0:  # Avoid division by zero
                recall = recall_score(y_val, y_pred_thresh)
                precision = precision_score(y_val, y_pred_thresh)
                f1 = f1_score(y_val, y_pred_thresh)
                pred_positive = (y_pred_thresh == 1).sum()

                print(f"{threshold:>9.1f} | {pred_positive:>12,} | {recall:>6.4f} | {precision:>9.4f} | {f1:>8.4f}")
            else:
                print(f"{threshold:>9.1f} | {0:>12} | {'N/A':>6} | {'N/A':>9} | {'N/A':>8}")


def main():
    """Run SMOTE investigation."""
    print("="*80)
    print("INVESTIGATING SMOTE POOR PERFORMANCE")
    print("="*80)

    # Load data
    print("\nLoading data...")
    data_dir = Path('data/processed')
    X_train = pd.read_csv(data_dir / 'X_train.csv')
    X_val = pd.read_csv(data_dir / 'X_val.csv')
    y_train = pd.read_csv(data_dir / 'y_train.csv').squeeze()
    y_val = pd.read_csv(data_dir / 'y_val.csv').squeeze()

    print(f"  Training: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Class distribution (train): {(y_train==0).sum():,} / {(y_train==1).sum():,}")

    # 1. Analyze SMOTE distribution
    X_smote, y_smote, original_minority, synthetic_minority = analyze_smote_distribution(X_train, y_train)

    # 2. Compare model predictions
    results = compare_model_predictions(X_train, y_train, X_val, y_val)

    # 3. Investigate threshold issue
    investigate_threshold_issue(results, y_val)

    print("\n" + "="*80)
    print("INVESTIGATION COMPLETE")
    print("="*80)

    print("\nKEY FINDINGS:")
    print("1. Compare feature statistics between original and synthetic samples")
    print("2. Check if SMOTE model predicts fewer positives than Balanced")
    print("3. Analyze if the issue is threshold-related or model calibration")
    print("\nPotential causes of poor SMOTE performance:")
    print("- Synthetic samples may be unrealistic in high-dimensional space (189 features)")
    print("- Model may be overfitting to training data (balanced classes)")
    print("- Model may be poorly calibrated, producing very low probabilities")
    print("- Default threshold (0.5) may be inappropriate for this use case")


if __name__ == "__main__":
    main()
