"""
Quick Test: Advanced Features vs Baseline
Target: 0.82 ROC-AUC

This script quickly compares:
1. Baseline (189 features) - Current: 0.778 ROC-AUC
2. Advanced Features (~300+ features) - Target: 0.82 ROC-AUC
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import time

sys.path.append(str(Path(__file__).parent.parent))

from src.advanced_features import create_all_advanced_features

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("ADVANCED FEATURES TEST - TARGET: 0.82 ROC-AUC")
print("="*80)

# Load data
print("\nLoading data...")
data_dir = Path('data/processed')
X_train = pd.read_csv(data_dir / 'X_train.csv')
X_val = pd.read_csv(data_dir / 'X_val.csv')
y_train = pd.read_csv(data_dir / 'y_train.csv').squeeze()
y_val = pd.read_csv(data_dir / 'y_val.csv').squeeze()

print(f"Training: {X_train.shape}")
print(f"Validation: {X_val.shape}")

# Model
model = LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1
)

print("\n" + "="*80)
print("TEST 1: BASELINE FEATURES")
print("="*80)
print(f"Features: {X_train.shape[1]}")

# Train baseline
start = time.time()
model.fit(X_train, y_train)
baseline_time = time.time() - start

# Evaluate
y_pred_proba = model.predict_proba(X_val)[:, 1]
baseline_roc = roc_auc_score(y_val, y_pred_proba)

print(f"\nBaseline Performance:")
print(f"  ROC-AUC: {baseline_roc:.4f}")
print(f"  Training time: {baseline_time:.2f}s")

print("\n" + "="*80)
print("TEST 2: ADVANCED FEATURES")
print("="*80)

# Create advanced features
print("\nApplying advanced feature engineering...")
X_train_advanced = create_all_advanced_features(X_train.copy())
X_val_advanced = create_all_advanced_features(X_val.copy())

print(f"\nFeatures after engineering: {X_train_advanced.shape[1]}")
print(f"New features created: {X_train_advanced.shape[1] - X_train.shape[1]}")

# Handle any inf/nan from feature engineering
X_train_advanced = X_train_advanced.replace([np.inf, -np.inf], np.nan)
X_val_advanced = X_val_advanced.replace([np.inf, -np.inf], np.nan)

# Fill missing values for new features
X_train_advanced = X_train_advanced.fillna(X_train_advanced.median())
X_val_advanced = X_val_advanced.fillna(X_train_advanced.median())

# Train with advanced features
start = time.time()
model.fit(X_train_advanced, y_train)
advanced_time = time.time() - start

# Evaluate
y_pred_proba_adv = model.predict_proba(X_val_advanced)[:, 1]
advanced_roc = roc_auc_score(y_val, y_pred_proba_adv)

print(f"\nAdvanced Features Performance:")
print(f"  ROC-AUC: {advanced_roc:.4f}")
print(f"  Training time: {advanced_time:.2f}s")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"\nBaseline ROC-AUC:     {baseline_roc:.4f}")
print(f"Advanced ROC-AUC:     {advanced_roc:.4f}")
print(f"Improvement:          {advanced_roc - baseline_roc:.4f} ({(advanced_roc - baseline_roc)/baseline_roc*100:.1f}%)")
print(f"\nTarget:               0.8200")
print(f"Gap to target:        {0.82 - advanced_roc:.4f}")

if advanced_roc >= 0.82:
    print(f"\nSUCCESS! Target of 0.82 ROC-AUC achieved!")
elif advanced_roc >= 0.80:
    print(f"\nCLOSE! With hyperparameter tuning, 0.82 is achievable.")
else:
    print(f"\nNEXT STEPS: Try hyperparameter optimization and feature selection.")

print("\n" + "="*80)
print("TOP 20 MOST IMPORTANT FEATURES")
print("="*80)

feature_importance = pd.DataFrame({
    'feature': X_train_advanced.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n", feature_importance.head(20).to_string(index=False))

# Check for EXT_SOURCE dominance
ext_features = feature_importance[feature_importance['feature'].str.contains('EXT_SOURCE', case=False)]
print(f"\nEXT_SOURCE features in top 20: {len(ext_features.head(20))}")

print("\n" + "="*80)
