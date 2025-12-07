# Project Scripts

This directory contains all scripts for the ML pipeline, analysis, and setup.

## 📂 pipeline/
**Core end-to-end workflow scripts.** Run these in order to reproduce the project results.

1.  `select_best_model.py`: Compares baseline models (Dummy, LR, RF, XGB, LGBM) on raw data.
2.  `run_feature_experiments.py`: Tests feature engineering strategies (Domain, Poly) and sampling strategies (SMOTE, Balanced).
3.  `optimize_model.py`: Tunes hyperparameters for the best configuration (LGBM + Domain + Balanced) to maximize F3.2 Score.
4.  `apply_best_model.py`: Retrains the optimized model on full data, predicts on test set, and generates final artifacts.

## 📂 analysis/
Exploratory scripts for understanding data and model behavior.
- `analyze_overfitting.py`
- `investigate_smote.py`

## 📂 setup/
Scripts to initialize the project environment and data.
- `create_processed_data.py`: Generates `data/processed/` from raw CSVs.
- `create_notebooks.py`: Regenerates notebook files.

## 📂 legacy/
Old or superseded scripts.
