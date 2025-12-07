import pandas as pd
from sklearn.metrics import confusion_matrix

try:
    print("Loading results/train_predictions.csv...")
    df = pd.read_csv('results/train_predictions.csv')
    print(f"Loaded {len(df)} rows.")
    
    y_true = df['TARGET']
    y_proba = df['PROBABILITY']
    
    threshold = 0.328
    y_pred = (y_proba >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix at {threshold}:")
    print(cm)
    
    tn, fp, fn, tp = cm.ravel()
    print(f"\nTP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}")
    
    cost = 10 * fn + 1 * fp
    print(f"Business Cost: {cost}")
    
    print("\n[SUCCESS] Data is valid for dashboard.")
    
except Exception as e:
    print(f"\n[ERROR] {e}")

