import pandas as pd
import sys
import os

# Add project root to path to ensure we can import src if needed
sys.path.append(os.getcwd())

try:
    df = pd.read_csv('data/application_train.csv', usecols=['CODE_GENDER'])
    print(f"Unique values in CODE_GENDER: {df['CODE_GENDER'].unique()}")
    print(f"Value counts:\n{df['CODE_GENDER'].value_counts()}")
except Exception as e:
    print(f"Error reading file: {e}")

