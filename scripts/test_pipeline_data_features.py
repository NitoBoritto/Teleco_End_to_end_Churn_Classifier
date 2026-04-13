import os
import pandas as pd
import sys

sys.path.append(os.path.abspath('src'))

from src.data.load_data import load_data                   # Data loading with error handling
from src.data.preprocess import preprocess_data            # Basic data cleaning
from src.features.build_features import build_features     # Feature engineering

Data_path = r'data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv'

def main():
    print("=== Testing Phase 1: Load → Preprocess → Build Features ===")
    
    # 1️⃣ Load data
    print('\n[1] Loading data...')
    df = load_data(Data_path)
    print(f"✅ Data loaded. Shape: {df.shape}")
    print(df.head(3))
    
    # 2️⃣ Preprocess
    print("\n[2] Preprocessing data...")
    df_clean = preprocess_data(df)
    print(f"Data after preprocessing. Shape: {df_clean.shape}")
    print(df_clean.head(3))
    
    # 3. Build Features
    print("\n[3] Building features...")
    df_features = build_features(df_clean)
    print(f"Data after feature engineering. Shape: {df_features.shape}")
    print(df_features.head(3))

    print("\n✅ Phase 1 pipeline completed successfully!")

if __name__ == "__main__":
    main()