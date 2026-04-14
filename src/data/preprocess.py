import pandas as pd
import numpy as np

def preprocess_data(df: pd.DataFrame, target_col: str = 'Churn') -> pd.DataFrame:
    """
    All cleaning steps made
    - trim columns names
    - drop ID columns
    - fix TotalCharges to numeric
    - map target Churn to 0/1 if needed
    - NA handling
    """
    df.columns = df.columns.str.strip()
    
    # Drop ID column if present
    for col in ['customerID', 'CustomerID', 'customer_df']:
        if col in df.columns:
            df = df.drop(columns = [col])
    
    # Binary encoding target variable        
    if target_col in df.columns and df[target_col].dtype == 'object':
        df[target_col] = df[target_col].str.strip()
        df[target_col] = df[target_col].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    
    # TotalCharges to numeric    
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce')
    
    # Fixing SeniorCitizen feature
    if 'SeniorCitizen' in df.columns:
        df['SeniorCitizen'] = df['SeniorCitizen'].fillna(0).astype(int)
        
    # NA strategy:
    # - numeric: fill with 0
    # - others: leave for encoders to handle
    num_cols = df.select_dtypes(include = 'number').columns
    df[num_cols] = df[num_cols].fillna(0)

    return df