"""
Inference pipeline - Production ML Model Serving with Feature Consistency

This module provides the core inference functionality for the Telco Churn prediction model.
It ensures that serving-time feature transformations exactly match training-time transformations,
which is CRITICAL for model accuracy in production.

Key Responsibilities:
1. Load MLflow-logged model and feature metadata from training
2. Apply identical feature transformations as used during training
3. Ensure correct feature ordering for model input
4. Convert model predictions to user-friendly output

"""

import os
import pandas as pd
import mlflow
import glob
import joblib

# Model loading configuration
# IMPORTANT: This path is set during Docker container build
# In development: uses local MLflow artifacts
# In production: uses model copied to container at build time

model_dir = '/app/model'

try:
    # Load the trained model in Mflow pyfunc format
    model = mlflow.pyfunc.load_model(model_dir)
    print('f✅ Model loaded successfully from {model_dr}')
except Exception as e:
    print(f'❌ Failed to load mdel from {model_dir}: {e}')
    # If this happens fallback for local development
    try:
        # Loading from local Mlflow tracking
        local_model_path = glob.glob('./mlruns/*/*/artifacts/model')
        if local_model_path:
            latest_model = max(local_model_path, key = os.path.getmtime)
            model = mlflow.pyfunc.load_model(latest_model)
            model_dir = latest_model
            print(f'✅ Fallback: Loaded model from {latest_model}')
        else:
            raise Exception('No model found in local mlruns')
    except Exception as fallback_error:
        raise Exception(f'Failed to load model: {e}. Fallback failed: {fallback_error}')

try:
    feature_file = os.path.join(model_dir, 'feature_columns.txt')
    with open(feature_file) as f:
        feature_cols = [ln.strip() for ln in f if ln.strip()]
    print(f'✅ Loaded {len(feature_cols)} feature columns from training')
except Exception as e:
    raise Exception(f'Failed to load feature columns: {e}')

# Feature transformation constraints
binary_map = {
    'bin_gender': {'Female': 0, 'Male': 1},
    'bin_Partner': {'No': 0, 'Yes': 1},
    'bin_Dependents': {'No': 0, 'Yes': 1},
    'bin_PhoneService': {'No': 0, 'Yes': 1},
    'bin_PaperlessBilling': {'No': 0, "Yes": 1}
}

numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

def service_transform(df : pd.DataFrame) -> pd.DataFrame:
    """
    Apply identical feature transformations as used during model training.
    
    This function is CRITICAL for production ML - it ensures that features are
    transformed exactly as they were during training to prevent train/serve skew.
    
    """
    df = df.copy()
    df.columns = df.columns.str.strip()