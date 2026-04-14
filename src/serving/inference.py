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
from src.data.preprocess import preprocess_data
# Model loading configuration
# IMPORTANT: This path is set during Docker container build
# In development: uses local MLflow artifacts
# In production: uses model copied to container at build time

model_dir = '/app/model'
artifacts_dir = './artifacts'

# Load the trained model in Mflow pyfunc format
try:
    
    model = mlflow.pyfunc.load_model(model_dir)
    print(f'✅ Model loaded successfully from {model_dir}')
except Exception as e:
    print(f'❌ Failed to load model from {model_dir}: {e}')
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

# Load transformer pipeline
try:
    transformers = joblib.load(os.path.join(artifacts_dir, 'transformers.pkl'))
    print('✅ Fitted transformers loaded successfully')
except Exception as e:
    raise Exception(f'Failed to load transformers: {e}')


def service_transform(df : pd.DataFrame) -> pd.DataFrame:
    """
    Apply identical feature transformations as used during model training.
    
    This function is CRITICAL for production ML - it ensures that features are
    transformed exactly as they were during training to prevent train/serve skew.
    
    """
    
    # Clean raw data
    df_cleaned = preprocess_data(df)
    
    df_transformed = transformers.transform(df_cleaned)
    
    return df_transformed


def predict(input_dict: dict) -> str:
    """
    Where the predictions happen
    
    """
    df = pd.DataFrame([input_dict])
    
    df_final = service_transform(df)
    
    prediction = model.predict(df_final)
    
    return 'Likely to churn' if prediction[0] == 1 else 'Not likely to churn'