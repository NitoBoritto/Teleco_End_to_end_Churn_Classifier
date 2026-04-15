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
import mlflow.sklearn
import glob

from src.utils.validate_data import validate_data

# Model loading configuration
# IMPORTANT: This path is set during Docker container build
# In development: uses local MLflow artifacts
# In production: uses model copied to container at build time

model_dir = '/app/model'

# Load the full ml pipeline in Mflow pyfunc format
try:
    
    model = mlflow.sklearn.load_model(model_dir)
    print(f'✅ Model loaded successfully from {model_dir}')
except Exception as e:
    print(f'❌ Failed to load model from {model_dir}: {e}')
    # If this happens fallback for local development
    try:
        # Loading from local Mlflow tracking
        local_model_path = glob.glob('./src/serving/model')
        if local_model_path:
            latest_model = max(local_model_path, key = os.path.getmtime)
            model = mlflow.sklearn.load_model(latest_model)
            model_dir = latest_model
            print(f'✅ Fallback: Loaded model from {latest_model}')
        else:
            raise Exception('No model found in local mlruns')
    except Exception as fallback_error:
        raise Exception(f'Failed to load model: {e}. Fallback failed: {fallback_error}')


def predict(input_dict: dict) -> str:
    """
    Where the predictions happen
    
    """
     
    df = pd.DataFrame([input_dict])
    
    if 'customerID' not in df.columns:
        df['customerID'] = 'ST-USER-0001'
    
    is_valid, error_message = validate_data(df)
    
    if not is_valid:
        raise Exception(f'Data validation failed: {error_message}')
    
    expected_order = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
            'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
            'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
        ]
    
    df = df[expected_order]
    
    prediction = model.predict(df)
    probability = model.predict_proba(df)
    
    churn_probability = float(probability[0][1])
    prediction_label = 'Likely to churn' if prediction[0] == 1 else 'Not likely to churn'
    
    return {'prediction': prediction_label,
            'probability': round(churn_probability * 100, 2)}