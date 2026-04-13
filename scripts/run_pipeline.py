"""
This is where everything comes together

"""
# Importing libraries
import os
import sys
import time
import json
import joblib
import argparse
import pandas as pd
import mlflow.sklearn
from posthog import project_root
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Fix import path for local modules
# Allows imports from src/ directory structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing local modules
from src.data.load_data import load_data                   # Data loading with error handling
from src.data.preprocess import preprocess_data            # Basic data cleaning
from src.features.build_features import build_features     # Feature engineering
from src.utils.validate_data import validate_data          # Data quality validation
from src.models.tune import tune_model                     # Hyper parameter tuning

def main(args):
    """
    Main training pipeline function that orchestrates the complete ML worflow
    
    """
    
    # MLflow setup for expirement tracking
    # Configuring MLflow to use local file-based tracking instead of a tracking server
    project_root = sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    mlruns_path = args.mlflow_uri or f'file://{project_root}/mlruns' # Local file-based tracking
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_expreiment(args.expirement) # Creates experiment if it doesn't exist
    
    # Start Mlflow run
    # All subsequent logging will be tracking under this run
    with mlflow.start_run():
        # Log hyperparameters and config
        # Required for model reproducibility
        mlflow.log_param('model', 'lgr')  # Model type for comparison
        mlflow.log_param('threshold', args.threshold)    # Classification threshold
        mlflow.log_param('test_size', args.test_size)    # Train/test split ratio
        
        
        
        # 1️⃣ Data loading & validation
        print('Loading data...')
        df = load_data(args.input) # Loading data using module
        print(f'✅ Data loaded successfully {df.shape[0]} rows, {df.shape[1]} columns')
        
        # Validating data using module
        print('Validating data using Great Expectations...')
        is_valid, failed = validate_data(df)
        mlflow.log_metric('data_quality_pass', int(is_valid))
        
        # log validation failures for debugging
        if not is_valid:
            mlflow.log_test(json.dumps(failed, indent = 2), artifact_file = 'failed_expectations.json')
            raise ValueError(f'❌ Data quality check failed. Issues: {failed}')
        else :
            print('✅ Validation passed, logging to MLflow')
            
            
        
        
        # 2️⃣ Preprocessing
        print('Preprocessing data...')
        df = preprocess_data(df) # Basic cleaning using imported module
        # Saving processed dataset for reproducibility and debugging
        processed_path = os.path.join(project_root, 'data', 'processed', 'teleco_churn_processed.csv')
        # Making directory with that name
        os.makedirs(os.path.dirname(processed_path), exist_ok = True)
        # Generating dataset
        df.to_csv(processed_path, index = False)
        print(f'✅ Processed dataset saved to {processed_path} | Shape: {df.shape}')
        
        
        
        
        # 3️⃣ Splitting
        print('Splitting data...')
        # Checking target validity
        target = args.target
        if target not in df.columns:
            raise ValueError(f'Target column `{target}` not found in data')
        
        X = df.drop(columns = target)   # Features
        y = df[target]                  # Target
        
        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size = args.test_size, # Default is 20%
            stratify = y,
            random_state = 30
        )
        print(f"✅ Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
        
        
        
        
        # 4️⃣ Feature engineering
        print('Building features...')
        # Apply feature engineering transformations
        feature_engineer = build_features(X_train)
        # BE, Bool encoding, OHE, and dropping correlated features using module
        
        
        
         
        # 5️⃣ Model training with optimized hyperparameters
        print('Training model pipeline...')
        best_params = tune_model(X_train, y_train)
        
        # Log the best params to MLflow so you can see them in the UI
        mlflow.log_params(best_params)
        
        print('Building final model with best parameters...')
        
        # Building the model
        model = LogisticRegression(**best_params,
                                    class_weight = 'balanced',
                                    random_state = 30,
                                    n_jobs = -1,
                                    max_iter = 1000)
        
        # Adding the model to the pipeline
        ml_pipeline = Pipeline([
            ('features', feature_engineer),
            ('lgr', model)
        ])
        
        # Train modela dn track performance
        t0 = time.time()
        ml_pipeline.fit(X_train, y_train)
        train_time = time.time() - t0
        mlflow.log_metric('train_time', train_time) # Tracking training performance
        print(f"✅ Model trained in {train_time:.2f} seconds")
        
        # Saving feature metadata for serving consistency
        artifacts_dir = os.path.join(project_root, 'artifacts')
        os.makedirs(artifacts_dir, exist_ok = True)
        
        # Get feature columns
        feature_cols = ml_pipeline.named_steps['features'].get_feature_names_out().tolist()
        
        # Save locally for development serving
        with open(os.path.join(artifacts_dir, 'feature_columns.json'), 'w') as f:
            json.dump(feature_cols, f)
            
        # Log to Mlflow for production serving
        mlflow.log_text('\n'.join(feature_cols), artifact_file = 'feature_columns.txt')
        
        # Save preprocessing artifacts for serving pipeline
        preprocessing_artifact = {
            'feature_columns': feature_cols,    # Exact feature order
            'target':target                     # Target column name
        }
        joblib.dump(preprocessing_artifact, os.path.join(artifacts_dir, 'preprocessing.pkl'))
        mlflow.log_artifact(os.path.join(artifacts_dir, 'preprocessing.pkl'))
        print(f'✅ Saved {len(feature_cols)} feature columns for serving consistency')
        
        
        
        
        # 6️⃣ Model evaluation
        print('Evaluating model performance')
        
        # Tracking predictions inference time
        t1 = time.time()
        y_proba = ml_pipeline.predict_proba(X_test)[:, 1] # Focusing on churners
        
        # Applying classification threshold
        y_pred = (y_proba >= args.threshold).astype(int)
        pred_time = time.time() - t1
        mlflow.log_metric('pred_time', pred_time) # Tracking prediction performance
        
        # Logging metrics
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        # Log all metrics for expirement tracking
        mlflow.log_metric('precision', prec)
        mlflow.log_metric('recall', rec)
        mlflow.log_metric('f1', f1)
        mlflow.log_metric('roc_auc', roc_auc)
        
        print("Model Performance:")
        print(f"Precision: {prec:.2f} | Recall: {rec:.2f}")
        print(f"F1 Score: {f1:.2f}    | ROC AUC: {roc_auc:.2f}")
        
        
        
        
        # 7️⃣ Model serializaation and logging
        print('Saving model to MLflow...')
        mlflow.sklearn.log_model(   # Log model in MLflow's standard format for serving
            ml_pipeline,
            artifact_path = 'model' # This creates a 'model/' folder in MLflow run artifacts
        )
        
        
        
        
        # 8️⃣ Final Performance Summary
        print('\nPerformance Summary:')
        print(f'Training time: {train_time:.2f}s')
        print(f'Inference time: {pred_time:.2f}s')
        print(f'Samples per second: {len(X_test)/pred_time:.0f}')
        print('-----------------------------------------------')
        print('Detailed Classification Report:')
        print(classification_report(y_test, y_pred, digits=3))
        
        
        
        
    if __name__ == '__main__':
        p = argparse.ArgumentParser(description = 'Run churn pipeline with LogisticRegression & MLflow')
        p.add_argument('--input', type = str, required = True,
                       default = r'data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv',
                       help = 'path to CSV (e.g., data/raw/Telco-Customer-Churn.csv)')
        p.add_argument('--target', type = str, default = 'Churn')
        p.add_argument('--threshold', type = float, default = 0.35)
        p.add_argument('--test_size', type = float, default = 0.2)
        p.add_argument('--experiment', type = str, default = 'Telco Churn')
        p.add_argument('--mlflow_uri', type = str, default = None,
                       help = 'override Mlflow tracking URI, else uses project_root/mlruns')
        
        args = p.parse_args()
        main(args)
        
"""
# Use this below to run the pipeline:

python scripts/run_pipeline.py \                                            
    --input data/raw/Telco-Customer-Churn.csv \
    --target Churn

"""