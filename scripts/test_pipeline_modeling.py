import time
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # Standard version for best_params
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, recall_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Your custom modules
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.models.tune import tune_model  # The Optuna module we just fixed

# Configuration
Data_path = r'data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv'
threshold = 0.35

def main():
    print("=== Phase 2: Modeling with Logistic Regression (Tuned) ===")
    
    

    # 1️⃣ Data Preparation
    df = load_data(Data_path)
    df = preprocess_data(df)

    # Ensure target is numeric for LogisticRegression
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"Target distribution in training set:\n{y_train.value_counts()}")

    # 2️⃣ Hyperparameter Tuning
    print("🎯 Tuning started...")
    # This finds the best 'C' using Optuna
    best_params = tune_model(X_train, y_train) 
    print(f"✅ Best Params Found: {best_params}")

    # 3️⃣ Build Final Pipeline
    # We define the feature engineer specifically for the training set
    feature_engineer = build_features(X_train)

    # Use standard LogisticRegression with the best params
    model = LogisticRegression(
        **best_params, 
        class_weight='balanced', 
        random_state=42, 
        max_iter=1000
    )

    ml_pipeline = Pipeline([
        ('features', feature_engineer),
        ('lgr', model)
    ])

    # 4️⃣ Final Fit and Evaluation
    print("Training final champion model...")
    start_time = time.time()
    ml_pipeline.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"✅ Model trained in {train_time:.2f} seconds")
    
    # Predict probabilities for the threshold
    start_time = time.time()
    y_proba = ml_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    predict_time = time.time() - start_time
    print(f"✅ Model predicted in {predict_time:.2f} seconds")
    
    print("\n--- Final Test Results ---")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()