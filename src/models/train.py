# Libraries
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import recall_score, f1_score
from sklearn.pipeline import Pipeline

# Custom transformers and pipeline
from src.data.preprocess import preprocess_data
from src.features import build_features
from src.utils import validate_data

def train_model(df: pd.DataFrame, target_col: str = 'Churn', model_artifact_path: str = 'models/churn_model'):
    """
    Trains a Logistic Regression model with custom feature
    engineering and logs everything to MLflow.
    
    """
    
    # Validating data
    print('Validating data...')
    is_valid, issues = validate_data(df)
    if not is_valid:
        raise ValueError(f'Data validation failed: {issues}')
    print('✅ Validation passed')
    
    # Preprocessing
    print('Preprocessing...')
    df = preprocess_data(df, target_col = target_col)
    
    # Splitting
    print('Splitting data...')
    X = df.drop(columns = [target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = .2, random_state = 30, stratify = y
    )
    
    # Building the model
    
    feature_engineer = build_features(X_train, target_col = target_col)
    
    skf = StratifiedKFold(n_splits = 5, 
                          shuffle = True,
                          random_state = 30)
    
    model = LogisticRegressionCV(C = .4,
                                 class_weight = 'balanced',
                                 random_state = 30,
                                 cv = skf)
    
    ml_pipeline = Pipeline([
        ('features', feature_engineer),
        ('lgr', model)
    ])
    
    with mlflow.start_run():
        # Train model
        ml_pipeline.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        
        # Log params, metrics and model
        mlflow.log_param('C', .4)
        mlflow.log_metric('F1', f1)
        mlflow.log_metric('Recall',rec)
        mlflow.sklearn.log_model(ml_pipeline.named_steps('lgr'), 'model')
        
        # Log dataset so it shows in MLflow UI
        train_ds = mlflow.data.from_pandas(df, source = 'training_data')
        mlflow.log_input(train_ds, context = 'training')
        
        print(f'Model trained, F1 Score: {f1:2f}, Recall: {rec:.2f}')