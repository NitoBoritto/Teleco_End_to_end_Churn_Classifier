# Libraries
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import recall_score, f1_score
from sklearn.pipeline import Pipeline

# Custom transformers and pipeline
from src.features import build_features

def train_model(df: pd.DataFrame, target_col: str = 'Churn'):
    """
    Trains a Logistic Regression model with custom feature
    engineering and logs everything to MLflow.
    
    """
    
    # Splitting
    print('Splitting data...')
    X = df.drop(columns = [target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = .2, random_state = 30, stratify = y
    )
    
    # Feature engineering
    feature_engineer = build_features(X_train, target_col = target_col)
    
    # Setting up cross validation
    skf = StratifiedKFold(n_splits = 5, 
                          shuffle = True,
                          random_state = 30)
    
    # Building the model
    model = LogisticRegressionCV(C = .4,
                                 class_weight = 'balanced',
                                 random_state = 30,
                                 cv = skf)
    
    # Adding the model to the pipeline
    ml_pipeline = Pipeline([
        ('features', feature_engineer),
        ('lgr', model)
    ])
    
    with mlflow.start_run():
        # Train model
        ml_pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = ml_pipeline.predict(X_test)
        
        # Calculate metrics
        f1 = f1_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        
        # Log params, metrics and model
        mlflow.log_params(model.get_params())
        mlflow.log_metric('F1', f1)
        mlflow.log_metric('Recall',rec)
        
        # Log model pipeline
        mlflow.sklearn.log_model(ml_pipeline, 'model_pipeline')
        
        # Log dataset so it shows in MLflow UI
        train_ds = mlflow.data.from_pandas(df, source = 'training_data')
        mlflow.log_input(train_ds, context = 'training')
        
        print(f'Model trained, F1 Score: {f1:2f}, Recall: {rec:.2f}')