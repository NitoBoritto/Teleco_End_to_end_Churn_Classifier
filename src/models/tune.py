import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

# Custom transformers and pipeline
from src.data.preprocess import preprocess_data
from src.features import build_features
from src.utils import validate_data

def tune_model(X, y):
    """
    Hyperparameter tuning using Optuna
    
    """
    # Feature engineering
    feature_engineer = build_features(X, target_col = y)
    
    # Tuning function
    def objective(trial):
        params = {
            'C': trial.suggest_float('C', .001, 1, log = True)
        }
        
        # Setting up cross validation
        skf = StratifiedKFold(n_splits = 5, 
                            shuffle = True,
                            random_state = 30)
        
        # Building the model
        model = LogisticRegression(**params,
                                   class_weight = 'balanced',
                                   random_state = 30,
                                   max_iter = 1000)
        
        # Adding the model to the pipeline
        ml_pipeline = Pipeline([
            ('features', feature_engineer),
            ('lgr', model)
        ])
        
        # Evaluting the model
        scores = cross_val_score(ml_pipeline, X, y, cv = skf, scoring = 'recall', n_jobs = -1)
        return scores.mean()
    
    study = optuna.create_study(direction = 'maximize')
    study.optimize(objective, n_trials = 20)
    
    print(f'Best parameters: {study.best_params}')
    return study.best_params