import optuna
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.base import clone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Custom transformers and pipeline
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.utils import validate_data

def tune_model(X, y):
    """
    Hyperparameter tuning using Optuna
    
    """
    
    # Tuning function
    def objective(trial):
        params = {
            'C': trial.suggest_float('C', .001, 1, log = True)
        }
        
        # Setting up cross validation
        skf = StratifiedKFold(n_splits = 5, 
                            shuffle = True,
                            random_state = 30)
        
        # Get the UNFITTED feature pipeline
        preprocessor, multicollinear = build_features(X)
        
        # Create a FLAT pipeline with all transformers + model
        ml_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('multicollinear', multicollinear),
            ('lgr', LogisticRegression(
                **params,
                class_weight='balanced',
                random_state=30,
                max_iter=1000
            ))
        ])
        
        # Evaluting the model
        scores = cross_val_score(ml_pipeline,
                                 X, y,
                                 cv = skf,
                                 scoring = 'recall',
                                 n_jobs = 1,
                                 error_score = 'raise')
        return scores.mean()
    
    study = optuna.create_study(direction = 'maximize')
    study.optimize(objective, n_trials = 20)
    
    print(f'✅ Optimization complete.\nBest parameters: {study.best_params}')
    return study.best_params