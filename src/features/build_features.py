import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer

class custom_binary_encoder(BaseEstimator, TransformerMixin):
    """
    Custom binary encoding transformer
    
    This class implements binary encoding logic that converts categorical Yes/No or Male/Female
    into 1/0 integers. the mappings must be consistent between training and serving
    
    """
    def __init__(self, binary_cols):
        self.binary_cols = binary_cols
        self.mapping = {'Yes': 1, 'No': 0, 'Female': 0, 'Male': 1}
        
    def fit(self, X, y = None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.binary_cols:
            X[col] = X[col].map(self.mapping).fillna(0).astype(int)
        return X

def build_features(df: pd.DataFrame, target_col: str = 'Churn') -> pd.DataFrame:
    """
    Appylying custom transformers and encoders to data
    
    This function seperates features from the dataframe to encode,
    drop nulls, and scale features depending on their type. Returns the
    result in a preprocessor pipeline to be used with the model
        
    """
    # Obtaining str columns to encode
    obj_cols = [c for c in df.select_dtypes(include = 'object').columns if c != target_col]    
    # If binary col then we use the custom encoder   
    binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
    # if 3 or more we ohe
    multi_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]
    # we scale all other numerical using RobustScaler
    numeric_cols = [c for c in df.select_dtypes(include = 'number').columns if c != target_col]
    
    # Preprocessor 
    preprocessor = ColumnTransformer(transformers = [
        ('bin', custom_binary_encoder(binary_cols), binary_cols),
        ('ohe', OneHotEncoder(drop = 'first'), multi_cols),
        ('scaler', RobustScaler(), numeric_cols)
    ], remainder = 'passthrough')
    
    # Returning the transformers and scalers with it's output set as a df
    return preprocessor.set_output('pandas')