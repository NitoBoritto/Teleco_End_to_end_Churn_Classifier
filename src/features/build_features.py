import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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

class aggregate_drop_multicollinear(BaseEstimator, TransformerMixin):
    """
    Handle multicollinearity by:
    1. Aggregating redundant 'No service' features into single columns
    2. Dropping other identified multicollinear features
    
    """
    def __init__(self, extra_drop_cols = None):
        # Features that should be aggregated
        self.no_internet_cols = [
            'OnlineSecurity_No internet service',
            'OnlineBackup_No internet service',
            'DeviceProtection_No internet service',
            'TechSupport_No internet service',
            'StreamingTV_No internet service',
            'StreamingMovies_No internet service'
        ]
        self.no_phone_col = 'MultipleLines_No phone service'
        
        # Features to drop
        self.extra_drop_cols = extra_drop_cols or []
        
    def fit(self, X, y = None):
        return self

    def transform(self, X):
        X = X.copy()

        # Aggregate `No_internet_service` columns
        no_internet_present = [col for col in self.no_internet_cols if col in X.columns]
        if no_internet_present:
            X['No_internet_service'] = X[no_internet_present].any(axis = 1).astype(int)
            X = X.drop(columns = no_internet_present)
            
        # Aggregate `No_phone_service` columns
        if self.no_phone_col in X.columns:
            X['No_phone_service'] = X[self.no_phone_col].astype(int)
            X = X.drop(columns = self.no_phone_col)
            
        # Drop other multicollinear features
        cols_to_drop = [col for col in self.extra_drop_cols if col in X.columns]
        if cols_to_drop:
            X = X.drop(columns = cols_to_drop)
            
        return X



def build_features(df: pd.DataFrame, target_col: str = 'Churn') -> pd.DataFrame:
    '''
    Appylying custom transformers and encoders to data
    
    This function seperates features from the dataframe to encode,
    drop nulls, and scale features depending on their type. Returns the
    result in a preprocessor pipeline to be used with the model
        
    '''
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
    
    # Multicollinear features identified
    extra_drops = ['InternetService_No', 'PhoneService', 'MonthlyCharges']
    
    feature_pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('multicollinear', aggregate_drop_multicollinear(extra_drop_cols = extra_drops))
    ])
    
    # Returning the transformers and scalers with it's output set as a df
    return feature_pipeline.set_output('pandas')