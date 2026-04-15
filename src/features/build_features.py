import pandas as pd
import numpy as np
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
    
    def get_feature_names_out(self, input_features=None):
        return np.array(self.binary_cols)
    
class boolean_encoder(BaseEstimator,  TransformerMixin):
    """
    Custom boolean encoder
    
    Transforms boolean features to binary 1/0 integers for consistence and
    overall model performance
    """
    def __init__(self, bool_cols):
        self.bool_cols = bool_cols
    
    def fit (self, X, y = None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if len(self.bool_cols) > 0:
            X[self.bool_cols] = X[self.bool_cols].astype(int)
        
        return X
    
    def get_feature_names_out(self, input_features=None):
        return np.array(self.bool_cols)
    
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
        X.columns = [str(c) for c in X.columns]

        # Find internet service columns (with any prefix)
        no_internet_cols = [
            col for col in X.columns 
            if any(keyword in col for keyword in self.no_internet_cols)
        ]
        if no_internet_cols:
            X['No_internet_service'] = X[no_internet_cols].any(axis=1).astype(int)
            X = X.drop(columns=no_internet_cols)
        
        # Find phone service columns
        no_phone_cols = [col for col in X.columns if 'MultipleLines_No phone service' in col]
        if no_phone_cols:
            X['No_phone_service'] = X[no_phone_cols].any(axis=1).astype(int)
            X = X.drop(columns=no_phone_cols)
        
        # Drop multicollinear with substring matching
        cols_to_drop = [col for col in X.columns if any(drop in col for drop in self.extra_drop_cols)]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
            
        return X
    
    def get_feature_names_out(self, input_features=None):
            """
            Calculates the feature names after aggregation and dropping.
            """
            if input_features is None:
                return np.array([])
                
            current_cols = list(input_features)

            # 1. Identify which columns will be removed (accounting for prefixes)
            to_remove = []
            for col in current_cols:
                # Match internet service columns
                if any(keyword in col for keyword in self.no_internet_cols):
                    to_remove.append(col)
                # Match phone service columns
                elif 'MultipleLines_No phone service' in col:
                    to_remove.append(col)
                # Match extra drops
                elif any(drop in col for drop in self.extra_drop_cols):
                    to_remove.append(col)

            # 2. Build the final list
            # Filter out the 'to_remove' list
            final_cols = [c for c in current_cols if c not in to_remove]
            
            # 3. Add the new feature names we created
            # We don't add prefixes here because this step isn't a ColumnTransformer
            final_cols.append('No_internet_service')
            final_cols.append('No_phone_service')

            return np.array(final_cols)



def build_features(df: pd.DataFrame, target_col: str = 'Churn') -> Pipeline:
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
    # If True/False columns we transform them into binary
    bool_cols = [c for c in df.select_dtypes(include = 'bool').columns if c != target_col]
    # if 3 or more we ohe
    multi_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]
    # we scale all other numerical using RobustScaler
    numeric_cols = [c for c in df.select_dtypes(include = 'number').columns if c != target_col]
    
    # Preprocessor 
    preprocessor = ColumnTransformer(transformers = [
        ('bin', custom_binary_encoder(binary_cols), binary_cols),
        ('bool', boolean_encoder(bool_cols), bool_cols),
        ('ohe', OneHotEncoder(drop = 'first', sparse_output = False), multi_cols),
        ('scaler', RobustScaler(), numeric_cols)
    ], remainder = 'passthrough')
    
    # Multicollinear features identified
    extra_drops = ['InternetService_No', 'PhoneService', 'MonthlyCharges']
    
    preprocessor.set_output(transform = 'pandas')
    
    multicollinear_transformer = aggregate_drop_multicollinear(extra_drop_cols = extra_drops)
    
    # Returning the transformers and scalers with it's output set as a df
    return preprocessor, multicollinear_transformer