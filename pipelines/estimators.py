from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
import pickle
import datetime as dt

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipelines.base_estimators import CustomBase

class CorrectOutliers(BaseEstimator, TransformerMixin):
    def __init__(self, verbose=False):
        self.verbose = verbose
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_encoded = X.copy()
        X_encoded["DAYS_EMPLOYED"] = X["DAYS_EMPLOYED"].replace({365243: np.nan})
        return X_encoded
    
class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=np.nan
            )
    
    def fit(self, X, y=None):
        self.X_ = X.copy()
        self.encoder.fit(self.X_[self.columns])
        return self
    
    def transform(self, X):
        X_encoded = X.copy()
        X_encoded = self.encoder.transform(X_encoded[self.columns])
        X_encoded = pd.DataFrame(X_encoded, columns=self.columns)
        return X_encoded
    
class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.columns_= None
        self.encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
            )
    
    def fit(self, X, y=None):
        self.X_ = X.copy()
        self.encoder.fit(self.X_[self.columns])
        return self
    
    def transform(self, X):
        X_encoded = X.copy()
        X_encoded = self.encoder.transform(X_encoded[self.columns])
        X_encoded = self.feature_adjust(X_encoded)
        self.columns_ = X_encoded.columns
        return X_encoded
    
    def feature_adjust(self, X_encoded):
        result = pd.DataFrame()
        last_len = 0
        for i, features in enumerate(self.encoder.categories_):
            len_feature = len(features)
            formated_features = [
                f"{self.columns[i]}_{feat}".replace(
                    " ", "_"
                )
                for feat in features
            ]
            builded_features = pd.DataFrame(
                X_encoded[:, last_len : last_len + len_feature],
                columns=formated_features,
            )
            result = pd.concat([result, builded_features], axis=1)
            last_len += len_feature
            
        result.columns = result.columns.str.replace(r"[^\w\s]", "_", regex=True).str.replace("__+", "_", regex=True)
        return result
        
class CustomImputer(BaseEstimator, TransformerMixin, CustomBase):
    def __init__(self, strategy="median"):
        self.strategy = strategy
        self.imputer = SimpleImputer(
            missing_values=np.nan,
            strategy=self.strategy)

    def fit(self, X, y=None):
        self.X_ = X.copy()
        self.imputer.fit(X)
        return self

    def transform(self, X):
        X_encoded = X.copy()
        X_encoded = self.imputer.transform(X_encoded)
        
        # Create DataFrame and correct columns types
        X_encoded = pd.DataFrame(X_encoded, columns=self.X_.columns)
        X_info = self.data_info(self.X_)
        X_dtypes = X_info.set_index("columna")["dtype"].to_dict()
        X_encoded = X_encoded.astype(X_dtypes)
        
        return X_encoded
        
class CustomBackup(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        self.X_ = X.copy()
        return self
    
    def transform(self, X):#TODO: df.name para identificar train, val, test 
        dm = dt.datetime.now().strftime("%d_%m")
        hms = dt.datetime.now().strftime("%H_%M_%S")
        with open(f"data-{dm}-{hms}-backup.pkl", "wb") as f:
            pickle.dump(self.X, f)
        # try:#Warning: X.name is not detected
        #     #Warning: downloading .csv have some issues downloading more than 1 file
        #     file_name = f"{X.name}-{dm}-{hms}-backup.csv"
        # except:
        #     file_name = f"data-{dm}-{hms}-backup.csv"
        # X.to_csv(file_name, index=False)
        return X
    
class CustomDropna(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        self.X_ = X.copy()
        return self
    
    def transform(self, X):
        X_encoded = X.copy()
        X_encoded = X_encoded.loc[:, [not col.endswith("nan") for col in X_encoded.columns]]
        return X_encoded
    
class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler=MinMaxScaler):
        self.scaler = scaler()
    
    def fit(self, X, y=None):
        self.X_ = X.copy()
        self.scaler.fit(self.X_)
        return self
    
    def transform(self, X):
        X_encoded = X.copy()
        X_encoded = self.scaler.transform(X_encoded)
        X_encoded = pd.DataFrame(X_encoded, columns=self.X_.columns)
        return X_encoded