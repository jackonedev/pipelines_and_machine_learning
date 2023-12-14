from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


class CustomBase:
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        
    def data_info(self, data:pd.DataFrame, sorted:bool=False) -> pd.DataFrame:
        """
        Function to describe the variables of a dataframe
        Analogous to the .describe() method of pandas.DataFrame
        """
        df = pd.DataFrame(pd.Series(data.columns))
        df.columns = ["columna"]
        df["NaNs"] = data.isna().sum().values
        df["pct_nan"] = round(df["NaNs"] / data.shape[0] * 100, 2)
        df["dtype"] = data.dtypes.values
        df["count"] = data.count().values
        df["count_unique"] = [
            len(data[elemento].value_counts()) for elemento in data.columns
        ]
        df["pct_unique"] = (df["count_unique"].values / data.shape[0] * 100).round(2)
        if sorted:
            df = df.reset_index(drop=False)
            df = df.sort_values(by=["dtype", "count_unique"])
        df = df.reset_index(drop=True)
        return df
    
    