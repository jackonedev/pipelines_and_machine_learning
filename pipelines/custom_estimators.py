from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


class CustomFunctions:
    
    def __init__(self, verbose=False):
        self.verbose = verbose
    
    def obtain_copy(self, dset):
        # Function to obtain a copy of the dataframes
        if self.verbose:
            print("Data shape: ", dset.shape)
        # assert isinstance(dset, pd.DataFrame), "The input isn't a pandas.DataFrame"
        return dset.copy()
    
    def create_objtype_serie(self, dset) -> pd.DataFrame:
        """
        Return a pandas.DataFrame with the object type columns of the dset_df as index
        and the amount of unique values as values.
        """

        working_df = self.obtain_copy(dset)
        return (
            working_df.loc[:, (working_df.dtypes == "object").values]
            .apply(lambda x: x.to_frame().drop_duplicates().value_counts(), axis=0)
            .sum()
        )
    
    def encode_set(self, model, dset, mask) -> pd.DataFrame:
        "Function used in the 'aggregate_encoded_set' function"
        # Encode set using the model
        encoded_features = model.transform(dset[mask])
        result = pd.DataFrame()
        last_len = 0
        for i, features in enumerate(model.categories_):
            len_feature = len(features)
            formated_features = [
                f"{dset[mask].columns[i]}_{feat}".replace(
                    " ", "_"
                )
                for feat in features
            ]
            builded_features = pd.DataFrame(
                encoded_features[:, last_len : last_len + len_feature],
                columns=formated_features,
            )
            result = pd.concat([result, builded_features], axis=1)
            last_len += len_feature
        return result

    def aggregate_encoded_set(self, model, set_df, mask, drop=True) -> pd.DataFrame:
        # Aggregate encoded set to the original set_df
        encoded_set = self.encode_set(model, set_df, mask)
        set_df = pd.concat([set_df.reset_index(), encoded_set], axis=1)
        if drop:
            set_df = set_df.drop(columns=mask)
        return set_df



class CustomOneHotEncoder(BaseEstimator, TransformerMixin, CustomFunctions):
    def __init__(self, sparse_output=False, handle_unknown="ignore", verbose=False):
        self.sparse_output = sparse_output
        self.handle_unknown = handle_unknown
        self.verbose = verbose
        self.one_hot_encoder = OneHotEncoder(sparse_output=self.sparse_output, handle_unknown=self.handle_unknown)

    def fit(self, X, y=None):
        objtype_values_df = self.create_objtype_serie(X, verbose=self.verbose)
        plus_two_categories_features = objtype_values_df[objtype_values_df > 2].index.to_list()
        self.one_hot_encoder.fit(X[plus_two_categories_features])
        return self

    def transform(self, X):
        objtype_values_df = self.create_objtype_serie(X, verbose=self.verbose)
        plus_two_categories_features = objtype_values_df[objtype_values_df > 2].index.to_list()
        result = self.aggregate_encoded_set(self.one_hot_encoder, X, plus_two_categories_features)
        if self.verbose:
            print(f"Transforming with OneHotEncoder.")
        return result

class MedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy="median", verbose=False):
        self.strategy = strategy
        self.verbose = verbose
        self.imputer = SimpleImputer(missing_values=np.nan, strategy=self.strategy)

    def fit(self, X, y=None):
        self.imputer.fit(X)
        return self

    def transform(self, X):
        result = self.imputer.transform(X)
        if self.verbose:
            print(f"Imputing with {self.strategy} strategy.")
        return pd.DataFrame(result, columns=X.columns)

class CustomScaler(BaseEstimator, TransformerMixin):
    # Inicializamos la clase con un parámetro opcional verbose
    def __init__(self, scaler=MinMaxScaler, verbose=False):
        self.scaler = scaler()
        self.verbose = verbose

    # La función fit calcula los valores mínimos y máximos de las columnas en el DataFrame de entrada
    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    # La función transform aplica la escala a los datos utilizando los valores calculados en fit
    def transform(self, X, y=None):
        X_copy = X.copy()
        # Reemplazamos los caracteres no alfanuméricos en los nombres de las columnas con guiones bajos
        columns_labels = X_copy.columns.str.replace(r"[^\w\s]", "_", regex=True).str.replace("__+", "_", regex=True)
        # Aplicamos la transformación y devolvemos un nuevo DataFrame con los nombres de las columnas modificados
        X_copy = pd.DataFrame(
            self.scaler.transform(X_copy),
            columns=columns_labels,
        )
        return X_copy