import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def obtain_copy(dset, verbose=False):
    # Function to obtain a copy of the dataframes
    if verbose:
        print("Input train data shape: ", dset.shape)
    # assert isinstance(dset, pd.DataFrame), "The input isn't a pandas.DataFrame"
    return dset.copy()

def correct_outliers(dset, verbose=False):
    # Correct outliers
    working_df = obtain_copy(dset, verbose=verbose)
    working_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    # assert isinstance(working_df, pd.DataFrame), "The returned object isn't a pandas.DataFrame"
    # assert working_df.shape == dset.shape, "The shape of the returned object isn't the same as the input"
    return working_df


def create_objtype_serie(dset, verbose=False) -> pd.DataFrame:
    """
    Return a pandas.DataFrame with the object type columns of the dset_df as index
    and the amount of unique values as values.
    """

    working_df = obtain_copy(dset, verbose=verbose)
    return (
        working_df.loc[:, (working_df.dtypes == "object").values]
        .apply(lambda x: x.to_frame().drop_duplicates().value_counts(), axis=0)
        .sum()
    )

# def replace_objtype_two_categories(dset, verbose=False):
#     working_df = obtain_copy(dset, verbose=verbose)
#     objtype_values_df = create_objtype_serie(working_df, verbose=verbose)
#     assert not isinstance(objtype_values_df, type(pd.DataFrame())), "The returned object isn't a pandas.DataFrame"
#     two_categories_features = objtype_values_df[
#         objtype_values_df == 2
#     ].index.to_list()
#     working_df[two_categories_features] = working_df[two_categories_features].replace(
        
#     assert isinstance(working_df, pd.DataFrame), "The returned object isn't a pandas.DataFrame"
#     return working_df

def ordinal_encoder_two_categories(dset, verbose=False):
    # Ordinal encoder for features with two unique categories
    working_df = obtain_copy(dset, verbose=verbose)

    objtype_values_df = create_objtype_serie(working_df, verbose=verbose)

    # assert not isinstance(objtype_values_df, type(pd.DataFrame())), "The returned object isn't a pandas.DataFrame"

    two_categories_features = objtype_values_df[
        objtype_values_df == 2
    ].index.to_list()

    ordinal_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=np.nan
    )
    ordinal_encoder.fit(working_df[two_categories_features])

    working_df[two_categories_features] = ordinal_encoder.transform(
        working_df[two_categories_features]
    )
    # assert isinstance(working_df, pd.DataFrame), "The returned object isn't a pandas.DataFrame"
    return working_df

def encode_set(model, dset_df, mask) -> pd.DataFrame:
    "Function used in the 'aggregate_encoded_set' function"
    # Encode set using the model
    encoded_features = model.transform(dset_df[mask])
    result = pd.DataFrame()
    last_len = 0
    for i, features in enumerate(model.categories_):
        len_feature = len(features)
        formated_features = [
            f"{dset_df[mask].columns[i]}_{feat}".replace(
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

def aggregate_encoded_set(model, set_df, mask, drop=True) -> pd.DataFrame:
    # Aggregate encoded set to the original set_df
    encoded_set = encode_set(model, set_df, mask)
    set_df = pd.concat([set_df.reset_index(), encoded_set], axis=1)
    if drop:
        set_df = set_df.drop(columns=mask)
    return set_df

def one_hot_encoder_plus_two_categories(dset_df, verbose=False) -> pd.DataFrame:
    # One-hot encoder for features with more than two unique categories
    working_df = obtain_copy(dset_df, verbose=verbose)

    objtype_values_df = create_objtype_serie(working_df, verbose=verbose)

    # assert not isinstance(objtype_values_df, type(pd.DataFrame())), "The 'obj_type_values_series' isn't a pandas.DataFrame"

    plus_two_categories_features = objtype_values_df[
        objtype_values_df > 2
    ].index.to_list()

    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    one_hot_encoder.fit(working_df[plus_two_categories_features])

    working_df = aggregate_encoded_set(
        one_hot_encoder, working_df, plus_two_categories_features
    )
    return working_df


def remove_nan_columns(dset, verbose=False):#FunctionTransformer
    # Function to remove columns with all NaN values
    if verbose:
        print("NaNs: {}".format(dset.loc[:, [col.endswith("nan") for col in dset.columns]].columns))
    working_df = obtain_copy(dset, verbose=verbose)
    working_df = working_df.loc[:, [not col.endswith("nan") for col in working_df.columns]]
    return working_df


def simple_imputer_set_df(dset_df, verbose=False) -> pd.DataFrame:
    # Impute missing values using SimpleImputer
    working_df = obtain_copy(dset_df, verbose=verbose)

    imputer = SimpleImputer(missing_values=np.nan, strategy="median")
    imputer.fit(working_df)

    return pd.DataFrame(
        imputer.transform(working_df), columns=working_df.columns
    )


def min_max_scaler_set_df(dset_df, verbose=False) -> pd.DataFrame:
    # Scale features using Min-Max scaler
    working_df = obtain_copy(dset_df, verbose=verbose)

    scaler = MinMaxScaler().fit(working_df)

    columns_labels = working_df.columns.str.replace(r"[^\w\s]", "_", regex=True).str.replace("__+", "_", regex=True)

    working_df = pd.DataFrame(
        scaler.transform(working_df),
        columns=columns_labels,
    )
    
    return working_df
