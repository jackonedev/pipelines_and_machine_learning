from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.reset_index().copy()
    working_val_df = val_df.reset_index().copy()
    working_test_df = test_df.reset_index().copy()

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    # 2. Encode string categorical features (dytpe `object`):
    obj_type_values_df = (
        working_train_df.loc[:, (working_train_df.dtypes == "object").values]
        .apply(lambda x: x.to_frame().drop_duplicates().value_counts(), axis=0)
        .sum()
    )

    # Two unique categories treatment
    assert obj_type_values_df[obj_type_values_df == 2].shape[0] == 4
    two_categories_features = obj_type_values_df[
        obj_type_values_df == 2
    ].index.to_list()
    ordinal_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=np.nan
    )
    ordinal_encoder.fit(working_train_df[two_categories_features])

    working_train_df[two_categories_features] = ordinal_encoder.transform(
        working_train_df[two_categories_features]
    )
    working_val_df[two_categories_features] = ordinal_encoder.transform(
        working_val_df[two_categories_features]
    )
    working_test_df[two_categories_features] = ordinal_encoder.transform(
        working_test_df[two_categories_features]
    )

    #     - If it has more than 2 categories, use one-hot encoding, please use
    # Rest of the Categories treatment
    assert obj_type_values_df[obj_type_values_df > 2].shape[0] == 12
    plus_two_categories_features = obj_type_values_df[
        obj_type_values_df > 2
    ].index.to_list()
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    one_hot_encoder.fit(working_train_df[plus_two_categories_features])

    # Train set Encoding #TODO: DRY
    encoded_features = one_hot_encoder.transform(
        working_train_df[plus_two_categories_features]
    )
    result = pd.DataFrame()
    last_len = 0
    for i, features in enumerate(one_hot_encoder.categories_):
        len_feature = len(features)
        formated_features = [
            f"{working_train_df[plus_two_categories_features].columns[i]}_{feat}".replace(
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
    working_train_df = pd.concat(
        [working_train_df.reset_index(drop=True), result], axis=1
    )
    working_train_df = working_train_df.drop(columns=plus_two_categories_features)

    # Validation set Encoding #TODO: DRY
    encoded_features = one_hot_encoder.transform(
        working_val_df[plus_two_categories_features]
    )
    result = pd.DataFrame()
    last_len = 0
    for i, features in enumerate(one_hot_encoder.categories_):
        len_feature = len(features)
        formated_features = [
            f"{working_val_df[plus_two_categories_features].columns[i]}_{feat}".replace(
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
    working_val_df = pd.concat([working_val_df.reset_index(drop=True), result], axis=1)
    working_val_df = working_val_df.drop(columns=plus_two_categories_features)

    # Test set Encoding #TODO: DRY
    encoded_features = one_hot_encoder.transform(
        working_test_df[plus_two_categories_features]
    )
    result = pd.DataFrame()
    last_len = 0
    for i, features in enumerate(one_hot_encoder.categories_):
        len_feature = len(features)
        formated_features = [
            f"{working_test_df[plus_two_categories_features].columns[i]}_{feat}".replace(
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
    working_test_df = pd.concat(
        [working_test_df.reset_index(drop=True), result], axis=1
    )
    working_test_df = working_test_df.drop(columns=plus_two_categories_features)

    # 3. Impute values for all columns with missing data or, just all the columns.
    # Use median as imputing value. Please use sklearn.impute.SimpleImputer().
    imputer = SimpleImputer(missing_values=np.nan, strategy="median")
    imputer.fit(working_train_df)

    working_train_df = pd.DataFrame(
        imputer.transform(working_train_df), columns=working_train_df.columns
    )
    working_val_df = pd.DataFrame(
        imputer.transform(working_val_df), columns=working_val_df.columns
    )
    working_test_df = pd.DataFrame(
        imputer.transform(working_test_df), columns=working_test_df.columns
    )

    # 4. Feature scaling with Min-Max scaler. Apply this to all the columns.
    # Please use sklearn.preprocessing.MinMaxScaler().
    scaler = MinMaxScaler().fit(working_train_df)

    working_train_df = pd.DataFrame(
        scaler.transform(working_train_df),
        columns=working_train_df.columns.str.replace(
            r"[^\w\s]", "_", regex=True
        ).str.replace("__+", "_", regex=True),
    )
    working_val_df = pd.DataFrame(
        scaler.transform(working_val_df),
        columns=working_val_df.columns.str.replace(
            r"[^\w\s]", "_", regex=True
        ).str.replace("__+", "_", regex=True),
    )
    working_test_df = pd.DataFrame(
        scaler.transform(working_test_df),
        columns=working_test_df.columns.str.replace(
            r"[^\w\s]", "_", regex=True
        ).str.replace("__+", "_", regex=True),
    )

    return (
        (train := working_train_df.values),
        (val := working_val_df.values),
        (test := working_test_df.values),
    )
