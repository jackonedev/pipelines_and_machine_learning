# main_pipeline_2.py: LightGBM Predictor
import sys, os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import data_utils
from pipelines.estimators import (
    CorrectOutliers,
    CustomOrdinalEncoder,
    CustomOneHotEncoder,
    CustomImputer,
    CustomBackup,
    CustomScaler,
    CustomDropna,
)
from pipelines.smart_pandas import PandasFeatureUnion
from lightgbm_custom.lgbm_estimators import LGBMPredictor

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler


## Main ##

# app_train, app_test, columns_description = data_utils.get_datasets()
app_train = pd.read_csv("dataset/application_train_aai.csv")
app_test = pd.read_csv("dataset/application_test_aai.csv")
columns_description  = pd.read_csv("dataset/HomeCredit_columns_description.csv")

X_train, y_train, X_test, y_test = data_utils.get_feature_target(app_train, app_test)
X_train, X_val, y_train, y_val = data_utils.get_train_val_sets(X_train, y_train)

dtype_object_info = (X_train.loc[:, (X_train.dtypes == "object").values]
        .apply(lambda x: x.to_frame().drop_duplicates().value_counts(), axis=0)
        .sum())

assert isinstance(dtype_object_info, pd.core.series.Series), "type error"


# Numerical or Datetime features
num_dt_ft = X_train.loc[:, (X_train.dtypes != "object").values].columns.tolist()
# Object type features with two unique categories
two_cat_ft = dtype_object_info[dtype_object_info == 2].index.to_list()
# Object type features with more than two unique categories
plus_two_cat_ft = dtype_object_info[dtype_object_info > 2].index.to_list()


workflow_1 = PandasFeatureUnion([
    ("num_dt_cat", FunctionTransformer(lambda X: X[num_dt_ft].reset_index())),
    ("binary_cat", CustomOrdinalEncoder(two_cat_ft)),
    ("multi_label_cat", CustomOneHotEncoder(plus_two_cat_ft)),
    ])

feature_enginering = Pipeline([
    ("outliers", CorrectOutliers()),
    ("numerical_transformation", workflow_1),
    ("impute_nan", CustomImputer(strategy="median")),
    ("backup", CustomBackup(shutdown=True)),
    ("remove_nan", CustomDropna()),
    ("scale", CustomScaler(scaler=StandardScaler)),
    ])

feature_enginering.fit(X_train)
val_data = feature_enginering.transform(X_val)


params = {
    "objective": "binary",
    "metric": "auc",
    "min_child_samples": 2000,
    "num_leaves": 14,
    "learning_rate": 0.1,
    "random_state": 88,
    "n_jobs": -1,
    "verbose": 0,
}

feature_model_pipeline = Pipeline([
    ("feature_enginering", feature_enginering),
    ("model", LGBMPredictor(params=params, val_data=val_data, y_val=y_val)),
])

feature_model_pipeline.fit(X_train, y_train)