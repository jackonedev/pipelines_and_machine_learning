import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src import data_utils
from pipelines.custom_estimators import CustomScaler, MedianImputer
from pipelines.custom_functions import correct_outliers, ordinal_encoder_two_categories, one_hot_encoder_plus_two_categories, remove_nan_columns

app_train, app_test, columns_description = data_utils.get_datasets()
X_train, y_train, X_test, y_test = data_utils.get_feature_target(app_train, app_test)
X_train, X_val, y_train, y_val = data_utils.get_train_val_sets(X_train, y_train)



pipe_fe = Pipeline([
    ("outliers", FunctionTransformer(correct_outliers)),
    ("binary_cat", FunctionTransformer(ordinal_encoder_two_categories)),
    ("cat", FunctionTransformer(one_hot_encoder_plus_two_categories)),
    ("impute_nan", MedianImputer),
    ("remove_nan", FunctionTransformer(remove_nan_columns)),
    ("scale", CustomScaler(scaler=StandardScaler)),
    ])


pipe_fe.fit_transform(X_train)