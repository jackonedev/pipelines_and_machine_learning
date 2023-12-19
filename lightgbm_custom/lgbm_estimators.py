# lightgbm_custom/lgbm_estimators.py
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin



class LGBMPredictor(BaseEstimator, TransformerMixin):
    def __init__(self, params, val_data, y_val):
        self.params = params
        self.val_data = val_data
        self.y_val = y_val
        self.model = lgb
        
    def fit(self, X, y=None):
        print("\nTraining LightGBM Model\n")
        self.X_ = X.copy() 
        self.y_= y.copy()
        self.train_set = lgb.Dataset(self.X_, label=self.y_)
        self.val_set = lgb.Dataset(self.val_data, label=self.y_val)
        self.model = self.model.train(self.params, self.train_set, num_boost_round=1000, callbacks=[lgb.early_stopping(100)], valid_sets=[self.train_set, self.val_set])
        return self
    
    def predict(self, X):
        X_ = X.copy()
        predictions = self.model.predict(X_)
        return predictions
    
    def score(self, X, y=None):
        y_pred = self.predict(X)
        return roc_auc_score(y_true=y, y_score=y_pred)
    

class LGBMFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, params, val_data, y_val, sorted=False, min_importance=0):
        self.params = params
        self.val_data = val_data
        self.y_val = y_val
        self.model = lgb
        self.sorted = sorted
        self.min_importance = min_importance
        
    def fit(self, X, y=None):
        print("\nTraining LGBM Feature Selector Model\n")
        self.X_ = X.copy() 
        self.y_= y.copy()
        self.train_set = lgb.Dataset(self.X_, label=self.y_)
        self.val_set = lgb.Dataset(self.val_data, label=self.y_val)
        self.model = self.model.train(self.params, self.train_set, num_boost_round=1000, callbacks=[lgb.early_stopping(100)], valid_sets=[self.train_set, self.val_set])
        return self
    
    def transform(self, X):
        X_ = X.copy()
        assert self.min_importance >= 0, "min_importance must be 0 or greater than 0"
        if self.sorted:
            relevant_features = sorted(self.model.feature_importance(), reverse=True) > self.min_importance
        else:
            relevant_features = self.model.feature_importance() > self.min_importance
        relevant_features = [col for cond, col in zip(relevant_features, self.X_.columns) if cond]
        return X_[relevant_features]
    