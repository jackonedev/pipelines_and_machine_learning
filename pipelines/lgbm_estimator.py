import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin


class LGBMEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, params, val_data, y_val):
        self.params = params
        self.val_data = val_data
        self.y_val = y_val
        self.model = lgb
        
    def fit(self, X, y=None):
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
    