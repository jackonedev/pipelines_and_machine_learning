# logistic_regression/log_reg_estimator.py
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pandas as pd


class CustomLogisticRegression(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs) -> None:
        
        self.model = LogisticRegression(**kwargs)
        
    def fit(self, X, y=None):
        self.X_ = X.copy()
        self.Y_= y.copy()
        
        self.model.fit(X, y)
        return self
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X):
        return self.model.predict(X)
    
    def roc_auc_score(self, X, y):
        return roc_auc_score(y, self.predict_proba(X))
    
    def transform(self, X):
        result = self.predict_proba(X)
        result = pd.DataFrame(result, columns=["log_reg"])
        return result