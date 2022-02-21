from pytorch_tabnet.tab_model import TabNetClassifier
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

class TNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.model = TabNetClassifier(**kwargs)
       
    def fit(self, train_X, train_y, eval_set= [None,None], max_epochs = 100):
        if isinstance(train_X, pd.DataFrame):
            if isinstance(eval_set[0], pd.DataFrame):
                self.model.fit(train_X.to_numpy(), train_y.values.flatten(), eval_set = 
                          eval_set, max_epochs = max_epochs,
                               eval_metric=['auc'])
            else:
                self.model.fit(train_X.to_numpy(), train_y.values.flatten(), max_epochs = max_epochs,
                               eval_metric=['auc'])
        else:
            if isinstance(test_X, pd.DataFrame):
                self.model.fit(train_X, train_y, eval_set = 
                          [(test_X, test_y)], max_epochs = max_epochs, eval_metric=['auc'])
            else:
                self.model.fit(train_X, train_y, max_epochs = max_epochs, eval_metric=['auc'])
    
    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            return self.model.predict_proba(X.to_numpy())
        else:
            return self.model.predict_proba(X)
    
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            return self.model.predict(X.to_numpy())
        else:
            return self.model.predict(X)
    
    def save_model(self, fp):
        self.model.save_model(fp)
    
    def load_model(self, fp):
        self.model.load_model(fp)
