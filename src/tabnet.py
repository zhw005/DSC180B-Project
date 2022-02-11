from pytorch_tabnet.tab_model import TabNetClassifier
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

class TabNetClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.model = TabNetClassifier(**kwargs)
       
    def fit(self, train_X, train_y, max_epochs = 100, test_X = None, test_y = None):
        if isinstance(train_X, pd.DataFrame):
            
            if X_eval and y_eval:
                self.model.fit(train_X.to_numpy(), train_y.values[:,0], eval_set = 
                          [(test_X.to_numpy(), test_y.values[:,0])], max_epochs,
                               eval_metric=['auc'])
            else:
                self.model.fit(train_X.to_numpy(), train_y.values[:,0], max_epochs,
                               eval_metric=['auc'])
        else:
            if X_eval and y_eval:
                self.model.fit(train_X, train_y, eval_set = 
                          [(test_X, test_y)], max_epochs, eval_metric=['auc'])
            else:
                self.model.fit(train_X, train_y, max_epochs, eval_metric=['auc'])
    
    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            return self.model.predict(X.to_numpy())
        else:
            return self.model.predict(X)
    
    def predict(self, X):
        return self.predict_proba(X) > 0.5
