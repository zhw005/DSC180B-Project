import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier
import pandas as pd
import zipfile
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay
import lime.lime_tabular
import shap

def PDP_XGBoost(x,y, model):
    fig, ax = plt.subplots(figsize=(10, 10))
    PartialDependenceDisplay.from_estimator(model, features=[i for i in range(9)], X=x,feature_names=x.columns,ax=ax)


def LIME_XGBoost(x,y,model):
    limeexplainer = lime.lime_tabular.LimeTabularExplainer(x.values, mode='classification', feature_selection = 'auto', feature_names=x.columns,class_names = [0,1],
                                                   kernel_width=None,discretize_continuous=True)
    # Now explain a prediction
    exp = limeexplainer.explain_instance(x.iloc[0],
                                     model.predict_proba,
                                     labels = [0,1])

    print('Predicted: ', y[0])

    exp.as_pyplot_figure(label = 0)
    plt.tight_layout()

    exp.as_pyplot_figure(label = 1)
    plt.tight_layout()

def SHAP_XGBoost(x,y,model):
    shapexplainer = shap.Explainer(model)
    shap_values2 = shapexplainer(x)
    shap.plots.waterfall(shap_values[0])


def Permutation_Feature_Importance(x, y, model):
    r = permutation_importance(model, x, y,n_repeats=10,random_state=0)
    sorted_idx = r.importances_mean.argsort()[::-1][:10][::-1]

    fig, ax = plt.subplots()
    ax.boxplot(
        r.importances[sorted_idx].T, vert=False, labels=x.columns[sorted_idx]
    )
    ax.set_title("Feature Importances")
    fig.tight_layout()
    plt.show()
