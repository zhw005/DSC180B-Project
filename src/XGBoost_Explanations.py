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

def run_model_explanations(config):
    if config['model'] == "svm":
        with open(config["model_path"], 'rb') as fp:
           model = pickle.load(fp)
    elif config['model'] == "xgboost":
        model = XGBClassifier()
        model.load_model(config["model_path"])
    elif config['model'] == "lgbm":
        model = LGBMClassifier()
        model.load_model(config["model_path"])
    elif config['model'] == "tabnet":
        model = TNClassifier()
        model.load_model(config["model_path"])
    else:
        raise NotImplementedError

    features = pd.read_csv(config["data_feature_path"],index_col = False)
    target = pd.read_csv(config["data_target_path"],index_col = False)
    features = features.drop(columns = ["Unnamed: 0"], errors='ignore')
    target = target.drop(columns = ["Unnamed: 0"], errors='ignore')
    
    train_X, test_X, train_y, test_y = train_test_split(features, target, random_state = 42)
    # predict
    pred_y = model.predict(test_X)
    os.makedirs(config["out_fp"], exist_ok=True)

    # Airbnb Example: PDP, Permutation Feature Importance, 
    if config['data'] == 'airbnb':
        PDP(train_X, model, [2, 3, 4, 7], fp = config["out_fp"] + "/" + config['data'] + "_partial_dependence_plots.png")
        Permutation_Feature_Importance(test_X, pred_y, model, fp = config["out_fp"] + "/" + config['data'] + "_permutation_feature_importance.png")

         
    # Loan Example: PDP, Permutation Feature Importance, LIME,SHAP
    if config['data'] == 'loan':
        PDP(train_X, model, [5, 23, 26, 40], fp = config["out_fp"] + "/" + config['data'] + "_partial_dependence_plots.png")
        Permutation_Feature_Importance(test_X, pred_y, model, fp = config["out_fp"] + "/" + config['data'] + "_permutation_feature_importance.png")
        LIME(train_X, test_X, pred_y, model,3,fp = config["out_fp"] + "/" + config['data'] + "_LIME.png")                               
        SHAP(train_X, model, 3, fp = config["out_fp"] + "/" + config['data'] + "_SHAP.png")                                

    # Healthcare Example: LIME, SHAP
    if config['data'] == 'healthcare':
        LIME(train_X, test_X, pred_y, model,0,fp = config["out_fp"] + "/" + config['data'] + "_LIME.png")                               
        SHAP(train_X, model, 0, fp = config["out_fp"] + "/" + config['data'] + "_SHAP.png")                            
   
    # Note: for SHAP figures,the code for shap.plots.waterfall has bug, so plots can't be saved. 
    
def PDP(x, model, feature_idx, fp = None):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = PartialDependenceDisplay.from_estimator(model, features=feature_idx, X=x,feature_names=x.columns,ax=ax)
    if fp:
        plt.savefig(fp)
    else:
        plt.show()
    return ax

def Permutation_Feature_Importance(x, y, model, fp):
    r = permutation_importance(model, x, y,n_repeats=10,random_state=0)
    sorted_idx = r.importances_mean.argsort()[::-1][:10][::-1]

    fig, ax = plt.subplots()
    ax.boxplot(
        r.importances[sorted_idx].T, vert=False, labels=x.columns[sorted_idx]
    )
    ax.set_title("Feature Importances")
    fig.tight_layout()

    if fp:
        plt.savefig(fp)
    else:
        plt.show()

def LIME(x1, x2, y,model,individual_idx,fp = None):
    limeexplainer = lime.lime_tabular.LimeTabularExplainer(x1.values, mode='classification', feature_selection = 'auto', feature_names=x1.columns,class_names = [0,1],
                                                   kernel_width=None,discretize_continuous=True)
    # Now explain a prediction
    exp = limeexplainer.explain_instance(x2.iloc[individual_idx],
                                     model.predict_proba,
                                     labels = [0,1])

    print('(LIME) Predicted outcome for the chosen individual: ', y[individual_idx])

    if y[individual_idx] == 0:
        exp.as_pyplot_figure(label = 0)
        plt.tight_layout()

    else:
        exp.as_pyplot_figure(label = 1)
        plt.tight_layout()

    if fp:
        plt.savefig(fp)
    else:
        plt.show()

def SHAP(x,model, individual_idx, fp = None):
    shapexplainer = shap.Explainer(model)
    shap_values = shapexplainer(x)
    shap.plots.waterfall(shap_values[individual_idx])

    if fp:
        plt.savefig(fp)
