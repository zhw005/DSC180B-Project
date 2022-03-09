import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
import pandas

from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier
from lightgbm import LGBMClassifier
from src.sk_tabnet import TNClassifier
from sklearn.model_selection import train_test_split

def run(config):
    """
    Run fairness evaluation methods
    
    config: configuration for the evaluation
    """

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
    
    features = pandas.read_csv(config["data_feature_path"],index_col = False)
    target = pandas.read_csv(config["data_target_path"],index_col = False)
    features = features.drop(columns = ["Unnamed: 0"], errors='ignore')
    target = target.drop(columns = ["Unnamed: 0"], errors='ignore')

    _, features, _, target = train_test_split(features, target, random_state = 42)
    
    os.makedirs(config["out_fp"], exist_ok=True)

    
    if "group_fairness" in config["methods"]:
        result = group_fairness(model, features, config['sensitive_variable'], target)
        with open(config["out_fp"] + "/group_fairness.txt", 'w') as fp:
            fp.write(result)

    if "predictive_parity" in config["methods"]:
        result = predictive_parity(model, features, config['sensitive_variable'], target)
        with open(config["out_fp"] + "/predictive_parity.txt", 'w') as fp:
            fp.write(result)

    if "conditional_frequencies" in config["methods"]:
        conditional_frequencies(model, features, config['sensitive_variable'], target, \
                                fp = config["out_fp"] + "/conditional_frequencies.png")

    if "causal_discrimination" in config["methods"]:
        if config['sensitive_variable'] == "CODE_GENDER: M":
            flipped = features.copy()
            idx_m = features["CODE_GENDER: M"] == 1
            idx_f = ~(features["CODE_GENDER: M"] == 1)
            flipped.loc[idx_m,"CODE_GENDER: M"] = 0
            flipped.loc[idx_m,"CODE_GENDER: F"] = 1
            flipped.loc[idx_f,"CODE_GENDER: M"] = 1
            flipped.loc[idx_f,"CODE_GENDER: F"] = 0
        else:
            raise NotImplementedError
        result =  causal_discrimination(model, features, flipped, config['sensitive_variable'])
        with open(config["out_fp"] + "/causal_discrimination.txt", 'w') as fp:
            fp.write(result)



def group_fairness(model, features, sensitive, target):
    """
    Check if P(Y = 1|S = si) = P(Y = 1|S = sj)
    
    model: trained model
    dataset: test dataset
    sensitive: column name of the sensitive variable
    target: target of the prediction task
    """
    S = list(set(features[sensitive]))
    assert len(S) == 2
    pred = model.predict(features) == 1
    p0 = pred[features[sensitive] == S[0]].mean()
    p1 = pred[features[sensitive] == S[1]].mean()
    result = ""
    result += f"Group with {sensitive} = {S[0]} has a probability of positive prediction at {p0}\n"
    result += f"Group with {sensitive} = {S[1]} has a probability of positive prediction at {p1}\n"
    result += f"The difference is {abs(p0-p1)}\n"
    
    return result
    
def predictive_parity(model, features, sensitive, target):
    """
    Check if P(T = 1|Y = 1, S= s_i) = P(T = 1|Y = 1, S = s_j)
    
    model: trained model
    dataset: test dataset
    sensitive: column name of the sensitive variable
    target: column name of the target of the prediction task
    """
    S = list(set(features[sensitive]))
    assert len(S) == 2
    pred = model.predict(features) == 1
    p0 = target[(features[sensitive] == S[0]) & (pred == 1)].mean()[0]
    p1 = target[(features[sensitive] == S[1]) & (pred == 1)].mean()[0]
    result = ""
    result += f"Group with {sensitive} = {S[0]} has a true positive rate of {p0}\n"
    result += f"Group with {sensitive} = {S[1]} has a true positive rate of {p1}\n"
    result += f"The difference is {abs(p0-p1)}\n"
    
    return result
    
def conditional_frequencies(model, features, sensitive, target, fp = None, k = 5):
    """
    Check if P(T = 1|Y ∈ bin_k, S= s_i) = P(T = 1|Y ∈ bink, S = s_j)
    
    model: trained model
    dataset: test dataset
    sensitive: column name of the sensitive variable
    target: column name of the target of the prediction task
    k: number of bins
    """
    S = list(set(features[sensitive]))
    assert len(S) == 2
    pred = model.predict_proba(features)[:,1]
    
    p0 = pred[(target.values[:,0] == 1) & (features[sensitive] == S[0]).values]
    p0_base = pred[(features[sensitive] == S[0])]
    hist0, bins = np.histogram(p0, np.arange(k+1) / k)
    base, _ = np.histogram(p0_base, bins)
    hist0 = hist0 / base
    
    p1 = pred[(target.values[:,0] == 1) & (features[sensitive] == S[1]).values]
    p1_base = pred[(features[sensitive] == S[1])]
    hist1, bins = np.histogram(p1, bins)
    base, _ = np.histogram(p1_base, bins)
    hist1 = hist1 / base
    
    plot_label = []
    for i in range(k):
        plot_label.append(f'{round(bins[i],2)} - {round(bins[i+1],2)}')
    
    plot_data = {
    'x': plot_label + plot_label,
    'y': np.concatenate([hist0, hist1]),
    'category': [f'{sensitive} = {S[0]}'] * k + [f'{sensitive} = {S[1]}'] * k
    }
    
    ax = sns.barplot(x='x', y='y', hue='category', data=plot_data)
    ax.set(xlabel = 'prediction', ylabel = 'P(Target = 1)')
    if fp:
        plt.savefig(fp)
    else:
        plt.show()
    
    return ax
    
    
def causal_discrimination(model, features, flipped_features, sensitive):
    """
    Flip sensitive attribute and check if the model output is the same
    
    model: trained model
    dataset: test dataset
    flipped_features: test dataset with the sensitive features flipped
    sensitive: column name of the sensitive variable
    """
    S = list(set(features[sensitive]))
    assert len(S) == 2
    pred_ori = model.predict(features)
    pred_flip = model.predict(flipped_features)
    diff = (pred_ori != pred_flip).mean()
    
    p_ori_0 = pred_ori[features[sensitive] == S[0]].mean()
    p_ori_1 = pred_ori[features[sensitive] == S[1]].mean()
    
    p_flip_0 = pred_flip[features[sensitive] == S[0]].mean()
    p_flip_1 = pred_flip[features[sensitive] == S[1]].mean()
    
    result = ""
    result += f"After flipping the sensitve attribute {sensitive}, {round(diff, 4)*100}% of the prediction changes\n"
    result += f"The average prediction of class {sensitive} = {S[0]} changed {p_flip_0 - p_ori_0}\n"
    result += f"The average prediction of class {sensitive} = {S[1]} changed {p_flip_1 - p_ori_1}\n"
    
    return result
    
def predictive_equality(test_data, col, sensitive_class, predicted_y = 'predicted_y', test_y = 'test_y'):
    """
    return: predictive_equality of one sensitive_class

    test_data: test data with true label and predictive label and features
    col: sensitive feature column name
    sensitive_class: sensitive attribute sensitive_class (e.g. m/f/0/1)
    predicted_y: predicted value column name
    test_y: true label column name
    """
    return test_data.loc[test_data[col] == sensitive_class].loc[test_data[test_y] == 0].loc[test_data[predicted_y] == 1].shape[0]\
                        /test_data.loc[test_data[col] == sensitive_class].loc[test_data[test_y] == 0].shape[0]

def predictive_opportunity(test_data, col, sensitive_class, predicted_y = 'predicted_y', test_y = 'test_y'):
    """
    return: predictive_equality of one sensitive_class

    test_data: test data with true label and predictive label and features
    col: sensitive feature column name
    sensitive_class: sensitive attribute sensitive_class (e.g. m/f/0/1)
    predicted_y: predicted value column name
    test_y: true label column name
    """
    return test_data.loc[test_data[col] == sensitive_class].loc[test_data[test_y] == 1].loc[test_data[predicted_y] == 0].shape[0]\
                        /test_data.loc[test_data[col] == sensitive_class].loc[test_data[test_y] == 1].shape[0]
