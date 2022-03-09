import dowhy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd

from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier
from lightgbm import LGBMClassifier
from src.sk_tabnet import TNClassifier
from sklearn.model_selection import train_test_split


def run(config):
    """
    Run FN and FP explanation examples

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

    X = pd.read_csv(config["data_feature_path"], index_col = 0)
    y = pd.read_csv(config["data_target_path"], index_col = 0)
    columns = X.columns
    X.columns = range(X.shape[1])

    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 42)
    pred_test_y = model.predict(test_X)
    train_X.columns = columns
    test_X.columns = columns

    # changing labels
    data = test_X.copy()
    data['test_y'] = test_y
    data = data.reset_index(drop = True)
    data['pred_test_y'] = pd.Series(pred_test_y)
    data['FN'] = data.apply(lambda x: 1 if x.test_y == 1 and x.pred_test_y == 0 else 0, axis = 1)
    data['FP'] = data.apply(lambda x: 1 if x.test_y == 0 and x.pred_test_y == 1 else 0, axis = 1)

    #os.makedirs(config["out_fp"], exist_ok=True)

    if config['outcome'] == "FN":
        out_fp = config["out_fp"] + "/FN-example.txt"
        result = do_why(data, config['treatment'], config['outcome'], config['common_causes'], out_fp)

    if config['outcome'] == "FP":
        out_fp = config["out_fp"] + "/FP-example.txt"
        result = do_why(data, config['treatment'], config['outcome'], config['common_causes'], out_fp)


def do_why(data, treatment, outcome, common_causes, out_fp):
    # 1. Model
    model= dowhy.CausalModel(
            data = data,
            treatment= treatment,
            outcome= outcome,
            common_causes = common_causes)

    f = open(out_fp, "w", encoding="utf-8")
    # 2. Identify
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print(identified_estimand, file=f)

    # 3. Estimate
    estimate = model.estimate_effect(identified_estimand,
                                     method_name="backdoor.linear_regression",target_units="ate")
    # ATE = Average Treatment Effect
    # ATT = Average Treatment Effect on Treated (i.e. those who were assigned a different room)
    # ATC = Average Treatment Effect on Control (i.e. those who were not assigned a different room)
    print(estimate, file=f)

    # 4. Refute

    # Radom Common Cause:- Adds randomly drawn covariates to data and re-runs the analysis to see if the causal estimate changes or not.
    # If our assumption was originally correct then the causal estimate shouldn’t change by much.
    refute1_results=model.refute_estimate(identified_estimand, estimate,
            method_name="random_common_cause")
    print(refute1_results, file=f)

    # Placebo Treatment Refuter:- Randomly assigns any covariate as a treatment and re-runs the analysis.
    # If our assumptions were correct then this newly found out estimate should go to 0.
    #refute2_results=model.refute_estimate(identified_estimand, estimate,
    #                                      method_name="placebo_treatment_refuter")
    #print(refute2_results)

    # Data Subset Refuter:- Creates subsets of the data(similar to cross-validation) and checks whether the causal estimates vary across subsets.
    # If our assumptions were correct there shouldn’t be much variation.
    refute3_results=model.refute_estimate(identified_estimand, estimate,
            method_name="data_subset_refuter")
    print(refute3_results, file=f)
    f.close()
