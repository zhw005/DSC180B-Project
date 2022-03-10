#!/usr/bin/env python

import sys
import json

from src.feature_engineering import features_airbnb
from src.feature_engineering import features_loan
from src.feature_engineering import features_diabetes
from src import fairness
from src import FN_FP
from src import XGBoost_Explanations

def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    '''

    if 'airbnb_features' in targets:
        with open('config/FeatureEng-params-airbnb.json') as fh:
            data_cfg = json.load(fh)

        print('start feature engineering for airbnb dataset')
        # feature engineering for airbnb dataset
        features_airbnb.airbnb_feature_engineer(**data_cfg)
        print('finished feature engineering for airbnb dataset')

    if 'loan_features' in targets:
        with open('config/FeatureEng-params-loan.json') as fh_loan:
            data_cfg_loan = json.load(fh_loan)

        print('start feature engineering for loan dataset')
        # feature engineering for airbnb dataset
        # The output is a dataset with both features and labels
        features_loan.loan_feature_engineer(**data_cfg_loan)
        print('finished feature engineering for loan dataset')

    if 'diabetes_features' in targets:
        with open('config/FeatureEng-params-diabetes.json') as fh:
            data_cfg_diabetes = json.load(fh)
            features_diabetes.feature_engineer(**data_cfg_diabetes)
            print('finished feature engineering for diabetes dataset')

    if 'fairness' in targets:
         with open('config/Fairness-example.json') as fh:
            fairness_cfg = json.load(fh)
            fairness.run(fairness_cfg)
            print(f'Fairness Analysis result generated at data/out/fairness/{fairness_cfg["out_fp"]}')

    if "FN_FP" in targets:
         with open('config/FN_FP-example.json') as fh:
            FN_FP_cfg = json.load(fh)
            FN_FP.run(FN_FP_cfg)
            print(f'FN_FP Analysis result generated at data/out/FN_FP/{FN_FP_cfg["out_fp"]}')
            
    if 'model_explanations' in targets:
        # loan
        with open('config/Model_Explanations_Example_loan.json') as fh:
            loan_ME_cfg = json.load(fh)
            XGBoost_Explanations.run_model_explanations(loan_ME_cfg)
            print(f'Loan model explanation Examples generated at data/out/model_explanations/{loan_ME_cfg["out_fp"]}')

        # airbnb
        with open('config/Model_Explanations_Example_airbnb.json') as fh:
            airbnb_ME_cfg = json.load(fh)
            XGBoost_Explanations.run_model_explanations(airbnb_ME_cfg)
            print(f'Airbnb model explanation Examples generated at data/out/model_explanations/{airbnb_ME_cfg["out_fp"]}')

        # healthcare
        with open('config/Model_Explanations_Example_healthcare.json') as fh:
            health_ME_cfg = json.load(fh)
            XGBoost_Explanations.run_model_explanations(health_ME_cfg)
            print(f'Healthcare model explanation Examples generated at data/out/model_explanations/{health_ME_cfg["out_fp"]}')

if __name__ == '__main__':
    # run via:
    # python main.py data model
    targets = sys.argv[1:]
    main(targets)
