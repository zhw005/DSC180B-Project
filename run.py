#!/usr/bin/env python

import sys
import json

from src import features_airbnb
from src import features_loan
from src import features_diabetes


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

if __name__ == '__main__':
    # run via:
    # python main.py data model
    targets = sys.argv[1:]
    main(targets)
