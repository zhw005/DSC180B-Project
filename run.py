#!/usr/bin/env python

import sys
import json

from src import features_airbnb


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


if __name__ == '__main__':
    # run via:
    # python main.py data model
    targets = sys.argv[1:]
    main(targets)
