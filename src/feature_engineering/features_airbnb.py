import pandas as pd
import numpy as np
import gzip
from sklearn.preprocessing import OneHotEncoder
import string
from nltk.corpus import stopwords
from collections import defaultdict
import os

def bow(sentence, words, wordId):
    """ helper function """
    feat = [0]*len(words)
    for w in sentence.split():
        if w in words:
            feat[wordId[w]] += 1
    return feat

def to_bow(col):
    """ helper function """
    wordCount = defaultdict(int)
    punctuation = set(string.punctuation)
    for d in col:
        r = ''.join([c for c in d.lower() if not c in punctuation])
        for w in r.split():
            wordCount[w] += 1

    counts = [(wordCount[w], w) for w in wordCount]
    counts.sort()
    counts.reverse()
    words = [x[1] for x in counts[:500]]
    wordId = dict(zip(words, range(len(words))))
    wordSet = set(words)

    return np.array([bow(s, words, wordId) for s in col]), words

def to_df(col):
    """ helper function """
    feat, words = to_bow(col)
    column_name = [col.name + ': ' + x for x in words]
    feat_text1 = pd.DataFrame(feat, columns = column_name)
    return feat_text1

def airbnb_feature_engineer(in_fp, out_fp):
    """
    Put them together, and return nothing
    in_fp: data file path
    out_fp: output file path
    """
    data = pd.read_csv(in_fp)

    # only keep columns below
    keep_columns = ['id', 'name', 'summary', 'neighborhood_overview', 'transit', 'host_response_time',
           'host_response_rate', 'host_is_superhost', 'host_neighbourhood',
           'host_listings_count', 'host_has_profile_pic', 'host_identity_verified', 'neighbourhood', 'zipcode',
           'is_location_exact', 'property_type', 'room_type', 'accommodates',
           'bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities',
           'review_scores_rating', 'security_deposit',
           'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights',
           'maximum_nights', 'has_availability',
           'availability_30', 'availability_60', 'availability_90',
           'availability_365', 'number_of_reviews', 'instant_bookable',
           'cancellation_policy', 'require_guest_profile_picture',
           'require_guest_phone_verification','reviews_per_month',
           'price']

    # only keep listings that have review_scores_rating
    listings = data.loc[data['review_scores_rating'].notnull()][keep_columns]

    # fillna
    listings.summary = listings.summary.fillna(' ')
    listings.neighborhood_overview = listings.neighborhood_overview.fillna(' ')
    listings.transit = listings.transit.fillna(' ')

    # fill null with mode
    listings.host_response_time = listings.host_response_time.fillna(listings.host_response_time.mode()[0])
    # fill null with mean
    listings.host_response_rate = listings.host_response_rate.fillna('999%').str[:-1].astype(int)
    listings.loc[listings.host_response_rate == 999, 'host_response_rate'] = int(listings.host_response_rate.mean())

    # impute with 0
    listings.security_deposit = listings.security_deposit.fillna('$0')
    listings.cleaning_fee = listings.cleaning_fee.fillna('$0')

    listings = listings.dropna().reset_index()

    # Features and target
    X = listings.iloc[:, 2:-1]
    y = listings.iloc[:, -1].str[1:].str.replace(',','').astype(float)
    y = y >= np.mean(y)
    y = y.astype(int)

    # Numerical Features
    columns_num = ['host_response_rate', 'host_listings_count', 'accommodates', 'bathrooms',
              'bedrooms', 'beds', 'security_deposit', 'cleaning_fee', 'guests_included',
              'extra_people', 'minimum_nights', 'maximum_nights',
               'availability_30', 'availability_60', 'availability_90',
               'availability_365', 'number_of_reviews', 'reviews_per_month', 'review_scores_rating']

    feat_num = X[columns_num]
    for i in ['security_deposit','cleaning_fee','extra_people']:
        feat_num[i] = feat_num[i].str[1:].str.replace(',','')
    feat_num = feat_num.astype(float)

    # Categorical Features
    columns_cat = ['host_response_time', 'host_is_superhost', 'host_neighbourhood',
               'host_has_profile_pic', 'host_identity_verified', 'neighbourhood', 'zipcode',
               'is_location_exact', 'property_type', 'room_type', 'bed_type', 'has_availability',
               'instant_bookable', 'cancellation_policy', 'require_guest_profile_picture',
               'require_guest_phone_verification']
    feat_cat = X[columns_cat]
    for i in feat_cat.columns:
        feat_cat[i] = i + ': ' + feat_cat[i]

    # one-hot
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(feat_cat)
    cat_array = enc.transform(feat_cat).toarray()
    cat_name = np.concatenate(enc.categories_).tolist()
    feat_cat = pd.DataFrame(cat_array, columns = cat_name)


    # Final Feature Matrix
    X_final = pd.concat([feat_num, feat_cat], axis = 1)


    # output the engineered features and targets
    X_final.to_csv(out_fp + 'airbnb_features.csv')
    y.to_csv(out_fp + 'airbnb_target.csv')
    return
