import pandas as pd 
import numpy as np


def male(row):
    val = 0
    if row['gender'] == 'Male':
        val = 1
    return val

def female(row):
    val = 0
    if row['gender'] == 'Female':
        val = 1
    return val

def caucasian(row):
    val = 0
    if row['race'] == 'Caucasian':
        val = 1
    return val

def african(row):
    val = 0
    if row['race'] == 'AfricanAmerican':
        val = 1
    return val

def hispanic(row):
    val = 0
    if row['race'] == 'Hispanic':
        val = 1
    return val

def other(row):
    val = 0
    if row['race'] == 'Other':
        val = 1
    return val

def asian(row):
    val = 0
    if row['race'] == 'Asian':
        val = 1
    return val

def no_insulin(row):
    val = 0
    if row['insulin'] == 'No':
        val = 1
    return val

def insulin_steady(row):
    val = 0
    if row['insulin'] == 'Steady':
        val = 1
    return val

def insulin_up(row):
    val = 0
    if row['insulin'] == 'Up':
        val = 1
    return val

def insulin_down(row):
    val = 0
    if row['insulin'] == 'Down':
        val = 1
    return val

def diabetes_med(val):
    if val == 'Yes':
        val = 1
    else:
        val = 0
    return val

def readmitted(val):
    if val == 'NO':
        val = 0
    else:
        val = 1
    return val

def over_50(val):
    splitter = val.split('-')
    if '50' in splitter[0]:
        return 1
    else:
        return 0
    return

def feature_engineer(in_fp):
    diabetic_data = pd.read_csv(in_fp)
    cols = ['race','gender','age','time_in_hospital','insulin','diabetesMed','readmitted']
    relevant_cols = diabetic_data[cols]
    male_col = relevant_cols.apply(male, axis = 1)
    female_col = relevant_cols.apply(female, axis = 1)
    caucasian_col = relevant_cols.apply(caucasian, axis = 1)
    african_col = relevant_cols.apply(african, axis = 1)
    hispanic_col = relevant_cols.apply(hispanic, axis = 1)
    asian_col = relevant_cols.apply(asian, axis = 1)
    other_col = relevant_cols.apply(other, axis = 1)
    no_insulin_col = relevant_cols.apply(no_insulin, axis = 1)
    insulin_steady_col = relevant_cols.apply(insulin_steady, axis = 1)
    insulin_up_col = relevant_cols.apply(insulin_up, axis = 1)
    insulin_down_col = relevant_cols.apply(insulin_down, axis = 1)
    diabetes_med_col = relevant_cols['diabetesMed'].apply(diabetes_med)
    readmitted_col = relevant_cols['readmitted'].apply(readmitted)
    
    over_50_col = relevant_cols['age'].apply(over_50)
    diabetes_features = pd.DataFrame()
    diabetes_features['Male'] = male_col
    diabetes_features['Female'] = female_col
    diabetes_features['Caucasian'] = caucasian_col
    diabetes_features['African American'] = african_col
    diabetes_features['Hispanic'] = hispanic_col
    diabetes_features['Asian'] = asian_col
    diabetes_features['Other'] = other_col
    diabetes_features['No Insulin'] = no_insulin_col
    diabetes_features['Insulin Steady'] = insulin_steady_col
    diabetes_features['Insulin Down'] = insulin_down_col
    diabetes_features['Insulin Up'] = insulin_up_col 
    diabetes_features['Diabetes Med'] = diabetes_med_col
    diabetes_target = pd.DataFrame()
    diabetes_target['Readmitted'] = readmitted_col
    
    diabetes_features.to_csv('diabetes features.csv')
    diabetes_target.to_csv('diabetes target.csv')
    return 