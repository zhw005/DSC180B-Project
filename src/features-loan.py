import pandas as pd
import numpy as np
import zipfile
from sklearn.preprocessing import OneHotEncoder

def loan_feature_engineer(in_fp,out_fp):
    zf = zipfile.ZipFile(in_fp) 
    loan = pd.read_csv(zf.open('application_data.csv'))
    # Drop columns that have > 35% missing values and are not relevant to the task.
    loan = loan.drop(['OWN_CAR_AGE','EXT_SOURCE_1','APARTMENTS_AVG','BASEMENTAREA_AVG','YEARS_BEGINEXPLUATATION_AVG','YEARS_BUILD_AVG','COMMONAREA_AVG','ELEVATORS_AVG','ENTRANCES_AVG','FLOORSMAX_AVG','FLOORSMIN_AVG','LANDAREA_AVG','LIVINGAPARTMENTS_AVG','LIVINGAREA_AVG','NONLIVINGAPARTMENTS_AVG','NONLIVINGAREA_AVG','APARTMENTS_MODE','BASEMENTAREA_MODE','YEARS_BEGINEXPLUATATION_MODE','YEARS_BUILD_MODE','COMMONAREA_MODE','ELEVATORS_MODE','ENTRANCES_MODE','FLOORSMAX_MODE','FLOORSMIN_MODE','LANDAREA_MODE','LIVINGAPARTMENTS_MODE','LIVINGAREA_MODE','NONLIVINGAPARTMENTS_MODE','NONLIVINGAREA_MODE','APARTMENTS_MEDI','BASEMENTAREA_MEDI','YEARS_BEGINEXPLUATATION_MEDI','YEARS_BUILD_MEDI','COMMONAREA_MEDI','ELEVATORS_MEDI','ENTRANCES_MEDI','FLOORSMAX_MEDI','FLOORSMIN_MEDI','LANDAREA_MEDI','LIVINGAPARTMENTS_MEDI','LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI','HOUSETYPE_MODE','TOTALAREA_MODE','WALLSMATERIAL_MODE','FONDKAPREMONT_MODE','EMERGENCYSTATE_MODE'], axis = 1)
    #OHE
    loan_ohe_features = ['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','WEEKDAY_APPR_PROCESS_START','OCCUPATION_TYPE','ORGANIZATION_TYPE']
    for i in loan_ohe_features:
        loan[i] = i + ': ' + loan[i].astype(str)
    loan_enc = OneHotEncoder(handle_unknown='ignore')
    loan_enc.fit(loan[loan_ohe_features])
    loan_ohe_features_result = loan_enc.transform(loan[loan_ohe_features]).toarray()
    # Create feature array
    loan_x = np.append(loan_ohe_features_result, loan.drop(loan_ohe_features, axis = 1).drop('TARGET', axis = 1).values, axis = 1)
    ohe_column_names = np.concatenate(loan_enc.categories_).tolist()
    other_column_names = loan.drop(loan_ohe_features, axis = 1).drop('TARGET', axis = 1).columns
    column_names = ohe_column_names + other_column_names.tolist()
    # Create target array
    loan_y = loan['TARGET']
    # Create output df
    all_columns = loan_y.tolist() + loan_x.tolist()
    df_x = pd.DataFrame(loan_x, columns = column_names)
    df_x['Target'] = loan_y
    feature_engineered_loan_data = df_x
    # Impute missing values
    fill_mean = ['AMT_ANNUITY','AMT_GOODS_PRICE','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_LAST_PHONE_CHANGE']
    fill_mode = ['CNT_FAM_MEMBERS', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE','AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']
    for i in fill_mean:
        feature_engineered_loan_data[i].fillna(int(feature_engineered_loan_data[i].mean()), inplace=True)
    for i in fill_mode:
        feature_engineered_loan_data[i].fillna(int(feature_engineered_loan_data[i].mode()), inplace=True)
    # save the df as csv to out_fp
    feature_engineered_loan_data.to_csv(out_fp)
