# DSC180B: Explainable AI
This is a repository that contains code for DSC180B section B06's Q2 Project: Explainable AI.

"build-script": "zhw005/dsc180a-project"

## Authors
- [Jerry (Yung-Chieh) Chan](https://github.com/JerryYC)
- [Apoorv Pochiraju](https://github.com/apochira)
- [Zhendong Wang](https://github.com/zhw005)
- [Yujie Zhang](https://github.com/yujiezhang0914)

## Introduction
In our project, we will be focusing on using different techniques from causal inferences and explainable AI to interpret various machine learning models across various domains. In particular, we are interested in three domains - healthcare, banking, and the housing market. Within each domain, we are going to train several machine learning models first:XGBoost, LightGBM, TabNet, and SVM. And we have four goals in general: 
1) Explaining black-box models in general and finding out to what extent the learning models agree and disagree with each other in terms of predictive multiplicity and FN/FP predictions; 
2) Assessing the fairness of each learning algorithm; 
3) Generating recourse for individuals - a set of minimal actions to change the prediction of those black-box models;
4) Evaluating explanations using domain knowledge.

## Running the project

 target | config | experiment |
| :---: | :---: | :---: |
| airbnb_features | 'config/FeatureEng-params-airbnb.json' | Do feature engineering for airbnb dataset |
| loan_features | 'config/FeatureEng-params-loan.json' | Do feature engineering for loan dataset |
| diabetes_features | 'config/FeatureEng-params-diabetes.json' | Do feature engineering for diabetes dataset |
| fairness | 'config/Fairness-example.json' | Do fairness evaluation |
| FN_FP | 'config/FN_FP-example.json' | Do False Negative and False Positive explanation |
