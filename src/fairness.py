import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
    
    print( f"Group with {sensitive} = {S[0]} has a probability of positive prediction at {p0}" )
    print( f"Group with {sensitive} = {S[1]} has a probability of positive prediction at {p1}" )
    print( f"The difference is {abs(p0-p1)}" )
    
    return abs(p0 - p1)
    
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
    
    print( f"Group with {sensitive} = {S[0]} has a true positive rate of {p0}" )
    print( f"Group with {sensitive} = {S[1]} has a true positive rate of {p1}" )
    print( f"The difference is {abs(p0-p1)}" )
    
    return abs(p0 - p1)
    
def conditional_frequencies(model, features, sensitive, target, k = 5):
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
    
    
    print( f"After flipping the sensitve attribute {sensitive}, {round(diff, 4)*100}% of the prediction changes" )
    print( f"The average prediction of class {sensitive} = {S[0]} changed {p_flip_0 - p_ori_0}" )
    print( f"The average prediction of class {sensitive} = {S[1]} changed {p_flip_1 - p_ori_1}" )
    
    return abs(diff)
    
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
