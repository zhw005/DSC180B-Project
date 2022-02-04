import pandas as pd

def group_fairness(test_data, col, class, predicted_y = 'predicted_y', test_y = 'test_y'):
    """
    return: group_fairness of one class

    test_data: test data with true label and predictive label and features
    col: sensitive feature column name
    class: sensitive attribute class (e.g. m/f/0/1)
    predicted_y: predicted value column name
    test_y: true label column name
    """
    return test_data.loc[test_data[col] == class].loc[test_data[predicted_y] == 1].shape[0]\
            /test_data.loc[test_data[col] == class].shape[0]

def predictive_parity(test_data, col, class, predicted_y = 'predicted_y', test_y = 'test_y'):
    """
    return: predictive_parity of one class

    test_data: test data with true label and predictive label and features
    col: sensitive feature column name
    class: sensitive attribute class (e.g. m/f/0/1)
    predicted_y: predicted value column name
    test_y: true label column name
    """
    return test_data.loc[test_data[col] == class].loc[test_data[predicted_y] == 1].loc[test_data[test_y] == 1].shape[0]\
                        /test_data.loc[test_data[col] == class].loc[test_data[predicted_y] == 1].shape[0]

def predictive_equality(test_data, col, class, predicted_y = 'predicted_y', test_y = 'test_y'):
    """
    return: predictive_equality of one class

    test_data: test data with true label and predictive label and features
    col: sensitive feature column name
    class: sensitive attribute class (e.g. m/f/0/1)
    predicted_y: predicted value column name
    test_y: true label column name
    """
    return test_data.loc[test_data[col] == class].loc[test_data[test_y] == 0].loc[test_data[predicted_y] == 1].shape[0]\
                        /test_data.loc[test_data[col] == class].loc[test_data[test_y] == 0].shape[0]

def predictive_opportunity(test_data, col, class, predicted_y = 'predicted_y', test_y = 'test_y'):
    """
    return: predictive_equality of one class

    test_data: test data with true label and predictive label and features
    col: sensitive feature column name
    class: sensitive attribute class (e.g. m/f/0/1)
    predicted_y: predicted value column name
    test_y: true label column name
    """
    return test_data.loc[test_data[col] == class].loc[test_data[test_y] == 1].loc[test_data[predicted_y] == 0].shape[0]\
                        /test_data.loc[test_data[col] == class].loc[test_data[test_y] == 1].shape[0]
