import numpy as np


def split_train_test(data, test_ratio):
    """A function that splits data to a train and test set, according 
    to a test_ratio.
    """    
    shuffled_indices = np.random.permutation(len(data)) 
    test_set_size = int(len(data) * test_ratio) 
    test_indices = shuffled_indices[:test_set_size] 
    train_indices = shuffled_indices[test_set_size:] 
    
    return data.iloc[train_indices], data.iloc[test_indices]


def select_col_missing(df, comparison_type, prop_missing):
    df_copy = df.copy()

    if comparison_type == "missing more than":
        selected_col = df_copy.columns[(df_copy.isnull().sum() / df_copy.shape[0]) > prop_missing]

    elif comparison_type == "missing less than equals to":
        selected_col = df_copy.columns[(df_copy.isnull().sum() / df_copy.shape[0]) <= prop_missing]

    elif comparison_type == "missing equals":
        selected_col = df_copy.columns[(df_copy.isnull().sum() / df_copy.shape[0]) == prop_missing]
        
    missing_no = len(selected_col)
    prop_missing_perc = int(prop_missing * 100)
    missing_no_prop = int((missing_no / df_copy.shape[1]) * 100)
    
    print("Checking data sparsity:")
    if comparison_type == "missing more than":
        print("{} variables or about {}% of features with more than {}% missing values".format(missing_no, missing_no_prop, prop_missing_perc))
    
    elif comparison_type == "missing less than equals to":
        print("{} variables or about {}% of features with less than equals to {}% missing values".format(missing_no, missing_no_prop, prop_missing_perc))
    
    elif comparison_type == "missing equals":
        print("{} variables or about {}% of features with {}% missing values".format(missing_no, missing_no_prop, prop_missing_perc))

    return selected_col