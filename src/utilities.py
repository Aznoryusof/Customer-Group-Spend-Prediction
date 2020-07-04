import numpy as np


def split_train_test(
    data: pd.DataFrame, test_ratio: float, seed: int
) -> tuple:
    """A function that splits data to a train and test set, according 
    to a test_ratio.

    Args:
        data (pd.DataFrame): A df that has to be split to train and test components.
        test_ratio (float): The proportion of test data out of total length of df.
        seed (int): The random seed to ensure replicability

    Returns:
        tuple: Contains the following data in a tuple: (train set, test set)
    """
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(data)) 
    test_set_size = int(len(data) * test_ratio) 
    test_indices = shuffled_indices[:test_set_size] 
    train_indices = shuffled_indices[test_set_size:] 
    
    return data.iloc[train_indices], data.iloc[test_indices]


def select_col_missing(
    df: pd.DataFrame, comparison_type: str, prop_missing: float
) -> list:
    """Returns a list of columns from input df, given a user specified condition
    of acceptable proportion of missing values in the dataset. 

    Args:
        df (pd.DataFrame): The df to be evaluated on.
        comparison_type (str): The condition for comparison. Allows for three values:
            - "missing more than"
            - "missing less than equals to"
            - "missing equals"
        prop_missing (float): The acceptable proportion of missing data.

    Returns:
        list: List of columns that satisfy the missing thresholds set above
    """    
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