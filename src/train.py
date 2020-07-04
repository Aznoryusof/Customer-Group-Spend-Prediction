import os
import sys

# Set system paths
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


import pandas as pd
import numpy as np

import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator

from src.utilities import *
from model_config import *


def _initialise_H2o():
    h2o.init(
        ip="localhost",
        port=54323,
        nthreads = -1, 
        max_mem_size = 8
    )


def _load_raw_data():
    spend = pd.read_csv("data/0_raw/spendata.csv")
    test_data = pd.read_csv("data/0_raw/testdata.csv")

    return spend, test_data


def _process_spend_data(spend):
    spend_copy = spend.copy()
    spend_clean = spend_copy.drop(
        ['t.158', "Unnamed: 0", "respondent.id"], axis=1
    )

    return spend_clean


def _process_test_data(test_data):
    test_data_copy = test_data.copy()
    test_data_clean = test_data_copy.drop(["Unnamed: 0", "respondent.id"], axis=1)

    return test_data_clean


def _select_columns(spend_clean, test_data_clean):
    selected_col = select_col_missing(
        spend_clean, "missing less than equals to", 0.30
    )

    return spend_clean[selected_col], test_data_clean[selected_col]


def _data_engineering_pipeline(spend, test_data):
    spend_clean = _process_spend_data(spend)
    test_data_clean = _process_test_data(test_data)
    spend_final, test_data_final = _select_columns(
        spend_clean, test_data_clean
    )

    return spend_final, test_data_final


def _generate_training_data(spend_final, seed):
    spend_hf = h2o.H2OFrame(spend_final)
    spend_hf["pov6"] = spend_hf["pov6"].asfactor()

    splits = spend_hf.split_frame(ratios=[0.8], seed = seed)  

    train = splits[0]
    valid = splits[1]

    return train, valid


def _tot_spend_model_features(spend_final):
    predictors = spend_final.columns[~(
        spend_final.columns.isin(["totshopping.rep", "pov6"])
    )].tolist()
    response = "totshopping.rep"

    return predictors, response


def _train_tot_spend(
    predictors,
    response,
    training_frame,
    validation_frame
):
    model_tot_spend = H2OGradientBoostingEstimator(
        seed = config_tot_spend["seed"], 
        stopping_metric = "RMSE",
        learn_rate = config_tot_spend["learn_rate"],
        max_depth = config_tot_spend["max_depth"],
        sample_rate = config_tot_spend["sample_rate"],
        col_sample_rate = config_tot_spend["col_sample_rate"]
    )

    model_tot_spend.train(
        x = predictors,
        y = response,
        training_frame = training_frame,
        validation_frame = validation_frame
    )

    return model_tot_spend


def _train_cust_grp(
    predictors, 
    response, 
    training_frame, 
    validation_frame
):
    model_cust_grp = H2OGradientBoostingEstimator(
        seed = config_cust_grp["seed"], 
        stopping_metric = "RMSE",
        learn_rate = config_cust_grp["learn_rate"],
        max_depth = config_cust_grp["max_depth"],
        sample_rate = config_cust_grp["sample_rate"],
        col_sample_rate = config_cust_grp["col_sample_rate"],
        balance_classes = True,
        distribution = "multinomial"
)

    model_cust_grp.train(
        x = predictors,
        y = response,
        training_frame = training_frame,
        validation_frame = validation_frame
    )

    return model_cust_grp


def _generate_tot_spend_predictions(model_tot_spend, test_data, predictors_tot_spend):
    
    test_data_hf = h2o.H2OFrame(test_data[predictors_tot_spend])
    
    pred_tot_spend = model_tot_spend.predict(
        test_data_hf
    )

    pred_tot_spend = pred_tot_spend.as_data_frame()
    pred_tot_spend = pred_tot_spend.rename(
        columns = {"predict":"totshopping.rep.pred"}
    )

    tot_spend_pred_output = pd.concat(
        [test_data, pred_tot_spend], axis=1
     )

    return tot_spend_pred_output


def _generate_cust_grp_predictions(model_cust_grp, test_data, predictors_cust_grp):
    test_data_hf = h2o.H2OFrame(test_data[predictors_cust_grp])

    pred_cust_grp = model_cust_grp.predict(
        test_data_hf
    )

    pred_cust_grp = pred_cust_grp.as_data_frame()
    pred_cust_grp = pred_cust_grp.rename(
        columns = {"predict":"pov6.pred"}
    )

    cust_grp_pred_output = pd.concat(
        [test_data, pred_cust_grp], axis=1
     )

    cust_grp_pred_output = cust_grp_pred_output.drop(
        ['p1', "p2", "p3", "p4", "p5", "p6"], axis=1
    )

    return cust_grp_pred_output


def _cust_grp_model_features(spend_final):
    predictors = spend_final.columns[~(
        spend_final.columns.isin(["totshopping.rep", "pov6"])
    )].tolist()
    response = "pov6"

    return predictors, response


def _message():
    print("\n")
    print("Model training in progress. Sit back, this may take a while...")
    print("\n")


def _print_performance_metrics(model_tot_spend, model_cust_grp, valid):
    print("\n\n")
    print("Prediction performance on validation data:")
    print("\n")
    print("The RMSE for the total spend prediction model is: {}".format(model_tot_spend.rmse(valid =True)))
    print("\n")
    print("The logloss for the customer group classifier is: {}".format(model_cust_grp.logloss(valid)))
    print("The confusion matrix for the customer group classifier is:")
    print(model_cust_grp.confusion_matrix(valid))


def pipeline():
    """This pipeline trains machine learning models to predict on the following
    variables:
    - totshopping.rep
    - pov6

    The algorithm used to train both model is based on H2o.ai's implementation
    of the Gradient Boosting Machine (GBM).
    """

    _message()

    # Setup, load, and clean data
    _initialise_H2o()
    spend, test_data = _load_raw_data()
    spend_final, test_data_final = _data_engineering_pipeline(
        spend, test_data
    )

    # Generate training and validation dataset
    train, valid = _generate_training_data(
        spend_final, config_tot_spend["seed"]
    )

    # Train models
    ## Train total spend model
    predictors_tot_spend, response_tot_spend = _tot_spend_model_features(spend_final)
    model_tot_spend = _train_tot_spend(
        predictors_tot_spend, response_tot_spend, train, valid
    )

    ## Train total customer groups model
    predictors_cust_grp, response_cust_grp = _cust_grp_model_features(spend_final)
    model_cust_grp = _train_cust_grp(
        predictors_cust_grp, response_cust_grp, train, valid
    )

    # Generate predictions on test_data
    ## Generate total spend predictions
    tot_spend_pred_output = _generate_tot_spend_predictions(
        model_tot_spend, test_data, predictors_tot_spend
    )

    ## Generate cust spend predictions
    cust_grp_pred_output = _generate_cust_grp_predictions(
        model_cust_grp, test_data, predictors_cust_grp
    )

    # Export predictions
    tot_spend_pred_output.to_csv("data/2_model_outputs/test_tot_spend_predictions.csv")
    cust_grp_pred_output.to_csv("data/2_model_outputs/test_cust_grp_predictions.csv")

    # Export model metrics
    _print_performance_metrics(model_tot_spend, model_cust_grp, valid)

    print("\n")
    print("Congratulations! Model trained and output generated!!")


if __name__ == "__main__":
    pipeline()
    