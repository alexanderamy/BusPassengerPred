import argparse
import pandas as pd
import datetime
from evaluation import Evaluation
from feature_sets import feature_set_with_trip_id, feature_set_without_trip_id
from utils import date_train_test_split,custom_train_test_split
import numpy as np
from sklearn.linear_model import LassoCV
from data_loader import load_global_feature_set

def run_experiment(
    global_feature_set,
    feature_extractor_fn,
    model,  
    stop_dict,
    dependent_variable="passenger_count",
    split_heuristic="date",
    test_size=0.1,
    split_date=(9, 27, 2021),
    split_time=(0, 0),
    num_test_periods=1,
    random_state=0
):
    # Feature selection
    print("Selecting features...")
    # split_date = datetime.date(year=2021, month=9, day=27)
    # train, test = feature_extractor_fn(date_train_test_split(global_feature_set, split_date, 1))

    train, test = custom_train_test_split(
        global_feature_set, 
        split_heuristic=split_heuristic, 
        test_size=test_size, 
        split_date=split_date, 
        split_time=split_time, 
        num_test_periods=num_test_periods, 
        random_state=random_state
    )

    train, test = feature_extractor_fn(train, test)

    train_x = train.drop(columns=[dependent_variable])
    train_y = train[dependent_variable]
    test_x = test.drop(columns=[dependent_variable])
    test_y = test[dependent_variable]

    # Fit
    print("Fitting model...")
    model.fit(train_x, train_y)

    # Inference
    print("Inference...")
    train_preds = model.predict(train_x)
    test_preds = model.predict(test_x)

    train['passenger_count_pred'] = train_preds
    test['passenger_count_pred'] = test_preds

    # Eval
    return Evaluation(global_feature_set=global_feature_set, train=train, test=test, stop_dict=stop_dict)


parser = argparse.ArgumentParser()

parser.add_argument(
    'data_dir',
    default="../data/",
    type=str
)

parser.add_argument(
    'route', 
    help='The route number, i.e. "B46"'
)


if __name__ == "__main__":
    args = parser.parse_args()

    data_dir = args.data_dir
    route_str = args.route

    ## Prepare globlal feature set
    df_route, stop_dict = load_global_feature_set()

    experiment_eval = run_experiment(
        df_route,
        feature_set_with_trip_id,
        LassoCV(),
        stop_dict
    )

    print("Evaluation")
    print(experiment_eval.basic_eval('train'))
