import argparse
import pandas as pd
import datetime
from evaluation import Evaluation
from feature_sets import feature_set_with_trip_id, feature_set_without_trip_id
from utils import date_train_test_split
import numpy as np
from sklearn.linear_model import LassoCV
from data_loader import load_global_feature_set

def run_experiment(
    global_feature_set,
    feature_extractor_fn,
    model,  
    stop_dict,
    dependent_variable="passenger_count",
    split_date=datetime.datetime(year=2021, month=9, day=27),
    test_period='1D',
    refit_interval=None
):    
    if refit_interval is None:
        train, test = feature_extractor_fn(date_train_test_split(global_feature_set, split_date, test_period))
        train_x = train.drop(columns=[dependent_variable])
        train_y = train[dependent_variable]
        test_x = test.drop(columns=[dependent_variable])
        test_y = test[dependent_variable]

        print("Fitting model on train set...")
        model.fit(train_x, train_y)

        print("Inference...")
        train_preds = model.predict(train_x)
        test_preds = model.predict(test_x)
        train['passenger_count_pred'] = train_preds
        test['passenger_count_pred'] = test_preds
    else:
        print(f"Refitting every {refit_interval}")
        initial_split_date = split_date
        refit_test_sets = []
        while split_date < initial_split_date + pd.Timedelta(test_period):
            print("Refitting...")
            train_refit, test_refit = feature_extractor_fn(date_train_test_split(global_feature_set, split_date, refit_interval))
            refit_test_sets.append(test_refit)
            train_x = train_refit.drop(columns=[dependent_variable])
            train_y = train_refit[dependent_variable]
            test_x = test_refit.drop(columns=[dependent_variable])
            test_y = test_refit[dependent_variable]
            model.fit(train_x, train_y)

            if (split_date == initial_split_date):
                train = train_refit
                train_preds = model.predict(train_x)
                train['passenger_count_pred'] = train_preds

            test_preds = model.predict(test_x)
            test_refit['passenger_count_pred'] = test_preds

            split_date += pd.Timedelta(refit_interval)
        
        test = pd.concat(refit_test_sets)

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

parser.add_argument(
    '-t',
    '--test_period',
    default='24H',
    help='Time period to hold out as test data'
)

parser.add_argument(
    '-r',
    '--refit_interval',
    default=None,
    help='Time period for refit'
)


if __name__ == "__main__":
    args = parser.parse_args()
    print("Arguments")
    print(args)

    data_dir = args.data_dir
    route_str = args.route

    ## Prepare globlal feature set
    df_route, stop_dict = load_global_feature_set(data_dir, route_str)

    experiment_eval = run_experiment(
        df_route,
        feature_set_without_trip_id,
        LassoCV(),
        stop_dict,
        test_period=args.test_period,
        refit_interval=args.refit_interval
    )

    print("Train evaluation")
    print(experiment_eval.basic_eval('train'))

    print("Test evaluation")
    print(experiment_eval.basic_eval('test'))
