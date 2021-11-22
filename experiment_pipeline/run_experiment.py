import argparse
from datetime import datetime
from evaluation import Evaluation
from feature_sets import compute_stop_stats, baseline_important_features
from utils import custom_train_test_split
from sklearn.linear_model import LassoCV
from data_loader import load_global_feature_set
import pandas as pd

def run_experiment(
    global_feature_set,
    feature_extractor_fn,
    model,  
    stop_id_ls,
    dependent_variable="passenger_count",
    split_heuristic="datetime",
    test_size=0.1,
    split_datetime=datetime(year=2021, month=9, day=27, hour=0, minute=0),
    test_period="1D",
    refit_interval=None,
    random_state=0
):
    train, test = custom_train_test_split(
        global_feature_set, 
        split_heuristic=split_heuristic, 
        test_size=test_size, 
        split_datetime=split_datetime,
        test_period=test_period, 
        random_state=random_state
    )

    stop_stats = compute_stop_stats(train, test)
    if refit_interval is None:
        train_x, train_y, test_x, test_y = feature_extractor_fn(train, test, dependent_variable, stop_stats)

        # Fit
        print("Fitting model...")
        model.fit(train_x, train_y)

        # Inference
        print("Inference...")
        train_preds = model.predict(train_x)
        test_preds = model.predict(test_x)

        train['passenger_count_pred'] = train_preds
        test['passenger_count_pred'] = test_preds
    else:
        print(f"Refitting every {refit_interval}")
        initial_split = split_datetime
        refit_test_sets = []
        total_refits = pd.Timedelta(test_period) / pd.Timedelta(refit_interval)
        counter = 0
        while split_datetime < initial_split + pd.Timedelta(test_period):
            print(f"Refitting {counter/int(total_refits):.0%}...")
            train_refit, test_refit = custom_train_test_split(
                global_feature_set, 
                split_heuristic=split_heuristic, 
                test_size=test_size, 
                split_datetime=split_datetime,
                test_period=refit_interval, 
                random_state=random_state
            )
            train_x, train_y, test_x, test_y = feature_extractor_fn(train_refit, test_refit, dependent_variable, stop_stats)
            refit_test_sets.append(test_refit)
            model.fit(train_x, train_y)

            # Run inference only once (when split datetime is the initial split)
            if (split_datetime == initial_split):
                train_preds = model.predict(train_x)
                train['passenger_count_pred'] = train_preds

            test_preds = model.predict(test_x)
            test_refit['passenger_count_pred'] = test_preds

            split_datetime += pd.Timedelta(refit_interval)
            counter += 1

        test = pd.concat(refit_test_sets)

    # Eval
    return Evaluation(global_feature_set=global_feature_set, train=train, test=test, stop_id_ls=stop_id_ls, stop_stats=stop_stats)


parser = argparse.ArgumentParser()

parser.add_argument(
    'data_dir',
    default="../data/",
    type=str
)

parser.add_argument(
    'route', 
    help='The route number, i.e. "B46"',
    type=str
)

parser.add_argument(
    'station', 
    help='The weather station, i.e. "JFK"',
    type=str
)

parser.add_argument(
    'direction', 
    help='The route direction, i.e. 0 or 1',
    type=int
)


parser.add_argument(
    '-r',
    '--refit_interval',
    default=None,
    help='Refit interval specified as a pandas timedelta'
)

parser.add_argument(
    '-t',
    '--test_period',
    default='1D',
    help='Time period for testing specified as a pandas timedelta'
)

if __name__ == "__main__":
    args = parser.parse_args()
    print("Arguments", args)

    data_dir = args.data_dir
    route_str = args.route
    station_str = args.station
    direction_int = args.direction

    ## Prepare globlal feature set
    df_route, stop_id_ls = load_global_feature_set(
        data_dir, 
        route_str, 
        station_str, 
        direction_int
    )

    ## run experiment
    experiment_eval = run_experiment(
        global_feature_set=df_route,
        feature_extractor_fn=baseline_important_features,
        model=LassoCV(),
        stop_id_ls=stop_id_ls,
        dependent_variable="passenger_count",
        split_heuristic="datetime",
        test_period=args.test_period,
        refit_interval=args.refit_interval,
        random_state=0
    )

    print("-- Evaluation on train --")
    print(experiment_eval.basic_eval('train'))
    print()
    print("-- Evaluation on test --")
    print(experiment_eval.basic_eval('test'))
