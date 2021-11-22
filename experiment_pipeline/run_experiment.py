import argparse
from evaluation import Evaluation
from feature_sets import compute_stop_stats, baseline_important_features
from utils import custom_train_test_split
from sklearn.linear_model import LassoCV
from data_loader import load_global_feature_set

def run_experiment(
    global_feature_set,
    feature_extractor_fn,
    model,  
    stop_id_ls,
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
    
    train, test = custom_train_test_split(
        global_feature_set, 
        split_heuristic=split_heuristic, 
        test_size=test_size, 
        split_date=split_date, 
        split_time=split_time, 
        num_test_periods=num_test_periods, 
        random_state=random_state
    )

    stop_stats = compute_stop_stats(train, test)

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

if __name__ == "__main__":
    args = parser.parse_args()

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
        split_heuristic="date",
        test_size=0.1,
        split_date=(9, 27, 2021),
        split_time=(0, 0),
        num_test_periods=0,
        random_state=0
    )

    print("Evaluation")
    print(experiment_eval.basic_eval('train'))
