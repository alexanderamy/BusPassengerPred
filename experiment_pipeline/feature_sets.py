import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def compute_stop_stats(train, test):
    # compute stop statistics using train set only
    # 25th Percentile
    def q25(x):
        return x.quantile(0.25)

    # 25th Percentile (median)
    def q50(x):
        return x.quantile(0.50)

    # 75th Percentile
    def q75(x):
        return x.quantile(0.75)

    stop_stats = train[['next_stop_id_pos', 'passenger_count']].groupby('next_stop_id_pos').agg({'passenger_count':['mean', 'std', q25, q50, q75]})
    return stop_stats


def add_stop_stats(train, test, stop_stats):
    train['avg_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'mean')].loc[x])
    train['std_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'std')].loc[x])
    test['avg_stop_passengers'] = test['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'mean')].loc[x])
    test['std_stop_passengers'] = test['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'std')].loc[x])
    return train, test


bus_features = [
    'vehicle_id',
    'next_stop_id_pos',
    'next_stop_est_sec',
    'month',    
    'DoW',  
    'hour',
    'minute',    
    'trip_id_comp_SDon_bool',
    'trip_id_comp_3_dig_id',
    # 'day',                   # always drop
    # 'year',                  # always drop
    # 'trip_id_comp_6_dig_id', # always drop
    # 'timestamp'              # always drop
]


weather_features = [
    'Precipitation',
    'Cloud Cover',
    'Relative Humidity',
    'Heat Index',
    'Max Wind Speed'
]


def bus_pos_and_obs_time(train, test, dependent_variable, stop_stats):
    # select features
    feature_set_bus = ['next_stop_id_pos', 'DoW','hour']
    feature_set_weather = []
    feature_set = feature_set_bus + feature_set_weather + [dependent_variable]
    train = train[feature_set].copy()
    test = test[feature_set].copy()

    # partition
    train_x = train.drop(columns=[dependent_variable])
    train_y = train[dependent_variable]
    test_x = test.drop(columns=[dependent_variable])
    test_y = test[dependent_variable]

    return train_x, train_y, test_x, test_y


def bus_features_with_stop_stats(train, test, dependent_variable, stop_stats):
    # select features
    feature_set_bus = bus_features
    feature_set_weather = []
    feature_set = feature_set_bus + feature_set_weather + [dependent_variable]
    train = train[feature_set].copy()
    test = test[feature_set].copy()

    # add stop stats
    train, test = add_stop_stats(train, test, stop_stats)

    # partition
    train_x = train.drop(columns=[dependent_variable])
    train_y = train[dependent_variable]
    test_x = test.drop(columns=[dependent_variable])
    test_y = test[dependent_variable]

    return train_x, train_y, test_x, test_y


def bus_and_weather_features_with_stop_stats(train, test, dependent_variable, stop_stats):
    # select features
    feature_set_bus = bus_features
    feature_set_weather = weather_features
    feature_set = feature_set_bus + feature_set_weather + [dependent_variable]
    train = train[feature_set].copy()
    test = test[feature_set].copy()

    # add stop stats
    train, test = add_stop_stats(train, test, stop_stats)

    # partition
    train_x = train.drop(columns=[dependent_variable])
    train_y = train[dependent_variable]
    test_x = test.drop(columns=[dependent_variable])
    test_y = test[dependent_variable]

    return train_x, train_y, test_x, test_y
