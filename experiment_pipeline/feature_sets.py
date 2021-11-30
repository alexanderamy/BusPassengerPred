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


def stop_pos_only(train, test, dependent_variable, stop_stats):
    non_features_bus = ['vehicle_id', 'next_stop_est_sec', 'day', 'month', 'year', 'DoW', 'hour', 'minute', 'trip_id_comp_SDon_bool', 'trip_id_comp_6_dig_id', 'trip_id_comp_3_dig_id', 'timestamp']
    non_features_weather = ['Precipitation', 'Cloud Cover', 'Relative Humidity', 'Heat Index', 'Max Wind Speed']
    non_features = non_features_bus + non_features_weather

    train = train.drop(columns=non_features)
    test = test.drop(columns=non_features)

    # partition train / test sets into features and targets
    train_x = train.drop(columns=[dependent_variable])
    train_y = train[dependent_variable]
    test_x = test.drop(columns=[dependent_variable])
    test_y = test[dependent_variable]

    return train_x, train_y, test_x, test_y


def bus_features_with_stop_stats(train, test, dependent_variable, stop_stats):
    # add columns for mean / std passenger_count per stop to train set
    train['avg_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'mean')].loc[x])
    train['std_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'std')].loc[x])

    # # add columns for q25 / q50 / q75 passenger_count per stop to train set
    # train['q25_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'q25')].loc[x])
    # train['q50_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'q50')].loc[x])
    # train['q75_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'q75')].loc[x])

    # add columns for mean / std passenger_count per stop to test set
    test['avg_stop_passengers'] = test['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'mean')].loc[x])
    test['std_stop_passengers'] = test['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'std')].loc[x])

    # # add columns for q25 / q50 / q75 passenger_count per stop to test set
    # test['q25_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'q25')].loc[x])
    # test['q50_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'q50')].loc[x])
    # test['q75_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'q75')].loc[x])

    # drop non_features from train / test sets
    non_features_bus = ['day', 'year', 'trip_id_comp_6_dig_id', 'timestamp']
    non_features_weather = ['Precipitation', 'Cloud Cover', 'Relative Humidity', 'Heat Index', 'Max Wind Speed']
    non_features = non_features_bus + non_features_weather

    train = train.drop(columns=non_features)
    test = test.drop(columns=non_features)

    # partition train / test sets into features and targets
    train_x = train.drop(columns=[dependent_variable])
    train_y = train[dependent_variable]
    test_x = test.drop(columns=[dependent_variable])
    test_y = test[dependent_variable]

    return train_x, train_y, test_x, test_y


def bus_and_weather_features_with_stop_stats(train, test, dependent_variable, stop_stats):
    # add columns for mean / std passenger_count per stop to train set
    train['avg_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'mean')].loc[x])
    train['std_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'std')].loc[x])

    # # add columns for q25 / q50 / q75 passenger_count per stop to train set
    # train['q25_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'q25')].loc[x])
    # train['q50_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'q50')].loc[x])
    # train['q75_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'q75')].loc[x])

    # add columns for mean / std passenger_count per stop to test set
    test['avg_stop_passengers'] = test['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'mean')].loc[x])
    test['std_stop_passengers'] = test['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'std')].loc[x])

    # # add columns for q25 / q50 / q75 passenger_count per stop to test set
    # test['q25_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'q25')].loc[x])
    # test['q50_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'q50')].loc[x])
    # test['q75_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'q75')].loc[x])

    # drop non_features from train / test sets
    non_features_bus = ['day', 'year', 'trip_id_comp_6_dig_id','timestamp']
    non_features_weather = []
    non_features = non_features_bus + non_features_weather

    train = train.drop(columns=non_features)
    test = test.drop(columns=non_features)

    # partition train / test sets into features and targets
    train_x = train.drop(columns=[dependent_variable])
    train_y = train[dependent_variable]
    test_x = test.drop(columns=[dependent_variable])
    test_y = test[dependent_variable]

    return train_x, train_y, test_x, test_y










# DON'T USE: TERRIBLE PERFORMANCE

def normalized_bus_features_with_stop_stats(train, test, dependent_variable, stop_stats):
    # add columns for mean / std passenger_count per stop to train set
    train['avg_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'mean')].loc[x])
    train['std_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'std')].loc[x])

    # # add columns for q25 / q50 / q75 passenger_count per stop to train set
    # train['q25_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'q25')].loc[x])
    # train['q50_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'q50')].loc[x])
    # train['q75_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'q75')].loc[x])

    # add columns for mean / std passenger_count per stop to test set
    test['avg_stop_passengers'] = test['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'mean')].loc[x])
    test['std_stop_passengers'] = test['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'std')].loc[x])

    # # add columns for q25 / q50 / q75 passenger_count per stop to test set
    # test['q25_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'q25')].loc[x])
    # test['q50_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'q50')].loc[x])
    # test['q75_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'q75')].loc[x])

    # drop non_features from train / test sets
    non_features_bus = ['timestamp', 'year', 'day', 'trip_id_comp_6_dig_id']
    non_features_weather = ['Precipitation', 'Cloud Cover', 'Relative Humidity', 'Heat Index', 'Max Wind Speed']
    non_features = non_features_bus + non_features_weather

    train = train.drop(columns=non_features)
    test = test.drop(columns=non_features)

    # fit MinMaxScaler on training data
    scaler = MinMaxScaler()
    scaler.fit(train)
    train = pd.DataFrame(scaler.transform(train), index=train.index, columns=train.columns)
    test = pd.DataFrame(scaler.transform(test), index=test.index, columns=test.columns)

    # partition train / test sets into features and targets
    train_x = train.drop(columns=[dependent_variable])
    train_y = train[dependent_variable]
    test_x = test.drop(columns=[dependent_variable])
    test_y = test[dependent_variable]

    return train_x, train_y, test_x, test_y
    

def normalized_bus_and_weather_features_with_stop_stats(train, test, dependent_variable, stop_stats):
    # add columns for mean / std passenger_count per stop to train set
    train['avg_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'mean')].loc[x])
    train['std_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'std')].loc[x])

    # # add columns for q25 / q50 / q75 passenger_count per stop to train set
    # train['q25_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'q25')].loc[x])
    # train['q50_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'q50')].loc[x])
    # train['q75_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'q75')].loc[x])

    # add columns for mean / std passenger_count per stop to test set
    test['avg_stop_passengers'] = test['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'mean')].loc[x])
    test['std_stop_passengers'] = test['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'std')].loc[x])

    # # add columns for q25 / q50 / q75 passenger_count per stop to test set
    # test['q25_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'q25')].loc[x])
    # test['q50_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'q50')].loc[x])
    # test['q75_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'q75')].loc[x])

    # drop non_features from train / test sets
    non_features_bus = ['timestamp', 'year', 'day', 'trip_id_comp_6_dig_id']
    non_features_weather = []
    non_features = non_features_bus + non_features_weather

    train = train.drop(columns=non_features)
    test = test.drop(columns=non_features)

    # fit scaler on training data
    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    scaler.fit(train)
    train = pd.DataFrame(scaler.transform(train), index=train.index, columns=train.columns)
    test = pd.DataFrame(scaler.transform(test), index=test.index, columns=test.columns)

    # partition train / test sets into features and targets
    train_x = train.drop(columns=[dependent_variable])
    train_y = train[dependent_variable]
    test_x = test.drop(columns=[dependent_variable])
    test_y = test[dependent_variable]

    return train_x, train_y, test_x, test_y