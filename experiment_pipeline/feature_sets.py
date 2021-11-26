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


def baseline_important_features(train, test, dependent_variable, stop_stats):
    # drop non_features from train / test sets
    non_features = ['timestamp', 'year', 'day', 'trip_id_comp_6_dig_id']
    train = train.drop(columns=non_features)
    test = test.drop(columns=non_features)

    # partition train / test sets into features and targets
    train_x = train.drop(columns=[dependent_variable])
    train_y = train[dependent_variable]
    test_x = test.drop(columns=[dependent_variable])
    test_y = test[dependent_variable]

    return train_x, train_y, test_x, test_y


def baseline_important_features_with_stop_stats(train, test, dependent_variable, stop_stats):
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
    # train['q25_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'q25')].loc[x])
    # train['q50_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'q50')].loc[x])
    # train['q75_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'q75')].loc[x])

    # drop non_features from train / test sets
    non_features = ['timestamp', 'year', 'day', 'trip_id_comp_6_dig_id']
    train = train.drop(columns=non_features)
    test = test.drop(columns=non_features)

    # partition train / test sets into features and targets
    train_x = train.drop(columns=[dependent_variable])
    train_y = train[dependent_variable]
    test_x = test.drop(columns=[dependent_variable])
    test_y = test[dependent_variable]

    return train_x, train_y, test_x, test_y