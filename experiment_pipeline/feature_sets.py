import pandas as pd

def feature_set_with_trip_id(dataset):
    train, test = dataset
    train = pd.get_dummies(train, columns=['route', 'trip_id'])

    # compute average passenger_count by next_stop_id (using training only so as not to bake in information about the test set)
    train_stop_stats = train[
        ['next_stop_id', 'passenger_count']
    ].groupby('next_stop_id').agg({'passenger_count':['mean', 'std']})
    train['avg_stop_passengers'] = train['next_stop_id'].apply(lambda x: train_stop_stats[('passenger_count', 'mean')].loc[x])
    test['avg_stop_passengers'] = test['next_stop_id'].apply(lambda x: train_stop_stats[('passenger_count', 'mean')].loc[x])

    test = pd.get_dummies(test, columns=['route'])

    non_features =  ['service_date', 'vehicle_id', 'timestamp', 'prior_stop_id', 'next_stop_id']

    return train.drop(columns=non_features), test.drop(columns=non_features)

def feature_set_without_trip_id(dataset):
    train, test = dataset
    train = pd.get_dummies(train, columns=['route'])

    # compute average passenger_count by next_stop_id (using training only so as not to bake in information about the test set)
    train_stop_stats = train[
        ['next_stop_id', 'passenger_count']
    ].groupby('next_stop_id').agg({'passenger_count':['mean', 'std']})
    train['avg_stop_passengers'] = train['next_stop_id'].apply(lambda x: train_stop_stats[('passenger_count', 'mean')].loc[x])
    test['avg_stop_passengers'] = test['next_stop_id'].apply(lambda x: train_stop_stats[('passenger_count', 'mean')].loc[x])

    test = pd.get_dummies(test, columns=['route'])

    non_features =  ['service_date', 'trip_id', 'vehicle_id', 'timestamp', 'prior_stop_id', 'next_stop_id']

    return train.drop(columns=non_features), test.drop(columns=non_features)