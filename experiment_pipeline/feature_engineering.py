import pandas as pd

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
