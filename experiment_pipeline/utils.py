import datetime
import pandas as pd
from sklearn.model_selection import train_test_split

def custom_train_test_split(data, split_heuristic='arbitrary', test_size=0.1, split_date=(9, 27, 2021), split_time=(0, 0), num_test_periods=1, random_state=0):
    '''
    Train-test split data based on various heuristics

    Args:
        split_heuristic (str):  "arbitrary" --> regular-way train-test split, with partition sizes determined by test_size;
                                "date" --> train up to but not including split_date | test on split_date to split_date + num_test_periods (days);
                                "time" --> train up to but not including split_date @ split_time | test on split_date @ split_time to split_date @ split_time + num_test_periods (mins)
        test_size (float <= 1): determines size of test set (required when split_heuristic == "arbitrary")
        split_date (tup == (day, month, year)): date on which data is split into train and test sets (required when split_heuristic == "date" or "time")
        split_time (tup == (hour, minute)): time at which data is split into train and test sets (required when split_heuristic == "time")
        num_test_periods (int): represents number of periods to include in test set (split_heuristic == 'date' --> n == days; split_heuristic == 'date' --> n == minutes);
                                note that num_test_periods == 0 --> test only the single day or minute corresponding to split_date / split_date
        random_state (int): random state for reproducibility
    
    Returns:
        train (DataFrame):  DataFrame corresponding to train set
        test (DataFrame):   DataFrame corresponding to test set
    '''

    data = data.copy()

    if split_heuristic == 'arbitrary':
        train, test = train_test_split(data, test_size=test_size, random_state=random_state)

    elif split_heuristic == 'date':
        month, day, year = split_date
        split_date = datetime.date(year=year, month=month, day=day)
        date_test_end = split_date + datetime.timedelta(days=num_test_periods)
        train = data[data['timestamp'].dt.date < split_date]
        test = data[(data['timestamp'].dt.date >= split_date) & (data['timestamp'].dt.date <= date_test_end)]

    elif split_heuristic == 'time':
        month, day, year = split_date
        hour, minute = split_time
        split_time = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute)
        time_test_end = split_time + datetime.timedelta(minutes=num_test_periods)
        train = data[data['timestamp'] < split_time]
        test = data[(data['timestamp'] >= split_time) & (data['timestamp'] <= time_test_end)]

    return train, test