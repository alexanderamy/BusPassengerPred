import datetime
import pandas as pd
from sklearn.model_selection import train_test_split

def custom_train_test_split(
    data, 
    split_heuristic='datetime', 
    test_size=0.1, 
    split_datetime=datetime.datetime(year=2021, month=9, day=27, hour=0, minute=0), 
    test_period='1D', 
    random_state=0
):
    '''
    Train-test split data based on various heuristics

    Args:
        split_heuristic (str):  "arbitrary" --> regular-way train-test split, with partition sizes determined by test_size;
                                "datetime" --> train up to but not including split_datetime | test on split_datetime to split_datetime + test_period (pandas timedelta)
        test_size (float <= 1): determines size of test set (required when split_heuristic == "arbitrary")
        split_datetime: datetime on which data is split into train and test sets (required when split_heuristic == "datetime")
        test_period (str): represents timeperiod to include in test set as a pandas timedelta (i.e. '1D', https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html)
        random_state (int): random state for reproducibility
    
    Returns:
        train (DataFrame):  DataFrame corresponding to train set
        test (DataFrame):   DataFrame corresponding to test set
    '''

    data = data.copy()

    if split_heuristic == 'arbitrary':
        train, test = train_test_split(data, test_size=test_size, random_state=random_state)

    elif split_heuristic == 'datetime':
        # Convert to timezone of dataframe
        split_datetime = pd.Timestamp(split_datetime).tz_localize(data['timestamp'].dt.tz)
        date_test_end = (split_datetime + pd.Timedelta(test_period))
        train = data[data['timestamp'] < split_datetime]
        test = data[(data['timestamp'] >= split_datetime) & (data['timestamp'] <= date_test_end)]

        print(f"split: fitting on train data until {split_datetime}: {train.shape[0]:,} rows")
        print(f"split: testing from {split_datetime} to {date_test_end}: {test.shape[0]:,} rows")

    return train, test