from datetime import timedelta
from evaluation import Evaluation

def date_train_test_split(segments_data, split_date, n_test_days):
    """ 
        Train: day 0 to split_date - 1
        Test: split_date to split_date + n_test_days 
    """
    date_test_end = split_date + timedelta(days=n_test_days)
    train = segments_data[segments_data['timestamp'].dt.date < split_date].copy()
    test = segments_data[
        (segments_data['timestamp'].dt.date >= split_date) &
        (segments_data['timestamp'].dt.date <= date_test_end)
    ].copy()

    return train, test

    