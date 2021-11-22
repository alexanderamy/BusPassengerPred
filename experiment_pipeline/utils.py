from datetime import datetime, timedelta
import pandas as pd

def date_train_test_split(segments_data, split_date: datetime, test_period):
    """ 
        Train: day 0 to split_date
        Test: split_date to split_date + test_period 
    """
    # Convert to timezone of dataframe
    split_date = pd.Timestamp(split_date).tz_localize(segments_data['timestamp'].dt.tz)
    date_test_end = (split_date + pd.Timedelta(test_period))
    train = segments_data[segments_data['timestamp'] < split_date].copy()
    test = segments_data[
        (segments_data['timestamp'] >= split_date) &
        (segments_data['timestamp'] <= date_test_end)
    ].copy()

    print(f"fitting on train data until {split_date}: {train.shape[0]:,} rows")
    print(f"testing from {split_date} to {date_test_end}: {test.shape[0]:,} rows")

    return train, test

    