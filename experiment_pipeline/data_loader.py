import pandas as pd
import json
import numpy as np

def load_global_feature_set(data_dir, route_str):
    read_segments = f'{data_dir}/Bus/Segment Data - Processed/{route_str}_2021-10-18.csv'
    read_stops = f'{data_dir}/Bus/Route Data/{route_str}_stops.json'

    df_route = pd.read_csv(read_segments)
    df_route['timestamp'] = pd.to_datetime(df_route['timestamp'])

    with open(read_stops, 'r') as f:
        stop_dict = json.load(f)
        stop_dict = {int(k): v for k, v in stop_dict.items()}

    # map cyclical components of timestamp (i.e. day of week and time of day) to numerical values
    df_route['timestamp_DoW'] = df_route['timestamp'].dt.dayofweek
    df_route['timestamp_sec_from_midnight'] = df_route['timestamp'].dt.time.apply(lambda x: (x.hour * 3600) + (x.minute * 60) + x.second + (x.microsecond / 1000000))

    # encode cyclical features. reference: https://towardsdatascience.com/cyclical-features-encoding-its-about-time-ce23581845ca 
    # normalize between 0 - 2pi
    df_route['timestamp_DoW_norm'] = 2 * np.pi * df_route['timestamp_DoW'] / df_route['timestamp_DoW'].max()
    df_route['timestamp_sec_from_midnight_norm'] = 2 * np.pi * df_route['timestamp_sec_from_midnight'] / df_route['timestamp_sec_from_midnight'].max()
    df_route = df_route.drop(columns=['timestamp_DoW', 'timestamp_sec_from_midnight'])

    # compute sin / cos 'coordinate' encodings; drop timestamp_DoW_norm
    df_route['timestamp_DoW_sin'] = np.sin(df_route['timestamp_DoW_norm'])
    df_route['timestamp_DoW_cos'] = np.cos(df_route['timestamp_DoW_norm'])
    df_route['timestamp_sec_from_midnight_sin'] = np.sin(df_route['timestamp_sec_from_midnight_norm'])
    df_route['timestamp_sec_from_midnight_cos'] = np.cos(df_route['timestamp_sec_from_midnight_norm'])
    df_route = df_route.drop(columns=['timestamp_DoW_norm', 'timestamp_sec_from_midnight_norm'])
    
    return df_route, stop_dict