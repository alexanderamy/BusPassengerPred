import pandas as pd
import json

def load_global_feature_set(data_dir, route_str, station_str, direction_int):
    read_combined = f'{data_dir}/Combined/bus_{route_str}_{direction_int}_weather_{station_str}.csv'
    read_stops = f'{data_dir}/Bus/Route Data/{route_str}_stops.json'

    route_weather_df = pd.read_csv(read_combined)

    # add 'timestamp' column (for use in utils/custom_train_test_split and evaluation/Evaluation)
    route_weather_df['timestamp'] = pd.to_datetime(route_weather_df[['year', 'month', 'day', 'hour', 'minute']])

    with open(read_stops, 'r') as f:
        stop_id_dict = json.load(f)
        stop_id_dict = {int(k): v for k, v in stop_id_dict.items()}
    
    stop_id_ls = stop_id_dict[direction_int]
    
    # return route_weather_df, stop_dict
    return route_weather_df, stop_id_ls
