import os
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd

parser = argparse.ArgumentParser()

parser.add_argument(
    'read_file', 
    help='Name and extension of file to be processed',
    type=str
)

if __name__ == "__main__":
    args = parser.parse_args()
    print("Arguments", args)

    read_file = args.read_file
    save_file, _ = root, ext = os.path.splitext(read_file)
    save_file = save_file + '.csv'
    read_path = 'data/Bus/API Call'
    save_path = 'data/Bus/Segment Data - Raw'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # read data
    df = gpd.read_file(os.path.join(read_path, read_file), ignore_geometry=True)

    # remove vehicles that never report passenger_count
    vehicles = set(df['vehicle_id'])
    for vehicle in vehicles:
        vehicle_data = df[df['vehicle_id'] == vehicle]
        num_non_nan_passenger_counts = vehicle_data['passenger_count'].notna().sum() 
        if num_non_nan_passenger_counts == 0:
            df = df[df['vehicle_id'] != vehicle]
    df.reset_index(drop=True, inplace=True)

    # cast 'timestamp' column values as DateTime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # create unique_trip_id column
    df['unique_trip_id'] = df['trip_id'] + '-' + df['service_date'] + '-' + df['vehicle_id']

    # generate segment_data_dict
    unique_trip_ids = list(set(df['unique_trip_id']))
    segment_data_dict = {}
    i = 0
    for unique_trip_id in unique_trip_ids:
        unique_trip_id_df = df.copy()
        unique_trip_id_df = unique_trip_id_df[unique_trip_id_df['unique_trip_id'] == unique_trip_id]
        unique_trip_id_stops = list(set(unique_trip_id_df['next_stop_id']))
        for unique_trip_id_stop in unique_trip_id_stops:
            unique_trip_id_stop_df = unique_trip_id_df.copy()
            if not pd.isna(unique_trip_id_stop):
                unique_trip_id_stop_df = unique_trip_id_stop_df[unique_trip_id_stop_df['next_stop_id'] == unique_trip_id_stop]
                unique_trip_id_stop_df.reset_index(drop=True, inplace=True)
                observation_count = unique_trip_id_stop_df.shape[0]
                duration = unique_trip_id_stop_df.timestamp.max() - unique_trip_id_stop_df.timestamp.min()
                middle = observation_count // 2
                segment_data = unique_trip_id_stop_df.loc[middle].to_dict()
                segment_data['observation_count'] = observation_count
                segment_data['duration'] = duration
                segment_data_dict[i] = segment_data
                i += 1
            else:
                unique_trip_id_stop_df = unique_trip_id_stop_df[unique_trip_id_stop_df['next_stop_id'].isna() == True]
                unique_trip_id_stop_df.reset_index(drop=True, inplace=True)
                unique_trip_id_stop_dict = unique_trip_id_stop_df.to_dict('index')
                for index in unique_trip_id_stop_dict:
                    segment_data = unique_trip_id_stop_dict[index]
                    segment_data['observation_count'] = np.nan
                    segment_data['duration'] = np.nan
                    segment_data_dict[i] = segment_data
                    i += 1
    segment_data_df = pd.DataFrame.from_dict(segment_data_dict, orient='index')

    # save segment_data_df to drive as csv
    segment_data_df.to_csv(os.path.join(save_path, save_file), index=False)
    