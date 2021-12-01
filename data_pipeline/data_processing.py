import json
import random
import requests
import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def remove_non_normalProgreess_observations(df):
    df[df['progress_rate'] == 'normalProgress']

def remove_unique_trip_ids_with_high_pct_nan_passenger_count_readings(df, pct):
    unique_trip_ids = set(df['unique_trip_id'])
    for uuid in unique_trip_ids:
        temp = df[df['unique_trip_id'] == uuid]
        num_nan =  temp['passenger_count'].isna().sum()
        if num_nan / temp.shape[0] >= pct:
            df = df[df['unique_trip_id'] != uuid].copy()

def remove_delinquent_stops(df, delinquent_stops_dict):
    for direction in [0, 1]:
        for delinquent_stop in delinquent_stops_dict[direction]:
            df = df[df['next_stop_id'] != delinquent_stop]





    
