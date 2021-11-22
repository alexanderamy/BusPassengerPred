import streamlit as st
import os
from experiment_pipeline.data_loader import load_global_feature_set
import pickle
import sys

# Hack for pickling evaluation class
from experiment_pipeline import evaluation
sys.modules['evaluation'] = evaluation

SAVED_EXPERIMENT_DIR = "saved_experiments/"

def st_report():
    st.header("Report")

    st.header("Explore Experiments")
    experiments = os.listdir(SAVED_EXPERIMENT_DIR)
    selected_experiment = st.selectbox("Select experiment", options=experiments)
    with open(SAVED_EXPERIMENT_DIR + selected_experiment, "rb") as f:
        loaded_experiment = pickle.load(f)

    st.write(loaded_experiment["eval"].plot_passenger_count_by_time_of_day('test'))

    bus_routes = ["B46", "Bx12"]
    selected_bus_route = st.selectbox("Select a bus route", options=bus_routes)
    try:
        df_route, stop_dict = load_global_feature_set("./data", selected_bus_route  )
    except:
        st.error(f"Failed to load data for route {selected_bus_route}")
        return
        
    st.write(f"**Route data:** {df_route.shape[0]:,} observations")
    st.dataframe(df_route.head())