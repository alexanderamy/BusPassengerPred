import streamlit as st
import os
from experiment_pipeline.data_loader import load_global_feature_set
from experiment_pipeline.run_experiment import load_pickled_experiment

SAVED_EXPERIMENT_DIR = "saved_experiments/"

def st_report():
    st.header("Report")

    st.header("Explore Experiments")
    experiments = os.listdir(SAVED_EXPERIMENT_DIR)
    selected_experiment = st.selectbox("Select experiment", options=experiments)
    experiment_eval, experiment_model = load_pickled_experiment(SAVED_EXPERIMENT_DIR + selected_experiment)

    st.write(experiment_eval.plot_passenger_count_by_time_of_day('test'))

    bus_routes = ["B46", "Bx12"]
    selected_bus_route = st.selectbox("Select a bus route", options=bus_routes)
    try:
        df_route, stop_dict = load_global_feature_set("./data", selected_bus_route  )
    except:
        st.error(f"Failed to load data for route {selected_bus_route}")
        return
        
    st.write(f"**Route data:** {df_route.shape[0]:,} observations")
    st.dataframe(df_route.head())