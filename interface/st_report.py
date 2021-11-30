import streamlit as st
import os
from experiment_pipeline.data_loader import load_global_feature_set
from run_experiment import load_pickled_experiment
import json
import pandas as pd

SAVED_EXPERIMENT_DIR = "saved_experiments/"
DATA_DIR = "data/streamlit/"

def load_bus_segment_data(route, processed=True):
    try:
        df_route = pd.read_pickle(os.path.join(
            DATA_DIR,
            "processed" if processed else "raw",
            f"{route}_2021-10-18.pickle"
        ))
    except Exception as e:
        st.error(f"Failed to load data for route {route}: {e}")
        return
    return df_route

def st_report():
    st.write("""
    # Demo Report
    *Made by Alexander Amy, Anton Abilov & Sanket Shah for Urban Data.*
    """)
    # st.subheader("Requirements")
    # st.write("""
    # **Final writeup:** A 1,500 to 2,000 word writeup, written for a general audience, that summarizes your analysis and findings. 
    # Your analysis should be understandable to a smart and curious non-specialist who is paying attention - 
    # imagine a regular reader of FiveThirtyEight or The New York Times Upshot. 

    # - [Example 1](https://fivethirtyeight.com/features/in-the-end-people-may-really-just-want-to-date-themselves/)
    # - [Example 2](https://www.nytimes.com/interactive/2021/06/30/opinion/environmental-inequity-trees-critical-infrastructure.html)
    # - [Example 3](https://www.nytimes.com/2014/03/23/opinion/sunday/the-geography-of-fame.html)
    # - [Example 4](https://www.washingtonpost.com/politics/2020/06/20/barr-says-theres-no-systemic-racism-policing-our-data-say-attorney-general-is-wrong/)

    # In addition to the writeup, please submit:

    # 1. all code which reproduces your analysis (this repository)
    # 2. a short (250 - 500 word) technical description of the analyses you ran.
    # """)

    st.header("Introduction / Motivation")
    st.write("We open source a repository for working with the NYC bus data, etc...")

    st.write("Something about the experiments we want to run, i.e. adding weather data.")

    st.header("Data Collection")
    st.write("""
    - How did we get the data
    - The value of open city data
    - Some technical details, link to notebook for in-depth
    """)


    bus_routes = ["B46", "Bx12", "M15"]
    selected_bus_route = st.selectbox("Select a bus route", options=bus_routes)
    df_route = load_bus_segment_data(selected_bus_route, processed=False)
        
    st.write(f"**Route data:** {df_route.shape[0]:,} observations")
    st.dataframe(df_route.head())

    st.map(
        data=df_route.sample(100)
    )

    st.header("Analysis")
    st.subheader("Feature selection")
    st.write("Correlation matrix")
    st.write("Streamlit table which shows features we focus on & their description")

    st.header("Experiments")
    st.write("Select to explore an experiment - loads experiment with according description & evaluation of results")
    experiments = os.listdir(SAVED_EXPERIMENT_DIR)
    selected_experiment = st.selectbox("Select experiment", options=experiments)
    experiment_eval, experiment_model = load_pickled_experiment(SAVED_EXPERIMENT_DIR + selected_experiment)

    fig1, fig2, fig3 = experiment_eval.plot_passenger_count_by_time_of_day('test')
    st.write(fig1)


    st.header("Conclusion")