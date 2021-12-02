import streamlit as st
import os
from experiment_pipeline.data_loader import load_global_feature_set
import pandas as pd

SAVED_EXPERIMENT_DIR = "saved_experiments/"
DATA_DIR = "data/streamlit/"

def st_demo_weather():
    st.header("Motivation")
    st.write("""
        The motivation behind a lot of this work was to understand how severe weather impacts bus ridership in NYC. Let’s see how the tools we developed can be used to gain insight into that question.
    """)

    st.header("Establising a Baseline")
    st.write("""
        A sensible first step in assessing the extent to which weather conditions influence the number of people who ride the bus on a given day (in the case of our analysis, the B46 between 8/1 and 9/30/2021) is to establish a baseline predictive model trained in the absence of weather features (e.g., bus position, observation time, and schedule details). Then, we can compare the performance of our baseline against that of a substantially similar model trained on exactly the same data plus some additional weather features (e.g., precipitation, temperature, and humidity). If the augmented model performs better than our baseline, we can say that the inclusion of weather features improves our model’s ability to predict bus ridership. Conversely, if the augmented model performs inline with (or worse than) our baseline, we might begin to question if weather has anything to do with people’s decision to ride the bus or, at the very least, attempt to diagnose why our data (or representation thereof) did not lend itself to the prediction task.
    """)

    st.subheader("An Aside on the Prediction Task")
    st.write("""
        Just to we are all on the same page, the specific task we are training our model to perform is the prediction of the number of passengers on a specific vehicle at a specific time and place. To help illustrate that, here are the first five rows of the combined bus and weather training data.
    """)

    

    