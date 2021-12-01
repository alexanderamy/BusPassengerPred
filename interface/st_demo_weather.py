import streamlit as st
import os
from experiment_pipeline.data_loader import load_global_feature_set
import pandas as pd

SAVED_EXPERIMENT_DIR = "saved_experiments/"
DATA_DIR = "data/streamlit/"

def st_demo_weather():
    st.header("Demo: Weather Data")
    st.write("""
        ...
    """)

    # This is how you add an image
    st.image("https://www.pismocoastvillage.com/wp-content/plugins/awesome-weather/img/awe-backgrounds/drizzle.jpg")