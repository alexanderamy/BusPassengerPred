import streamlit as st
from interface.st_data import st_data
from interface.st_demo_weather import st_demo_weather
from interface. st_eval_harness import st_eval_harness

# Main page
def render_landing_page():
    st.subheader("Landing page")
    st.write("Todo: write a description of the project")

    st.write("We open source a repository for working with the NYC bus data, etc...")

    st.write("Something about the experiments we want to run, i.e. adding weather data.")


    st.write("**[Github repository](https://github.com/Cornell-Tech-Urban-Tech-Hub/buswatcher-insights/)**")

PAGES = {
    "Landing page": render_landing_page,
    "Data Pipeline": st_data,
    "Experiment & Evaluation Harness": st_eval_harness,
    "Demo: Weather Data": st_demo_weather,
}

PAGE_OPTIONS = list(PAGES.keys())

# Sidebar
st.sidebar.header("BusWatcher | Insights")
st.sidebar.write("**By the [Urban Hub at Cornell Tech](https://urban.tech.cornell.edu/)**")
page_selection = st.sidebar.radio("Page navigation", PAGE_OPTIONS)
page = PAGES[page_selection]
page()
st.sidebar.write("""
**Contributors:**  
Alexander Amy  
Anton Abilov  
Lars Kouwenhoven  
Sanket Shah  
""")