import streamlit as st
from interface.st_utils import load_bus_segment_data

def st_data():
    # TODO:
    # - Say something about the value of open city data
    # - Link to notebook, api documentation for the data pipeline
    # - Mention Bx12 and M15 (/other bus routes) in the intro
    # - Say something about other cities

    st.header("Data Collection")
    st.write("""
    We leveraged Dr. Anthony Townsend’s [public NYCBusWatcher API](https://github.com/Cornell-Tech-Urban-Tech-Hub/nycbuswatcher) 
    to generate 398,096 rows of data comprising once-a-minute observations of the location, occupancy, and other metrics reported
    by 85 individual vehicles serving NYC’s B46 bus route between August 1 and September 30, 2021.

    Because the data is collected once a minute, one of our first preprocessing steps was to filter it such that each route segment
    (i.e., the space between stops) corresponded with at most one observation per unique vehicle-trip ID 
    (i.e., the sequence of stops a particular vehicle is scheduled to make on a given day). 
    We made this decision on the basis that passenger count (our ultimate regression target) does not change between stops. 
    The process (visualized below) had the net effect of reducing the total number of observations in our dataset by about 40%.
    """)

    st.image("./interface/images/data-segments.png")

    # TODO: streamlit map… maybe incorporate functionality to show raw vs. processed data for a given unique trip id?
    # st.write("""
    # On a map this looks like:
    # """)

    st.write("""
    The next important decision we made was to think about the two directions a bus could travel along a given route 
    (e.g., uptown vs. downtown) as two distinct routes. 
    This decision facilitated a shift from 100+-dimensional one-hot encodings of segment IDs, 
    to one-dimensional ordinal representations corresponding to their positions along a particular 
    route in a given direction that dramatically reduced the training times and increased the interpretability of our downstream models. 
    """)

    st.subheader("Segment Data Visualization")
    bus_routes = ["B46", "Bx12", "M15"]
    selected_bus_route = st.selectbox("Select a bus route", options=bus_routes)
    df_route = load_bus_segment_data(selected_bus_route, processed=False)
        
    st.write(f"**Route data:** {df_route.shape[0]:,} observations")
    st.dataframe(df_route.head())

    st.map(
        data=df_route.sample(100)
    )

    st.subheader("Joining with Weather Data")
    st.write("""
    Finally, we associated each cleaned and processed BusWatcher datapoint with the nearest sub-hourly weather 
    observation collected by the station at JFK and accessed through the VisualCrossing Weather API.
    """)