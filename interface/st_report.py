import streamlit as st

from experiment_pipeline.data_loader import load_global_feature_set

def st_report():
    st.header("Report")
    bus_routes = ["B46", "Bx12"]
    selected_bus_route = st.selectbox("Select a bus route", options=bus_routes)

    try:
        df_route, stop_dict = load_global_feature_set("./data", selected_bus_route  )
    except:
        st.error(f"Failed to load data for route {selected_bus_route}")
        return
        
    st.write(f"**Route data:** {df_route.shape[0]:,} observations")
    st.dataframe(df_route.head())