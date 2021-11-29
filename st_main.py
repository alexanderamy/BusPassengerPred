import streamlit as st
from interface.st_report import st_report


# Main page
def render_landing_page():
    st.subheader("Landing page")
    st.write("Todo: write a description of the project")

    st.write("**[Github repository](https://github.com/Cornell-Tech-Urban-Tech-Hub/buswatcher-insights/)**")

PAGES = {
    "Landing page": render_landing_page,
    "Demo Report": st_report,
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