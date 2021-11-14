import streamlit as st
from interface.st_report import st_report

# Sidebar
st.sidebar.subheader("Urban Data, 2021")
st.sidebar.write("Made by Alexander Amy, Anton Abilov & Sanket Shah")

# Main page
st.subheader("Requirements")
st.write("""
**Final writeup:** A 1,500 to 2,000 word writeup, written for a general audience, that summarizes your analysis and findings. 
Your analysis should be understandable to a smart and curious non-specialist who is paying attention - 
imagine a regular reader of FiveThirtyEight or The New York Times Upshot. 

- [Example 1](https://fivethirtyeight.com/features/in-the-end-people-may-really-just-want-to-date-themselves/)
- [Example 2](https://www.nytimes.com/interactive/2021/06/30/opinion/environmental-inequity-trees-critical-infrastructure.html)
- [Example 3](https://www.nytimes.com/2014/03/23/opinion/sunday/the-geography-of-fame.html)
- [Example 4](https://www.washingtonpost.com/politics/2020/06/20/barr-says-theres-no-systemic-racism-policing-our-data-say-attorney-general-is-wrong/)

In addition to the writeup, please submit:

1. all code which reproduces your analysis (this repository)
2. a short (250 - 500 word) technical description of the analyses you ran.
""")

st.subheader("Streamlit examples")
st.write("""
- [NY City Airbnb Listing Analysis](https://ny-airbnb-app.herokuapp.com/)
""")

st_report()