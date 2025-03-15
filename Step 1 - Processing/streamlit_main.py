import streamlit as st
from utils import load_main_dataset

# Load Data
df = load_main_dataset()

st.set_page_config(
    page_title="Home",
)

st.title("Analysis of Steam Store Data")
st.write("To begin, choose a page from the sidebar.")

if df is None or df.empty:
    st.error(f"Data could not be loaded. Please ensure the path is correct and the data is available.")
    st.stop()
