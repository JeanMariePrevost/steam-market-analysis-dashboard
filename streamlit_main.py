import streamlit as st

import utils

st.set_page_config(page_title="Homepage")

utils.display_streamlit_custom_navigation()

st.title("Homepage")
st.write("Welcome to the Steam Market Analysis dashboard.")
st.write("This tool allows you to explore and analyze data from the Steam Market through various visualizations and analysis, stemming from a combined dataset of more than 82,000 products.")
st.write("To begin, choose a page from the sidebar.")


with st.spinner("Testing connection to the datasets..."):
    # Load Data
    df = utils.load_main_dataset()

    if df is None or df.empty:
        st.error(f"Main dataset could not be loaded. Please ensure the path is correct and the data is available.\n\n Expected path: ./output_preprocessed/combined_df_preprocessed_dense.parquet")
        st.stop()
