from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Load Data
df = pd.read_csv(r"F:\OneDrive\MyDocs\Study\TELUQ\Session 8 - Hiver 2025\SCI 1402\Step 1 - Processing\combined_df_cleaned.csv")

df = df.convert_dtypes()  # Convert columns to best possible types


# # Prep / type inference
# def safe_convert(val):
#     try:
#         return ast.literal_eval(val) if isinstance(val, str) else val
#     except:
#         return []  # Return empty list if conversion fails


# df["genres"] = df["genres"].apply(safe_convert)

# Sidebar filters
# st.set_page_config(layout="wide")

st.set_page_config(
    page_title="Home",
)


st.title("Styled Streamlit App")
st.header("Choose an analysis from the sidebar")
