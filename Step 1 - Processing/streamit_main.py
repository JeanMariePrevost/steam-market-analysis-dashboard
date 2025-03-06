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

# Inject custom CSS with st.markdown
st.markdown(
    """
    <style>
    .stMainBlockContainer {
        max-width: 1000px;
    }     
    </style>
    """,
    unsafe_allow_html=True,
)

# Add some content to see the styling
st.title("Styled Streamlit App")

st.sidebar.title("Filters")
genre_options = ["All"] + sorted(df["genres"].explode().unique().astype(str).tolist())
selected_genre = st.sidebar.selectbox("Select Genre", genre_options)
year_options = ["All"] + sorted(df["release_year"].dropna().unique().tolist())
selected_year = st.sidebar.selectbox("Select Release Year", year_options)

# Apply selected filters
filtered_df = df.copy()
if selected_genre != "All":
    filtered_df = filtered_df[filtered_df["genres"].apply(lambda x: selected_genre in x)]

if selected_year != "All":
    filtered_df = filtered_df[filtered_df["release_year"] == int(selected_year)]

# Remove price outliers using IQR
# Q1 = filtered_df["price_original"].quantile(0.25)
# Q3 = filtered_df["price_original"].quantile(0.75)
# IQR = Q3 - Q1
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR
lower_bound = 0
# upper_bound = filtered_df["price_original"].quantile(0.95)
upper_bound = 100
filtered_df = filtered_df[(filtered_df["price_original"] >= lower_bound) & (filtered_df["price_original"] <= upper_bound)]


# Plot Histogram + Scaled KDE with Log Y-Axis
st.write("### Price Distribution of Games")

# fig, ax = plt.subplots(figsize=(12, 6), dpi=200)  # Larger figure for better scaling
fig, ax = plt.subplots()

# Histogram (bars)
hist_values, bin_edges, _ = ax.hist(filtered_df["price_original"], bins=20, alpha=0.6, color="blue", edgecolor="black")

# KDE Plot (Scaled to Histogram)
if len(filtered_df) > 1:
    kde = gaussian_kde(filtered_df["price_original"])
    x_range = np.linspace(min(filtered_df["price_original"]), max(filtered_df["price_original"]), 200)  # More points for smoother KDE

    # Scale KDE to match histogram counts
    bin_width = bin_edges[1] - bin_edges[0]  # Width of each histogram bin
    kde_scaled = kde(x_range) * len(filtered_df) * bin_width  # Scale KDE to match histogram counts

    ax.plot(x_range, kde_scaled, color="red", linewidth=2, label="KDE (Scaled)")


# Increase the number of X-axis ticks dynamically
num_ticks = 20  # Adjust the number of ticks
ax.set_xticks(np.arange(0, 110, 10))  # Fixed ticks from 0 to 100 (step = 10)
ax.set_xticklabels([f"${int(tick)}" for tick in np.arange(0, 110, 10)])  # Format labels as prices


ax.set_xlabel("Price ($)")
ax.set_ylabel("Count")
ax.set_title("Game Releases by Launch Price")
ax.legend()

st.pyplot(fig)
