import ast
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Load Data
df = pd.read_parquet(r"F:\OneDrive\MyDocs\Study\TELUQ\Session 8 - Hiver 2025\SCI 1402\Step 1 - Processing\combined_df_cleaned.parquet")


df = df.convert_dtypes()  # Convert columns to best possible types


# Prep / type inference for arrays
def safe_convert(val):
    try:
        return ast.literal_eval(val) if isinstance(val, str) else val
    except:
        return []  # Return empty list if conversion fails


df["genres"] = df["genres"].apply(safe_convert)

# Sidebar filters
# st.set_page_config(layout="wide")


st.set_page_config(
    page_title="Releases by Price",
    page_icon="👋",
)


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

# Debug, print the type of the genres column
print(df["genres"].dtype)

st.sidebar.title("Filters")
# Genre options with counts (sorted alphabetically)
genre_counts = df["genres"].explode().value_counts()
genre_options = ["All"] + [f"{genre} ({genre_counts[genre]})" for genre in sorted(genre_counts.index)]  # Sort alphabetically
genre_mapping = {f"{genre} ({genre_counts[genre]})": genre for genre in genre_counts.index}
selected_genre_display = st.sidebar.selectbox("Select Genre", genre_options)
selected_genre = genre_mapping.get(selected_genre_display, "All")  # Map back to actual value

# Tag options with counts (sorted alphabetically)
tag_counts = df["tags"].explode().value_counts()
tag_options = ["All"] + [f"{tag} ({tag_counts[tag]})" for tag in sorted(tag_counts.index)]  # Sort alphabetically
tag_mapping = {f"{tag} ({tag_counts[tag]})": tag for tag in tag_counts.index}
selected_tag_display = st.sidebar.selectbox("Select Tag", tag_options)
selected_tag = tag_mapping.get(selected_tag_display, "All")  # Map back to actual value

# Year options (sorted numerically)
year_counts = df["release_year"].value_counts()
year_options = ["All"] + [f"{year} ({year_counts[year]})" for year in sorted(year_counts.index)]  # Sort numerically
year_mapping = {f"{year} ({year_counts[year]})": year for year in year_counts.index}
selected_year_display = st.sidebar.selectbox("Select Release Year", year_options)
selected_year = year_mapping.get(selected_year_display, "All")  # Map back to actual value

# Apply selected filters
filtered_df = df.copy()
if selected_genre != "All":
    filtered_df = filtered_df[filtered_df["genres"].apply(lambda x: selected_genre in x)]

if selected_tag != "All":
    filtered_df = filtered_df[filtered_df["tags"].apply(lambda x: selected_tag in x)]

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
genre_string = selected_genre_display.split(" (")[0] if selected_genre != "All" else "All"
tag_string = selected_tag_display.split(" (")[0] if selected_tag != "All" else "All"
year_string = selected_year_display.split(" (")[0] if selected_year != "All" else "All"
st.write(f"Filtered by Genre: _{genre_string}_, Tag: _{tag_string}_, Year: _{year_string}_")
st.write(f"({len(filtered_df)} items)")

# fig, ax = plt.subplots(figsize=(12, 6), dpi=200)  # Larger figure for better scaling
fig, ax = plt.subplots(figsize=(12, 6))

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
