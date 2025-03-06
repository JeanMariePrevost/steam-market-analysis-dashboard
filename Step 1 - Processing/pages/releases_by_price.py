import ast
from matplotlib.ticker import AutoMinorLocator
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

# Year range slider
min_year = int(df["release_year"].min())
max_year = int(df["release_year"].max())
selected_year_range = st.sidebar.slider("Select Release Year Range", min_year, max_year, (min_year, max_year))

# Apply selected filters
filtered_df = df.copy()
if selected_genre != "All":
    filtered_df = filtered_df[filtered_df["genres"].apply(lambda x: selected_genre in x)]

if selected_tag != "All":
    filtered_df = filtered_df[filtered_df["tags"].apply(lambda x: selected_tag in x)]

filtered_df = filtered_df[(filtered_df["release_year"] >= selected_year_range[0]) & (filtered_df["release_year"] <= selected_year_range[1])]


st.sidebar.title("Options")
# Add a "bins" slider to the sidebar
num_bins = st.sidebar.slider("Select Number of Bins", 5, 50, 20)

# Add a "price range" min max input to the sidebar
price_range = st.sidebar.slider("Select Price Range", 0, 120, (0, 100))


filtered_df = filtered_df[(filtered_df["price_original"] >= price_range[0]) & (filtered_df["price_original"] <= price_range[1])]


# Plot Histogram + Scaled KDE with Log Y-Axis
st.write("### Price Distribution of Games")
genre_string = selected_genre_display.split(" (")[0] if selected_genre != "All" else "All"
tag_string = selected_tag_display.split(" (")[0] if selected_tag != "All" else "All"
year_string = f"{selected_year_range[0]} - {selected_year_range[1]}"
st.write(f"Filtered by Genre: _{genre_string}_, Tag: _{tag_string}_, Year: _{year_string}_")
st.write(f"({len(filtered_df)} items)")

# Halt if no data
if filtered_df.empty:
    st.warning("No data available for the selected filters. Please adjust the filters.")
    st.stop()

# fig, ax = plt.subplots(figsize=(12, 6), dpi=200)  # Larger figure for better scaling
fig, ax = plt.subplots(figsize=(12, 6))

# Histogram (bars)
hist_values, bin_edges, _ = ax.hist(filtered_df["price_original"], bins=num_bins, alpha=0.6, color="blue", edgecolor="black")

# KDE Plot (Scaled to Histogram)
if len(filtered_df["price_original"]) < 2 or np.var(filtered_df["price_original"]) == 0:
    st.warning("Insufficient data variance for KDE computation.")
else:
    kde = gaussian_kde(filtered_df["price_original"])
    x_range = np.linspace(min(filtered_df["price_original"]), max(filtered_df["price_original"]), max((price_range[1] - price_range[0]), 10))  # More points for smoother KDE

    # Scale KDE to match histogram counts
    bin_width = bin_edges[1] - bin_edges[0]  # Width of each histogram bin
    kde_scaled = kde(x_range) * len(filtered_df) * bin_width  # Scale KDE to match histogram counts

    ax.plot(x_range, kde_scaled, color="red", linewidth=2, label="KDE (Scaled)")


# X axis ticks
num_ticks = 11
tick_step = (price_range[1] - price_range[0]) / (num_ticks - 1)  # Step size for ticks
ticks = np.arange(price_range[0], price_range[1] + tick_step, tick_step)  # Generate ticks
ax.set_xticks(ticks)  # Set dynamic ticks
ax.set_xticklabels([f"${int(tick)}" for tick in ticks])  # Format labels as prices
ax.xaxis.set_minor_locator(AutoMinorLocator(2))


ax.set_xlabel("Price ($)")
ax.set_ylabel("Count")
ax.set_title("Game Releases by Launch Price")
ax.legend()

st.pyplot(fig)
