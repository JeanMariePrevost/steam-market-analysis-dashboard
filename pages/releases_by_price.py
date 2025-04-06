import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import gaussian_kde

import utils
from utils import load_main_dataset, triangular_weighted_mean

# Load Data
df = load_main_dataset()


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
    page_title="Game Price Analysis",
    # page_icon="👋",
)
utils.display_streamlit_custom_navigation()


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
st.title("Game Price Analysis")
st.write("This page allows you to explore the distribution of game releases by their launch price. You can filter by genre, tag, release year, and price range to see how the distribution changes.")

if df is None or df.empty:
    st.error(f"Data could not be loaded. Please ensure the path is correct and the data is available.")
    st.stop()

##############################
# Sidebar Options
##############################
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

##############################
# Dynamic page header
##############################
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

##############################
# 1 - Total releases by price
##############################
with st.spinner("Analyzing..."):
    # Histogram (bars)
    fig, ax = plt.subplots(figsize=(12, 6))
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

    ax.set_xlabel("Price (USD)")
    ax.set_ylabel("Count")
    ax.set_title("Game Releases by Launch Price")
    ax.legend()

    st.pyplot(fig)

##############################
# Most common prices
##############################
with st.spinner("Analyzing..."):
    # Identify the most common price ranges by using narrow bins, and extracting the mode of each bin as the "label price" for that group
    # Build bins piecewise
    bin_bounds = np.array([])
    highest_price = filtered_df["price_original"].max()

    # Create bin boundaries for each segment.
    bins_a = np.arange(0, 2.1, 0.1)  # 0.1 increments until $2 (includes 2.0)
    bins_b = np.arange(2, 5.25, 0.25)  # 0.25 increments from $2 to $5 (includes 5.0)
    bins_c = np.arange(5, 15.5, 0.5)  # 0.5 increments from $5 to $15 (includes 14.0)
    bins_d = np.arange(15, highest_price + 1, 1)  # $1 increments from $15 onward

    # Concatenate the segments and remove duplicates.
    bins = np.unique(np.concatenate([bins_a, bins_b, bins_c, bins_d]))

    # Apply the bins to your DataFrame column.
    filtered_df["precise_price_bin"] = pd.cut(filtered_df["price_original"], bins=bins, right=False)

    # Group by the new price bin label, and count the number of games in each group.
    price_bin_counts = filtered_df.groupby("precise_price_bin").size().reset_index(name="count")

    # Sort by count and get the top N price bins.
    number_of_prices_to_display = 25
    top_price_bins = price_bin_counts.sort_values(by="count", ascending=False).head(number_of_prices_to_display)

    # Calculate median review score and estimated owners for each bin.
    top_price_bins["median_review_score"] = top_price_bins["precise_price_bin"].apply(
        lambda bin_range: (
            filtered_df[(filtered_df["price_original"] >= bin_range.left) & (filtered_df["price_original"] < bin_range.right)]["steam_positive_review_ratio"].median()
            if not filtered_df[(filtered_df["price_original"] >= bin_range.left) & (filtered_df["price_original"] < bin_range.right)]["steam_positive_review_ratio"].empty
            else 0
        )
    )
    top_price_bins["median_estimated_owners"] = top_price_bins["precise_price_bin"].apply(
        lambda bin_range: (
            filtered_df[(filtered_df["price_original"] >= bin_range.left) & (filtered_df["price_original"] < bin_range.right)]["estimated_owners_boxleiter"].median()
            if not filtered_df[(filtered_df["price_original"] >= bin_range.left) & (filtered_df["price_original"] < bin_range.right)]["estimated_owners_boxleiter"].empty
            else 0
        )
    )

    # Get the mode price for each bin and print a new DataFrame with the results.
    # Start, End, Mode, Count, proportion of all games
    total_games = len(filtered_df)
    top_price_bins["bin_mode_price"] = top_price_bins["precise_price_bin"].apply(
        lambda bin_range: (
            filtered_df[(filtered_df["price_original"] >= bin_range.left) & (filtered_df["price_original"] < bin_range.right)]["price_original"].mode().iloc[0]
            if not filtered_df[(filtered_df["price_original"] >= bin_range.left) & (filtered_df["price_original"] < bin_range.right)]["price_original"].mode().empty
            else bin_range.mid
        )
    )
    top_price_bins["start_price"] = top_price_bins["precise_price_bin"].apply(lambda x: x.left)  # Get the start of each bin
    top_price_bins["end_price"] = top_price_bins["precise_price_bin"].apply(lambda x: x.right)  # Get the end of each bin
    top_price_bins["proportion"] = top_price_bins["count"] / total_games  # proportion of all games
    # Copyinto a column as strings, with 2 decimals, and make ones that round to 0 into "<0.01"
    top_price_bins["proportion"] = top_price_bins["proportion"].apply(lambda x: f"<0.01" if x < 0.01 else f"{x:.2f}")
    # Create a "display label" for each biun, being the mode + the actual range in parentheses
    top_price_bins["bin_label"] = top_price_bins.apply(lambda row: f"${row['bin_mode_price']:.2f} ({row['start_price']:.2f} - {row['end_price']:.2f})", axis=1)
    top_price_bins = top_price_bins[["bin_label", "proportion", "count", "median_review_score", "median_estimated_owners"]]
    top_price_bins = top_price_bins.rename(
        columns={"bin_label": "Price", "count": "Count", "proportion": "Proportion of All Games", "median_review_score": "Median Review Score", "median_estimated_owners": "Median Estimated Owners"}
    )
    top_price_bins = top_price_bins.sort_values(by="Count", ascending=False)  # Sort by count
    top_price_bins = top_price_bins.reset_index(drop=True)  # Reset index

    st.write("### Most Common Prices")
    st.write("The following table shows the most common prices, grouping prices that are very similar.")
    st.write("The label for each bin is the mode price for that bin, followed by the actual price range in parentheses.")
    st.write(top_price_bins)


##############################
# 2 - Total Estimated Owners by price and Gross Revenue by price
##############################
def plot_weighted_histogram(filtered_df, price_range, num_bins, weight_column, ylabel, title):
    """ "
    Plots a histogram using the "sum of X" as the weight for each bin
    E.g. get the sum of game owners by year of release _rather than_ the _count_ of these games
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Copy the filtered DataFrame and remove NA values in the weight column
    filtered_df = filtered_df.copy()
    filtered_df = filtered_df.dropna(subset=[weight_column])

    # Plot weighted histogram
    hist_vals, bin_edges, _ = ax.hist(filtered_df["price_original"], bins=num_bins, weights=filtered_df[weight_column], alpha=0.6, color="blue", edgecolor="black")

    # Configure x-axis ticks and labels
    ticks = np.linspace(price_range[0], price_range[1], 11)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"${int(t)}" for t in ticks])
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))

    # Set labels, title, and legend
    ax.set(xlabel="Price (USD)", ylabel=ylabel, title=title)
    ax.legend()

    st.pyplot(fig)


st.write("### Total Estimated Owners and Gross Revenue by Launch Price")
st.write("The following graphs show the total estimated owners and gross revenue across all games in each price bin.")
st.write("Note that these are estimates based on the Boxleiter model, and may not reflect actual sales.")
with st.spinner("Analyzing..."):
    # Graph 1: Total players (estimated_owners_boxleiter)
    plot_weighted_histogram(filtered_df, price_range, num_bins, weight_column="estimated_owners_boxleiter", ylabel="Total Estimated Owners", title="Total Estimated Owners by Launch Price")

    # Graph 2: Gross revenue (estimated_gross_revenue_boxleiter)
    plot_weighted_histogram(
        filtered_df, price_range, num_bins, weight_column="estimated_gross_revenue_boxleiter", ylabel="Total Estimated Gross Revenue", title="Total Estimated Gross Revenue by Launch Price"
    )


##############################
# 3 - Impact of launch price on steam_positive_review_ratio
##############################
with st.spinner("Analyzing..."):
    # Define the custom bins and corresponding labels
    bin_bounds = [-0.001, 0.001, 1, 5, 10, 20, 30, 45, 60, np.inf]
    labels = ["0", "0.01-1", "1-5", "5-10", "10-20", "20-30", "30-45", "45-60", "60+"]

    # Create the price bin column. (cut assigns a bin label to each row based on the corresponding price)
    filtered_df["price_bin"] = pd.cut(filtered_df["price_original"], bins=bin_bounds, labels=labels)

    # Group by the new price bin label, and sum up the positive and negative reviews for each group
    grouped = filtered_df.groupby("price_bin").agg({"steam_positive_reviews": "sum", "steam_negative_reviews": "sum"}).reset_index()

    # Calculate the ratio of positive reviews
    grouped["steam_positive_review_ratio"] = grouped["steam_positive_reviews"] / (grouped["steam_positive_reviews"] + grouped["steam_negative_reviews"])

    # Compute the overall positive review ratio for the entire dataset to make the histogram relative instead of absolute
    total_positive = filtered_df["steam_positive_reviews"].sum()
    total_negative = filtered_df["steam_negative_reviews"].sum()
    overall_ratio = total_positive / (total_positive + total_negative)

    # difference relative to the overall mean for each group
    grouped["ratio_diff"] = grouped["steam_positive_review_ratio"] - overall_ratio

    # Set colors based on ratio_diff
    colors = grouped["ratio_diff"].apply(lambda x: "green" if x >= 0 else "red")

    # Plot the thing
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(grouped["price_bin"], grouped["ratio_diff"], color=colors, edgecolor="black")
    ax.axhline(0, color="black", lw=1)  # Reference line at zero

    ax.set_xlabel("Launch Price Range (USD)")
    ax.set_ylabel("Difference to Overall Positive Review Ratio")
    ax.set_title("Effect of Launch Price on Review Scores")
    st.write("### Effect of Launch Price on Review Scores")
    st.write("And the following graph shows the difference in positive review ratio for each price bin compared to the overall average.")
    st.pyplot(fig)


#################################################################
#################################################################
#################################################################
with st.spinner("Analyzing..."):
    # 4 - Sliding Window Analysis with Smoothed Data
    percentile_range = 5  # How much the "area" around the line covers, in percentiles

    # Determine the range of prices available in your data
    min_price = filtered_df["price_original"].min()
    max_price = filtered_df["price_original"].max()

    ###### The grid
    # Define breakpoints (adjust these based on your data)
    low_break = 5.0  # Upper bound for low-price resolution
    mid_break = 25.0  # Upper bound for mid-price resolution

    # Define the number of points for each segment
    n_points_low = 30  # More points for the low range
    n_points_mid = 30  # Moderate points for the mid range
    n_points_high = 30  # Fewer points for the high range

    # Create piecewise linear grids
    price_grid_low = np.linspace(min_price, low_break, n_points_low, endpoint=False)
    price_grid_mid = np.linspace(low_break, mid_break, n_points_mid, endpoint=False)
    price_grid_high = np.linspace(mid_break, max_price, n_points_high)

    # Combine them into a single grid
    price_grid = np.concatenate([price_grid_low, price_grid_mid, price_grid_high])

    # Lists to store the computed statistics for each grid point
    mean_ratios = []
    lower_percentiles = []
    upper_percentiles = []

    # Get the median steam_positive_review_ratio as a whole
    mean_positive_ratio = filtered_df["steam_positive_review_ratio"].dropna().mean()

    # Introduce a "difference to mean"
    filtered_df["ratio_diff"] = filtered_df["steam_positive_review_ratio"] - mean_positive_ratio

    # Loop over each grid price and compute the desired statistics using a dynamic window
    for p in price_grid:
        # Calculate the statistics if there is any data in the window
        # current_window = max(0.2 * p, 0.80)
        current_window = 0.7 + 0.15 * p
        mean_ratio = triangular_weighted_mean(filtered_df, "steam_positive_review_ratio", "price_original", p, current_window)
        mean_ratios.append(mean_ratio)

        # Do the percentiles
        window_df = filtered_df[(filtered_df["price_original"] >= p - current_window) & (filtered_df["price_original"] <= p + current_window)]

        # Calculate the distance from median to +/- percentile_range
        median = window_df["steam_positive_review_ratio"].median()

        # 50 + percentile_range percentile miunus the median
        distance_up = np.percentile(window_df["steam_positive_review_ratio"].dropna(), 50 + percentile_range) - median
        distance_down = median - np.percentile(window_df["steam_positive_review_ratio"].dropna(), 50 - percentile_range)

        lower_bound = mean_ratio - distance_down
        upper_bound = mean_ratio + distance_up
        lower_percentiles.append(lower_bound)
        upper_percentiles.append(upper_bound)

    # plot the data
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(price_grid, mean_ratios, color="red", alpha=0.5, label="Weighted Average Score")
    ax.fill_between(price_grid, lower_percentiles, upper_percentiles, color="lightblue", alpha=0.5, label=f"±{percentile_range} Percentiles")
    ax.set_xlabel("Launch Price (USD)")
    ax.set_ylabel("Steam Positive Review Ratio")
    ax.set_title("Steam Positive Review Ratio by Launch Price")
    ax.legend()
    st.write("The following graph shows the a more granular correlation between steam positive review ratio and launch price, using a sliding window approach.")
    st.pyplot(fig)
