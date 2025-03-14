import ast
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator
import numpy as np
import pandas as pd
import streamlit as st
import os
import matplotlib.ticker as mticker
import utils

# Page configuration & custom CSS
st.set_page_config(page_title="Tags And Genres Trends")
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


df = utils.load_main_dataset()

# Page Title & Description
st.title("Tags And Genres Trends")
st.write("Select a tag in the sidebar to begin.")

if df is None or df.empty:
    st.warning(f"Data could be loaded. Please ensure the path is correct and the data is available.")
    st.stop()

############################################
# Sidebar
############################################
# Sidebar Filter: Release Year Range
st.sidebar.title("Filters")
# Tag options with counts (sorted alphabetically)
tag_counts = df["tags"].explode().value_counts()  # Count occurrences of each tag (for display only)
tag_options = "Select a Tag", *sorted([f"{tag} ({tag_counts[tag]})" for tag in tag_counts.index])
tag_mapping = {f"{tag} ({tag_counts[tag]})": tag for tag in tag_counts.index}  # Map display tag (with count) to actual value (used in df)
selected_tag_display = st.sidebar.selectbox("Select Tag", tag_options)  # Create dropdown with display values
selected_tag = tag_mapping.get(selected_tag_display, "Select a Tag")  # Map back from selected_display_tag to actual tag

# Sidebar Filter: Release Year Range
min_year = int(df["release_year"].min())
max_year = int(df["release_year"].max())
selected_year_range = st.sidebar.slider("Select Release Year Range", min_year, max_year, (2007, max_year - 1))


# Debug, pick a random tag if none is selected
if selected_tag == "Select a Tag":
    random_tag = random.choice(tag_options[1:])  # Exclude the first element "Select a Tag"
    selected_tag_display = random_tag
    selected_tag = tag_mapping.get(random_tag, "Select a Tag")


if selected_tag == "Select a Tag":
    st.warning("Please select a tag from the sidebar to begin analysis.")
    st.stop()


# Apply release year filter
df_all_in_year_range = df[(df["release_year"] >= selected_year_range[0]) & (df["release_year"] <= selected_year_range[1])]
df_filtered = df_all_in_year_range[df_all_in_year_range["tags"].apply(lambda x: selected_tag in x)]

if df_filtered.empty:
    st.warning("No data available for the selected release year range. Please adjust the filters.")
    st.stop()


st.write(f"Debug - Selected Tag = {selected_tag}")
st.write(f"Debug - Entries in filtered_df = {df_filtered.shape[0]}")


if selected_year_range[0] < 2007:
    st.warning("Limited data available before 2007. Consider adjusting the release year range for more meaningful insights.")

# ##############################
# # Analysis 1: Relative Releases Across Years
# ##############################
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np

st.write("### 1. Relative Releases Across Years")
st.write(
    "This analysis shows the relative number of releases for the selected tag compared to the expected count based on the overall distribution of tags, hinting us at trends in the popularity of the tag over time, independent of the overall growth of the platform."
)

confidence_area_level = 0.95

# Calculate the counts and expected counts for the selected tag to use as the reference
tag_counts = df_filtered.groupby("release_year").size().reset_index(name="tag_count")
total_counts = df_all_in_year_range.groupby("release_year").size().reset_index(name="total_count")
counts_df = pd.merge(tag_counts, total_counts, on="release_year", how="outer").fillna(0)
total_count_tag = counts_df["tag_count"].sum()
total_count_all = counts_df["total_count"].sum()
global_proportion = total_count_tag / total_count_all if total_count_all > 0 else 0
counts_df["expected_for_tag"] = counts_df["total_count"] * global_proportion

# Compute the percent deviation from the expected count, the thing we want to analyze
counts_df["percent_deviation"] = ((counts_df["tag_count"] - counts_df["expected_for_tag"]) / counts_df["expected_for_tag"]) * 100

# Compute 95% confidence intervals based on the binomial distribution
lower_bounds, upper_bounds = utils.binom_confidence_interval(counts_df["tag_count"], counts_df["total_count"], confidence=confidence_area_level)
# Convert the bounds to scale as proportions of the total count
lower_bounds = ((lower_bounds - global_proportion) / global_proportion) * 100
upper_bounds = ((upper_bounds - global_proportion) / global_proportion) * 100


# Warn if the sample size is small
if total_count_tag < 100:
    st.warning(f"Warning: The sample size is small ({total_count_tag:.0f} releases). Results may be noisy or unreliable.")

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(counts_df["release_year"], counts_df["percent_deviation"], marker="o", linestyle="-", label="Percent Deviation")
ax.fill_between(counts_df["release_year"], lower_bounds, upper_bounds, alpha=0.3, color="gray", label=f"{confidence_area_level * 100:.0f}% Confidence Interval")
ax.axhline(0, color="gray", linestyle="--")

# # Fit a linear trend line using np.polyfit
# coeffs = np.polyfit(counts_df["release_year"], counts_df["percent_deviation"], 1)
# trend_line = np.poly1d(coeffs)
# ax.plot(counts_df["release_year"], trend_line(counts_df["release_year"]), linestyle="--", color="red", label="Trend Line")

# Fit a quadratic (polynomial degree 2) trend line
coeffs = np.polyfit(counts_df["release_year"], counts_df["percent_deviation"], 2)
trend_line = np.poly1d(coeffs)
ax.plot(counts_df["release_year"], trend_line(counts_df["release_year"]), linestyle="--", color="red", label="Trend Line (Quadratic)")


ax.set_xlabel("Release Year")
ax.set_ylabel("Percent Deviation (%)")
ax.set_title("Evolution of {} Releases Relative to Expected Count".format(selected_tag))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.legend()

st.pyplot(fig)

# ##############################
# # Analysis 2: Rception / Sentiment analysis
# ##############################
# Plots the median review score for the tg across the years

st.write("### 2. Reception/Sentiment Analysis")
st.write(
    "This analysis shows the median positive review ratio for the selected tag across the years, along with a confidence interval to indicate the range of values. This, paired with number of releases, can help us understand how the sentiment or reception of games with this tag has evolved over time."
)


percentile_range = 5  # How much the "area" around the line covers, in percentiles
minimum_review_count = 10  # Minimum number of reviews to consider a game for analysis
trash_threshold = 0.1  # Filter out "garbage" games to not have an influx of shovelware drag down the average.

df_ana2 = df_filtered[df_filtered["steam_total_reviews"] >= minimum_review_count]  # Filter out games with too few reviews
df_ana2 = df_ana2[df_filtered["steam_positive_review_ratio"] >= trash_threshold]  # Filter out outlier games that are likely trash (below the threshold)

# Group the remaining data by release_year and calculate the mean positive review ratio.
median_reception_by_year = df_ana2.groupby("release_year")["steam_positive_review_ratio"].median().reset_index()

lower_percentiles = []
upper_percentiles = []
for year in median_reception_by_year["release_year"]:
    year_data = df_ana2[df_ana2["release_year"] == year]
    lower_percentiles.append(year_data["steam_positive_review_ratio"].quantile(0.5 - (percentile_range / 100)))
    upper_percentiles.append(year_data["steam_positive_review_ratio"].quantile(0.5 + (percentile_range / 100)))


# Plot the trend over time.
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(median_reception_by_year["release_year"], median_reception_by_year["steam_positive_review_ratio"], label="Median Positive Review Ratio", marker="o", linestyle="-")
ax.fill_between(median_reception_by_year["release_year"], lower_percentiles, upper_percentiles, alpha=0.2, label=f"{percentile_range}% Percentile Range")

# # Fit a linear trend line using np.polyfit
# coeffs = np.polyfit(median_reception_by_year["release_year"], median_reception_by_year["steam_positive_review_ratio"], 1)
# trend_line = np.poly1d(coeffs)
# ax.plot(median_reception_by_year["release_year"], trend_line(median_reception_by_year["release_year"]), linestyle="--", color="red", label="Trend Line")

# Fit a quadratic (polynomial degree 2) trend line
coeffs = np.polyfit(median_reception_by_year["release_year"], median_reception_by_year["steam_positive_review_ratio"], 2)
trend_line = np.poly1d(coeffs)
ax.plot(median_reception_by_year["release_year"], trend_line(median_reception_by_year["release_year"]), linestyle="--", color="red", label="Trend Line (Quadratic)")


ax.set_xlabel("Release Year")
ax.set_ylabel("Average Steam Positive Review Ratio")
ax.set_title("Reception/Sentiment Analysis for Tag")
ax.grid(True)
ax.legend()

st.pyplot(fig)

# ##############################
# # Analysis 2: Top N Genres Across Years (Stacked Bar Chart - Ratio)
# ##############################
# st.write("### 2. Proportion of Releases by Top Genres Across Years")

# # Define parameter
# top_n_genres = 15  # Change this value to adjust the number of top genres displayed

# # Explode genres and filter to top N overall
# genres_exploded = filtered_df.explode("genres")
# top_genres = genres_exploded["genres"].value_counts().head(top_n_genres).index.tolist()
# genres_exploded = genres_exploded[genres_exploded["genres"].isin(top_genres)]

# # Group by release year and genre
# genre_year_counts = genres_exploded.groupby(["release_year", "genres"]).size().reset_index(name="count")

# # Pivot for stacked bar chart
# pivot_genres = genre_year_counts.pivot(index="release_year", columns="genres", values="count").fillna(0)

# # Convert counts to proportions (normalize per year)
# pivot_genres = pivot_genres.div(pivot_genres.sum(axis=1), axis=0)  # Normalize row-wise

# # Plot stacked bar chart
# fig2, ax2 = plt.subplots(figsize=(10, 5))
# pivot_genres.plot(kind="bar", stacked=True, ax=ax2, width=0.8)

# ax2.set_xlabel("Release Year")
# ax2.set_ylabel("Proportion of Releases")
# ax2.set_title(f"Proportion of Releases by Top {top_n_genres} Genres Over Time")
# ax2.legend(title="Genre", bbox_to_anchor=(1.05, 1), loc="upper left")

# st.pyplot(fig2)


# ##############################
# # Analysis 3: Top N Tags Across Years (Stacked Bar Chart - Ratio)
# ##############################
# st.write("### 3. Proportion of Releases by Top Tags Across Years")

# # Define parameter
# top_n_tags = 15  # Change this value to adjust the number of top tags displayed

# # Explode tags and filter to top N overall
# tags_exploded = filtered_df.explode("tags")
# top_tags = tags_exploded["tags"].value_counts().head(top_n_tags).index.tolist()
# tags_exploded = tags_exploded[tags_exploded["tags"].isin(top_tags)]

# # Group by release year and tag
# tag_year_counts = tags_exploded.groupby(["release_year", "tags"]).size().reset_index(name="count")

# # Pivot for stacked bar chart
# pivot_tags = tag_year_counts.pivot(index="release_year", columns="tags", values="count").fillna(0)

# # Convert counts to proportions (normalize per year)
# pivot_tags = pivot_tags.div(pivot_tags.sum(axis=1), axis=0)  # Normalize row-wise

# # Plot stacked bar chart
# fig3, ax3 = plt.subplots(figsize=(10, 5))
# pivot_tags.plot(kind="bar", stacked=True, ax=ax3, width=0.8)

# ax3.set_xlabel("Release Year")
# ax3.set_ylabel("Proportion of Releases")
# ax3.set_title(f"Proportion of Releases by Top {top_n_tags} Tags Over Time")
# ax3.legend(title="Tag", bbox_to_anchor=(1.05, 1), loc="upper left")

# st.pyplot(fig3)


# ##############################
# # Analysis 4: Median & Average Estimated Owners per Year (Line Chart)
# ##############################
# st.write("### 4. Median Estimated Playerbase per Title per Year")
# st.write("This analysis presents a sharp decline starting in 2014, which most likely points to the explosion of Steam Greenlight titles and the subsequent flood of low-quality games to the market.")
# st.write("Sources:")
# st.write("https://steamcommunity.com/games/593110/announcements/detail/1328973169870947116, https://www.theguardian.com/technology/2017/feb/13/valve-kills-steam-greenlight-heres-why-it-matters")
# # Compute median & mean estimated owners per year
# owners_stats = filtered_df.groupby("release_year")["estimated_owners_boxleiter"].agg(["median", "mean"]).reset_index()

# # Plot
# fig4, ax4 = plt.subplots(figsize=(10, 5))
# ax4.plot(owners_stats["release_year"], owners_stats["median"], marker="o", linewidth=2, label="Median Owners")
# # ax4.plot(owners_stats["release_year"], owners_stats["mean"], marker="s", linewidth=2, label="Average Owners")

# # Keep only the top 10% of titles in a new dataframe
# top_10_percent = filtered_df[filtered_df["estimated_owners_boxleiter"] > filtered_df["estimated_owners_boxleiter"].quantile(0.9)]
# top_10_percent = top_10_percent.groupby("release_year")["estimated_owners_boxleiter"].median().reset_index()
# # Plot the top 10% of titles
# ax4.plot(top_10_percent["release_year"], top_10_percent["estimated_owners_boxleiter"], marker="s", linewidth=2, label="Median owners (top 10% titles only)", color="red")


# # Set labels and title
# ax4.set_xlabel("Release Year")
# ax4.set_ylabel("Estimated Owners")
# ax4.set_title("Estimated Playerbase: Median per title per Year")
# ax4.legend()

# # Ensure x-axis and y-axis ticks are integers
# ax4.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))  # Force integer x-ticks
# ax4.xaxis.set_minor_locator(AutoMinorLocator(2))
# ax4.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))  # Force integer y-ticks

# st.pyplot(fig4)


# ##############################
# # Analysis 5: Average Review Score per Year (Line Chart)
# ##############################
# st.write("### 5. Average Review Score per Year")
# st.write(
#     "Coinciding with the decline in estimated playerbase and the peak of the Steam Greenlight crisis in 2016, the average review score per year shows a marked decline starting in between 2014 and 2018."
# )

# # Ensure positive/negative review columns exist and drop rows where either is missing
# filtered_reviews = filtered_df.dropna(subset=["steam_positive_reviews", "steam_negative_reviews"])

# # Ensure numeric types (avoid issues if columns were stored as strings)
# filtered_reviews["steam_positive_reviews"] = pd.to_numeric(filtered_reviews["steam_positive_reviews"], errors="coerce")
# filtered_reviews["steam_negative_reviews"] = pd.to_numeric(filtered_reviews["steam_negative_reviews"], errors="coerce")

# # Drop any remaining rows where review values are still NaN (e.g., failed conversions)
# filtered_reviews = filtered_reviews.dropna(subset=["steam_positive_reviews", "steam_negative_reviews"])


# # Calculate review score per game (avoid division by zero)
# def compute_review_score(pos, neg):
#     total = pos + neg
#     return pos / total if total > 0 else np.nan  # Avoid division by zero


# filtered_reviews["review_score"] = filtered_reviews.apply(lambda row: compute_review_score(row["steam_positive_reviews"], row["steam_negative_reviews"]), axis=1)

# # Group by release year and compute the mean review score
# review_stats = filtered_reviews.groupby("release_year")["review_score"].mean().reset_index()

# # Plot the results
# fig5, ax5 = plt.subplots(figsize=(10, 5))
# ax5.plot(review_stats["release_year"], review_stats["review_score"], marker="o", linewidth=2, color="green")
# ax5.set_xlabel("Release Year")
# ax5.set_ylabel("Average Review Score")
# ax5.set_title("Average Review Score per Year")
# ax5.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))  # Force integer x-ticks
# ax5.xaxis.set_minor_locator(AutoMinorLocator(2))
# ax5.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))  # Force integer y-ticks
# st.pyplot(fig5)


# ##############################
# # Analysis 6: Proportion of Games by Price Bracket (Stacked Bar Chart - Proportion)
# ##############################
# st.write("### 6. Proportion of Games by Price Bracket Over Time (Ratio)")


# # Define price brackets
# def price_bracket(price):
#     if price == 0:
#         return "Free"
#     elif price < 5:
#         return "<10"
#     elif price < 10:
#         return "5-10"
#     elif price < 20:
#         return "10-20"
#     elif price < 40:
#         return "20-40"
#     elif price < 60:
#         return "40-60"
#     elif price < 80:
#         return "60-80"
#     elif price >= 80:
#         return "80+"
#     else:
#         return "Unknown"


# # Apply price bracket classification
# filtered_df["price_bracket"] = filtered_df["price_original"].apply(price_bracket)

# # Group by release year and price bracket
# price_year_counts = filtered_df.groupby(["release_year", "price_bracket"]).size().reset_index(name="count")

# # Pivot for stacked bar chart
# pivot_prices = price_year_counts.pivot(index="release_year", columns="price_bracket", values="count").fillna(0)

# # Ensure the correct order of price brackets
# price_bracket_order = ["Free", "<10", "5-10", "10-20", "20-40", "40-60", "60-80", "80+"]
# pivot_prices = pivot_prices.reindex(columns=price_bracket_order, fill_value=0)  # Enforce order

# # Convert to proportions (normalize by total games per year)
# pivot_prices = pivot_prices.div(pivot_prices.sum(axis=1), axis=0)  # Normalize each row to sum to 1 (ratio)

# # Plot stacked bar chart
# fig6, ax6 = plt.subplots(figsize=(10, 5))
# pivot_prices.plot(kind="bar", stacked=True, ax=ax6, width=0.8)

# ax6.set_xlabel("Release Year")
# ax6.set_ylabel("Proportion of Games")
# ax6.set_title("Proportion of Games by Price Bracket Over Time")
# ax6.legend(title="Price Bracket", bbox_to_anchor=(1.05, 1), loc="upper left")

# st.pyplot(fig6)
