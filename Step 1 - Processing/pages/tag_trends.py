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

# Sidebar, minimum number of reviews for analysis 2
st.sidebar.markdown("---")
min_review_count = st.sidebar.number_input("Minimum number of reviews for analysis 2", value=10, min_value=1)
st.sidebar.caption("Median review scores for games with fewer reviews may not be representative. Suggested value: 10.")

# Sidebar, minimum number of owners for analysis 3
min_owners_count = st.sidebar.number_input("Minimum number of owners for analysis 3", value=10000, min_value=0)
st.sidebar.caption(
    'Since 2014 steam has had an issue with shovelware and fake games, setting a reasonable minimum can help focus the analysis on more "serious" releases. Suggested value: 10,000 for general trends, 2,000 to include more unsuccessful titles.'
)


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
ax.set_title(f'Evolution of "{selected_tag}" Releases Relative to Expected Count')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.legend()

st.pyplot(fig)

# ##############################
# # Analysis 2: Reception / Sentiment analysis
# ##############################
# Plots the median review score for the tg across the years

st.write("### 2. Reception/Sentiment Analysis")
st.write(
    "This analysis shows the median positive review ratio for the selected tag across the years, along with a confidence interval to indicate the range of values. This, paired with number of releases, can help us understand how the sentiment or reception of games with this tag has evolved over time."
)


percentile_range = 5  # How much the "area" around the line covers, in percentiles
minimum_review_count = min_review_count  # Minimum number of reviews to consider a game for analysis
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

# trend line
coeffs = np.polyfit(median_reception_by_year["release_year"], median_reception_by_year["steam_positive_review_ratio"], 2)
trend_line = np.poly1d(coeffs)
ax.plot(median_reception_by_year["release_year"], trend_line(median_reception_by_year["release_year"]), linestyle="--", color="red", label="Trend Line (Quadratic)")


ax.set_xlabel("Release Year")
ax.set_ylabel("Average Steam Positive Review Ratio")
ax.set_title(f'Median review score for tag "{selected_tag}"')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.grid(True)
ax.legend()

st.pyplot(fig)

##############################
# Analysis 3: Median / percential number of owners through the years
##############################
st.write("### 3. Number of Owners Analysis")
st.write(
    "This analysis shows the median number of owners for games with the selected tag across the years, along with a confidence interval to indicate the range of values. This can help us understand how the popularity of games with this tag has evolved over time."
)
st.write(
    '**Note**: The introduction of Steam Greenlight in 2012 and the "indipocalypse" of 2014-2015 caused a dramatic shift in playerbase dillution. It may be more useful to filter from 2015 onwards for this analysis in particular.'
)

st.warning('This analysis relies on the "Boxleiter Method" for estimating the number of owners, which may not be accurate for all games.')

# quantiles = [0.75, 0.9, 0.95, 0.99]  # Percentiles to plot
# quantiles = [0.01, 0.05, 0.1, 0.25, 0.75, 0.9, 0.95, 0.99]  # Percentiles to plot
#

df_ana3 = df_filtered[df_filtered["estimated_owners_boxleiter"] > min_owners_count]  # Filter out games with too few owners

# Filter out the top 1% of games, as they are likely outliers
top_1_percentile = df_ana3["estimated_owners_boxleiter"].quantile(0.99)
df_ana3 = df_ana3[df_ana3["estimated_owners_boxleiter"] < top_1_percentile]

# Gorup by releasy year and calculate the values for each percentile
# quantile_owners_by_year = df_ana3.groupby("release_year")["estimated_owners_boxleiter"].quantile(quantiles).unstack().reset_index()
median_owners_by_year = df_ana3.groupby("release_year")["estimated_owners_boxleiter"].median().reset_index()

# Debug, add the _count_ of elements in each year
median_owners_by_year["count"] = df_ana3.groupby("release_year").size().reset_index()[0]

# Get confidence intervals through utils
df_ana3_confidence_int = utils.median_confidence_intervals(df_ana3, "estimated_owners_boxleiter", "release_year", confidence_area_level)

# Clip the confidence intervals above 0
df_ana3_confidence_int["ci_lower"] = df_ana3_confidence_int["ci_lower"].clip(lower=0)
df_ana3_confidence_int["ci_upper"] = df_ana3_confidence_int["ci_upper"].clip(lower=0)

# Debug, print the median owners by year
st.write(median_owners_by_year)
# st.write(quantile_owners_by_year)

# Warn user if any years have a small sample size
ana3_min_sample_size = 4
if median_owners_by_year["count"].min() < ana3_min_sample_size:
    ana3_warning_message = f"Warning: Some years have a very small sample size (<{ana3_min_sample_size} games). Results may be noisy or unreliable."
    for year, count in median_owners_by_year[median_owners_by_year["count"] < ana3_min_sample_size][["release_year", "count"]].values:
        ana3_warning_message += f"\n- Year {year}: {count} games"
    ana3_warning_message += "\n\nConsider adjusting the filters in the sidebar for more meaningful results."
    st.warning(ana3_warning_message)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(median_owners_by_year["release_year"], median_owners_by_year["estimated_owners_boxleiter"], label="Median Estimated Owners", marker="o", linestyle="-")
# for q in quantiles:
#     ax.plot(quantile_owners_by_year["release_year"], quantile_owners_by_year[q], label=f"{int(q * 100)}% Percentile", linestyle="--")

ax.fill_between(
    df_ana3_confidence_int["release_year"], df_ana3_confidence_int["ci_lower"], df_ana3_confidence_int["ci_upper"], alpha=0.2, label=f"{confidence_area_level * 100:.0f}% Confidence Interval"
)

# trend line
coeffs = np.polyfit(median_owners_by_year["release_year"], median_owners_by_year["estimated_owners_boxleiter"], 2)
trend_line = np.poly1d(coeffs)
ax.plot(median_owners_by_year["release_year"], trend_line(median_owners_by_year["release_year"]), linestyle="--", color="red", label="Trend Line (Quadratic)")

# ax.set_yscale("log")

ax.set_xlabel("Release Year")
ax.set_ylabel("Estimated Number of Owners")
ax.set_title(f'Estimated number of owners for "{selected_tag}"')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.grid(True)
ax.legend()

st.pyplot(fig)

##############################
# Analysis 4: playtime_median through the years
##############################

##############################
# Analysis 5: Pricing distribution (histogram?) (NOT through the years)
##############################

##############################
# Analysis 6: Best and worst tag interactions (NOT through the years)
##############################
