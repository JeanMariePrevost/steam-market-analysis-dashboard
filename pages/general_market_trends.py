import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.ticker import AutoMinorLocator

import utils
from utils import load_main_dataset

# Page configuration & custom CSS
st.set_page_config(page_title="Overview of Releases & Trends")
utils.display_streamlit_custom_navigation()
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


df = load_main_dataset()

# Page Title & Description
st.title("Overview: Game Releases & Market Trends")
st.write("This page presents several static analyses that offer a market-wide overview of game releases, trends in genres and tags, audience reach, user review quality, and pricing strategies.")

if df is None or df.empty:
    st.error(f"Data could not be loaded. Please ensure the path is correct and the data is available.")
    st.stop()

# Sidebar Filter: Release Year Range
st.sidebar.title("Filters")
min_year = int(df["release_year"].min())
max_year = int(df["release_year"].max())
selected_year_range = st.sidebar.slider("Select Release Year Range", min_year, max_year, (2007, max_year))

# Apply release year filter
filtered_df = df[(df["release_year"] >= selected_year_range[0]) & (df["release_year"] <= selected_year_range[1])]

if filtered_df.empty:
    st.warning("No data available for the selected release year range. Please adjust the filters.")
    st.stop()

if selected_year_range[0] < 2007:
    st.warning("Limited data available before 2007. Consider adjusting the release year range for more meaningful insights.")

##############################
# Analysis 1: Releases Across Years (Line Chart)
##############################
st.write("### 1. General Trends Across Time")
st.write("This analysis presents the total number of game releases per year followed by the total number of players across all titles released in each year.")
st.write("We notice a strong increase in the number of releases starting in 2014, which may be related to the rise of indie developers and the introduction of Steam Greenlight.")
with st.spinner("Analyzing..."):
    release_counts = filtered_df.groupby("release_year").size().reset_index(name="count")

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(release_counts["release_year"], release_counts["count"], marker="o", linewidth=2)
    ax1.set_xlabel("Release Year")
    ax1.set_ylabel("Number of Releases")
    ax1.set_title("Total Game Releases per Year")
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))  # Force integer x-ticks
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))  # Force integer y-ticks
    st.pyplot(fig1)

##############################
# Analysis 1.1: Total Players Across Years (Line Chart)
##############################
# st.write("### 1. Total Players Across Years")
st.write("And below we see the total number of owners across all games released in each year.")
st.write("We see that even though the number of titles grow, and Steam's MAU grows, the purchases of games in recent years appear to decline, suggesting a displacement towards older titles.")
st.write("It is also considering that this could be te result of potentially incomplete data for recent years.")
with st.spinner("Analyzing..."):
    release_counts = filtered_df.groupby("release_year")["estimated_owners_boxleiter"].sum().reset_index(name="count")

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(release_counts["release_year"], release_counts["count"], marker="o", linewidth=2)
    ax1.set_xlabel("Release Year")
    ax1.set_ylabel("Number of Players")
    ax1.set_title("Total Number of Title Owners per Year of Release")
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))  # Force integer x-ticks
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))  # Force integer
    st.pyplot(fig1)


##############################
# Analysis 2: Top N Genres Across Years (Stacked Bar Chart - Ratio)
##############################
st.write("### 2. Proportion of Releases by Top Genres Across Years")

with st.spinner("Analyzing..."):
    # Define parameter
    top_n_genres = 15  # Change this value to adjust the number of top genres displayed

    # Explode genres and filter to top N overall
    genres_exploded = filtered_df.explode("genres")
    top_genres = genres_exploded["genres"].value_counts().head(top_n_genres).index.tolist()
    genres_exploded = genres_exploded[genres_exploded["genres"].isin(top_genres)]

    # Group by release year and genre
    genre_year_counts = genres_exploded.groupby(["release_year", "genres"]).size().reset_index(name="count")

    # Pivot for stacked bar chart
    pivot_genres = genre_year_counts.pivot(index="release_year", columns="genres", values="count").fillna(0)

    # Convert counts to proportions (normalize per year)
    pivot_genres = pivot_genres.div(pivot_genres.sum(axis=1), axis=0)  # Normalize row-wise

    # Plot stacked bar chart
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    pivot_genres.plot(kind="bar", stacked=True, ax=ax2, width=0.8)

    ax2.set_xlabel("Release Year")
    ax2.set_ylabel("Proportion of Releases")
    ax2.set_title(f"Proportion of Releases by Top {top_n_genres} Genres Over Time")
    ax2.legend(title="Genre", bbox_to_anchor=(1.05, 1), loc="upper left")

    st.pyplot(fig2)


##############################
# Analysis 3: Top N Tags Across Years (Stacked Bar Chart - Ratio)
##############################
st.write("### 3. Proportion of Releases by Top Tags Across Years")

with st.spinner("Analyzing..."):
    # Define parameter
    top_n_tags = 15  # Change this value to adjust the number of top tags displayed

    # Explode tags and filter to top N overall
    tags_exploded = filtered_df.explode("tags")
    top_tags = tags_exploded["tags"].value_counts().head(top_n_tags).index.tolist()
    tags_exploded = tags_exploded[tags_exploded["tags"].isin(top_tags)]

    # Group by release year and tag
    tag_year_counts = tags_exploded.groupby(["release_year", "tags"]).size().reset_index(name="count")

    # Pivot for stacked bar chart
    pivot_tags = tag_year_counts.pivot(index="release_year", columns="tags", values="count").fillna(0)

    # Convert counts to proportions (normalize per year)
    pivot_tags = pivot_tags.div(pivot_tags.sum(axis=1), axis=0)  # Normalize row-wise

    # Plot stacked bar chart
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    pivot_tags.plot(kind="bar", stacked=True, ax=ax3, width=0.8)

    ax3.set_xlabel("Release Year")
    ax3.set_ylabel("Proportion of Releases")
    ax3.set_title(f"Proportion of Releases by Top {top_n_tags} Tags Over Time")
    ax3.legend(title="Tag", bbox_to_anchor=(1.05, 1), loc="upper left")

    st.pyplot(fig3)


##############################
# Analysis 4: Median & Average Estimated Owners per Year (Line Chart)
##############################
st.write("### 4. Median Estimated Playerbase per Title per Year")
st.write("This analysis presents a sharp decline starting in 2014, which most likely points to the explosion of Steam Greenlight titles and the subsequent flood of low-quality games to the market.")
st.write("Sources:")
st.write("https://steamcommunity.com/games/593110/announcements/detail/1328973169870947116, https://www.theguardian.com/technology/2017/feb/13/valve-kills-steam-greenlight-heres-why-it-matters")
with st.spinner("Analyzing..."):
    # Compute median & mean estimated owners per year
    owners_stats = filtered_df.groupby("release_year")["estimated_owners_boxleiter"].agg(["median", "mean"]).reset_index()

    # Plot
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    ax4.plot(owners_stats["release_year"], owners_stats["median"], marker="o", linewidth=2, label="Median Owners")
    # ax4.plot(owners_stats["release_year"], owners_stats["mean"], marker="s", linewidth=2, label="Average Owners")

    # Keep only the top 10% of titles in a new dataframe
    top_10_percent = filtered_df[filtered_df["estimated_owners_boxleiter"] > filtered_df["estimated_owners_boxleiter"].quantile(0.9)]
    top_10_percent = top_10_percent.groupby("release_year")["estimated_owners_boxleiter"].median().reset_index()
    # Plot the top 10% of titles
    ax4.plot(top_10_percent["release_year"], top_10_percent["estimated_owners_boxleiter"], marker="s", linewidth=2, label="Median owners (top 10% titles only)", color="red")

    # Set labels and title
    ax4.set_xlabel("Release Year")
    ax4.set_ylabel("Estimated Owners")
    ax4.set_title("Estimated Playerbase: Median per title per Year")
    ax4.legend()

    # Ensure x-axis and y-axis ticks are integers
    ax4.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))  # Force integer x-ticks
    ax4.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax4.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))  # Force integer y-ticks

    st.pyplot(fig4)


##############################
# Analysis 5: Average Review Score per Year (Line Chart)
##############################
st.write("### 5. Average Review Score per Year")
st.write(
    "Coinciding with the decline in estimated playerbase and the peak of the Steam Greenlight crisis in 2016, the average review score per year shows a marked decline starting in between 2014 and 2018."
)

with st.spinner("Analyzing..."):
    # Ensure positive/negative review columns exist and drop rows where either is missing
    filtered_reviews = filtered_df.dropna(subset=["steam_positive_reviews", "steam_negative_reviews"])

    # Ensure numeric types (avoid issues if columns were stored as strings)
    filtered_reviews["steam_positive_reviews"] = pd.to_numeric(filtered_reviews["steam_positive_reviews"], errors="coerce")
    filtered_reviews["steam_negative_reviews"] = pd.to_numeric(filtered_reviews["steam_negative_reviews"], errors="coerce")

    # Drop any remaining rows where review values are still NaN (e.g., failed conversions)
    filtered_reviews = filtered_reviews.dropna(subset=["steam_positive_reviews", "steam_negative_reviews"])

    # Calculate review score per game (avoid division by zero)
    def compute_review_score(pos, neg):
        total = pos + neg
        return pos / total if total > 0 else np.nan  # Avoid division by zero

    filtered_reviews["review_score"] = filtered_reviews.apply(lambda row: compute_review_score(row["steam_positive_reviews"], row["steam_negative_reviews"]), axis=1)

    # Group by release year and compute the mean review score
    review_stats = filtered_reviews.groupby("release_year")["review_score"].mean().reset_index()

    # Plot the results
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    ax5.plot(review_stats["release_year"], review_stats["review_score"], marker="o", linewidth=2, color="green")
    ax5.set_xlabel("Release Year")
    ax5.set_ylabel("Average Review Score")
    ax5.set_title("Average Review Score per Year")
    ax5.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))  # Force integer x-ticks
    ax5.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax5.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))  # Force integer y-ticks
    st.pyplot(fig5)


##############################
# Analysis 6: Proportion of Games by Price Bracket (Stacked Bar Chart - Proportion)
##############################
st.write("### 6. Proportion of Games by Price Bracket Over Time (Ratio)")

with st.spinner("Analyzing..."):
    # Define price brackets
    def price_bracket(price):
        if price == 0:
            return "Free"
        elif price < 5:
            return "<10"
        elif price < 10:
            return "5-10"
        elif price < 20:
            return "10-20"
        elif price < 40:
            return "20-40"
        elif price < 60:
            return "40-60"
        elif price < 80:
            return "60-80"
        elif price >= 80:
            return "80+"
        else:
            return "Unknown"

    # Apply price bracket classification
    filtered_df["price_bracket"] = filtered_df["price_original"].apply(price_bracket)

    # Group by release year and price bracket
    price_year_counts = filtered_df.groupby(["release_year", "price_bracket"]).size().reset_index(name="count")

    # Pivot for stacked bar chart
    pivot_prices = price_year_counts.pivot(index="release_year", columns="price_bracket", values="count").fillna(0)

    # Ensure the correct order of price brackets
    price_bracket_order = ["Free", "<10", "5-10", "10-20", "20-40", "40-60", "60-80", "80+"]
    pivot_prices = pivot_prices.reindex(columns=price_bracket_order, fill_value=0)  # Enforce order

    # Convert to proportions (normalize by total games per year)
    pivot_prices = pivot_prices.div(pivot_prices.sum(axis=1), axis=0)  # Normalize each row to sum to 1 (ratio)

    # Plot stacked bar chart
    fig6, ax6 = plt.subplots(figsize=(10, 5))
    pivot_prices.plot(kind="bar", stacked=True, ax=ax6, width=0.8)

    ax6.set_xlabel("Release Year")
    ax6.set_ylabel("Proportion of Games")
    ax6.set_title("Proportion of Games by Price Bracket Over Time")
    ax6.legend(title="Price Bracket", bbox_to_anchor=(1.05, 1), loc="upper left")

    st.pyplot(fig6)
