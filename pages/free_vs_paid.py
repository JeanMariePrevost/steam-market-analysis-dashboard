import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import streamlit as st
from matplotlib.ticker import AutoMinorLocator

import utils
from utils import load_main_dataset, remove_outliers_iqr

##############################
# Load & Prepare Data
##############################
df = load_main_dataset()

# Page configuration & custom CSS
st.set_page_config(page_title="Free vs F2P vs Paid: Market Trends on Steam")
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

# Page Title & Description
st.title("Free vs F2P vs Paid: Market Trends on Steam")
st.write(
    "This page presents analyses on the number of releases, total playerbase, gross revenue generated over time, "
    "as well as tag correlations and rating breakdowns for Free, Free-to-Play (F2P), and Paid games on Steam, respectively."
)

if df is None or df.empty:
    st.error(f"Data could not be loaded. Please ensure the path is correct and the data is available.")
    st.stop()

##############################
# Sidebar Filter: Release Year Range
##############################
st.sidebar.title("Filters")
min_year = int(df["release_year"].min())
max_year = int(df["release_year"].max())
selected_year_range = st.sidebar.slider("Select Release Year Range", min_year, max_year, (2007, max_year))

# Apply release year filter
filtered_df = df[(df["release_year"] >= selected_year_range[0]) & (df["release_year"] <= selected_year_range[1])]

if filtered_df.empty:
    st.warning("No data available for the selected release year range. Please adjust the filters.")
    st.stop()


##############################
# Analysis 1: Number of Releases per Type Over Time (Line Plot)
##############################
st.write("### 1. Number of Releases per Monetization Model Over Time")
st.write(
    "This chart shows the number of releases for each monetization model (Free, F2P, Paid) over the years,\
    revealin overall similar trends across models, though paid titles are by far the most prominent on the platform."
    "\n\nWe do note a relative decrease in the number of free and F2P titles compared to paid titles in recent years, potentially showing a change in consumer preferences."
)
with st.spinner("Analyzing..."):
    release_counts = filtered_df.groupby(["release_year", "monetization_model"]).size().reset_index(name="count")
    pivot_releases = release_counts.pivot(index="release_year", columns="monetization_model", values="count").fillna(0)

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    for col in pivot_releases.columns:
        ax1.plot(pivot_releases.index, pivot_releases[col], marker="o", linewidth=2, label=col)
    ax1.set_xlabel("Release Year")
    ax1.set_ylabel("Number of Releases")
    ax1.set_title("Number of Releases per Type Over Time")
    ax1.legend()
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    st.pyplot(fig1)

##############################
# Analysis 2: Total Playerbase per Type Over Time (Line Plot)
##############################
st.write("### 2. Total Playerbase per Type Over Time")
st.write(
    "Unsurprisingly, we observe a similar trend in the playerbase, though non-paid titles manage to acquire \
    a larger part of the playerbase per title, certainly due to low barrier to entry."
)

with st.spinner("Analyzing..."):
    df_2 = remove_outliers_iqr(filtered_df, "estimated_owners_boxleiter", cap_instead_of_drop=True)
    playerbase = df_2.groupby(["release_year", "monetization_model"])["estimated_owners_boxleiter"].sum().reset_index()

    pivot_playerbase = playerbase.pivot(index="release_year", columns="monetization_model", values="estimated_owners_boxleiter").fillna(0) / 1e6  # Convert to millions

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for col in pivot_playerbase.columns:
        ax2.plot(pivot_playerbase.index, pivot_playerbase[col], marker="o", linewidth=2, label=col)
    ax2.set_xlabel("Release Year")
    ax2.set_ylabel("Total Playerbase (in millions)")
    ax2.set_title("Total Playerbase per Type Over Time")
    ax2.legend()
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    st.pyplot(fig2)

# ##############################
# # Analysis 3: Total Revenue Generated per Type Over Time (Line Plot)
# ##############################
st.write("### 3. Total Revenue Generated per Type Over Time")
st.warning(
    "This analysis makes many assumptions, using naive estimates of revenue based on playerbase and average price. \
    This is especially true of F2P titles, which ar enotoriously difficult to estimate revenue for due to the importance of marketing strartegies employed."
)
with st.spinner("Analyzing..."):
    df_3 = remove_outliers_iqr(filtered_df, "estimated_gross_revenue_boxleiter", cap_instead_of_drop=True)
    revenue = df_3.groupby(["release_year", "monetization_model"])["estimated_gross_revenue_boxleiter"].sum().reset_index()
    pivot_revenue = revenue.pivot(index="release_year", columns="monetization_model", values="estimated_gross_revenue_boxleiter").fillna(0)

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    for col in pivot_revenue.columns:
        ax3.plot(pivot_revenue.index, pivot_revenue[col], marker="o", linewidth=2, label=col)
    ax3.set_xlabel("Release Year")
    ax3.set_ylabel("Total Revenue")
    ax3.set_title("Total Revenue Generated per Type Over Time")
    ax3.legend()
    ax3.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax3.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax3.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    st.pyplot(fig3)

# ##############################
# # Analysis 4: Stacked Bar Charts (Ratios)
# ##############################
st.write("### 4. Stacked Bar Charts: Ratio of Releases, Playerbase & Revenue Over Time")
st.write(
    "In line with the previous analyses, we observe a growing interest in non-paid titles up to around 2020, \
    at which point the trend seems to reverse in favor of paid titles."
)
st.write(
    'This suggests either a decrease in interest in free-to-play titles, or the increasing difficulty for new titles \
    to gain traction in an increasingly crowded market dominated by a few major "live service" giants.'
)
st.warning('Not that "year" here refer to the _release year_ of titles, and that these chart do not show global trends of the martker across time.')

with st.spinner("Analyzing..."):
    # a. Releases Ratio
    pivot_releases_ratio = pivot_releases.div(pivot_releases.sum(axis=1), axis=0)
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    pivot_releases_ratio.plot(kind="bar", stacked=True, ax=ax4, width=0.8)
    ax4.set_xlabel("Release Year")
    ax4.set_ylabel("Proportion of Releases")
    ax4.set_title("Proportion of Releases by Type Over Time")
    ax4.legend(title="Game Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    st.pyplot(fig4)

    # b. Playerbase Ratio
    pivot_playerbase_ratio = pivot_playerbase.div(pivot_playerbase.sum(axis=1), axis=0)
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    pivot_playerbase_ratio.plot(kind="bar", stacked=True, ax=ax5, width=0.8)
    ax5.set_xlabel("Release Year")
    ax5.set_ylabel("Proportion of Total Playerbase")
    ax5.set_title("Proportion of Total Playerbase by Type Over Time")
    ax5.legend(title="Game Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    st.pyplot(fig5)

    # c. Revenue Ratio
    pivot_revenue_ratio = pivot_revenue.div(pivot_revenue.sum(axis=1), axis=0)
    fig6, ax6 = plt.subplots(figsize=(10, 5))
    pivot_revenue_ratio.plot(kind="bar", stacked=True, ax=ax6, width=0.8)
    ax6.set_xlabel("Release Year")
    ax6.set_ylabel("Proportion of Total Revenue")
    ax6.set_title("Proportion of Total Revenue by Type Over Time")
    ax6.legend(title="Game Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    st.pyplot(fig6)

# ##############################
# # Analysis 5: Top/Worst Tags by Various Metrics
# ##############################
# TODO - Move the number of tags to a sidebar slider?
# TODO - Add a dropdown to select the metric to analyze instead of showing multiple?
# TODO - Set an input for the smoothing parameter k in the Bayesian shrinkage (and does a value of 0 disable it?)
st.write("### 5. Best and Worst Tags by Metrics")
st.write("This section lists the various tags most associated with the strongest increases and decreases of review scores and estimated playerbase.")


def get_relative_impact_by_tag(df, monetization_model, metric) -> pd.DataFrame:
    # Filter for the current monetization model
    sub_df = df[df["monetization_model"] == monetization_model]

    # Explode tags list into individual rows and drop any missing values
    df_exploded_by_tag = sub_df.explode("tags")
    df_exploded_by_tag = df_exploded_by_tag[df_exploded_by_tag["tags"].notna()]

    # Debug: print all tags with the number of times they appear in the dataset
    tag_counts = df_exploded_by_tag["tags"].value_counts()
    print(tag_counts)

    # Compute the average value of the metric per tag, along with the count of occurrences
    df_per_tag_stats = df_exploded_by_tag.groupby("tags").agg(mean_metric=(metric, "mean"), count=("tags", "count")).reset_index()

    # Compute the global mean for the metric
    global_mean = df[metric].mean()

    #########################################################
    # Bayesian shrinkage
    # i.e. adjust each tag's mean by blending it with the global mean
    # This reduces the impact of tags with very few occurrences
    # https://kiwidamien.github.io/shrinkage-and-empirical-bayes-to-improve-inference.html
    ################################################
    k = 10  # Smoothing parameter; adjust based on your needs, 10 is
    df_per_tag_stats["smoothed_mean"] = (df_per_tag_stats["count"] * df_per_tag_stats["mean_metric"] + k * global_mean) / (df_per_tag_stats["count"] + k)

    # Compute the relative difference using the smoothed mean
    # E.g. if "rpg" has "stem_positive_review_ratio" of 0.90 and the overall average is 0.80, the relative difference is (0.90 - 0.80) / 0.80 = 0.125
    # E.g. if "bowling" has "stem_positive_review_ratio" of 0.60 and the overall average is 0.80, the relative difference is (0.60 - 0.80) / 0.80 = -0.25
    # BUT tags with very few occurrences will be smoothed out towards the global mean
    df_per_tag_stats["relative_diff"] = (df_per_tag_stats["smoothed_mean"] - global_mean) / global_mean

    # Sort by relative difference in descending order
    df_per_tag_stats = df_per_tag_stats.sort_values(by="relative_diff", ascending=False)

    return df_per_tag_stats


# Define a helper function to compute top tags per game type based on a given metric
def top_tags_by_metric_relative(df, monetization_model, metric, top_n=5):
    df_per_tag_stats = get_relative_impact_by_tag(df, monetization_model, metric)
    df_per_tag_stats = df_per_tag_stats.sort_values(by="relative_diff", ascending=False)
    return df_per_tag_stats.head(top_n)


def worst_tags_by_metric_relative(df, monetization_model, metric, top_n=5):
    df_per_tag_stats = get_relative_impact_by_tag(df, monetization_model, metric)
    df_per_tag_stats = df_per_tag_stats.sort_values(by="relative_diff", ascending=True)
    return df_per_tag_stats.head(top_n)


st.write("#### By Impact on Steam Positive Review Ratio")
with st.spinner("Analyzing..."):
    # a. By Steam Positive Review Ratio (all types)
    monetization_models = ["free", "f2p", "paid"]
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10), sharex=True)  # 3 stacked subplots, same x-axis
    for i, monetization_model in enumerate(monetization_models):
        # Fetch top and bottom N tags by a given metric.
        metric = "steam_positive_review_ratio"
        n_value = 5
        top_tags = top_tags_by_metric_relative(filtered_df, monetization_model, metric, top_n=n_value)
        bottom_tags = worst_tags_by_metric_relative(filtered_df, monetization_model, metric, top_n=n_value)

        # Label categories
        top_tags["Category"] = "Top"
        bottom_tags["Category"] = "Bottom"

        df_top_and_bottom = pd.concat([top_tags, bottom_tags])
        df_top_and_bottom = df_top_and_bottom.sort_values(by="relative_diff", ascending=True)  # Sort bars

        # Assign colors based on best/worst
        colors = ["green" if category == "Top" else "red" for category in df_top_and_bottom["Category"]]

        # Horizontal Bar Plot
        axes[i].barh(df_top_and_bottom["tags"], df_top_and_bottom["relative_diff"], color=colors)
        axes[i].set_title(f"{monetization_model.capitalize()} Games")
        axes[i].set_ylabel("Tags")

    axes[-1].set_xlabel("Relative Difference from Average")  # Global x-axis label
    plt.tight_layout()  # Prevent overlap
    st.pyplot(fig)

# b. By Estimated Playerbase (all types)
st.write("#### By Impact on Estimated Number of Owners")
with st.spinner("Analyzing..."):
    monetization_models = ["free", "f2p", "paid"]
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10), sharex=True)  # 3 stacked subplots, same x-axis
    for i, monetization_model in enumerate(monetization_models):
        # Fetch top and bottom N tags by a given metric.
        metric = "estimated_owners_boxleiter"
        n_value = 5
        top_tags = top_tags_by_metric_relative(filtered_df, monetization_model, metric, top_n=n_value)
        bottom_tags = worst_tags_by_metric_relative(filtered_df, monetization_model, metric, top_n=n_value)

        # Label categories
        top_tags["Category"] = "Top"
        bottom_tags["Category"] = "Bottom"

        df_top_and_bottom = pd.concat([top_tags, bottom_tags])
        df_top_and_bottom = df_top_and_bottom.sort_values(by="relative_diff", ascending=True)  # Sort bars

        # Assign colors based on best/worst
        colors = ["green" if category == "Top" else "red" for category in df_top_and_bottom["Category"]]

        # Horizontal Bar Plot
        axes[i].barh(df_top_and_bottom["tags"], df_top_and_bottom["relative_diff"], color=colors)
        axes[i].set_title(f"{monetization_model.capitalize()} Games")
        axes[i].set_ylabel("Tags")

    axes[-1].set_xlabel("Relative Difference from Average")  # Global x-axis label
    plt.tight_layout()  # Prevent overlap
    st.pyplot(fig)
