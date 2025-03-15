"""
This script contains the code for the "Success Factors" page of the app.
It displays a bunch of static analyses for the impact of various elements on the success of a game,
either in terms of sales or review score.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.ticker import MaxNLocator, MultipleLocator
from scipy.signal import savgol_filter

import utils

# Page configuration & custom CSS
st.set_page_config(page_title="Tags Trends")
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


############################################
# Sidebar
############################################
# Sidebar Filter: Release Year Range
st.sidebar.title("Filters")
# Tag options with counts (sorted alphabetically)
tag_counts = df["tags"].explode().value_counts()  # Count occurrences of each tag (for display only)
tag_options = "All", *sorted([f"{tag} ({tag_counts[tag]})" for tag in tag_counts.index])
tag_mapping = {f"{tag} ({tag_counts[tag]})": tag for tag in tag_counts.index}  # Map display tag (with count) to actual value (used in df)
selected_tag_display = st.sidebar.selectbox("All", tag_options)  # Create dropdown with display values
selected_tag = tag_mapping.get(selected_tag_display, "All")  # Map back from selected_display_tag to actual tag

# Sidebar Filter: Release Year Range
min_year = int(df["release_year"].min())
max_year = int(df["release_year"].max())
selected_year_range = st.sidebar.slider("Select Release Year Range", min_year, max_year, (2010, max_year - 1))

# Sidebar, minimum number of reviews for analysis 2
st.sidebar.markdown("---")
# min_review_count = st.sidebar.number_input("Minimum number of reviews for analysis 2", value=10, min_value=1)
# st.sidebar.caption("Median review scores for games with fewer reviews may not be representative. Suggested value: 10.")

# # Sidebar, minimum number of owners for analysis 3
# min_owners_count = st.sidebar.number_input("Minimum number of owners for analysis 3", value=10000, min_value=0)
# st.sidebar.caption(
#     """Since 2014 steam has had an issue with shovelware and fake games,
#     setting a reasonable minimum can help focus the analysis on more "serious" releases.
#     Suggested value: 10,000 for general trends, 2,000 to include more unsuccessful titles."""
# )

############################################
# Preparation and setup
############################################
# Page Title & Description
st.title(f"Factors of Success Analysis")

if df is None or df.empty:
    st.error(f"Data could not be loaded. Please ensure the path is correct and the data is available.")
    st.stop()

if selected_tag == "Select a Tag":
    st.warning("Please select a tag from the sidebar to begin analysis.")
    st.stop()

# Apply release year filter
df_all_in_year_range = df[(df["release_year"] >= selected_year_range[0]) & (df["release_year"] <= selected_year_range[1])]
if selected_tag != "All":
    df_filtered = df_all_in_year_range[df_all_in_year_range["tags"].apply(lambda x: selected_tag in x)]
else:
    df_filtered = df_all_in_year_range

if df_filtered.empty:
    st.warning("No data available for the selected release year range. Please adjust the filters.")
    st.stop()

st.write(
    f"""This analysis focuses on the tag **'{selected_tag}'** and the release years between
         **{selected_year_range[0]} and {selected_year_range[1]}**, based on **{df_filtered.shape[0]} unique titles**."""
)


# Warn user if there are fewer than N games in df_filtered
if df_filtered.shape[0] < 50:
    st.warning(f"**Warning**: There are fewer than 50 titles matching the selected tag and year range. Usefulnes of the analysis will be limited.")
elif df_filtered.shape[0] < 250:
    st.info(f"**Note**: There are fewer than 250 titles matching the selected tag and year range. Results may be noisy or unreliable.")


##############################
# Achievements
##############################
def do_achievements():
    st.header("Achievements")
    st.write("Here we analyze the impact of achievements on the success of games.")

    # Separate the unknown (Nan/missing) from known
    df_known_achievements = df_filtered[df_filtered["achievements_count"] >= 0]
    # Unkown is everything _not_ in known
    df_unknown_achievements = df_filtered[~df_filtered.index.isin(df_known_achievements.index)]

    # Filter out games with < N reviews or a zero review score
    df_known_achievements = df_known_achievements[df_known_achievements["steam_total_reviews"] >= 10]
    df_unknown_achievements = df_unknown_achievements[df_unknown_achievements["steam_total_reviews"] >= 10]
    df_known_achievements = df_known_achievements[df_known_achievements["steam_positive_review_ratio"] > 0]

    # Print average review score for games unknown, zero and non-zero achievements
    st.write(
        f"""First, we notice that games with achievement have a higher average review score than games without achievements:
    - Average review score for games with unknown achievements: {df_unknown_achievements['steam_positive_review_ratio'].mean():.2f}
    - Average review score for games with zero achievements: {df_known_achievements[df_known_achievements['achievements_count'] == 0]['steam_positive_review_ratio'].mean():.2f}
    - Average review score for games with achievements: {df_known_achievements[df_known_achievements['achievements_count'] > 0]['steam_positive_review_ratio'].mean():.2f}"""
    )

    # Filter out top and bottom 1% of games by achievements count
    achievements_count_quantiles = df_known_achievements["achievements_count"].quantile([0.01, 0.99])
    df_known_achievements = df_known_achievements[
        (df_known_achievements["achievements_count"] >= achievements_count_quantiles[0.01]) & (df_known_achievements["achievements_count"] <= achievements_count_quantiles[0.99])
    ]

    # calculate a trend line
    x = df_known_achievements["achievements_count"]
    y = df_known_achievements["steam_positive_review_ratio"]
    trend_line, r2 = utils.get_trend_line_and_r2(x, y)

    st.write(f"""We do not however observe a strong correlation (R^2={r2:.3f}) between the number of achievements and the review score:""")

    # # Scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df_known_achievements["achievements_count"], df_known_achievements["steam_positive_review_ratio"], label="Achievements", alpha=0.5)

    ax.plot(x, trend_line(x), label=f"Trend Line (R^2={r2:.3f})", color="red")

    ax.set_xlabel("Number of Achievements")
    ax.set_ylabel("Mean Review Score")
    ax.set_title("Mean Review Score by Number of Achievements")
    ax.legend()
    st.pyplot(fig)


with st.spinner("Running...", show_time=True):
    do_achievements()


##############################
# average_time_to_beat
##############################
def do_time_to_beat():
    st.header("Game Duration (time to beat)")
    st.write("Here we analyze the impact of game duration on their reception.")

    # keep only the non-null, > N values
    df_known_time_to_beat = df_filtered[df_filtered["average_time_to_beat"].notnull() & (df_filtered["average_time_to_beat"] > 0.1)]

    # Filter out games with < 10 reviews or a zero review score
    df_known_time_to_beat = df_known_time_to_beat[df_known_time_to_beat["steam_total_reviews"] >= 10]
    df_known_time_to_beat = df_known_time_to_beat[df_known_time_to_beat["steam_positive_review_ratio"] > 0]

    # Filter out top and bottom 1% of games by average_time_to_beat
    achievements_count_quantiles = df_known_time_to_beat["average_time_to_beat"].quantile([0.01, 0.99])
    df_known_time_to_beat = df_known_time_to_beat[
        (df_known_time_to_beat["average_time_to_beat"] >= achievements_count_quantiles[0.01]) & (df_known_time_to_beat["average_time_to_beat"] <= achievements_count_quantiles[0.99])
    ]

    # calculate a trend line
    x = df_known_time_to_beat["average_time_to_beat"]
    y = df_known_time_to_beat["steam_positive_review_ratio"]
    trend_line, r2 = utils.get_trend_line_and_r2(x, y)

    st.write(f"""We do not however observe a strong correlation (R^2={r2:.3f}) between the duration of a game and its review score:""")

    # # Scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df_known_time_to_beat["average_time_to_beat"], df_known_time_to_beat["steam_positive_review_ratio"], label="Achievements", alpha=0.5)

    ax.plot(x, trend_line(x), label=f"Trend Line (R^2={r2:.3f})", color="red")

    ax.set_xlabel("Average Time to Beat (hours)")
    ax.set_ylabel("Mean Review Score")
    ax.set_title("Mean Review Score by Game Duration")
    ax.legend()
    st.pyplot(fig)


with st.spinner("Running...", show_time=True):
    do_time_to_beat()


##############################
# controller_support
##############################
def do_controller_support():
    st.header("Controller Support")
    st.write("Here we analyze the impact of controller support on the success of games.")

    df_controller_support = df_filtered.copy()

    # Filter out games with < 10 reviews or a zero review score
    df_controller_support = df_controller_support[df_controller_support["steam_total_reviews"] >= 10]
    df_controller_support = df_controller_support[df_controller_support["steam_positive_review_ratio"] > 0]

    # Replace NaN with "Unknown"
    df_controller_support["controller_support"] = df_controller_support["controller_support"].fillna("unknown")

    # group by controller support values
    df_controller_support = (
        df_controller_support.groupby("controller_support")
        .agg(
            {
                "steam_positive_review_ratio": "mean",
                "steam_total_reviews": "count",
                "release_year": "mean",
            }
        )
        .reset_index()
    )

    message = """We observe that controller support itself does _not_ seem to have a strong impact on the review score of a game.
    We suggest an cancelling out of the advantages of controller support by the negative impact of poorly ported games.
    However, games with unknown controller support score surprisingly higher than those with this information, but this
    is likely due to the fact that the controller support data is missing for many more recent games, suggesting perhaps
    an upward shift in review scores over time instead of a direct positive impact of "unknown" controller support:"""

    mean_release_year_for_unknown = df_controller_support[df_controller_support["controller_support"] == "unknown"]["release_year"].mean()
    mean_release_year_for_known = df_controller_support[df_controller_support["controller_support"] != "unknown"]["release_year"].mean()
    message += f"\n- Average release year for games with unknown controller support: **{mean_release_year_for_unknown:.2f}**"
    message += f"\n- Average release year for games with known controller support: **{mean_release_year_for_known:.2f}**"

    st.write(message)

    # Plot as a bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df_controller_support["controller_support"], df_controller_support["steam_positive_review_ratio"], alpha=0.7)
    ax.set_xlabel("Controller Support")
    ax.set_ylabel("Mean Review Score")
    ax.set_title("Mean Review Score by Controller Support")

    # ax.set_xlabel("Average Time to Beat (hours)")
    # ax.set_ylabel("Mean Review Score")
    # ax.set_title("Mean Review Score by Game Duration")
    ax.legend()
    st.pyplot(fig)


with st.spinner("Running...", show_time=True):
    do_controller_support()
