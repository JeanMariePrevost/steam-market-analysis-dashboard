"""
This script contains the code for the "Success Factors" page of the app.
It displays a bunch of static analyses for the impact of various elements on the success of a game,
either in terms of sales or review score.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.ticker import MaxNLocator, MultipleLocator
from scipy.signal import savgol_filter

import utils

# Page configuration & custom CSS
st.set_page_config(page_title="Factors of Success")
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

st.sidebar.markdown("---")

target_metric = st.sidebar.radio("Select a target metric:", ["Review Scores", "Estimated Owners"])

st.sidebar.markdown("---")
min_review_count = st.sidebar.number_input("Minimum number of reviews per game", value=10, min_value=1)
st.sidebar.caption("Games with very few reviews have a high chance of being fake or having skewed review scores. Suggested value: 10.")


############################################
# Helper Functions
############################################
def plot_categorical(
    df: pd.DataFrame,
    metric_column: str,
    category_column: str,
    header: str,
    body_before: str = None,
    body_after: str = None,
    control_for_yearly_trends: bool = True,
    metric_label: str = None,
    category_label: str = None,
):
    """
    Plots a bar chart of a categorical variable against a metric variable.
    """

    df = df.copy()

    if metric_label is None:
        metric_label = metric_column
    if category_label is None:
        category_label = category_column

    st.header(header)

    with st.spinner("Running...", show_time=True):
        # introduce a temporary column of the unique values of the category column, and "unknown" for NaN
        temp_category_column = f"{category_column}_temp"
        df[temp_category_column] = df[category_column].astype(str).fillna("unknown")
        # normalize review scores against yearly trends (e.g. ratings tend to go up over time)
        if control_for_yearly_trends:
            df[metric_column] = utils.normalize_metric_across_groups(df, metric_column, "release_year", method="diff")

        # Test for significance
        f_stat, p_value, eta_squared = utils.anova_categorical(df, metric_column, temp_category_column)

        if body_before:
            st.write(body_before)

        # Print a description of the results
        # Determine significance level and effect size
        precision = 6
        format_string = f".{precision}f"

        p_value = round(p_value, precision)
        if p_value > 0:
            p_value_string = f"{p_value:{format_string}}"
        else:
            p_value_string = f"< {10**-precision:{format_string}}"  # If p-value rounds to zero, display like "< 0.000001" (using correct precision)

        eta_squared = round(eta_squared, precision)
        if eta_squared > 0:
            eta_squared_string = f"{eta_squared:{format_string}}"
        else:
            eta_squared_string = f"< {10**-precision:{format_string}}"

        if p_value < 0.05:
            message = f'We note a **statistically significant** impact of "{category_label}" on "{metric_label}" (p-value: {p_value_string})'
        elif p_value < 0.1:
            message = f'We note a **potentially significant** impact of "{category_label}" on "{metric_label}" (p-value: {p_value_string})'
        else:
            message = f'We observe **no statistically significant** impact of "{category_label}" on "{metric_label}" (p-value: {p_value_string})'

        if p_value < 0.1:
            if eta_squared >= 0.14:
                message += f" explaining {eta_squared_string} of the variance, indicating a **strong relationship**."
            elif eta_squared >= 0.06:
                message += f" explaining {eta_squared_string} of the variance, indicating a **moderate relationship**."
            elif eta_squared >= 0.01:
                message += f", however, only {eta_squared_string} of the variance is explained, indicating a **weak relationship**."
            else:
                message += f", however, only {eta_squared_string} of the variance is explained, indicating a **negligible relationship**."
        else:
            if eta_squared >= 0.14:
                message += f" explaining {eta_squared_string} of the variance, indicating a **strong relationship**."
            elif eta_squared >= 0.06:
                message += f" explaining {eta_squared_string} of the variance, indicating a **moderate relationship**."
            elif eta_squared >= 0.01:
                message += f" explaining only {eta_squared_string} of the variance, indicating a **weak relationship**."
            else:
                message += f" explaining only {eta_squared_string} of the variance, indicating a **negligible relationship**."

        # Combine the messages
        st.write(message)

        if body_after:
            st.write(body_after)

        # group by category_column
        df = (
            df.groupby(temp_category_column)
            .agg(
                {
                    metric_column: "mean",
                    "release_year": "mean",
                }
            )
            .reset_index()
        )

        # Plot as a bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(df[temp_category_column], df[metric_column], alpha=0.7)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)  # Add a zero line
        ax.set_xlabel(category_label)
        ax.set_ylabel(metric_label)
        ax.set_title(f"Mean {metric_label} by {category_label}")
        ax.legend()
        st.pyplot(fig)


def plot_numerical(
    df: pd.DataFrame,
    metric_column: str,
    independent_var_column: str,
    header: str,
    body_before: str = None,
    body_after: str = None,
    metric_label: str = None,
    independent_var_label: str = None,
):
    """
    Plots a line chart of a metric against a numerical independent variable.
    """

    if metric_label is None:
        metric_label = metric_column
    if independent_var_label is None:
        independent_var_label = independent_var_column

    st.header(header)

    with st.spinner("Running...", show_time=True):
        if body_before:
            st.write(body_before)

        x = df[independent_var_column]
        y = df[metric_column]
        trend_line, r2, individual_p_values, p_value, cohen_f2 = utils.polynomial_regression_analysis(x, y, 1)

        # Print a description of the results
        # Determine significance level
        precision = 3
        format_string = f".{precision}f"

        p_value = round(p_value, precision)
        if p_value > 0:
            p_value_string = f"{p_value:{format_string}}"
        else:
            p_value_string = f"< {10**-precision:{format_string}}"  # If p-value rounds to zero, display like "< 0.000001" (using correct precision)

        r2 = round(r2, precision)
        if r2 > 0:
            r2_string = f"{r2:{format_string}}"
        else:
            r2_string = f"< {10**-precision:{format_string}}"

        if p_value < 0.05:
            message = f'We note a **statistically significant** impact of "{independent_var_label}" on "{metric_label}" (p-value: {p_value_string})'
        elif p_value < 0.1:
            message = f'We note a **potentially significant** impact of "{independent_var_label}" on "{metric_label}" (p-value: {p_value_string})'
        else:
            message = f'We observe **no statistically significant** impact of "{independent_var_label}" on "{metric_label}" (p-value: {p_value_string})'

        if cohen_f2 < 0.02:
            message += f", with a **negligible effect size** (Cohen's f²: {cohen_f2:.3f}),"
        elif cohen_f2 < 0.15:
            message += f", with a **small effect size** (Cohen's f²: {cohen_f2:.3f}),"
        elif cohen_f2 < 0.35:
            message += f", with a **moderate effect size** (Cohen's f²: {cohen_f2:.3f}),"
        else:
            message += f", with a **large effect size** (Cohen's f²: {cohen_f2:.3f}),"

        if p_value < 0.1 and cohen_f2 >= 0.15:
            ## _Seems_ like t should be a strong relationship
            if r2 >= 0.5:
                message += f" explaining {r2_string} of the variance, indicating a **strong relationship**."
            elif r2 >= 0.25:
                message += f" explaining {r2_string} of the variance, indicating a **moderate relationship**."
            elif r2 >= 0.05:
                message += f" however, only {r2_string} of the variance is explained, indicating a **weak relationship**."
            else:
                message += f" however, only {r2_string} of the variance is explained, indicating a **negligible relationship**."
        else:
            if r2 >= 0.14:
                message += f" explaining {r2_string} of the variance, indicating a **strong relationship**."
            elif r2 >= 0.06:
                message += f" explaining {r2_string} of the variance, indicating a **moderate relationship**."
            elif r2 >= 0.01:
                message += f" explaining only {r2_string} of the variance, indicating a **weak relationship**."
            else:
                message += f" explaining only {r2_string} of the variance, indicating a **negligible relationship**."

        # # Combine the messages
        st.write(message)

        if body_after:
            st.write(body_after)

        # Plot as a scatter + trend line
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x, y, label="Data", alpha=0.5)
        ax.plot(x, trend_line(x), label=f"Trend Line (R^2={r2:.3f})", color="red")
        ax.set_xlabel(independent_var_label)
        ax.set_ylabel(metric_label)
        ax.set_title(f"Mean {metric_label} by {independent_var_label}")
        ax.legend()
        st.pyplot(fig)


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

st.warning(
    "**Note:** This analysis identifies statistical relationships between variables but does not establish causation or the direction of influence. "
    "Any observed associations should not be interpreted as direct cause-and-effect relationships."
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
    trend_line, r2, individual_p_values, p_value, cohen_f2 = utils.polynomial_regression_analysis(x, y)

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
    trend_line, r2, individual_p_values, p_value, cohen_f2 = utils.polynomial_regression_analysis(x, y)

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

    # normalize review scores against yearly trends
    df_controller_support["steam_positive_review_ratio_norm"] = utils.normalize_metric_across_groups(df_controller_support, "steam_positive_review_ratio", "release_year", method="diff")

    f_stat, p_value, eta_squared = utils.anova_categorical(df_controller_support, "steam_positive_review_ratio_norm", "controller_support")

    st.write(f"We observe no measurable impact of controller support on the review score of games when controlling for release year (p-value: {p_value:.2f}).")

    # group by controller support values
    df_controller_support = (
        df_controller_support.groupby("controller_support")
        .agg(
            {
                "steam_positive_review_ratio_norm": "mean",
                "steam_total_reviews": "count",
                "release_year": "mean",
            }
        )
        .reset_index()
    )

    # Plot as a bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df_controller_support["controller_support"], df_controller_support["steam_positive_review_ratio_norm"], alpha=0.7)
    ax.set_xlabel("Controller Support")
    ax.set_ylabel("Mean Review Score")
    ax.set_title("Mean Review Score by Controller Support")
    ax.legend()
    st.pyplot(fig)


with st.spinner("Running...", show_time=True):
    do_controller_support()


##############################
# early_access
##############################
def do_early_access():
    st.header("Early Access")
    st.write("Here we analyze the impact of early access on the success of games.")

    df_early_access = df_filtered.copy()

    # Filter out games with < 10 reviews or a zero review score
    df_early_access = df_early_access[df_early_access["steam_total_reviews"] >= 10]
    df_early_access = df_early_access[df_early_access["steam_positive_review_ratio"] > 0]

    # Convert all early_access values to strings, mapping booleans appropriately
    df_early_access["early_access"] = df_early_access["early_access"].apply(lambda x: "yes" if x is True else ("no" if x is False else "unknown"))

    # normalize review scores against yearly trends
    df_early_access["steam_positive_review_ratio_norm"] = utils.normalize_metric_across_groups(df_early_access, "steam_positive_review_ratio", "release_year", method="diff")

    f_stat, p_value, eta_squared = utils.anova_categorical(df_early_access, "steam_positive_review_ratio_norm", "early_access")
    st.write(f"We observe no significant impact of early access status on review scores when controlling for year of release, and no meaningful coorelation (p-value: {p_value:.2f}).")

    # group by controller support values
    df_early_access = (
        df_early_access.groupby("early_access")
        .agg(
            {
                "steam_positive_review_ratio_norm": "mean",
                "steam_total_reviews": "count",
                "release_year": "mean",
            }
        )
        .reset_index()
    )

    # Plot as a bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df_early_access["early_access"], df_early_access["steam_positive_review_ratio_norm"], alpha=0.7)
    ax.set_xlabel("Early Access")
    ax.set_ylabel("Mean Review Score")
    ax.set_title("Mean Review Score by Early Access")
    ax.legend()
    st.pyplot(fig)


with st.spinner("Running...", show_time=True):
    do_early_access()


##############################
# gamefaqs_difficulty_rating
##############################
def do_early_access():
    st.header("Game Difficulty")
    st.write("Here we analyze the relationship between game difficulty ratings (as defined by GameFAQs) and critical reception.")

    df_difficulty = df_filtered.copy()

    # Filter out games with < 10 reviews or a zero review score
    df_difficulty = df_difficulty[df_difficulty["steam_total_reviews"] >= 10]
    df_difficulty = df_difficulty[df_difficulty["steam_positive_review_ratio"] > 0]

    # Convert all early_access values to strings, mapping booleans appropriately
    df_difficulty["gamefaqs_difficulty_rating"].fillna("unknown", inplace=True)

    # normalize review scores against yearly trends
    df_difficulty["steam_positive_review_ratio_norm"] = utils.normalize_metric_across_groups(df_difficulty, "steam_positive_review_ratio", "release_year", method="diff")

    f_stat, p_value, eta_squared = utils.anova_categorical(df_difficulty, "steam_positive_review_ratio_norm", "gamefaqs_difficulty_rating")
    st.write(f"(p-value: {p_value:.2f}).")

    # group by controller support values
    df_difficulty = (
        df_difficulty.groupby("gamefaqs_difficulty_rating")
        .agg(
            {
                "steam_positive_review_ratio_norm": "mean",
                "steam_total_reviews": "count",
                "release_year": "mean",
            }
        )
        .reset_index()
    )

    # Rename the difficulty ratings
    rename_dict = {
        "unknown": "unknown",
        "Simple": "1. Simple",
        "Simple-Easy": "2. Simple-Easy",
        "Easy": "3. Easy",
        "Easy-Just Right": "4. Easy-Just Right",
        "Just Right": "5. Just Right",
        "Just Right-Tough": "6. Just Right-Tough",
        "Tough": "7. Tough",
        "Tough-Unforgiving": "8. Tough-Unforgiving",
        "Unforgiving": "9. Unforgiving",
    }
    df_difficulty["gamefaqs_difficulty_rating"] = df_difficulty["gamefaqs_difficulty_rating"].map(rename_dict)

    # Sort by difficulty rating
    df_difficulty = df_difficulty.sort_values("gamefaqs_difficulty_rating", ascending=False)

    # Plot as a bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df_difficulty["gamefaqs_difficulty_rating"], df_difficulty["steam_positive_review_ratio_norm"], alpha=0.7)
    ax.set_ylabel("Game Difficulty Rating")
    ax.set_xlabel("Mean Review Score")
    ax.set_title("Mean Review Score by Game Difficulty Rating")
    ax.legend()
    st.pyplot(fig)


with st.spinner("Running...", show_time=True):
    do_early_access()


##############################
# has_demos
##############################

plot_categorical(
    df=df_filtered,
    metric_column="steam_positive_review_ratio",
    category_column="has_demos",
    header="Game Demos",
    body_before="This suggests that offering a demo has no significant impact on the review score of a game.",
    metric_label="Review Score",
    category_label="Has a Demo",
)


##############################
# languages_supported _count_
##############################
temp_df = df_filtered.copy()

temp_df = temp_df.dropna(subset=["languages_supported"])
temp_df["languages_supported_count"] = temp_df["languages_supported"].apply(lambda x: len(x))

# Drop games with < 10 reviews
temp_df = temp_df[temp_df["steam_total_reviews"] >= 10]

# Drop games with zero review score
temp_df = temp_df[temp_df["steam_positive_review_ratio"] > 0]

# Drop games with more than N languages
temp_df = temp_df[temp_df["languages_supported_count"] < 20]

# Rerun the analysis
plot_numerical(
    df=temp_df,
    metric_column="steam_positive_review_ratio",
    independent_var_column="languages_supported_count",
    header="Languages Supported",
    body_before="Note that games with an extremely high number of languages supported have been excluded from this analysis due to the assumption that they are fake games or lying about their language support, but the effect was still negligible.",
    metric_label="Review Score",
    independent_var_label="Number of Languages Supported",
)


##############################
# languages_with_full_audio _count_
##############################
temp_df = df_filtered.copy()

temp_df = temp_df.dropna(subset=["languages_with_full_audio"])
temp_df["languages_with_full_audio_count"] = temp_df["languages_with_full_audio"].apply(lambda x: len(x))

# Drop games with < 10 reviews
temp_df = temp_df[temp_df["steam_total_reviews"] >= 10]

# Drop games with zero review score
temp_df = temp_df[temp_df["steam_positive_review_ratio"] > 0]

# Drop games with more than N languages
temp_df = temp_df[temp_df["languages_with_full_audio_count"] < 20]

# Sanity check, prind a bunch fo random game names, their number of languages and the review score
st.write(temp_df.sample(10)[["name", "languages_with_full_audio_count", "steam_positive_review_ratio"]])

# Rerun the analysis
plot_numerical(
    df=temp_df,
    metric_column="steam_positive_review_ratio",
    independent_var_column="languages_with_full_audio_count",
    header="Languages Full Audio Supported",
    body_before="Note that games with an extremely high number of audio languages have been excluded from this analysis due to the assumption that they are fake games or lying about their language support, but the effect was still negligible.",
    metric_label="Review Score",
    independent_var_label="Number of Languages with Full Audio",
)


##############################
# price_original
##############################

temp_df = df_filtered.copy()

# Drop games with < 10 reviews
temp_df = temp_df[temp_df["steam_total_reviews"] >= 10]

# Drop games with zero review score
temp_df = temp_df[temp_df["steam_positive_review_ratio"] > 0]

# Drop games with zero price
temp_df = temp_df[temp_df["price_original"] > 0]

# Drop games with top/bottom 1% price
q_low, q_high = temp_df["price_original"].quantile([0.01, 0.99])
temp_df = temp_df[(temp_df["price_original"] >= q_low) & (temp_df["price_original"] <= q_high)]


plot_numerical(
    df=temp_df,
    metric_column="steam_positive_review_ratio",
    independent_var_column="price_original",
    header="Launch Price",
    body_after="This suggests that the launch price of a game has no significant impact on its review score.",
    metric_label="Review Score",
    independent_var_label="Original Price",
)


##############################
# "Runs on"
##############################
# runs_on_linux	runs_on_mac	runs_on_steam_deck	runs_on_windows


def do_runs_on_platform():
    st.header("Platform Support")

    with st.spinner("Running...", show_time=True):

        df_runs_on = df_filtered.copy()

        # normalize review scores against yearly trends
        df_runs_on["steam_positive_review_ratio"] = utils.normalize_metric_across_groups(df_runs_on, "steam_positive_review_ratio", "release_year", method="diff")

        # Test for significance / detemination
        win_f_stat, win_p_value, win_r_squared, win_cohen_d = utils.ttest_two_groups(df_runs_on, "steam_positive_review_ratio", "runs_on_windows")
        mac_f_stat, mac_p_value, mac_r_squared, mac_cohen_d = utils.ttest_two_groups(df_runs_on, "steam_positive_review_ratio", "runs_on_mac")
        linux_f_stat, linux_p_value, linux_r_squared, linux_cohen_d = utils.ttest_two_groups(df_runs_on, "steam_positive_review_ratio", "runs_on_linux")
        deck_f_stat, deck_p_value, deck_r_squared, deck_cohen_d = utils.ttest_two_groups(df_runs_on, "steam_positive_review_ratio", "runs_on_steam_deck")

        # Debug, just print each
        format_string = ".3f"
        st.write(
            f"""
            The findings are as follows:

            Though all platform support had a small to moderate effect size, only the Linux and Mac support had a statistically significant association with review scores.

            - **Windows**: p-value: {win_p_value:{format_string}}, r-squared: {win_r_squared:{format_string}}, Cohen's d: {win_cohen_d:{format_string}}
            - **Mac**: p-value: {mac_p_value:{format_string}}, r-squared: {mac_r_squared:{format_string}}, Cohen's d: {mac_cohen_d:{format_string}}
            - **Linux**: p-value: {linux_p_value:{format_string}}, r-squared: {linux_r_squared:{format_string}}, Cohen's d: {linux_cohen_d:{format_string}}
            - **Steam Deck**: p-value: {deck_p_value:{format_string}}, r-squared: {deck_r_squared:{format_string}}, Cohen's d: {deck_cohen_d:{format_string}}
            """
        )
    # List of platform columns to compare
    platform_columns = ["runs_on_windows", "runs_on_mac", "runs_on_linux", "runs_on_steam_deck"]

    # Melt the DataFrame so that we have a row per game per platform. ("unpivot" the data)
    # The "Supported" column will have the True/False values.
    df_melted = df_runs_on.melt(id_vars=["steam_positive_review_ratio"], value_vars=platform_columns, var_name="Platform", value_name="Supported")

    # Group by Platform and Supported to compute the mean normalized review score.
    df_means = df_melted.groupby(["Platform", "Supported"])["steam_positive_review_ratio"].mean().reset_index()

    # Create the bar plot using seaborn (because it automates the grouping)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_means, x="Platform", y="steam_positive_review_ratio", hue="Supported", ax=ax)
    ax.set_title("Mean Normalized Review Score by Platform Support")
    ax.set_xlabel("Platform")
    ax.set_ylabel("Mean Normalized Steam Positive Review Ratio")

    # Display the plot in Streamlit
    st.pyplot(fig)


do_runs_on_platform()


##############################
# steam_store_movie_count
##############################

df_temp = df_filtered.copy()

# remove N% outliers in terms of number of movies
q_low, q_high = df_temp["steam_store_movie_count"].quantile([0.01, 0.99])
df_temp = df_temp[(df_temp["steam_store_movie_count"] >= q_low) & (df_temp["steam_store_movie_count"] <= q_high)]


plot_numerical(
    df=df_temp,
    metric_column="steam_positive_review_ratio",
    independent_var_column="steam_store_movie_count",
    header="Number of Trailers",
    body_before="This suggests that the number of trailers a game has has no significant impact on the review score.",
    metric_label="Review Score",
    independent_var_label="Number of Trailers",
)


##############################
# steam_store_screenshot_count
##############################

df_temp = df_filtered.copy()

# remove N% outliers in terms of number of screenshots
q_low, q_high = df_temp["steam_store_screenshot_count"].quantile([0.01, 0.99])
df_temp = df_temp[(df_temp["steam_store_screenshot_count"] >= q_low) & (df_temp["steam_store_screenshot_count"] <= q_high)]

plot_numerical(
    df=df_temp,
    metric_column="steam_positive_review_ratio",
    independent_var_column="steam_store_screenshot_count",
    header="Number of Screenshots",
    body_before="This suggests that the number of screenshots a game has has no significant impact on the review score.",
    metric_label="Review Score",
    independent_var_label="Number of Screenshots",
)


##############################
# tags _count_
##############################

# Introduce temporary column of the number of tags
df_filtered["tags_count"] = df_filtered["tags"].apply(lambda x: len(x))

plot_numerical(
    df=df_filtered,
    metric_column="steam_positive_review_ratio",
    independent_var_column="tags_count",
    header="Number of Tags",
    body_after="This suggests that the number of tags a game has is weakly associated with higher review scores.",
    metric_label="Review Score",
    independent_var_label="Number of Tags",
)


##############################
# vr_supported
##############################

plot_categorical(
    df=df_filtered,
    metric_column="steam_positive_review_ratio",
    category_column="vr_supported",
    header="VR Support",
    # body_before="This suggests that VR support has no significant impact on the review score of a game.",
    metric_label="Review Score",
    category_label="VR Supported",
)
