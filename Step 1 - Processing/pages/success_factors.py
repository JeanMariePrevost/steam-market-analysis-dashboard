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
# Set page defaults
############################################


variance_explained_per_variable = {}

default_review_score_range = (0.1, 1.0)
default_tag_display_value = "All"
default_min_year = 2010
default_max_year = 2024
default_target_metric_display_name = "Review Scores"
default_min_review_count = 10

# If not already set, set the session state variables to the default values
if "selected_tag_display" not in st.session_state:
    st.session_state["selected_tag_display"] = default_tag_display_value
    st.session_state["selected_year_range"] = (default_min_year, default_max_year)
    st.session_state["target_metric_display_name"] = default_target_metric_display_name
    st.session_state["min_review_count"] = default_min_review_count
    st.session_state["review_score_range"] = default_review_score_range

############################################
# Sidebar
############################################
# Sidebar Filter: Release Year Range
st.sidebar.title("Filters")

# Tag options with counts (sorted alphabetically)
tag_counts = df["tags"].explode().value_counts()  # Count occurrences of each tag (for display only)
tag_options = "All", *sorted([f"{tag} ({tag_counts[tag]})" for tag in tag_counts.index])
tag_mapping = {f"{tag} ({tag_counts[tag]})": tag for tag in tag_counts.index}  # Map display tag (with count) to actual value (used in df)
selected_tag_display = st.sidebar.selectbox("All", tag_options, key="selected_tag_display")  # Create dropdown with display values
selected_tag = tag_mapping.get(selected_tag_display, "All")  # Map back from selected_display_tag to actual tag

# Map the slider range from the data's min and max years
min_year = int(df["release_year"].min())
max_year = int(df["release_year"].max())

selected_year_range = st.sidebar.slider("Select Release Year Range", min_year, max_year, key="selected_year_range")

target_metric_display_name = st.sidebar.radio("Select a target metric:", ["Review Scores", "Estimated Owners"], key="target_metric_display_name")

if st.sidebar.button("Reset Filters"):
    print("Hello!")
    # Clear the session state variables which will make them reset to their default values on rerun
    st.session_state.clear()
    st.rerun()


st.sidebar.markdown("---")
st.sidebar.title("Data Cleaning")
min_review_count = st.sidebar.number_input("Minimum number of reviews per game", min_value=1, key="min_review_count")
st.sidebar.caption("Games with very few reviews have a high chance of being fake or having skewed review scores. Suggested value: 10.")

review_score_range = st.sidebar.slider("Review Scores Range", step=0.01, min_value=0.0, max_value=1.0, key="review_score_range")
st.sidebar.caption("Games with very low scores are likely to be fake or shovelware. Suggested value: 0.1 to 1.0, or 0.1 to 0.99 to exclude perfect scores.")

############################################
# Helper Functions
############################################


def is_default_review_score_analysis():
    if target_metric_display_name != "Review Scores":
        return False
    if selected_year_range[0] != default_min_year or selected_year_range[1] != default_max_year:
        return False
    if selected_tag != "All":
        return False
    if min_review_count != default_min_review_count:
        return False
    if review_score_range != default_review_score_range:
        return False
    return True


def is_default_estimated_owners_analysis():
    if target_metric_display_name != "Estimated Owners":
        return False
    if selected_year_range[0] != default_min_year or selected_year_range[1] != default_max_year:
        return False
    if selected_tag != "All":
        return False
    if min_review_count != default_min_review_count:
        return False
    if review_score_range != default_review_score_range:
        return False
    return True


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
    horizontal: bool = False,
    dropna: bool = True,
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

        if dropna:
            # Drop rows wheremetric or category column is NaN
            df = df.dropna(subset=[metric_column, category_column])
            df[temp_category_column] = df[category_column].astype(str)
        else:
            df[temp_category_column] = df[category_column].astype(str).fillna("unknown")

        # normalize review scores against yearly trends (e.g. ratings tend to go up over time)
        if control_for_yearly_trends:
            df[metric_column] = utils.adjust_metric_for_group_trends(df, metric_column, "release_year")

        # Test for significance
        f_stat, p_value, eta_squared = utils.anova_categorical(df, metric_column, temp_category_column)

        if body_before:
            st.write(body_before)

        # Print a description of the results
        # Determine significance level and effect size
        precision = 3
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

        # Store the result
        variance_explained_per_variable[category_column] = eta_squared

        if body_after:
            st.write(body_after)

        # Use raw data for proper box plot distributions
        if horizontal:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x=metric_column, y=temp_category_column, data=df, ax=ax, showfliers=False, order=sorted(df[temp_category_column].unique()))
            ax.set_xlabel(f"{metric_label} (Adjusted for {category_label})")
            ax.set_ylabel(category_label)
            ax.set_title(f"Adjusted {metric_label} by {category_label}")
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x=temp_category_column, y=metric_column, data=df, ax=ax, showfliers=False, order=sorted(df[temp_category_column].unique()))
            ax.set_xlabel(category_label)
            ax.set_ylabel(f"{metric_label} (Adjusted for {category_label})")
            ax.set_title(f"Adjusted {metric_label} by {category_label}")
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
    trend_line_degree: int = 1,
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
        if body_before and body_before != "":
            st.write(body_before)

        x = df[independent_var_column]
        y = df[metric_column]
        # Sort values of x for trend line
        x, y = zip(*sorted(zip(x, y)))
        trend_line, r2, individual_p_values, p_value, cohen_f2 = utils.polynomial_regression_analysis(x, y, trend_line_degree)

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

        # Combine the messages
        st.write(message)

        # Store the result
        variance_explained_per_variable[independent_var_column] = r2

        if body_after and body_after != "":
            st.write(body_after)

        # Plot as a scatter + trend line
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x, y, label="Data", alpha=0.5)
        ax.plot(x, trend_line(x), label=f"Trend Line (R^2={r2:.3f})", color="red")
        ax.set_xlabel(independent_var_label)
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} by {independent_var_label}")
        ax.legend()
        st.pyplot(fig)


###################################################################################################
# Preparation and setup
###################################################################################################
# Page Title & Description
st.title(f"Factors of Success Analysis")

if df is None or df.empty:
    st.error(f"Data could not be loaded. Please ensure the path is correct and the data is available.")
    st.stop()


# Apply release year filter
df_all_in_year_range = df[(df["release_year"] >= selected_year_range[0]) & (df["release_year"] <= selected_year_range[1])]
if selected_tag != "All":
    df_filtered = df_all_in_year_range[df_all_in_year_range["tags"].apply(lambda x: selected_tag in x)]
else:
    df_filtered = df_all_in_year_range

# Filter out games with < N reviews
df_filtered = df_filtered[df_filtered["steam_total_reviews"] >= min_review_count]

# Filter out games with too low/high review scores
df_filtered = df_filtered[df_filtered["steam_positive_review_ratio"] >= review_score_range[0]]
df_filtered = df_filtered[df_filtered["steam_positive_review_ratio"] <= review_score_range[1]]

target_metric = "steam_positive_review_ratio" if target_metric_display_name == "Review Scores" else "estimated_owners_boxleiter"

# remove 0 owners and nan owners games
df_filtered = df_filtered[df_filtered["estimated_owners_boxleiter"] > 0]

# remove estimated_owners_boxleiter
q_low, q_high = df_filtered["estimated_owners_boxleiter"].quantile([0.01, 0.99])
df_filtered = df_filtered[(df_filtered["estimated_owners_boxleiter"] >= q_low) & (df_filtered["estimated_owners_boxleiter"] <= q_high)]


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


###################################################################################################
# Analysis
###################################################################################################

##############################
# Achievements
##############################
df_temp = df_filtered.copy()

# Drop zero and >100 achievements
df_temp = df_temp[(df_temp["achievements_count"] > 0) & (df_temp["achievements_count"] < 100)]

if is_default_review_score_analysis():
    comment = "So while we do see an upward trend up to a point, it is largely drowned out by the individual variance by title."
elif is_default_estimated_owners_analysis():
    comment = "So while noise and individual variance is very high, we do see positive relationship between the number of achievements and the number of estimated owners."
else:
    comment = ""

plot_numerical(
    df=df_temp,
    metric_column=target_metric,
    independent_var_column="achievements_count",
    header="Achievements",
    body_after=comment,
    metric_label=target_metric_display_name,
    independent_var_label="Number of Achievements",
    trend_line_degree=2,
)


##############################
# average_time_to_beat
##############################

# Drop outliers
q_low, q_high = df_filtered["average_time_to_beat"].quantile([0.01, 0.99])
df_temp = df_filtered[(df_filtered["average_time_to_beat"] >= q_low) & (df_filtered["average_time_to_beat"] <= q_high)]

if is_default_review_score_analysis():
    comment = "So while there is a pattern of seeing fewer games with low review scores as the average time to beat increases, the statistical correlation is virtually non-existent."
elif is_default_estimated_owners_analysis():
    comment = ""
else:
    comment = ""


plot_numerical(
    df=df_temp,
    metric_column="steam_positive_review_ratio",
    independent_var_column="average_time_to_beat",
    header="Game Duration (time to beat)",
    metric_label=target_metric_display_name,
    body_after=comment,
    independent_var_label="Average Time to Beat (hours)",
)


##############################
# controller_support
##############################

df_temp = df_filtered.copy()

# Fill controller support NaNs with "unknown"
df_temp["controller_support"] = df_temp["controller_support"].fillna("unknown")

# Rename for sorting:
rename_dict = {
    "unknown": "unknown",
    "partial": "2. Partial",
    "full": "3. Full",
    "none": "1. None",
}

df_temp["controller_support"] = df_temp["controller_support"].map(rename_dict)

plot_categorical(
    df=df_temp,
    metric_column="steam_positive_review_ratio",
    category_column="controller_support",
    header="Controller Support",
    metric_label=target_metric_display_name,
    category_label="Controller Support",
)


##############################
# early_access
##############################


df_temp = df_filtered.copy()

# Fill NaNs with "unknown"
df_temp["early_access"] = df_temp["early_access"].fillna("unknown")

plot_categorical(
    df=df_temp,
    metric_column="steam_positive_review_ratio",
    category_column="early_access",
    header="Early Access",
    metric_label=target_metric_display_name,
    category_label="Early Access",
)


##############################
# gamefaqs_difficulty_rating
##############################

df_temp = df_filtered.copy()

# Fill missing difficulty ratings with "unknown"
df_temp["gamefaqs_difficulty_rating"] = df_temp["gamefaqs_difficulty_rating"].fillna("unknown")

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
df_temp["gamefaqs_difficulty_rating"] = df_temp["gamefaqs_difficulty_rating"].map(rename_dict)


plot_categorical(
    df=df_temp,
    metric_column="steam_positive_review_ratio",
    category_column="gamefaqs_difficulty_rating",
    header="Game Difficulty",
    metric_label=target_metric_display_name,
    category_label="Game Difficulty Rating",
    horizontal=True,
)


##############################
# has_demos
##############################

plot_categorical(
    df=df_filtered,
    metric_column="steam_positive_review_ratio",
    category_column="has_demos",
    header="Game Demos",
    metric_label=target_metric_display_name,
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

if is_default_review_score_analysis():
    comment = "While there is a visible pattern of higher review scores for games with more languages supported, the statistical correlation is virtually non-existent."
elif is_default_estimated_owners_analysis():
    comment = ""
else:
    comment = ""


# Rerun the analysis
plot_numerical(
    df=temp_df,
    metric_column="steam_positive_review_ratio",
    independent_var_column="languages_supported_count",
    header="Languages Supported",
    body_before="Note that games with an extremely high number of languages supported have been excluded from this analysis due to the assumption that they are fake games or lying about their language support, but the effect was still negligible at any range.",
    body_after=comment,
    metric_label=target_metric_display_name,
    independent_var_label="Number of Languages Supported",
)


##############################
# languages_with_full_audio _count_
##############################
temp_df = df_filtered.copy()

temp_df = temp_df.dropna(subset=["languages_with_full_audio"])
temp_df["languages_with_full_audio_count"] = temp_df["languages_with_full_audio"].apply(lambda x: len(x))

# Drop those with 0 languages, essentially NaNs
temp_df = temp_df[temp_df["languages_with_full_audio_count"] > 0]

# Drop games with more than N languages, which are likely fake
temp_df = temp_df[temp_df["languages_with_full_audio_count"] < 20]

# Rerun the analysis
plot_numerical(
    df=temp_df,
    metric_column="steam_positive_review_ratio",
    independent_var_column="languages_with_full_audio_count",
    header="Languages Full Audio Supported",
    body_before="Note that games with an extremely high number of audio languages have been excluded from this analysis due to the assumption that they are fake games or lying about their language support, but the effect was still negligible.",
    metric_label=target_metric_display_name,
    independent_var_label="Number of Languages with Full Audio",
    trend_line_degree=2,
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
    metric_label=target_metric_display_name,
    independent_var_label="Original Price",
)


##############################
# "Runs on" / Platform Support
##############################
# runs_on_linux	runs_on_mac	runs_on_steam_deck	runs_on_windows

plot_categorical(
    df=df_filtered,
    metric_column="steam_positive_review_ratio",
    category_column="runs_on_linux",
    header="Platform Support (Linux)",
    metric_label=target_metric_display_name,
    category_label="Linux Support",
)

plot_categorical(
    df=df_filtered,
    metric_column="steam_positive_review_ratio",
    category_column="runs_on_mac",
    header="Platform Support (Mac)",
    metric_label=target_metric_display_name,
    category_label="Mac Support",
)

plot_categorical(
    df=df_filtered,
    metric_column="steam_positive_review_ratio",
    category_column="runs_on_steam_deck",
    header="Platform Support (Steam Deck)",
    metric_label=target_metric_display_name,
    category_label="Steam Deck Support",
)

plot_categorical(
    df=df_filtered,
    metric_column="steam_positive_review_ratio",
    category_column="runs_on_windows",
    header="Platform Support (Windows)",
    metric_label=target_metric_display_name,
    category_label="Windows Support",
)


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
    metric_label=target_metric_display_name,
    independent_var_label="Number of Trailers",
)


##############################
# steam_store_screenshot_count
##############################

df_temp = df_filtered.copy()

# remove N% outliers in terms of number of screenshots
q_low, q_high = df_temp["steam_store_screenshot_count"].quantile([0.00, 0.99])
df_temp = df_temp[(df_temp["steam_store_screenshot_count"] >= q_low) & (df_temp["steam_store_screenshot_count"] <= q_high)]

plot_numerical(
    df=df_temp,
    metric_column="steam_positive_review_ratio",
    independent_var_column="steam_store_screenshot_count",
    header="Number of Screenshots",
    body_before="This suggests that the number of screenshots a game has has no significant impact on the review score.",
    metric_label=target_metric_display_name,
    independent_var_label="Number of Screenshots",
    trend_line_degree=2,
)


##############################
# tags _count_
##############################

# Introduce temporary column of the number of tags
df_filtered["tags_count"] = df_filtered["tags"].apply(lambda x: len(x))

if is_default_review_score_analysis():
    comment = """This surprisingly suggests that the number of tags a game has would be weakly associated with higher review scores,
     though it is worth remembering that tags are partly community-driven, and so this could be a reflection of the community's engagement towards the game.
      It is also possible that having more — and more accurate — tags has a positive impact on the review score through being able to target the right audience."""
elif is_default_estimated_owners_analysis():
    comment = ""
else:
    comment = ""


plot_numerical(
    df=df_filtered,
    metric_column="steam_positive_review_ratio",
    independent_var_column="tags_count",
    header="Number of Tags",
    body_after=comment,
    metric_label=target_metric_display_name,
    independent_var_label="Number of Tags",
    # trend_line_degree=2,
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
    metric_label=target_metric_display_name,
    category_label="VR Supported",
)


##############################
# Recap
##############################

if target_metric_display_name == "Review Scores" and selected_year_range[0] == default_min_year and selected_year_range[1] == default_max_year and selected_tag == "All":
    ## Default "full" analysis for review scores
    st.header("Summary of Findings")

    st.info(
        "Remember that the results are based on retrospective data and do not imply causation. They are merely statistical observations, as directionality of relationship cannot be established through these methods."
    )

    if is_default_review_score_analysis():
        st.write("Disappointingly, meta-features such as the number of screenshots, achievements or tags all have but very limited impacts on the review score of a game.")
        st.write("We observe that, if anything, the only significant correlations are the additions of Mac and Linux platform support, and having sufficient tags.")

    results_message = "Here are the proportion of variance explained by each feature, in descending order:"

    # Sort the variance explained by each variable
    variance_explained_per_variable = {k: v for k, v in sorted(variance_explained_per_variable.items(), key=lambda item: item[1], reverse=True)}

    # Print the variance explained by each variable
    for variable, variance in variance_explained_per_variable.items():
        results_message += f"\n- **{variable}**: {variance:.3f}"

    st.write(results_message)
elif target_metric_display_name == "Estimated Owners" and selected_year_range[0] == default_min_year and selected_year_range[1] == default_max_year and selected_tag == "All":
    ## Default "full" analysis for estimated owners
    st.header("Summary of Findings")
    st.write("TODO")

    # st.write("Disappointingly, meta-features such as the number of screenshots, achievements or tags all have but very limited impacts on the number of estimated owners of a game.")
    # st.write("We suggest that, if anything, the only significant factors are the platform support and having sufficient tags.")
    # results_message = "Here are the proportion of variance explained by each feature, in descending order:"

    # # Sort the variance explained by each variable
    # variance_explained_per_variable = {k: v for k, v in sorted(variance_explained_per_variable.items(), key=lambda item: item[1], reverse=True)}

    # # Print the variance explained by each variable
    # for variable, variance in variance_exvariableplained_per_variable.items():
    #     results_message += f"\n- **{variable}**: {variance:.3f}"

    # st.write(results_message)
else:
    st.header("Summary of Findings")
    # Dynamic summary
    st.write("TODO")
