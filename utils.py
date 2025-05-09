import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import streamlit as st
from scipy.stats import norm
from statsmodels.formula.api import ols


def load_main_dataset() -> pd.DataFrame:
    """
    Load the preprocessed data from the parquet file.
    Returns None if an error occurs.
    """
    current_file = Path(__file__).resolve()
    root_dir = current_file.parent
    data_path = root_dir / "output_preprocessed" / "combined_df_preprocessed_dense.parquet"

    return load_dataset(data_path)


def load_feature_engineered_dataset_with_na() -> pd.DataFrame:
    """
    Load the NA-tolerant feature-engineered data from the parquet file.
    This version retains rows with missing values and does not impute anything.
    Returns None if an error occurs.
    """
    current_file = Path(__file__).resolve()
    root_dir = current_file.parent
    data_path = root_dir / "output_feature_engineered" / "processed_with_na.parquet"

    return load_dataset(data_path)


def load_feature_engineered_dataset_no_na() -> pd.DataFrame:
    """
    Load the "no NA" feature-engineered data from the parquet file.
    This version imputes missing values in various ways.
    Returns None if an error occurs.
    """
    current_file = Path(__file__).resolve()
    root_dir = current_file.parent
    data_path = root_dir / "output_feature_engineered" / "processed_no_na.parquet"

    df = load_dataset(data_path)
    if df is not None:
        if df.isnull().sum().sum() > 0:
            raise ValueError("The dataset contains missing values. Double-check the preprocessing steps leading to processed_no_na.parquet.")

    return df


def load_inference_dataset() -> pd.DataFrame:
    """
    Loads the further preprocessed dataset for "game success" inference.

    This dataset lacks all:
    - irrelevant columns (e.g. name, appid, publishers...)
    - directly correlated columns (e.g. steam_total_reviews, steam_positive_review_ratio, peak_ccu...)
    - outcome-related columns (e.g. , playtime_avg, achievements_count...)

    Returns None if an error occurs.
    """
    current_file = Path(__file__).resolve()
    root_dir = current_file.parent
    data_path = root_dir / "output_feature_engineered" / "inference_dataset.parquet"

    df = load_dataset(data_path)
    if df is not None:
        if df.isnull().sum().sum() > 0:
            raise ValueError("The dataset contains missing values. Double-check the preprocessing steps leading to processed_for_inference.parquet.")

    return df


def load_baseline_element() -> pd.DataFrame:
    """
    Load the baseline element for inference.

    This dataset is a single row with all columns set to the data's median or mode values.

    Returns None if an error occurs.
    """
    current_file = Path(__file__).resolve()
    root_dir = current_file.parent
    data_path = root_dir / "output_feature_engineered" / "baseline_element.parquet"

    df = load_dataset(data_path)
    if df is not None:
        if df.isnull().sum().sum() > 0:
            raise ValueError("The dataset contains missing values. Double-check the preprocessing steps leading to baseline_element.parquet.")

    return df


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a dataset from a file.
    Returns None if an error occurs.
    """
    try:
        df = pd.read_parquet(path)
        return df
    except Exception as e:
        print(f"An error occurred while reading the data: {e}")
        return None


def remove_outliers_iqr(df, value_col, threshold=1.5, cap_instead_of_drop=False):
    """
    Removes or caps outliers from a DataFrame based on the IQR method.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        value_col (str): The numerical column to process.
        threshold (float): The IQR multiplier for detecting outliers, i.e. how many times the IQR to go above Q3 or below Q1 to be considered an outlier.
        cap_instead_of_drop (bool): If True, caps outliers instead of removing them.

    Returns:
        pd.DataFrame: Processed DataFrame with outliers removed or capped.
    """

    if not pd.api.types.is_numeric_dtype(df[value_col]):
        raise ValueError(f"The column '{value_col}' must be numeric to remove outliers.")

    is_int = pd.api.types.is_integer_dtype(df[value_col])

    q1 = df[value_col].quantile(0.25)
    q3 = df[value_col].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    if is_int:
        lower_bound = np.floor(lower_bound)
        upper_bound = np.ceil(upper_bound)

    if cap_instead_of_drop:
        df.loc[:, value_col] = np.clip(df[value_col], lower_bound, upper_bound)
        return df
    else:
        return df[(df[value_col] >= lower_bound) & (df[value_col] <= upper_bound)]


def collapse_pseudo_duplicate_games(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove pseudo-duplicate games from the DataFrame by selecting the best candidate per group.

    Pseudo-duplicates are defined as having the same 'name' and 'release_date'.
    Values will "squashed" together, with non-null values from higher-score candidates overwriting lower-score values.

    Example usage:
    collapsed_df = collapse_duplicate_games(df) # And just like that, pseudo-duplicates are combined
    """

    def normalized_score(series: pd.Series) -> pd.Series:
        """
        Returns a 0-1 score for a series of numerical values.
        e.g. the length of the tags lists for competing rows
        """
        if series.empty:
            return pd.Series([0] * len(series), index=series.index)
        if series.max() == series.min():
            return pd.Series([1] * len(series), index=series.index)
        return (series - series.min()) / (series.max() - series.min())

    def compute_length(series: pd.Series) -> pd.Series:
        return series.fillna("").apply(len)

    def compute_group_scores(group: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate scores for each group of pseudo-duplicates.

        We use things like "most reviews" and "most tags" to determine the best candidate.
        """
        group = group.copy()  # Avoid SettingWithCopyWarning

        # Score numerical features.
        group["score_reviews"] = normalized_score(group["steam_total_reviews"].dropna()) * 2  # Most important feature
        group["score_recommendations"] = normalized_score(group["recommendations"].dropna())

        # Score text features based on string length.
        group["score_tags"] = normalized_score(compute_length(group["tags"].dropna()))
        group["score_genres"] = normalized_score(compute_length(group["genres"].dropna()))
        group["score_languages"] = normalized_score(compute_length(group["languages_supported"].dropna()))

        # Aggregate score.
        group["total_score"] = group["score_reviews"] + group["score_recommendations"] + group["score_tags"] + group["score_genres"] + group["score_languages"]
        return group

    def collapse_group_rows(group: pd.DataFrame) -> pd.Series:
        # Sort the rows by total_score (lowest to highest).
        group_sorted = group.sort_values("total_score")
        accum = group_sorted.iloc[0].copy()

        # Columns to ignore when updating: grouping keys and score columns.
        ignore_cols = {"name", "release_date", "total_score", "score_reviews", "score_recommendations", "score_tags", "score_genres", "score_languages"}
        cols_to_update = [col for col in group.columns if col not in ignore_cols]

        # Iterate over the remaining rows to update with non-null values.
        for _, row in group_sorted.iloc[1:].iterrows():
            for col in cols_to_update:
                val = row[col]
                if isinstance(val, (list, np.ndarray)):
                    if not pd.isnull(val).all():
                        accum[col] = val
                else:
                    if pd.notnull(val):
                        accum[col] = val
        return accum

    print("Collapsing pseudo-duplicate entries... This may take a while...")
    # Reset index to make sure it's treated as a other columns such that appid is kept
    df = df.reset_index()

    # Group by ["name", "release_date"] to identify pseudo-duplicates.
    group_cols = ["name", "release_date"]

    # This section has been reworked to be vectorized, though a little less readable imo
    # # Separate duplicate groups from unique entries to reduce processing.
    # print("Grouping by name and release date...")
    # groups = df.groupby(group_cols)
    # print("Identifying duplicates...")
    # duplicates = groups.filter(lambda x: len(x) > 1)
    # print("Identifying unique entries...")
    # singles = groups.filter(lambda x: len(x) == 1)
    # =======================================================
    print("Identifying duplicates...")
    df["group_count"] = df.groupby(group_cols)[group_cols[0]].transform("size")

    # Vectorized filtering for duplicates and singles
    duplicates = df[df["group_count"] > 1].copy()
    singles = df[df["group_count"] == 1].copy()
    # =======================================================

    # Score duplicates
    print("Scoring duplicates...")
    duplicates_scored = duplicates.groupby(group_cols).progress_apply(compute_group_scores).reset_index(drop=True)

    # Squash them by overwriting lower-score values with higher-score values progressively (which lets us keep non-null values the better candidates might not have)
    print("Collapsing duplicates...")
    collapsed_duplicates = duplicates_scored.groupby(group_cols).progress_apply(collapse_group_rows).reset_index(drop=True)

    # Combine the unique entries with now collapsed duplicates.
    final_df = pd.concat([singles, collapsed_duplicates]).reset_index(drop=True)

    # Drop temporary columns used for scoring.
    final_df = final_df.drop(columns=["score_reviews", "score_recommendations", "score_tags", "score_genres", "score_languages", "total_score"])

    # set 'appid' as the index again if needed:
    final_df = final_df.set_index("appid")

    # drop the helper column
    final_df = final_df.drop(columns=["group_count"])

    return final_df


def triangular_weighted_mean(df, value_column, selector_column, center, window_size):
    """
    Compute a triangular weighted mean for the values in `value_column` based on the `x_column`.
    I.e. samples near the edge of the window will have less weight than those near the center.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        value_column (str): Name of the column containing the values to average (e.g., steam_positive_review_ratio).
        x_column (str): Name of the column to base the weights on (e.g., price_original).
        center (float): The center value for the triangular window.
        window_size (float): The half-width of the triangular window.

    Returns:
        float: The triangular weighted mean of the values. Returns np.nan if no valid weights exist.
    """

    # Exlude rows with missing values in either column
    df = df.dropna(subset=[value_column, selector_column])

    # Keep only the rows for which the selector_column is within the window
    df = df[(df[selector_column] >= center - window_size) & (df[selector_column] <= center + window_size)]

    # Compute the triangular weights
    df["distance"] = np.abs(df[selector_column] - center)
    df["weights"] = 1 - df["distance"] / window_size

    # If no weights are positive, return np.nan to avoid division by zero
    if df["weights"].sum() == 0:
        return np.nan

    # Compute and return the weighted average
    return np.average(df[value_column], weights=df["weights"])


def binom_confidence_interval(successes, totals, confidence=0.95):
    """
    Calculate the confidence interval for a binomial proportion (i.e. rates that an event occurs or not, "binary" outcomes).

    Parameters:
        successes (np.ndarray or pd.Series): Array of success counts.
        totals (np.ndarray or pd.Series): Array of total counts.
        confidence (float): Confidence level (default 0.95).

    Returns:
        lower_bounds, upper_bounds (tuple of np.ndarray): Arrays of lower and upper confidence bounds.

    Example usage (scalar):
        df["ci_lower"], df["ci_upper"] = binom_confidence_interval(df["rpg_releases_count"], df["total_releases_count"])
    """
    # Convert to numpy arrays
    successes = np.asarray(successes, dtype=float)
    totals = np.asarray(totals, dtype=float)

    # Calculate the proportion of successes
    # Avoid division by zero; when total == 0, set proportion to 0
    p_hat = np.divide(successes, totals, out=np.zeros_like(successes), where=totals > 0)

    # Calculate the alpha level from the confidence level
    z = norm.ppf(1 - (1 - confidence) / 2)

    # Calculate the standard error
    se = np.sqrt(np.divide(p_hat * (1 - p_hat), totals, out=np.zeros_like(p_hat), where=totals > 0))

    # Calculate the confidence interval bounds
    lower_bounds = p_hat - z * se
    upper_bounds = p_hat + z * se

    # Keep bounds within [0, 1]
    lower_bounds = np.clip(lower_bounds, 0, 1)
    upper_bounds = np.clip(upper_bounds, 0, 1)

    return lower_bounds, upper_bounds


import numpy as np
from scipy.stats import norm


def median_confidence_intervals(df, value_col, group_col, confidence=0.95):
    """
    Compute confidence intervals for the median of a continuous dataset using standard error of the median.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        value_col (str): Column name containing the continuous values (e.g., estimated owners).
        group_col (str): Column name to group by (e.g., release_year).
        confidence (float): Confidence level (default is 0.95).

    Returns:
        pd.DataFrame: DataFrame with group_col, median values, and lower/upper confidence bounds.
    """
    grouped = df.groupby(group_col)[value_col].apply(list).reset_index()

    lower_bounds, upper_bounds, medians = [], [], []

    z = norm.ppf((1 + confidence) / 2)  # Z-score for two-tailed CI (e.g., 1.96 for 95% CI)

    for values in grouped[value_col]:
        values = np.array(values)
        n = len(values)

        if n == 0:
            medians.append(np.nan)
            lower_bounds.append(np.nan)
            upper_bounds.append(np.nan)
            continue

        median_value = np.median(values)
        sigma = np.std(values, ddof=1)  # Sample standard deviation
        sem = (1.2533 * sigma) / np.sqrt(n)  # Standard error of the median

        lower_bounds.append(median_value - z * sem)
        upper_bounds.append(median_value + z * sem)
        medians.append(median_value)

    result_df = grouped[[group_col]].copy()
    result_df["median"] = medians
    result_df["ci_lower"] = lower_bounds
    result_df["ci_upper"] = upper_bounds

    return result_df


def polynomial_regression_analysis(x, y, degree=1):
    """
    Calculate the trend line, r² value, individual p-values, global p-value and Cohen's f² effect size for a polynomial fit.

    Parameters:
        x (array-like): Independent variable data.
        y (array-like): Dependent variable data.
        degree (int): Degree of the polynomial fit.

    Returns:
        trend_line (np.poly1d): The polynomial function representing the trend line.
        r2 (float): The coefficient of determination (r²).
        p_values (list): List of p-values for each model parameter (starting with the intercept at index 0).
        global_p_value (float): Overall p-value for model significance.
        cohen_f2 (float): Effect size using Cohen's f².

    Example usage:
        trend_line, r2, p_values, global_p = get_trend_line_and_r2_p(x, y, degree=2)
        print(f"R²: {r2:.3f}, Global p-value: {global_p:.3f}, Individual p-values: {p_values}")
    """
    # Convert inputs to numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Drop nan values
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    # Fit a polynomial of the specified degree using np.polyfit (coefficients in descending order)
    coeffs = np.polyfit(x, y, degree)
    trend_line = np.poly1d(coeffs)

    # Compute predicted values and R²
    y_pred = trend_line(x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # Compute effect size using Cohen's f²: f² = r² / (1 - r²)
    cohen_f2 = r2 / (1 - r2) if r2 != 1 else np.inf

    # Construct polynomial features for regression EXCLUDING the constant (i.e. how each individual parameter affects the model)
    # For degree=1, this yields just the x values; for degree=2, [x, x^2], etc.
    X = np.column_stack([x**i for i in range(1, degree + 1)])
    X = sm.add_constant(X)  # Add intercept column

    # Fit the regression model using OLS
    model = sm.OLS(y, X).fit()

    # Extract individual p-values (first is for the intercept, then for each polynomial term)
    p_values = model.pvalues.tolist()
    # Global p-value from the F-test for _overall_ model significance (for degree>1)
    global_p_value = model.f_pvalue

    return trend_line, r2, p_values, global_p_value, cohen_f2


def adjust_metric_for_group_trends(df, metric_col, group_col) -> pd.Series:
    """
    Standardizes a metric for group-specific trends by removing the group-level mean and variance,
    and then re-projecting the values onto the overall distribution.

    Basically, each value is transformed into its z-score _within its group_ and then converted
    back to original units using the overall mean and standard deviation. This "adjusts" the
    metric as if all groups shared the same baseline (e.g. adjusting for inflation).

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        metric_col (str): The name of the metric column (e.g. 'price_original').
        group_col (str): The column name for grouping (e.g. 'release_year').

    Returns:
        pd.Series: A new Series with the metric adjusted for group trends.

    Example usage:
        df["price_adjusted"] = adjust_metric_for_group_trends(df, "price_original", "release_year")
    """
    # Compute group-specific mean and standard deviation
    group_mean = df.groupby(group_col)[metric_col].transform("mean")
    group_std = df.groupby(group_col)[metric_col].transform("std")

    # Calculate the z-score for each value within its group
    z_scores = (df[metric_col] - group_mean) / group_std

    # Compute overall mean and standard deviation (across all groups)
    overall_mean = df[metric_col].mean()
    overall_std = df[metric_col].std()

    # Convert the z-scores back to the original units using overall statistics
    adjusted_metric = z_scores * overall_std + overall_mean

    return adjusted_metric


def ttest_two_groups(df, numeric_col, cat_col):
    """
    Performs Welch's t-test to determine if the mean of a numeric column differs significantly
    between two groups defined by a categorical variable.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        numeric_col (str): The name of the numeric column to test, e.g. "steam_positive_review_ratio".
        cat_col (str): The name of the categorical column that must have exactly 2 unique values, e.g. "early_access=True/False".

    Returns:
        - 't_statistic': The computed t statistic.
        - 'p_value': The p-value from the test.

    Example usage:
        _, p_value = compare_two_groups_ttest(df, "steam_positive_review_ratio", "early_access")
        print(f"p-value: {p_value:.3f}")
    """

    # Confirm columns exist
    if numeric_col not in df.columns or cat_col not in df.columns:
        raise ValueError(f"Columns must exist in DataFrame. Check '{numeric_col}' and '{cat_col}'")

    # Filter out rows where either column has NaN values
    valid_data = df.dropna(subset=[numeric_col, cat_col])

    # Confirm exactly 2 groups
    group_labels = valid_data[cat_col].unique()
    if len(group_labels) != 2:
        raise ValueError(f"Column '{cat_col}' must have exactly 2 unique groups. Found: {len(group_labels)}")

    # Split the data into two groups
    group1 = valid_data[valid_data[cat_col] == group_labels[0]][numeric_col]
    group2 = valid_data[valid_data[cat_col] == group_labels[1]][numeric_col]

    # Debug
    print(f"Group 1 ({group_labels[0]}) size: {len(group1)}, mean: {group1.mean() if len(group1) > 0 else 'N/A'}")
    print(f"Group 2 ({group_labels[1]}) size: {len(group2)}, mean: {group2.mean() if len(group2) > 0 else 'N/A'}")

    if len(group1) < 2 or len(group2) < 2:
        print("ERROR: Each group must have at least 2 valid observations for t-test.")
        return None, None

    if not pd.api.types.is_numeric_dtype(valid_data[numeric_col]):
        print("ERROR: The numeric column must contain numerical data.")
        return None, None

    # More robust check for zero variance
    if group1.std() < 1e-10 or group2.std() < 1e-10:
        print("ERROR: One of the groups has zero variance. Cannot perform t-test.")
        return None, None

    # Perform Welch's t-test
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False, nan_policy="raise")

    # Calculate r_squared (effect size)
    df = len(group1) + len(group2) - 2
    r_squared = t_stat**2 / (t_stat**2 + df)

    # Calculate Cohen's d using pooled standard deviation
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = group1.mean(), group2.mean()
    s1, s2 = group1.std(ddof=1), group2.std(ddof=1)
    pooled_std = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
    pooled_std = pooled_std**0.5
    cohen_d = (mean1 - mean2) / pooled_std

    return t_stat, p_value, r_squared, cohen_d


from scipy import stats


def anova_categorical(df, numeric_col, cat_col):
    """
    Performs a one-way ANOVA to determine if the mean of a numeric column differs significantly
    across the groups defined by a categorical variable with more than two levels.
    Note: Drops NaN values, so handle them manually if they have to be included.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        numeric_col (str): The name of the numeric column to test (e.g., "steam_positive_review_ratio").
        cat_col (str): The name of the categorical column (e.g., "controller_support") which contains multiple groups (e.g., "none", "partial", "full").

    Returns:
        tuple: A tuple containing the F-statistic, p-value, and eta-squared (η²) proportion of variance explained.

    Example usage:
        F_stat, p_value, eta_sq = anova_categorical(df, "steam_positive_review_ratio", "controller_support")
        print(f"F-statistic: {F_stat:.3f}, p-value: {p_value:.3f}, eta²: {eta_sq:.4f}")
    """
    # Drop NaNs
    df = df[[numeric_col, cat_col]].dropna()

    # Group the data by the categorical column and extract the numeric data for each group
    groups = [group[numeric_col] for _, group in df.groupby(cat_col)]

    if len(groups) < 2:
        raise ValueError(f"Column '{cat_col}' must have at least 2 groups. Found only {len(groups)} group(s).")

    # Perform one-way ANOVA
    F_stat, p_value = stats.f_oneway(*groups)

    # Compute Eta-Squared (η²) for effect size
    model = ols(f"{numeric_col} ~ C({cat_col})", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    eta_sq = anova_table["sum_sq"][0] / anova_table["sum_sq"].sum()

    return F_stat, p_value, eta_sq


def find_closest_string_in_list(input: str, list: list, number_of_matches: int = 1) -> list:
    """
    Find the closest string(s) in a list to the input string using "thefuzz" library.
    This function uses the Levenshtein distance to measure string similarity.

    Parameters:
        input (str): The input string to compare against.
        list (list): The list of strings to search through.
        number_of_matches (int): The number of closest matches to return.

    Returns:
        list: A list of the closest matching strings from the input list.
    """
    from thefuzz import process

    # Get the closest matches
    closest_matches = process.extract(input, list, limit=number_of_matches)

    # Extract only the matched strings from the tuples
    return [match[0] for match in closest_matches]


def step_round(value):
    """
    Returns numbers rounded to "visually pleasing" values based on their scale.
    E.g. 29 -> 30, 479 -> 500, 48713 -> 50000...
    """
    # Everything <= a certain value is rounded to the nearest "value" for that step
    steps = [
        (100, 10),
        (500, 50),
        (1000, 100),
        (2000, 250),
        (5000, 500),
        (10000, 1000),
        (np.inf, 5000),
    ]
    for threshold, step in steps:
        if value < threshold:
            return int(round(value / step) * step)


def display_streamlit_custom_navigation():
    """
    Displays a custom navigation menu in the Streamlit sidebar
    Must be called in each page, after the st.set_page_config() call.
    """
    with st.sidebar.expander("Navigation", expanded=True):

        # Inject custom CSS
        st.markdown(
            """
            <style>
            [data-testid="stSidebarHeader"] {
                display: none;
            }
            
            hr {
                margin-top: 0.5rem !important;
                margin-bottom: -0.2rem !important;
            }
            
            .stVerticalBlock {
                gap: 0.7rem; /* Reduce overall gap, to have the separators not take up so much space, we replace it with per-item margins */
            }
            
            .stVerticalBlock .stHeading {
                margin: 0.25rem 0;
            }
            

            """,
            unsafe_allow_html=True,
        )

        # st.header("Navigation")
        st.page_link("streamlit_main.py", label="Homepage")

        st.markdown("---")
        st.subheader("Prospective Analysis")
        st.page_link("pages/project_evaluator.py", label="Project Evaluation Tool")

        st.markdown("---")
        st.subheader("Retrospective Analysis")
        st.page_link("pages/general_market_trends.py", label="Market Overview & General Trends")
        st.page_link("pages/free_vs_paid.py", label="Free vs Paid Model Analysis")
        st.page_link("pages/releases_by_price.py", label="Game Price Analysis")
        st.page_link("pages/tag_trends.py", label="Keyword Analysis")
        st.page_link("pages/success_factors.py", label="Success Factor Analysis")
