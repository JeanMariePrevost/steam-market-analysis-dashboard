import os

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm


def load_main_dataset() -> pd.DataFrame:
    """
    Load the preprocessed data from the parquet file.
    Returns None if an error occurs.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script itself
    data_path = f"{script_dir}\\preprocessed_output\\combined_df_preprocessed_dense.parquet"

    return load_dataset(data_path)


def load_feature_engineered_dataset_with_na() -> pd.DataFrame:
    """
    Load the NA-tolerant feature-engineered data from the parquet file.
    This version retains rows with missing values and does not impute anything.
    Returns None if an error occurs.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script itself
    data_path = f"{script_dir}\\feature_engineered_output\\processed_allow_na.parquet"

    return load_dataset(data_path)


def load_feature_engineered_dataset_no_na() -> pd.DataFrame:
    """
    Load the "no NA" feature-engineered data from the parquet file.
    This version imputes missing values in various ways.
    Returns None if an error occurs.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script itself
    data_path = f"{script_dir}\\feature_engineered_output\\processed_no_na.parquet"

    df = load_dataset(data_path)
    if df is not None:
        if df.isnull().sum().sum() > 0:
            raise ValueError("The dataset contains missing values. Double-check the preprocessing steps leading to processed_no_na.parquet.")

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


def get_trend_line_and_r2(x, y, degree=1):
    """
    Calculate the trend line and r² value for the provided data.

    Parameters:
        x (array-like): Independent variable data.
        y (array-like): Dependent variable data.
        degree (int): Degree of the polynomial fit (default is 1 for linear).

    Returns:
        trend_line (np.poly1d): The polynomial function representing the trend line.
        r2 (float): The coefficient of determination (r²) for the fit.

    Example usage:
        x = df_sorted["achievements_count"]
        y = df_sorted["steam_positive_review_ratio"]
        trend_line, r2 = get_trend_line_and_r2(x, y)
        print("r²:", r2)
        ax.plot(x, trend_line(x), label="Trend Line", color="red", linestyle="--")
    """
    # Fit a polynomial of the specified degree
    coeffs = np.polyfit(x, y, degree)
    trend_line = np.poly1d(coeffs)

    # Predicted y-values from the trend line
    y_pred = trend_line(x)

    # Calculate the sum of squared residuals and total sum of squares
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    # Compute the coefficient of determination (r²)
    r2 = 1 - (ss_res / ss_tot)

    return trend_line, r2


def normalize_metric_across_groups(df, metric_col, group_col="release_year", method="zscore"):
    """
    Normalizes a metric column based on grouping.
    E.g. get the infliation-adjusted price for each year as opposed to the raw price.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        metric_col (str): The column name of the metric to normalize.
        group_col (str): The column to group by (default: 'release_year').
        method (str): Normalization method. Options:
            - 'zscore': Standardize using z-score (meaning you get the number of standard deviations from the group mean, and lose the original units).
            - 'diff': Compute the difference from the group mean (adjusts for shifts, but not variance).

    Returns:
        pd.Series: The normalized metric as a new pandas Series.

    Example usage:
        df_filtered["review_zscore"] = normalize_metric(df_filtered, "steam_positive_review_ratio", method="zscore")
        df_filtered["review_diff"] = normalize_metric(df_filtered, "steam_positive_review_ratio", method="diff")
    """
    # Calculate group statistics
    group_mean = df.groupby(group_col)[metric_col].transform("mean")

    if method == "zscore":
        group_std = df.groupby(group_col)[metric_col].transform("std")
        normalized = (df[metric_col] - group_mean) / group_std
    elif method == "diff":
        normalized = df[metric_col] - group_mean
    else:
        raise ValueError("Method must be either 'zscore' or 'diff'.")

    return normalized


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
    groups = df[cat_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError(f"Column '{cat_col}' must have exactly 2 unique groups. Found: {groups}")

    group1 = df[df[cat_col] == groups[0]][numeric_col]
    group2 = df[df[cat_col] == groups[1]][numeric_col]

    # Perform Welch's t-test
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)

    return t_stat, p_value
