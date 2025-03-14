from scipy.stats import norm
import numpy as np
import pandas as pd
import os
from tqdm import tqdm


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
