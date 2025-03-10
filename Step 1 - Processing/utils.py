import numpy as np
import pandas as pd


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

    # Group by ["name", "release_date"] to identify pseudo-duplicates.
    group_cols = ["name", "release_date"]

    # Separate duplicate groups from unique entries to reduce processing.
    duplicates = df.groupby(group_cols).filter(lambda x: len(x) > 1)
    singles = df.groupby(group_cols).filter(lambda x: len(x) == 1)

    # Score duplicates
    duplicates_scored = duplicates.groupby(group_cols).apply(compute_group_scores).reset_index(drop=True)

    # Squash them by overwriting lower-score values with higher-score values progressively (which lets us keep non-null values the better candidates might not have)
    collapsed_duplicates = duplicates_scored.groupby(group_cols).apply(collapse_group_rows).reset_index(drop=True)

    # Combine the unique entries with now collapsed duplicates.
    final_df = pd.concat([singles, collapsed_duplicates]).reset_index(drop=True)

    # Drop temporary columns used for scoring.
    final_df = final_df.drop(columns=["score_reviews", "score_recommendations", "score_tags", "score_genres", "score_languages", "total_score"])

    return final_df
