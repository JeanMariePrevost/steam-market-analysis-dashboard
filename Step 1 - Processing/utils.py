import numpy as np


def remove_outliers_iqr(df, value_col, threshold=1.5, cap_instead_of_drop=False):
    """
    Removes or caps outliers from a DataFrame based on the IQR method.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        value_col (str): The numerical column to process.
        threshold (float): The IQR multiplier for detecting outliers, i.e. how many times the IQR to go above Q3 or below Q1 to be considered an outlier. (default: 1.5).
        cap_instead_of_drop (bool): If True, caps outliers instead of removing them.

    Returns:
        pd.DataFrame: Processed DataFrame with outliers removed or capped.
    """
    q1 = df[value_col].quantile(0.25)
    q3 = df[value_col].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    if cap_instead_of_drop:
        df[value_col] = np.clip(df[value_col], lower_bound, upper_bound)
        return df
    else:
        return df[(df[value_col] >= lower_bound) & (df[value_col] <= upper_bound)]
