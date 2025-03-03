"""
This module provides a way to merge dataframes by using various strategies, such as
defining which set takes precendence over the other, or combining lists of values.
"""

import pandas as pd
import numpy as np


def is_empty(value):
    """
    Defines what is considered an "empty" value for the purpose of merging.
    Currntly, empty values are NaN, empty strings, 0s, and empty lists.
    """
    # Handle empty arrays as empty, non-empty arrays as non-empty
    if isinstance(value, list):
        return len(value) == 0
    # Handle other types (e.g., strings, numbers) as empty if they are NaN, empty strings, or 0
    return pd.isna(value) or value == "" or value == 0


################################
# Merge strategies
################################


def overwrite_master_if_value_null(master_col, slave_col):
    """Overwrites master value with slave value only if master is empty."""
    return master_col.mask(master_col.apply(is_empty), slave_col)


def combine_list_unique_values(master_col, slave_col):
    """Merges array values as sets to avoid duplicates (assumes lists)."""
    return master_col.combine(slave_col, lambda a, b: list(set(a) | set(b)) if isinstance(a, list) and isinstance(b, list) else a)


def merge_dataframes(master_df, slave_df, column_mappings):
    """
    Merges two DataFrames based on a set of column-specific merge strategies.
    The column_mappings dictionary should contain column names as keys and merge functions as values.

    Note: this module provides general merge functions already.
    """

    # Step 1: Perform an outer merge on "appid" (keep all rows from both DataFrames, overwrite nothing since we use suffixes)
    merged_df = pd.merge(master_df, slave_df, on="appid", how="outer", suffixes=("_master", "_slave"))

    # Step 2: Setup to handle all columns and differentiate between new, mapped, and unmapped columns
    all_columns = set(merged_df.columns)
    specifically_mapped_columns = set(column_mappings.keys())

    new_columns = set(slave_df.columns) - set(master_df.columns)  # columns that were in slave_df but not in master_df
    unmapped_columns = all_columns - specifically_mapped_columns - {"appid"}

    # Define default behavior (e.g., overwrite_master_if_value_null for all unmapped columns)
    default_merge_strategy = overwrite_master_if_value_null

    # Process explicitly mapped columns
    for col, merge_func in column_mappings.items():
        merged_df[col] = merge_func(merged_df[f"{col}_master"], merged_df[f"{col}_slave"])

    # Handle all unmapped columns with the default strategy
    for col in unmapped_columns:
        if col.endswith("_master") or col.endswith("_slave"):  # Meaning only for columns which were "conflicting"
            base_col = col.rsplit("_", 1)[0]  # Get the original column name (without _master or _slave)
            if base_col not in merged_df.columns:  # Skip columns that were already handled by a specific merge strategy
                merged_df[base_col] = default_merge_strategy(
                    merged_df.get(f"{base_col}_master", pd.Series(dtype="object")),
                    merged_df.get(f"{base_col}_slave", pd.Series(dtype="object")),
                )

    # Handle completely new columns that only exist in slave_df by simply copying them over
    for col in new_columns:
        merged_df[col] = slave_df[col]

    # Step 4: Drop temporary _master and _slave columns that were introduced in the process
    merged_df = merged_df[[col for col in merged_df.columns if not col.endswith(("_master", "_slave"))]]

    return merged_df
