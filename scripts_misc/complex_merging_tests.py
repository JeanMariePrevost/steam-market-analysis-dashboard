import pandas as pd
import numpy as np

# Sample Master (Base) DataFrame
master_df = pd.DataFrame(
    {
        "appid": [1, 2, 3],
        "name": ["Game A", "Game B", "Game C"],
        "release_date": [np.nan, "2022-05-01", ""],  # Some missing values
        "supported_languages": [["English", "French"], [], ["Spanish"]],  # List columns
    }
)

# Sample Slave (Incoming) DataFrame
slave_df = pd.DataFrame(
    {
        "appid": [1, 2, 3, 4],
        "name": ["Game A", "Game B", "Game C", "Game D"],
        "release_date": ["2021-11-15", "", "2023-08-11", "2023-08-10"],
        "supported_languages": [["German", "English"], ["Spanish"], ["French", "Spanish"], []],
        "new_column": [5, 6, 7, 8],
        "2nd_new_column": ["", "", "", "here"],
    }
)


# Define "empty" values
def is_empty(value):
    # Handle empty arrays as empty, non-empty arrays as non-empty
    if isinstance(value, list):
        return len(value) == 0
    # Handle other types (e.g., strings, numbers) as empty if they are NaN, empty strings, or 0
    return pd.isna(value) or value == "" or value == 0


# Merge Strategy Functions
def overwrite_master_if_value_null(master_col, slave_col):
    """Overwrites master value with slave value only if master is empty."""
    return master_col.mask(master_col.apply(is_empty), slave_col)


def combine_list_unique_values(master_col, slave_col):
    """Merges array values as sets to avoid duplicates (assumes lists)."""
    return master_col.combine(slave_col, lambda a, b: list(set(a) | set(b)) if isinstance(a, list) and isinstance(b, list) else a)


# Step 1: Perform an outer merge on "appid" (keep all rows from both DataFrames, overwrite nothing since we use suffixes)
merged_df = pd.merge(master_df, slave_df, on="appid", how="outer", suffixes=("_master", "_slave"))

# Step 2: Define column-specific merge strategies, defaulting to overwrite_if_null
column_mappings = {
    "name": overwrite_master_if_value_null,
    # "release_date": overwrite_master_if_value_null,
    "supported_languages": combine_list_unique_values,
}

# Step 3: Setup to handle all columns and differentiate between new, mapped, and unmapped columns
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

# Handle completely new columns that only exist in slave_df (just copy them over)
for col in new_columns:
    merged_df[col] = slave_df[col]

# Step 4: Drop temporary _master and _slave columns that were introduced
merged_df = merged_df[[col for col in merged_df.columns if not col.endswith(("_master", "_slave"))]]

# Save test results as a CSV file
output_file = "complex_merge_test_results.csv"
merged_df.to_csv(output_file, index=False)
print(f"Results saved to '{output_file}'.")
