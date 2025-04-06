"""
This script takes the merged and preprocessed "human readable" data and preprocesses it into a format that is more suitable for machine learning.
E.g. turning genres and tags into one-hot encoded columns and so on.

NOTE: This processing does _not_ normalize all numerical fields, which is left to the model-specific pipelines.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Allow importing modules from the parent directory of this script
current_file = Path(__file__).resolve()
parent_dir = current_file.parent.parent
sys.path.insert(0, str(parent_dir))

from utils import load_main_dataset


def median_fillna(col):
    median_value = col.median()
    return col.fillna(median_value)


def first_of_list(col, default="none"):
    """
    Extracts the first element from a list or NumPy array within a column.
    If the input is not a list/NumPy array or is empty, returns the default value.
    """

    def inner(value):
        if not isinstance(value, (list, np.ndarray)) or len(value) == 0:
            return default
        return value[0]

    return col.map(inner)


def zero_fillna(col):
    return col.fillna(0)


def false_fillna(col):
    return col.fillna(False)


def true_fillna(col):
    return col.fillna(True)


def value_fillna(col, value):
    return col.fillna(value)


def empty_list_fillna(col):
    # return col.fillna([]) # Doesn't work, can't pass a list to fillna
    return col.apply(lambda x: [] if not isinstance(x, (list, np.ndarray)) else x)  # Workaround


def convert_difficulty_rating(col):
    """Converts the difficulty rating to a numerical value."""
    # Ratings are
    # -1 -> NA / mising / unknown
    # 0 -> Simple
    # 1 -> Simple-Easy
    # 2 -> easy
    # 3 -> Easy-Just Right
    # 4 -> Just Right
    # 5 -> Just Right-Tough
    # 6 -> Tough
    # 7 -> Tough-Unforgiving
    # 8 -> Unforgiving

    mapping = {
        "simple": 0,
        "simple-easy": 1,
        "easy": 2,
        "easy-just right": 3,
        "just right": 4,
        "just right-tough": 5,
        "tough": 6,
        "tough-unforgiving": 7,
        "unforgiving": 8,
    }

    def inner(value):
        if isinstance(value, str):
            lowercased = value.lower()
            if lowercased in mapping:
                return mapping[lowercased]
        return -1

    return col.map(inner)


def convert_controller_support(col):
    """Converts the controller_support string to a numerical value."""

    def inner(value):
        if value == "full":
            return 1.0
        if value == "partial":
            return 0.5
        if value == "none":
            return 0.0
        return -1

    return col.map(inner)


# Mappings for the "NA-intolerant" set
# NOTE: NAs left with "None" will later have their entire rows dropped
fill_na_mappings = {
    "appid": None,
    "achievements_count": median_fillna,
    "average_non_steam_review_score": median_fillna,
    "average_time_to_beat": median_fillna,
    "categories": empty_list_fillna,
    "controller_support": convert_controller_support,
    "developers": None,
    "dlc_count": median_fillna,
    "early_access": false_fillna,
    "estimated_gross_revenue_boxleiter": zero_fillna,
    "estimated_ltarpu": zero_fillna,
    "estimated_owners_boxleiter": zero_fillna,
    "gamefaqs_difficulty_rating": None,
    "genres": empty_list_fillna,
    "has_demos": false_fillna,
    "has_drm": false_fillna,
    "is_released": true_fillna,
    "languages_supported": empty_list_fillna,
    "languages_with_full_audio": empty_list_fillna,
    "monetization_model": lambda x: value_fillna(x, "unknown"),
    "name": None,
    "peak_ccu": zero_fillna,
    "playtime_avg": zero_fillna,
    "playtime_avg_2_weeks": zero_fillna,
    "playtime_median": zero_fillna,
    "playtime_median_2_weeks": zero_fillna,
    "price_latest": zero_fillna,
    "price_original": None,
    "publishers": None,
    "recommendations": zero_fillna,
    "release_year": zero_fillna,
    "release_month": zero_fillna,
    "release_day_of_month": zero_fillna,
    "required_age": zero_fillna,
    "runs_on_linux": false_fillna,
    "runs_on_mac": false_fillna,
    "runs_on_steam_deck": true_fillna,
    "runs_on_windows": true_fillna,
    "steam_negative_reviews": zero_fillna,
    "steam_positive_review_ratio": zero_fillna,
    "steam_positive_reviews": zero_fillna,
    "steam_store_movie_count": zero_fillna,
    "steam_store_screenshot_count": zero_fillna,
    "steam_total_reviews": zero_fillna,
    "tags": empty_list_fillna,
    "vr_only": false_fillna,
    "vr_supported": false_fillna,
}


# Mappings for the "NA-tolerant" set
allow_na_mappings = {
    "appid": None,
    "achievements_count": None,
    "average_non_steam_review_score": None,
    "average_time_to_beat": None,
    "categories": empty_list_fillna,
    "controller_support": convert_controller_support,
    "developers": first_of_list,
    "dlc_count": None,
    "early_access": None,
    "estimated_gross_revenue_boxleiter": None,
    "estimated_ltarpu": None,
    "estimated_owners_boxleiter": None,
    "gamefaqs_difficulty_rating": convert_difficulty_rating,
    "genres": empty_list_fillna,
    "has_demos": None,
    "has_drm": None,
    "is_released": None,
    "languages_supported": empty_list_fillna,
    "languages_with_full_audio": empty_list_fillna,
    "monetization_model": None,
    "name": None,
    "peak_ccu": None,
    "playtime_avg": None,
    "playtime_avg_2_weeks": None,
    "playtime_median": None,
    "playtime_median_2_weeks": None,
    "price_latest": None,
    "price_original": None,
    "publishers": first_of_list,
    "recommendations": None,
    "release_year": None,
    "release_month": None,
    "release_day_of_month": None,
    "required_age": None,
    "runs_on_linux": None,
    "runs_on_mac": None,
    "runs_on_steam_deck": None,
    "runs_on_windows": None,
    "steam_negative_reviews": None,
    "steam_positive_review_ratio": None,
    "steam_positive_reviews": None,
    "steam_store_movie_count": None,
    "steam_store_screenshot_count": None,
    "steam_total_reviews": None,
    "tags": empty_list_fillna,
    "vr_only": None,
    "vr_supported": None,
}

###################################################################################
# Work starts here
###################################################################################


tqdm.pandas()
df = load_main_dataset()


###################################################################################
# Date processing
###################################################################################
# Convert date columns to separete year, month, day columns
# We already have the release_year column, so we can skip that one
df["release_date"] = pd.to_datetime(df["release_date"])
df["release_month"] = df["release_date"].dt.month
df["release_day_of_month"] = df["release_date"].dt.day
df.drop(columns=["release_date"], inplace=True)


###################################################################################
# Custom mappings of "NA-tolerant" set
###################################################################################
df_allow_na = df.copy()

# Apply mappings
for column_from_df in df.columns:
    print(f"Processing column {column_from_df}...")

    # Apply the NA-tolerant mappings
    if column_from_df in allow_na_mappings:
        mapping = allow_na_mappings[column_from_df]
        if mapping is not None:
            df_allow_na[column_from_df] = mapping(df[column_from_df])
    else:
        raise ValueError(f"No allow_na_mappings for {column_from_df}")

###################################################################################
# Label encoding (for categorical columns)
###################################################################################
# Convert categorical columns to numerical values (including missing values) using pandas' factorize method
columns_to_label_encode = [
    "developers",
    "early_access",
    "has_demos",
    "has_drm",
    "monetization_model",
    "publishers",
    "runs_on_linux",
    "runs_on_mac",
    "runs_on_steam_deck",
    "runs_on_windows",
    "vr_only",
    "vr_supported",
]

# Debug, print all unique values of the "developers" column
print(len(df_allow_na["publishers"].unique()))
print(df_allow_na["publishers"].unique())

for column in columns_to_label_encode:
    df_allow_na[column] = pd.factorize(df_allow_na[column], use_na_sentinel=True)[0]


# Debug, print all unique values of the "developers" column
print(len(df_allow_na["publishers"].unique()))
print(df_allow_na["publishers"].unique())


###################################################################################
# Custom mappings of "No-NA" set
###################################################################################
df_no_na = df_allow_na.copy()

# Apply mappings
for column_from_df in df_allow_na.columns:
    print(f"Processing column {column_from_df}...")

    # Apply the NA-intolerant mappings
    if column_from_df in fill_na_mappings:
        mapping = fill_na_mappings[column_from_df]
        if mapping is not None:
            df_no_na[column_from_df] = mapping(df_allow_na[column_from_df])
    else:
        raise ValueError(f"No fill_na_mappings for {column_from_df}")


# Debug, print all unique values of the "developers" column
print(len(df_allow_na["publishers"].unique()))
print(df_allow_na["publishers"].unique())

print(len(df_no_na["publishers"].unique()))
print(df_no_na["publishers"].unique())


###################################################################################
# Introducing "meta" rows
###################################################################################
# "Count" columns
df_no_na["categories_count"] = df_no_na["categories"].apply(len)
df_no_na["genres_count"] = df_no_na["genres"].apply(len)
df_no_na["languages_supported_count"] = df_no_na["languages_supported"].apply(len)
df_no_na["languages_with_full_audio_count"] = df_no_na["languages_with_full_audio"].apply(len)
df_no_na["tags_count"] = df_no_na["tags"].apply(len)

df_allow_na["categories_count"] = df_allow_na["categories"].apply(len)
df_allow_na["genres_count"] = df_allow_na["genres"].apply(len)
df_allow_na["languages_supported_count"] = df_allow_na["languages_supported"].apply(len)
df_allow_na["languages_with_full_audio_count"] = df_allow_na["languages_with_full_audio"].apply(len)
df_allow_na["tags_count"] = df_allow_na["tags"].apply(len)


###################################################################################
# Dropping rows
###################################################################################
# Drop _all_ rows that have _any_ NA values left in the no-NA set (if any)
print(f"Df contains {df_no_na.isna().sum().sum()} NaNs")

print(f"Number of rows before dropping NA rows: {len(df_no_na)}")

df_no_na = df_no_na.dropna()

print(f"Number of rows after dropping NA rows: {len(df_no_na)}")

# Sanity check
print("Df contains NaNs: ", df_no_na.isna().sum().sum() > 0)

# If any, show where the first is
if df_no_na.isna().sum().sum() > 0:
    print(df_no_na.isna().sum().idxmax())
    raise ValueError("There are still NaNs in the no-NA set")


# Debug, print all unique values of the "developers" column
print(len(df_allow_na["publishers"].unique()))
print(df_allow_na["publishers"].unique())

###################################################################################
# One-Hot Encoding
###################################################################################
# Here we explode every "list of strings" column into a set of one-hot encoded columns
# This is done for both the NA-intolerant and NA-tolerant sets

prefix_for_sorting = "~"  # Prefix used to sort the one-hot encoded columns to the _end_ of the DataFrame for better readability

# Define the columns to explode
columns_to_explode = {
    "categories": prefix_for_sorting + "category_",
    "genres": prefix_for_sorting + "genre_",
    "languages_supported": prefix_for_sorting + "lang_",
    "languages_with_full_audio": prefix_for_sorting + "lang_audio_",
    "tags": prefix_for_sorting + "tag_",
}


def explode_column_unique_values(df, column_name):
    """Iterates across all the values found in a column to extract the unique values found in the lists"""

    unique_values = set()
    for row in df[column_name]:
        # Raise error if value is not a list
        if not isinstance(row, (list, np.ndarray)):
            raise ValueError(f"explode_column_unique_values: Expected list, got {type(row)}")

        unique_values.update(row)
    return unique_values


def explode_list_to_one_hot(reference_df, column_name, prefix_for_exploded_columns):
    """
    Extracts unique values from a column of lists, creates them as new columns with a prefix, and fills them with 1 if the value is present in the list for each row.
    """
    # Get all unique values from the column (each cell is expected to be a list)
    unique_values = explode_column_unique_values(reference_df, column_name)

    # Create an empty DataFrame with the same index and with columns for each unique value, with a prefix
    one_hot_df = pd.DataFrame(0, index=reference_df.index, columns=[f"{prefix_for_exploded_columns}{val}" for val in unique_values])

    # # Populate the one-hot DataFrame: for each row, mark 1 if the unique value is present in the list
    # tqdm is used to show a progress bar
    for idx, row in tqdm(reference_df[column_name].items(), total=len(reference_df)):
        for val in row:
            one_hot_df.loc[idx, f"{prefix_for_exploded_columns}{val}"] = 1

    return one_hot_df


print("Exploding columns to one-hot encoding... This may take a while...")
for column, prefix in columns_to_explode.items():
    print(f"Exploding column [{column}] with prefix [{prefix}]...")

    # # Explode the "list columns" into one-hot encoded columns
    exploded_column_df = explode_list_to_one_hot(df, column, prefix)
    df_no_na = pd.concat([df_no_na, exploded_column_df], axis=1)
    df_allow_na = pd.concat([df_allow_na, exploded_column_df], axis=1)
    df_no_na.drop(columns=[column], inplace=True)
    df_allow_na.drop(columns=[column], inplace=True)


print("One-hot encoding complete ✅")

# Drop _all_ rows that have _any_ NA values left in the first set (if we introduced any, something went wrong...)
df_no_na = df_no_na.dropna()

# Sanity check
print("Df contains NaNs: ", df_no_na.isna().sum().sum() > 0)

# If any, show where the first is
if df_no_na.isna().sum().sum() > 0:
    print(df_no_na.isna().sum().idxmax())
    raise ValueError("There are still NaNs in the no-NA set")


###################################################################################
# Final prep and Saving
###################################################################################

# Sort columns alphabetically for better readability
df_no_na = df_no_na.reindex(sorted(df_no_na.columns), axis=1)
df_allow_na = df_allow_na.reindex(sorted(df_allow_na.columns), axis=1)


# Debug, print all unique values of the "developers" column
print(len(df_allow_na["publishers"].unique()))
print(df_allow_na["publishers"].unique())
print(len(df_no_na["publishers"].unique()))
print(df_no_na["publishers"].unique())

# Define and create output directory if it doesn't exist
root_dir = current_file.parent.parent
output_dir = root_dir / "output_feature_engineered"
os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

# Save the processed data
output_filename = "processed_no_na"
df_no_na.to_csv(f"{output_dir}/{output_filename}.csv", index=True)
df_no_na.to_parquet(f"{output_dir}/{output_filename}.parquet")
output_filename = "processed_allow_na"
df_allow_na.to_csv(f"{output_dir}/{output_filename}.csv", index=True)
df_allow_na.to_parquet(f"{output_dir}/{output_filename}.parquet")

print("Feature engineering complete ✅")

# Open the output directory in Windows Explorer
os.startfile(output_dir)  # Open the output directory in Windows Explorer

# subprocess.Popen(f'explorer /select,"{output_dir}\\{output_filename}.parquet"')  # Open the output directory in Windows Explorer
