"""
This script contains all the logic to sequentially process, trim and merge the datasets one by one.
The process is not particularly clean or optimized, but it should be a one-time operation, so anything more would be over-engineering.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow importing modules from the parent directory of this script
current_file = Path(__file__).resolve()
parent_dir = current_file.parent.parent
sys.path.insert(0, str(parent_dir))

from scripts_data_processing.complex_merge import combine_list_unique_values, merge_dataframes_with_mappings

# Selected datasets paths
current_dir = Path(__file__).resolve().parent
datasets_directory = current_dir.parent / "source_datasets"
# Great general metadata foundation
set1_path = f"{datasets_directory}\\artermiloff_steam-games-dataset\\games_may2024_cleaned.csv"
# grab price_final and price_original
set2_path = f"{datasets_directory}\\antonkozyriev_game-recommendations-on-steam\\games.csv"
# Has interesting features like metacritic score, achievement, recommandations, genres, tags (combine across sets on these?)
set3_path = f"{datasets_directory}\\fronkongames_steam-games-dataset\\games.csv"
# Unsure. Includes game descriptions an such (NLP?) DLCs, genres, and especially "estimated_owners"
set4_path = f"{datasets_directory}\\fronkongames_steam-games-dataset\\games.json"
# playtime stats and owners
set5_path = f"{datasets_directory}\\nikdavis_steam-store-games_steamspy\\steam.csv"
# Similar to base set, with 2025 data, but we have to ensure ALL its columns can fully merge in
set6_path = f"{datasets_directory}\\srgiomanhes_steam-games-dataset-2025\\steam_games.csv"
# Some VERY interesting "meta" fields from gamefaqs, stsp, hltb, metacritic and igdb IF complete enough
set7_path = f"{datasets_directory}\\sujaykapadnis_games-on-steam\\steamdb.json"
# For the vr tags, whch I THINK other sets don't offer
set8_path = f"{datasets_directory}\\vicentearce_steamdata\\game_data.csv"
# Coop, online, workshop support, languages, precise rating, playtime, peak player, owners, has dlc, has demos....
set9_path = f"{datasets_directory}\\vicentearce_steamdata\\final\\steam_games.csv"
# Owners, CCU, detailled tags
set10_path = f"{datasets_directory}\\souyama_steam-dataset\\steam_dataset\\steamspy\detailed\\steam_spy_detailed.json"
# F2P, required age, what is price_overview?, whether games have demos...
set11_path = f"{datasets_directory}\\souyama_steam-dataset\\steam_dataset\appinfo\store_data\\steam_store_data.json"
# Acually good time to complete, but have to merge on name...
set12_path = f"{datasets_directory}\\kasumil5x_howlongtobeat-games-completion-times\\games.csv"

# Define and create output directory if it doesn't exist
output_dir = current_dir.parent / "output_merging"
os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

combined_df = pd.DataFrame()  # The "container" for the final merged DataFrame


#####################################
# Helper functions
#####################################
def apply_mappings(df, mappings, keep_only_mapped_columns=False) -> pd.DataFrame:
    """
    Renames and transforms columns of a DataFrame based on a given mappings list.
    Does not copy columns with no mapping if keep_only_mapped_columns is True.
    """
    print(f"Starting columns: {df.columns}")

    # Validate source names exist in the DataFrame
    for source_name, _, _ in mappings:
        if source_name not in df.columns:
            raise ValueError(f"Source column '{source_name}' not found in the DataFrame.")

    if keep_only_mapped_columns:
        print("Keeping only mapped columns")
        mapped_columns = {src for src, _, _ in mappings}
        df = df[df.columns.intersection(mapped_columns)]  # Keep only the columns that have a mapping by keeping only the intersection
        print(f"Columns after filtering: {df.columns}")

    rename_dict = {source_name: target_name for source_name, target_name, _ in mappings}  # rename mappings-only dict for DataFrame.rename
    df = df.rename(columns=rename_dict)  # Apply the renames

    print(f"Columns after renaming: {df.columns}")

    # Applying transofrmations if specified
    for _, target_name, transform_function in mappings:  # We use only the target name here since we already renamed the columns
        if transform_function is not None and target_name in df.columns:
            print(f"Applying transformation to column {target_name}")
            df[target_name] = df[target_name].apply(transform_function)
            # df[target_name] = df[target_name].map(transform_function, na_action=None)
    return df


#####################################
# In-place tranform functions
#####################################
def split_string_to_list(value, separator=","):
    """Converts a string into a list by splitting on the given separator."""
    return value.split(separator) if isinstance(value, str) else []


def count_items_in_string(value, separator=","):
    """Counts the number of items in a string by splitting on the given separator."""
    return len(value.split(separator)) if isinstance(value, str) else 0


def average_range_string(value, separator=" - "):
    """Converts a range string (e.g., '100 - 200') into its average."""
    if isinstance(value, str):
        if value == "0.0" or value == "0":
            return 0
        try:
            numbers = list(map(int, value.split(separator)))
            return sum(numbers) / len(numbers) if numbers else 0
        except ValueError:
            print(f'Error converting range string to average: ({value}). (Is the separator "{separator}" correct?)')
            return 0
    return 0


# ===================================================================================================
# ===================================================================================================
# Set 1 processing
# ===================================================================================================
# ===================================================================================================

print("Processing set 1...")

if True:  # Temporarily disable this block while working on the others
    df_set1 = pd.read_csv(set1_path)

    # DEBUG - Print column names
    # print("Original Columns:")
    # print(df.columns)

    # Rename "AppID" to all lowercase
    df_set1.rename(columns={"AppID": "appid"}, inplace=True)

    # Rename "price" to "price_2024_05" to reflect the date of the dataset and not conflict with other pricing info
    df_set1.rename(columns={"price": "price_2024_05"}, inplace=True)

    columns_to_drop = [
        "detailed_description",
        "about_the_game",
        "short_description",
        "packages",
        "reviews",
        "header_image",
        "website",
        "support_url",
        "support_email",
        "metacritic_url",
        "notes",
        "user_score",
        "score_rank",
    ]

    df_set1.drop(columns=columns_to_drop, inplace=True)

    # Print new column names
    print("Remaining Columns:")
    print(df_set1.columns)

    # Turn "screenshots" string representation of an array of URLs into its count (by splitting on commas)
    df_set1["screenshots"] = df_set1["screenshots"].apply(lambda x: len(x.split(",")) if isinstance(x, str) else 0)

    # Same thing for "movies", turn it into a count
    df_set1["movies"] = df_set1["movies"].apply(lambda x: len(x.split(",")) if isinstance(x, str) else 0)

    # Turn "estimated_owners" from a range (e.g. "100000000 - 200000000") to the average (or the floor?)
    df_set1["estimated_owners"] = df_set1["estimated_owners"].apply(lambda x: sum(map(int, x.split(" - "))) / 2 if isinstance(x, str) else 0)

    # Strip the weights from tags and turn it from a dict to a flat list, e.g. {'FPS': 90076, 'Shooter': 64786} -> ['FPS', 'Shooter']
    df_set1["tags"] = df_set1["tags"].apply(lambda x: list(x.keys()) if isinstance(x, dict) else [])

    # DEBUG - Preview those 3 columns for appid 578080, which has the required data
    print(df_set1.loc[df_set1["appid"] == 578080, ["screenshots", "movies", "estimated_owners"]])

    # Save the processed dataset
    df_set1.to_csv(f"{output_dir}/debug_set1_processed.csv", index=False)
    df_set1.to_csv(f"{output_dir}/debug_set1_processed.csv", index=False)
    print("Processed dataset saved to debug_set1_processed.csv")

    combined_df = df_set1  # Set the combined DataFrame to the processed set 1


# ===================================================================================================
# ===================================================================================================
# Set 2 processing
# ===================================================================================================
# ===================================================================================================
print("Processing set 2...")

if True:  # Temporarily disable this block while working on the others
    df_set2 = pd.read_csv(set2_path)

    # DEBUG - Print column names
    print("Original Columns:")
    print(df_set2.columns)

    columns_to_drop = [
        "rating",
        "discount",
        "positive_ratio",
        "user_reviews",
    ]

    df_set2.drop(columns=columns_to_drop, inplace=True)

    mappings = [
        ("app_id", "appid", None),
        ("title", "name", None),
        ("date_release", "release_date", None),
        ("win", "windows", None),
        ("mac", "mac", None),
        ("linux", "linux", None),
        ("price_final", "price_2024_08", None),
        ("price_original", "price_original", None),
        ("steam_deck", "steam_deck", None),
    ]

    # TODO - Consider calculating positive and negative columns from "positive_ratio" and "user_reviews"

    print(f"Head before: {df_set2.head()}")
    df_set2 = apply_mappings(df_set2, mappings, keep_only_mapped_columns=True)
    print(f"Head after: {df_set2.head()}")

    # Save the processed dataset
    df_set2.to_csv(f"{output_dir}/debug_set2_processed.csv", index=False)
    print("Processed dataset saved to debug_set2_processed.csv")

    # Merge the processed set 2 into the combined DataFrame
    combined_df = merge_dataframes_with_mappings(combined_df, df_set2, {})

    # DEBUG - Save the combined DataFrame to a CSV file
    combined_df.to_csv(f"{output_dir}/combined_df_at_step2.csv", index=False)


# ===================================================================================================
# ===================================================================================================
# Set 3 processing
# ===================================================================================================
# ===================================================================================================
print("Processing set 3...")

df_set3 = pd.read_csv(set3_path)  # HEADS UP! This dataset had a missing comma between "Discount" and "DLC Count", causing a shift in the columns. I fixed it manually in the file.

# Mappings of (set 3 column, set 1 column, transform function)
mappings = [
    ("AppID", "appid", None),
    ("Name", "name", None),
    ("Release date", "release_date", None),
    ("Required age", "required_age", None),
    ("Price", "price_2024_09", None),
    ("DLC count", "dlc_count", None),
    ("Windows", "windows", None),
    ("Mac", "mac", None),
    ("Linux", "linux", None),
    ("Achievements", "achievements", None),
    ("Supported languages", "supported_languages", None),
    ("Full audio languages", "full_audio_languages", None),
    ("Recommendations", "recommendations", None),
    ("Average playtime forever", "average_playtime_forever", None),
    ("Average playtime two weeks", "average_playtime_2weeks", None),
    ("Median playtime forever", "median_playtime_forever", None),
    ("Median playtime two weeks", "median_playtime_2weeks", None),
    ("Peak CCU", "peak_ccu", None),
    ("Developers", "developers", split_string_to_list),
    ("Publishers", "publishers", split_string_to_list),
    ("Categories", "categories", split_string_to_list),
    ("Genres", "genres", split_string_to_list),
    ("Tags", "tags", split_string_to_list),
    ("Screenshots", "screenshots", count_items_in_string),
    ("Movies", "movies", count_items_in_string),
    ("Positive", "positive", None),
    ("Negative", "negative", None),
    ("Estimated owners", "estimated_owners", average_range_string),
]


print(f"Head before: {df_set3.head()}")
df_set3 = apply_mappings(df_set3, mappings, keep_only_mapped_columns=True)
print(f"Head after: {df_set3.head()}")


# Save the processed dataset
df_set3.to_csv(f"{output_dir}/debug_set3_processed.csv", index=False)


# Merge the processed set 3 into the combined DataFrame


merge_mappings = {
    "supported_languages": combine_list_unique_values,
    "full_audio_languages": combine_list_unique_values,
    "developers": combine_list_unique_values,
    "publishers": combine_list_unique_values,
    "categories": combine_list_unique_values,
    "genres": combine_list_unique_values,
    "tags": combine_list_unique_values,
}

combined_df = merge_dataframes_with_mappings(combined_df, df_set3, merge_mappings)

# DEBUG - Save the combined DataFrame to a CSV file
combined_df.to_csv(f"{output_dir}/combined_df_at_step3.csv", index=False)

# ===================================================================================================
# ===================================================================================================
# Set 4 processing
# ===================================================================================================
# ===================================================================================================
print("Processing set 4...")
print("NOTE: Skipping set 4 processing due to difficulties in properly formatting the JSON data into compatible tabular data.")
# NOTE: Currently skipped due to difficulties in properly formatting the JSON _nested_ data structures into compatible tabular data.
# df_set4 = pd.read_csv(set3_path)


# ===================================================================================================
# ===================================================================================================
# Set 5 processing
# ===================================================================================================
# ===================================================================================================
print("Processing set 5...")

df_set5 = pd.read_csv(set5_path)

mappings = [
    ("appid", "appid", None),
    ("name", "name", None),
    ("release_date", "release_date", None),
    ("developer", "developers", lambda x: split_string_to_list(x, ";")),
    ("publisher", "publishers", lambda x: split_string_to_list(x, ";")),
    ("categories", "categories", lambda x: split_string_to_list(x, ";")),
    ("genres", "genres", lambda x: split_string_to_list(x, ";")),
    ("positive_ratings", "positive", None),
    ("negative_ratings", "negative", None),
    ("average_playtime", "average_playtime_forever", None),
    ("median_playtime", "median_playtime_forever", None),
    ("owners", "estimated_owners", lambda x: average_range_string(x, "-")),
    ("price", "price_2019_06", None),
]


print(f"Head before: {df_set5.head()}")
df_set5 = apply_mappings(df_set5, mappings, keep_only_mapped_columns=True)
print(f"Head after: {df_set5.head()}")

# Save the processed dataset
df_set5.to_csv(f"{output_dir}/debug_set5_processed.csv", index=False)


merge_mappings = {
    "developers": combine_list_unique_values,
    "publishers": combine_list_unique_values,
    "categories": combine_list_unique_values,
    "genres": combine_list_unique_values,
}

combined_df = merge_dataframes_with_mappings(combined_df, df_set5, merge_mappings)

# DEBUG - Save the combined DataFrame to a CSV file
combined_df.to_csv(f"{output_dir}/combined_df_at_step5.csv", index=False)


# ===================================================================================================
# ===================================================================================================
# Set 6 processing
# ===================================================================================================
# ===================================================================================================
# NOTE -  Set dates 2025-01
# NOTE - release_date can be "Not Released" instead of a date

print("Processing set 6...")
df_set6 = pd.read_csv(set6_path)

# Split "platforms" (e.g. ['windows', 'mac', "linux']) into separate boolean columns
df_set6["windows"] = df_set6["platforms"].apply(lambda x: "windows" in x)
df_set6["mac"] = df_set6["platforms"].apply(lambda x: "mac" in x)
df_set6["linux"] = df_set6["platforms"].apply(lambda x: "linux" in x)


mappings = [
    ("steam_appid", "appid", None),
    ("name", "name", None),
    ("release_date", "release_date", None),
    ("developers", "developers", None),
    ("publishers", "publishers", None),
    ("categories", "categories", None),
    ("genres", "genres", None),
    ("required_age", "required_age", None),
    ("n_achievements", "achievements", None),
    ("windows", "windows", None),
    ("mac", "mac", None),
    ("linux", "linux", None),
    ("total_positive", "positive", None),
    ("total_negative", "negative", None),
    ("metacritic", "metacritic_score", None),
    ("price_initial (USD)", "price_original", None),
    ("is_free", "is_free", None),
]


print(f"Head before: {df_set6.head()}")
df_set6 = apply_mappings(df_set6, mappings, keep_only_mapped_columns=True)
print(f"Head after: {df_set6.head()}")


# Save the processed dataset
df_set6.to_csv(f"{output_dir}/debug_set6_processed.csv", index=False)


merge_mappings = {
    "developers": combine_list_unique_values,
    "publishers": combine_list_unique_values,
    "categories": combine_list_unique_values,
    "genres": combine_list_unique_values,
}

combined_df = merge_dataframes_with_mappings(combined_df, df_set6, merge_mappings)

# DEBUG - Save the combined DataFrame to a CSV file
combined_df.to_csv(f"{output_dir}/combined_df_at_step6.csv", index=False)


# ===================================================================================================
# ===================================================================================================
# Set 7 processing
# ===================================================================================================
# ===================================================================================================
# NOTE -  Set 7 is a simpler JSON file that easily converts to a DataFrame
print("Processing set 7...")

with open(set7_path, "r", encoding="utf-8") as f:
    data = json.load(f)

df_set7 = pd.DataFrame(data)


# Split "platforms" (e.g. ['windows', 'mac', "linux']) into separate boolean columns
df_set7["windows"] = df_set7["platforms"].apply(lambda x: "WIN" in x)
df_set7["mac"] = df_set7["platforms"].apply(lambda x: "MAC" in x)
df_set7["linux"] = df_set7["platforms"].apply(lambda x: "LNX" in x)


mappings = [
    ("sid", "appid", None),
    ("published_store", "release_date", None),
    ("name", "name", None),
    ("full_price", "price_original", lambda x: float(x) / 100 if isinstance(x, (int, float)) else np.nan),  # Convert cents to dollars
    ("current_price", "price_2023_11", lambda x: float(x) / 100 if isinstance(x, (int, float)) else np.nan),  # Convert cents to dollars
    ("windows", "windows", None),
    ("mac", "mac", None),
    ("linux", "linux", None),
    ("developers", "developers", split_string_to_list),
    ("publishers", "publishers", split_string_to_list),
    ("languages", "supported_languages", split_string_to_list),
    ("voiceovers", "full_audio_languages", split_string_to_list),
    ("categories", "categories", split_string_to_list),
    ("genres", "genres", split_string_to_list),
    ("tags", "tags", split_string_to_list),
    ("gfq_difficulty", "gamefaqs_difficulty_rating", None),
    ("gfq_rating", "gamefaqs_review_score", None),
    ("gfq_length", "gamefaqs_game_length", None),
    ("stsp_owners", "steam_spy_estimated_owners", None),
    ("hltb_single", "hltb_single", None),
    ("hltb_complete", "hltb_complete", None),
    ("meta_uscore", "metacritic_score", None),
    ("igdb_uscore", "igdb_review_score", None),
    ("igdb_single", "igdb_single", None),
    ("igdb_complete", "igdb_complete", None),
]


print(f"Head before: {df_set7.head()}")
df_set7 = apply_mappings(df_set7, mappings, keep_only_mapped_columns=True)
print(f"Head after: {df_set7.head()}")

# Save the processed dataset
df_set7.to_csv(f"{output_dir}/debug_set7_processed.csv", index=False)

merge_mappings = {
    "developers": combine_list_unique_values,
    "publishers": combine_list_unique_values,
    "supported_languages": combine_list_unique_values,
    "full_audio_languages": combine_list_unique_values,
    "categories": combine_list_unique_values,
    "genres": combine_list_unique_values,
    "tags": combine_list_unique_values,
}

combined_df = merge_dataframes_with_mappings(combined_df, df_set7, merge_mappings)

# DEBUG - Save the combined DataFrame to a CSV file
combined_df.to_csv(f"{output_dir}/combined_df_at_step7.csv", index=False)


# ===================================================================================================
# ===================================================================================================
# Set 8 processing
# ===================================================================================================
# ===================================================================================================
print("Processing set 8...")

df_set8 = pd.read_csv(set8_path)

mappings = [
    ("game_id", "appid", None),
    ("early_access", "early_access", None),
    ("title", "name", None),
    ("tag_list", "tags", split_string_to_list),
    ("vr_only", "vr_only", None),
    ("vr_supported", "vr_supported", None),
]


df_set8 = apply_mappings(df_set8, mappings, keep_only_mapped_columns=True)

# Save the processed dataset
df_set8.to_csv(f"{output_dir}/debug_set8_processed.csv", index=False)

merge_mappings = {
    "tags": combine_list_unique_values,
}

combined_df = merge_dataframes_with_mappings(combined_df, df_set8, merge_mappings)

# DEBUG - Save the combined DataFrame to a CSV file
combined_df.to_csv(f"{output_dir}/combined_df_at_step8.csv", index=False)


# ===================================================================================================
# ===================================================================================================
# Set 9 processing
# ===================================================================================================
# ===================================================================================================
print("Processing set 9...")

df_set9 = pd.read_csv(set9_path)


# Split "platforms" (e.g. ['windows', 'mac', "linux']) into separate boolean columns
df_set9["windows"] = df_set9["platforms"].apply(lambda x: "windows" in x)
df_set9["mac"] = df_set9["platforms"].apply(lambda x: "mac" in x)
df_set9["linux"] = df_set9["platforms"].apply(lambda x: "linux" in x)

# Fix the weird "owners" column format that has ranges like "10,000,000 .. 20,000,000"
df_set9["owners"] = df_set9["owners"].apply(lambda x: x.replace(",", "").replace(" .. ", " - ") if isinstance(x, str) else x)

# Exclude multiple faulty "initialprice" values that did not have their decimal in the right place (e.g. 8999.0 instead of 89.99)
# Cutoff set at 150.0 after manual verification of multiple titles
numeric_prices = pd.to_numeric(df_set9["initialprice"], errors="coerce")  # First, convert the column to numeric, coercing errors to NaN
mask = numeric_prices.isna() | (numeric_prices >= 150)  # Create a mask: True if the value is NaN or (if numeric) greater than or equal to 150.
df_set9.loc[mask, "initialprice"] = np.nan  # Set the values in the "initialprice" column to NaN where the mask is True


mappings = [
    ("id", "appid", None),
    ("name", "name", None),
    ("developers", "developers", None),
    ("publishers", "publishers", None),
    ("release_date", "release_date", None),
    ("initialprice", "price_original", None),
    ("drm", "has_drm", None),
    ("controller_support", "controller_support", None),
    ("demos", "has_demos", None),
    ("vr_only", "vr_only", None),
    ("vr_supported", "vr_supported", None),
    ("supported_languages", "supported_languages", split_string_to_list),
    ("voice_languages", "full_audio_languages", split_string_to_list),
    ("metacritic", "metacritic_score", lambda x: np.nan if (x == "FALSE" or x == "False" or x == False) else float(x)),  # Convert to float, or np.nan if "FALSE"
    ("total_positive", "positive", None),
    ("total_negative", "negative", None),
    ("playtime_mean_forever", "average_playtime_forever", None),
    ("playtime_median_forever", "median_playtime_forever", None),
    ("playtime_mean_last2weeks", "average_playtime_2weeks", None),
    ("playtime_median_last2weeks", "median_playtime_2weeks", None),
    ("peak_players_17April2022", "peak_ccu", None),
    ("owners", "estimated_owners", average_range_string),
]

df_set9 = apply_mappings(df_set9, mappings, keep_only_mapped_columns=True)

# Save the processed dataset
df_set9.to_csv(f"{output_dir}/debug_set9_processed.csv", index=False)

merge_mappings = {
    "developers": combine_list_unique_values,
    "publishers": combine_list_unique_values,
    "supported_languages": combine_list_unique_values,
    "full_audio_languages": combine_list_unique_values,
}

combined_df = merge_dataframes_with_mappings(combined_df, df_set9, merge_mappings)

# DEBUG - Save the combined DataFrame to a CSV file
combined_df.to_csv(f"{output_dir}/combined_df_at_step9.csv", index=False)


# ===================================================================================================
# ===================================================================================================
# Set 10 processing
# ===================================================================================================
# ===================================================================================================
print("Processing set 10...")
print("NOTE: Skipping set 10 processing due to difficulties in properly formatting the JSON data into compatible tabular data.")
# NOTE: Currently skipped due to difficulties in properly formatting the JSON _nested_ data structures into compatible tabular data.


# ===================================================================================================
# ===================================================================================================
# Set 11 processing
# ===================================================================================================
# ===================================================================================================
print("Processing set 11...")
print("NOTE: Skipping set 11 processing due to difficulties in properly formatting the JSON data into compatible tabular data.")
# NOTE: Currently skipped due to difficulties in properly formatting the JSON _nested_ data structures into compatible tabular data.


# ===================================================================================================
# ===================================================================================================
# Set 12 processing (HLTB times, by full game name, which makes us lose a bunch of them...)
# ===================================================================================================
# ===================================================================================================

df_set12 = pd.read_csv(set12_path)

# Keep only title, main_story, main_plus_extras, completionist
df_set12 = df_set12[["title", "main_story", "main_plus_extras", "completionist"]]

# Prefix all columns with "hltb_" to avoid conflicts
df_set12.columns = [f"hltb_{col}" for col in df_set12.columns]

# # Merge df_merged  ("left" join)  with df_hltb on game title/name
combined_df = combined_df.merge(df_set12, left_on="name", right_on="hltb_title", how="left")

# # Drop the "title" column
combined_df.drop(columns=["hltb_title"], inplace=True)

# Save the processed dataset
df_set12.to_csv(f"{output_dir}/debug_set9_processed.csv", index=False)

# DEBUG - Save the combined DataFrame to a CSV file
combined_df.to_csv(f"{output_dir}/combined_df_at_step12.csv", index=False)


# ===================================================================================================
# Finally, save the combined fully-merged DataFrame
# ===================================================================================================
final_output_filename = "combined_df_final.csv"
combined_df.to_csv(f"{output_dir}/combined_df_final.csv", index=False)

print("Processing complete. Final combined DataFrame saved to combined_df_final.csv")

# Open output directory
print(f"Opening output directory: {output_dir}")
print(f"Focusing on file: {final_output_filename}")

subprocess.Popen(f'explorer /select,"{output_dir}\\{final_output_filename}"')
