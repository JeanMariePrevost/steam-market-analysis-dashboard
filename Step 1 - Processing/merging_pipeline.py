"""
This script contains all the logic to sequentially process, trim and merge the datasets one by one.
The process is not particularly clean or optimized, but it should be a one-time operation, so anything more would be over-engineering.
"""

import json
import random
import pandas as pd

# Selected datasets paths
set1_path = r"F:\OneDrive\MyDocs\Study\TELUQ\Session 8 - Hiver 2025\SCI 1402\datasets\artermiloff_steam-games-dataset\games_may2024_cleaned.csv"  # Great general metadata foundation
set2_path = r"F:\OneDrive\MyDocs\Study\TELUQ\Session 8 - Hiver 2025\SCI 1402\datasets\antonkozyriev_game-recommendations-on-steam\games.csv"  # grab price_final and price_original
set3_path = r"F:\OneDrive\MyDocs\Study\TELUQ\Session 8 - Hiver 2025\SCI 1402\datasets\fronkongames_steam-games-dataset\games.csv"  # Has interesting features like metacritic score, achievement, recommandations, genres, tags (combine across sets on these?)
set4_path = r"F:\OneDrive\MyDocs\Study\TELUQ\Session 8 - Hiver 2025\SCI 1402\datasets\fronkongames_steam-games-dataset\games.json"  # Unsure. Includes game descriptions an such (NLP?) DLCs, genres, and especially "estimated_owners"
set5_path = r"F:\OneDrive\MyDocs\Study\TELUQ\Session 8 - Hiver 2025\SCI 1402\datasets\nikdavis_steam-store-games_steamspy\steam.csv"  # playtime stats and owners
set6_path = r"F:\OneDrive\MyDocs\Study\TELUQ\Session 8 - Hiver 2025\SCI 1402\datasets\srgiomanhes_steam-games-dataset-2025\steam_games.csv"  # Similar to base set, with 2025 data, but we have to ensure ALL its columns can fully merge in
set7_path = r"F:\OneDrive\MyDocs\Study\TELUQ\Session 8 - Hiver 2025\SCI 1402\datasets\sujaykapadnis_games-on-steam\steamdb.json"  # Some VERY interesting "meta" fields from gamefaqs, stsp, hltb, metacritic and igdb IF complete enough
set8_path = r"F:\OneDrive\MyDocs\Study\TELUQ\Session 8 - Hiver 2025\SCI 1402\datasets\vicentearce_steamdata\game_data.csv"  # For the vr tags, whch I THINK other sets don't offer
set9_path = r"F:\OneDrive\MyDocs\Study\TELUQ\Session 8 - Hiver 2025\SCI 1402\datasets\vicentearce_steamdata\final\steam_games.csv"  # Coop, online, workshop support, languages, precise rating, playtime, peak player, owners, has dlc, has demos....
set10_path = r"F:\OneDrive\MyDocs\Study\TELUQ\Session 8 - Hiver 2025\SCI 1402\datasets\souyama_steam-dataset\steam_dataset\steamspy\detailed\steam_spy_detailed.json"  # Owners, CCU, detailled tags
set11_path = r"F:\OneDrive\MyDocs\Study\TELUQ\Session 8 - Hiver 2025\SCI 1402\datasets\souyama_steam-dataset\steam_dataset\appinfo\store_data\steam_store_data.json"  # F2P, required age, what is price_overview?, whether games have demos...

# TO consider: antonkozyriev_game-recommendations-on-steam	games_metadata.json for its long tags list (unweighted)


def apply_mappings(df, mappings, keep_only_mapped_columns=False) -> pd.DataFrame:
    """
    Renames and transforms columns of a DataFrame based on a given mappings list.
    Does not copy columns with no mapping if keep_only_mapped_columns is True.
    """
    print(f"Starting columns: {df.columns}")

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


# ===================================================================================================
# ===================================================================================================
# Set 1 processing
# ===================================================================================================
# ===================================================================================================

if False:  # Temporarily disable this block while working on the others
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
    df_set1.to_csv("set1_processed.csv", index=False)
    print("Processed dataset saved to set1_processed.csv")


# ===================================================================================================
# ===================================================================================================
# Set 2 processing
# ===================================================================================================
# ===================================================================================================

if False:  # Temporarily disable this block while working on the others
    df_set2 = pd.read_csv(set2_path)

    # DEBUG - Print column names
    print("Original Columns:")
    print(df_set2.columns)

    # Rename "app_id" to "appid" to match the other datasets
    df_set2.rename(columns={"app_id": "appid"}, inplace=True)

    # Keep only "price_original" and "steam_deck" columns, with app_id as index to merge on
    columns_to_keep = ["appid", "price_original", "steam_deck"]
    df_set2 = df_set2[columns_to_keep]

    # Print new column names
    print("Remaining Columns:")
    print(df_set2.columns)

    # Save the processed dataset
    df_set2.to_csv("set2_processed.csv", index=False)
    print("Processed dataset saved to set2_processed.csv")


# ===================================================================================================
# ===================================================================================================
# Set 3 processing
# ===================================================================================================
# ===================================================================================================

df_set3 = pd.read_csv(set3_path)  # HEADS UP! This dataset had a missing comma between "Discount" and "DLC Count", causing a shift in the columns. I fixed it manually in the file.

# DEBUG - Print column names
# print("Original Columns:")
# print(df_set3.columns)

# "Fill in the blanks" with the data from set1
# Anything this set has that the others don't, we add in a way that matches the other sets

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
    ("Developers", "developers", lambda x: x.split(",") if isinstance(x, str) else []),  # Turn the string into a list
    ("Publishers", "publishers", lambda x: x.split(",") if isinstance(x, str) else []),  # Turn the string into a list
    ("Categories", "categories", lambda x: x.split(",") if isinstance(x, str) else []),  # Turn the string into a list
    ("Genres", "genres", lambda x: x.split(",") if isinstance(x, str) else []),  # Turn the string into a list
    ("Tags", "tags", lambda x: x.split(",") if isinstance(x, str) else []),  # Turn the string into a list
    ("Screenshots", "screenshots", lambda x: len(x.split(",")) if isinstance(x, str) else 0),  # Turn it into the NUMBER of screenshots
    ("Movies", "movies", lambda x: len(x.split(",")) if isinstance(x, str) else 0),  # Turn it into the NUMBER of movies
    ("Positive", "positive", None),
    ("Negative", "negative", None),
    ("Estimated owners", "estimated_owners", lambda x: sum(map(int, x.split(" - "))) / 2 if isinstance(x, str) else 0),  # Turn the range into its average
]


print(f"Head before: {df_set3.head()}")
df_set3 = apply_mappings(df_set3, mappings, keep_only_mapped_columns=True)
print(f"Head after: {df_set3.head()}")


# Save the processed dataset
df_set3.to_csv("set3_processed.csv", index=False)
