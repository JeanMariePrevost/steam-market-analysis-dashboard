"""
This script contains all the logic to sequentially preprocess the merged dataset generated from merging_pipeline.py.
The end result is a cleaner, more usable dataset with enforced types and value formats.
It also introduces new columns and normalizes existing ones for better usability and analysis.
However, it retains "human-readable" features such as tags, genres and languages as lists of strings.
"""

import pandas as pd
import numpy as np
import os
import subprocess
import ast

from tqdm import tqdm


# Define and create output directory if it doesn't exist
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script itself
output_dir = os.path.join(script_dir, "preprocessed_output")  # Define the output directory relative to the script's location
os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

source_csv_path = os.path.join(script_dir, "merge_output/combined_df_final.csv")


####################################################################
# Helper functions
####################################################################
def enforce_float_or_nan_column(df, column_name):
    """Enforces a column as floats, squashing all non-numeric values to NaN."""
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' does not exist in the DataFrame.")
    df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
    print(f"{column_name} fully numerical (floats) ✅")


def enforce_int_or_nan_column(df, column_name, verbose=True):
    """Enforces a column as integers, squashing all non-numeric values to NaN. Any floats will also be converted to integers (truncated)."""
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' does not exist in the DataFrame.")
    df[column_name] = pd.to_numeric(df[column_name], errors="coerce").astype(
        "Int64",
    )
    print(f"{column_name} fully numerical (integers) ✅")


def enforce_boolean_or_nan_column(df, column_name):
    """Enforces a column as boolean, squashing all non-boolean values to NaN."""
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' does not exist in the DataFrame.")

    # Replace non-boolean values with pd.NA
    df[column_name] = df[column_name].map(lambda x: x if isinstance(x, bool) else pd.NA).astype("boolean")  # Convert to nullable boolean dtype

    print(f"{column_name} fully boolean ✅")


def enforce_string_or_nan_column(df, column_name):
    """Ensures a column is fully string, coercing invalid values to NaN."""
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' does not exist in the DataFrame.")
    df[column_name] = df[column_name].map(lambda x: x if isinstance(x, str) else np.nan)
    print(f"{column_name} fully string ✅")


def enforce_datetime_column(df, column_name):
    """Ensures a column is fully datetime, coercing invalid values to NaT. Tries multiple formats."""

    date_formats = [
        "%Y-%m-%d",  # e.g. 1998-11-08
        "%Y-%m-%d %H:%M:%S",  # e.g. 2018-04-05 00:00:00
        "%b %d, %Y",  # e.g. Dec 22, 2023
        "%B %d, %Y",  # e.g. February 9, 2024
        "%B %Y",  # e.g. January 2020 (Will assume day = 1)
        "%b %Y",  # e.g. Jan 2020 (Will assume day = 1)
    ]

    def try_parsing_date(value):
        if pd.isna(value) or value == "" or (isinstance(value, str) and value.lower() in ["not released", "not_released", "coming soon", "coming_soon", "unknown"]):
            return pd.NaT  # Keep NaN as NaT
        if isinstance(value, pd.Timestamp):
            return value  # Already correct format

        for fmt in date_formats:
            try:
                parsed = pd.to_datetime(value, format=fmt)
                return parsed
            except ValueError:
                continue

        # Special case: Month-Year (e.g., "May 2020") – Assume the 1st day
        try:
            return pd.to_datetime(f"1 {value}", format="%d %B %Y")
        except ValueError:
            pass

        raise ValueError(f"Invalid date format in {column_name}: {repr(value)}")

    # Apply the parsing function
    print(f"Normalizing {column_name}...")
    df[column_name] = df[column_name].progress_map(try_parsing_date)

    # Ensure the column is explicitly datetime64[ns]
    df[column_name] = pd.to_datetime(df[column_name], errors="coerce")

    print(f"{column_name} fully datetime ✅")


def normalize_lists_string_values(df, column_name, force_lowercase=True, remove_duplicate_elements=True, sort_elements=True):
    """
    Normalizes all string lists in a column by optionally forcing lowercase and removing duplicates.
    """

    def normalize_cell(value):
        if isinstance(value, list):
            value_as_list = value
        elif pd.isna(value) or value == "":
            return []
        else:
            try:
                value_as_list = ast.literal_eval(value)
                if not isinstance(value_as_list, list):
                    raise ValueError(f"Value is not a list: {repr(value)}")
                if not all(isinstance(x, str) for x in value_as_list):
                    raise ValueError(f"List contains non-string elements: {repr(value)}")
            except:
                raise ValueError(f"Invalid entry: {repr(value)}")

        if force_lowercase:
            value_as_list = [x.lower() for x in value_as_list]
        if remove_duplicate_elements:
            # Remove duplicates while preserving order
            seen = set()
            value_as_list = [x for x in value_as_list if not (x in seen or seen.add(x))]
        if sort_elements:
            value_as_list.sort()
        return value_as_list

    print(f"Normalizing {column_name}...")
    df[column_name] = df[column_name].progress_map(normalize_cell)
    print(f"{column_name} fully normalized ✅")


def normalize_languages_column(df, column_name):
    """
    Normalizes a column containing lists of languages, ensuring all languages are in a predefined list.
    Tries to match languages that are "close enough" to the allowed ones.
    Removes duplicates and sorts the final lists.
    """
    allowed_languages = {
        "afrikaans",
        "albanian",
        "amharic",
        "arabic",
        "armenian",
        "assamese",
        "azerbaijani",
        "bangla",
        "basque",
        "belarusian",
        "bosnian",
        "bulgarian",
        "catalan",
        "cherokee",
        "chinese_simplified",
        "chinese_traditional",
        "croatian",
        "czech",
        "danish",
        "dari",
        "dutch",
        "english",
        "estonian",
        "filipino",
        "finnish",
        "french",
        "galician",
        "georgian",
        "german",
        "greek",
        "gujarati",
        "hausa",
        "hebrew",
        "hindi",
        "hungarian",
        "hungarian",
        "icelandic",
        "igbo",
        "indonesian",
        "irish",
        "italian",
        "japanese",
        "kannada",
        "kazakh",
        "khmer",
        "kinyarwanda",
        "konkani",
        "korean",
        "kyrgyz",
        "latvian",
        "lithuanian",
        "luxembourgish",
        "macedonian",
        "malay",
        "malayalam",
        "maltese",
        "maori",
        "marathi",
        "mongolian",
        "nepali",
        "norwegian",
        "odia",
        "persian",
        "polish",
        "portuguese_brazil",
        "portuguese_portugal",
        "punjabi_gurmukhi",
        "punjabi_shahmukhi",
        "quechua",
        "romanian",
        "russian",
        "scots",
        "serbian",
        "sindhi",
        "sinhala",
        "slovak",
        "slovenian",
        "sorani",
        "sotho",
        "spanish_latin_america",
        "spanish_spain",
        "swahili",
        "swedish",
        "tajik",
        "tamil",
        "tatar",
        "telugu",
        "thai",
        "tigrinya",
        "tswana",
        "turkish",
        "turkmen",
        "ukrainian",
        "urdu",
        "uyghur",
        "uzbek",
        "valencian",
        "vietnamese",
        "welsh",
        "wolof",
        "xhosa",
        "yoruba",
        "zulu",
    }

    def normalize_cell(value):
        if isinstance(value, list):
            languages_list = value
        elif pd.isna(value) or value == "":
            return []
        else:
            try:
                languages_list = ast.literal_eval(value)
                if not isinstance(languages_list, list):
                    raise ValueError(f"Value is not a list: {repr(value)}")
                if not all(isinstance(x, str) for x in languages_list):
                    raise ValueError(f"List contains non-string elements: {repr(value)}")
            except:
                raise ValueError(f"Invalid entry: {repr(value)}")

        # Normalize each language
        normalized_languages = []
        for element in languages_list:
            element = element.lower()
            if element in allowed_languages:
                normalized_languages.append(element)
            else:
                for allowed_lang in allowed_languages:
                    if allowed_lang in element.lower():
                        normalized_languages.append(allowed_lang)  # Replace what "looks like" the language with the correct one
                        break

        # Remove duplicates
        normalized_languages = list(set(normalized_languages))

        normalized_languages.sort()
        return normalized_languages

    df[column_name] = df[column_name].map(normalize_cell)
    print(f"{column_name} fully normalized ✅")


def add_average_column(df, columns, new_column_name):
    """
    Adds a new column to the DataFrame that contains the average of the specified columns, excluding NA/NaN values.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of column names to average. E.g. ["column1", "column2", "column3"]
        new_column_name (str): Name of the new column to be added. E.g. "salary_avg"
    """
    # Row-wise mean on the selected columns, skipping NaN values
    df[new_column_name] = df[columns].mean(axis=1, skipna=True)
    return df


def normalize_datetime_column(df, column_name):
    """
    Normalizes a datetime column to a consistent format, squashing all invalid values to NA.
    """
    df[column_name] = pd.to_datetime(df[column_name], errors="coerce")
    print(f"{column_name} fully normalized ✅")


##########################################################################################################################
##########################################################################################################################
# Cleaning begins here
##########################################################################################################################
##########################################################################################################################
tqdm.pandas()
df = pd.read_csv(source_csv_path)

####################################################################
# Index
####################################################################
df.set_index("appid", inplace=True)  # Make "appid" the index
df.sort_index(inplace=True)  # Sort by index
# Explicitly set the index name if it was "appid"
df.index.name = "appid"

if df.index.duplicated().any():
    raise ValueError("Duplicate appids found in the dataset. ❌")

####################################################################
# Per column special cleaning (exlcuding type enforcement)
####################################################################
# achivements
df.rename(columns={"achievements": "achievements_count"}, inplace=True)


# playtime columns (avg, median...)
df.rename(columns={"average_playtime_forever": "playtime_avg"}, inplace=True)
df.rename(columns={"average_playtime_2weeks": "playtime_avg_2_weeks"}, inplace=True)
df.rename(columns={"median_playtime_forever": "playtime_median"}, inplace=True)
df.rename(columns={"median_playtime_2weeks": "playtime_median_2_weeks"}, inplace=True)


# languages stuff
df.rename(columns={"supported_languages": "languages_supported"}, inplace=True)
df.rename(columns={"full_audio_languages": "languages_with_full_audio"}, inplace=True)

# "Time to beat" stuff:
#   - gamefaqs_game_length
#   - hltb_single
#   - hltb_complete
#   - igdb_complete
#   - igdb_single
#   - hltb_main_story
#   - hltb_main_plus_extras
#   - hltb_completionist
add_average_column(
    df, ["gamefaqs_game_length", "hltb_single", "hltb_complete", "igdb_complete", "igdb_single", "hltb_main_story", "hltb_main_plus_extras", "hltb_completionist"], "average_time_to_beat"
)
df.drop(columns=["gamefaqs_game_length", "hltb_single", "hltb_complete", "igdb_complete", "igdb_single", "hltb_main_story", "hltb_main_plus_extras", "hltb_completionist"], inplace=True)

# External review scores
#   - gamefaqs_review_score (out of 5!)
#   - igdb_review_score (out of 100)
#   - metacritic_score (out of 100)

# Normalize each rating field to a 0-1 range
df["gamefaqs_review_score"] /= 5
df["igdb_review_score"] /= 100
df["metacritic_score"] /= 100

add_average_column(df, ["gamefaqs_review_score", "igdb_review_score", "metacritic_score"], "average_non_steam_review_score")
df.drop(columns=["gamefaqs_review_score", "igdb_review_score", "metacritic_score"], inplace=True)

# platforms stuff
df.rename(columns={"linux": "runs_on_linux"}, inplace=True)
df.rename(columns={"mac": "runs_on_mac"}, inplace=True)
df.rename(columns={"windows": "runs_on_windows"}, inplace=True)

# movies and screenshots
df.rename(columns={"screenshots": "steam_store_screenshot_count"}, inplace=True)
df.rename(columns={"movies": "steam_store_movie_count"}, inplace=True)

# Steam reviews
df.rename(columns={"negative": "steam_negative_reviews"}, inplace=True)
df.rename(columns={"positive": "steam_positive_reviews"}, inplace=True)


# release_date stuff (we have YYYY-MM-DD format, "Feb 27, 2018" format, some with time (2020-04-05 00:00:00), and "Not Released", "coming_soon" and "unknown")...
# First, extract "is_released" bool column from "release_date"
df["is_released"] = df["release_date"].progress_map(lambda x: isinstance(x, str) and x.lower() not in ["not released", "coming soon"])

# steam_deck
df.rename(columns={"steam_deck": "runs_on_steam_deck"}, inplace=True)

# Drop columns too weak, irrelevant, or sparse
df.drop(columns=["num_reviews_recent", "num_reviews_total", "pct_pos_recent", "pct_pos_total"], inplace=True)

df.drop(columns=["is_released"], inplace=True)  # Not accurate at all

####################################################################
# Combine all punctual (by date) pricing info
####################################################################
# Replace all "price_YYYY_MM" with a single, more usable "price_latest"
ordered_price_columns = [
    "price_2024_09",
    "price_2024_08",
    "price_2024_05",
    "price_2023_11",
    "price_2019_06",
    "price_original",
]

# Fill missing price_original values with the highest available price
enforce_float_or_nan_column(df, "price_original")
enforce_float_or_nan_column(df, "price_2024_09")
enforce_float_or_nan_column(df, "price_2024_08")
enforce_float_or_nan_column(df, "price_2024_05")
enforce_float_or_nan_column(df, "price_2023_11")
enforce_float_or_nan_column(df, "price_2019_06")

# Set price_original to the highest possible value _IF_ if is NaN or 0
df.loc[df["price_original"].isna() | (df["price_original"] == 0), "price_original"] = df[ordered_price_columns].astype(float).max(axis=1)

# Introduced a "price_latest" column that is the most recent price available
df["price_latest"] = df[ordered_price_columns].bfill(axis=1).iloc[:, 0]

# Drop the old "by date" price columns
columns_to_drop = [
    "price_2024_09",
    "price_2024_08",
    "price_2024_05",
    "price_2023_11",
    "price_2019_06",
]
df.drop(columns=columns_to_drop, inplace=True)


# Fix invalid pricing info and introduce "is_f2p" for the "free to play" tag / genre
normalize_lists_string_values(df, "genres")
normalize_lists_string_values(df, "tags")
df["is_f2p"] = False
df.loc[df["genres"].map(lambda x: "free to play" in x), "is_f2p"] = True
df.loc[df["tags"].map(lambda x: "free to play" in x), "is_f2p"] = True
# Set "is_free" to False for games that don't have any "free to play" tage/genre _and_ that have both a price_original and a price_latest > 0, which are extremely unlikely to actually be free
enforce_float_or_nan_column(df, "price_original")
enforce_float_or_nan_column(df, "price_latest")
df.loc[(df["is_f2p"] == False) & (df["price_original"] > 0) & (df["price_latest"] > 0), "is_free"] = False

# set F2P to false for games where is_free is True, to differentiate between "free" and "free to play", which the tags don't do
df.loc[df["is_free"] == True, "is_f2p"] = False
enforce_boolean_or_nan_column(df, "is_free")
enforce_boolean_or_nan_column(df, "is_f2p")


# Fix all " that aren't aligned with the "is_free" tag
df.loc[df["is_f2p"], "price_latest"] = 0.0
df.loc[df["is_f2p"], "price_original"] = 0.0

# Turn all text (e.g. unlreleased) and empties into NaN
enforce_float_or_nan_column(df, "price_latest")
enforce_float_or_nan_column(df, "price_original")

# Cross-populate prices columns where one is missing, as current or original price should be a good proxy of the other
print("Cross-populating price columns...")
df.loc[df["price_latest"].isna(), "price_latest"] = df["price_original"]
df.loc[df["price_original"].isna(), "price_original"] = df["price_latest"]


####################################################################
# Fixing the broken/duplicate tags manually
#     'base building' -> 'base-building'
#     'dystopian ' -> 'dystopian'
#     'parody ' -> 'parody'
#     'puzzle-platformer' -> 'puzzle platformer'
#     'rogue-like' -> 'roguelike'
#     'rogue-lite' -> 'roguelite'


# Function to replace a specific string inside lists
def replace_in_list(lst, before, after):
    if isinstance(lst, str):
        lst = ast.literal_eval(lst)
    if lst is None or not isinstance(lst, list) or lst == "":
        return []
    return [item.replace(before, after) if isinstance(item, str) else item for item in lst]


# Apply the function with "rogue-like" -> "roguelike"
print("Fixing pseudo-duplicated tags...")
df["tags"] = df["tags"].map(lambda lst: replace_in_list(lst, "base building", "base-building"))
df["tags"] = df["tags"].map(lambda lst: replace_in_list(lst, "dystopian ", "dystopian"))
df["tags"] = df["tags"].map(lambda lst: replace_in_list(lst, "parody ", "parody"))
df["tags"] = df["tags"].map(lambda lst: replace_in_list(lst, "puzzle-platformer", "puzzle platformer"))
df["tags"] = df["tags"].map(lambda lst: replace_in_list(lst, "rogue-like", "roguelike"))
df["tags"] = df["tags"].map(lambda lst: replace_in_list(lst, "rogue-lite", "roguelite"))


####################################################################
# "Usability/qol" columns to introduce
####################################################################
# Introduce a "release_year" column from "release_date"
enforce_datetime_column(df, "release_date")
df["release_year"] = df["release_date"].dt.year

df["steam_total_reviews"] = df["steam_negative_reviews"] + df["steam_positive_reviews"]

df["steam_positive_review_ratio"] = df["steam_positive_reviews"] / df["steam_total_reviews"]

# Add a "monetization_model" column (free, f2p, paid) to replace the "is_free" and "is_f2p" columns
# Set it to "paid" where the game has a price >0
df.loc[df["price_original"] > 0, "monetization_model"] = "paid"
df.loc[df["price_latest"] > 0, "monetization_model"] = "paid"
# Set it to "free" where is_free
df.loc[df["is_free"] == True, "monetization_model"] = "free"
# Set it to "f2p" where is_f2p
df.loc[df["is_f2p"] == True, "monetization_model"] = "f2p"

# Drop the old "is_free" and "is_f2p" columns
df.drop(columns=["is_free", "is_f2p"], inplace=True)

####################################################################
# Inferred or imputed columns to introduce
####################################################################
# Estimated owners were positively useless and rather sparse. Even review numbers often made them technically impossible.
# I've decided to rely on the conservative Boxleiter method

minimum_total_reviews = 1  # Minimum total reviews to consider the estimate (don't use?)

df.drop(columns=["estimated_owners"], inplace=True)
df.drop(columns=["steam_spy_estimated_owners"], inplace=True)

# temp_scale_factor = 1, or 0.75 if steam_negative_reviews + steam_positive_reviews > 100000
df["temp_scale_factor"] = 1.0  # Default to 1.0
df.loc[df["steam_total_reviews"] > 100000, "temp_scale_factor"] = 0.75


df["temp_review_inflation_factor"] = 1.0
df.loc[df["release_year"] == 2013, "temp_review_inflation_factor"] = 79 / 80
df.loc[df["release_year"] == 2014, "temp_review_inflation_factor"] = 72 / 80
df.loc[df["release_year"] == 2015, "temp_review_inflation_factor"] = 62 / 80
df.loc[df["release_year"] == 2016, "temp_review_inflation_factor"] = 52 / 80
df.loc[df["release_year"] == 2017, "temp_review_inflation_factor"] = 43 / 80
df.loc[df["release_year"] == 2018, "temp_review_inflation_factor"] = 38 / 80
df.loc[df["release_year"] == 2019, "temp_review_inflation_factor"] = 36 / 80
df.loc[df["release_year"] > 2020, "temp_review_inflation_factor"] = 31 / 80

boxleiter_number = 80  # Boxleiter's Method "owner per review" multipler

# User Commitment Bias : Expensive games get more reviews per owner, F2P games get the least
# Closest match was roughly `3 + 2 * np.exp(-0.2 * price) -2`, where F2P games get nearly 3 _TIMES_ fewer reviews per player (see marvel Rivals, TF2, etc), and the effect vanishing around $30
df["temp_commitment_bias_mult"] = df["price_original"].map(lambda x: 3 + 2 * np.exp(-0.2 * x) - 2)


df["estimated_owners_boxleiter"] = (
    (df["steam_total_reviews"] * boxleiter_number * df["temp_scale_factor"] * df["temp_review_inflation_factor"] * df["temp_commitment_bias_mult"]).round().astype("Int64")
)

# Remove esitmates for rows where the total reviews are too low
df.loc[df["steam_total_reviews"] < minimum_total_reviews, "estimated_owners_boxleiter"] = np.nan

# Drop the temporary columns
df.drop(columns=["temp_scale_factor", "temp_review_inflation_factor", "temp_commitment_bias_mult"], inplace=True)

# Add Boxleiter based conservative revenue estimates
# Implements heuristics described in the Boxleiter method's modifications in regards to game reviews and age influencing the degree of discount applied

# First, calculate an "average sale discount" based on age (from current year) and negative review ratio
# most_recent_year = df["release_year"].max()
# df["years_since_release"] = most_recent_year - df["release_year"]
# Calculate years_since_release as a float including month (and day) fractions.
current_date = pd.Timestamp.today()
df["years_since_release"] = (current_date - df["release_date"]).dt.days / 365.25

# Use `0.5 * np.exp(-b * years_since_release) + 0.3` as the discount factor
# where b is a the dislike ratio, that determines how quickly the discount decreases with age, from 80% (typical launch discount) down to a semi-arbitrary 30% average discount
# This was validated against "known" game revenues as shared or public (Stardew Valley, Cyberpunk 2077, Withcer 3 Wild Hunt...), and found to surprisingly accurate
# The dislike ratio is calculated as the ratio of negative reviews to total reviews
df["dislike_ratio"] = (df["steam_negative_reviews"] / df["steam_total_reviews"]).pow(2)  # Makes the dislike ratio less impactful near the positive end
df["discount_factor"] = 0.5 * np.exp(-df["dislike_ratio"] * df["years_since_release"]) + 0.3
# Fill NaNs (missing years_since_release) with a simpler discount factor based on reviews only
df["discount_factor"].fillna(0.8 - 0.3 * df["dislike_ratio"], inplace=True)

####################################################################
# Calculate the estimated gross revenue using an estimated LTV and the Boxleiter method
# For paid titles
df.loc[df["monetization_model"] == "paid", "estimated_ltarpu"] = (df["price_original"] * df["discount_factor"]).round().astype("float")

# Do the same with a conservative LTV / ARPU for free to play games, but the monetization strategies would play a _massive_ part here
base_ltarpu_for_estimate = 1.0

# Custom curve created through https://mycurvefit.com/
# Older games had more time to monetize players, meaning higher LTV
df["f2p_release_years_score"] = 1.287535 + (0.500978 - 1.287535) / (1 + (df["years_since_release"] / 0.7431369) ** 1.998668)

# Certain tags are known to be more monetizable than others
# https://rocketbrush.com/blog/most-popular-video-game-genres-in-2024-revenue-statistics-genres-overview
# https://premortem.games/2024/02/28/based-on-the-data-from-2023-what-genre-of-casual-games-will-perform-well-in-2024/
# https://www.linkedin.com/pulse/mixing-genres-key-game-monetisation-success-samuel-huber

tags_value_mappings = {
    "battle royale": 3.0,
    "card battler": 3.0,
    "card game": 3.0,
    "casual": -0.5,
    "character action game": 0.5,
    "character customization": 1.0,
    "competitive": 2.0,
    "cozy": -0.5,
    "deckbuilding": 0.5,
    "e-sports": 3.0,
    "esports": 3.0,
    "gambling": 4.0,
    "games workshop": -0.25,
    "indie": -0.25,
    "loot": 0.5,
    "looter shooter": 0.5,
    "massively multiplayer": 3.0,
    "mmorpg": 3.0,
    "moba": 3.0,
    "mod": -0.25,
    "moddable": -0.25,
    "multiplayer": 1.5,
    "puzzle": 0.5,
    "pvp": 3.0,
    "rpg": 0.5,
}


def compute_tag_scores(tags, mapping=tags_value_mappings, decay=0.25):
    """
    Given a list of tags, this function:
      - Filters the tags based on the provided mapping.
      - Separates positive and negative tags.
      - Sorts each group by impact (largest impact first).
      - Applies a decaying weight (0.25 for each extra tag).

    Returns a dict with:
      - 'positive': cumulative score from positive tags.
      - 'negative': cumulative score from negative tags.
      - 'net': overall score (positive + negative).
    """
    pos_values = []
    neg_values = []

    for tag in tags:
        if tag in mapping:
            value = mapping[tag]
            if value > 0:
                pos_values.append(value)
            elif value < 0:
                neg_values.append(value)

    # Sort positive tags in descending order (largest first).
    pos_values.sort(reverse=True)
    # For negatives, sort by absolute value (largest penalty first).
    neg_values.sort(key=lambda x: abs(x), reverse=True)

    pos_score = sum(val * (decay**i) for i, val in enumerate(pos_values))
    neg_score = sum(val * (decay**i) for i, val in enumerate(neg_values))

    return 1 + pos_score + neg_score


# for each title, go through its df["tags"] and cumulate all the positive and negative tags
# then apply their value from highest impact to lowest impact, with every extra tag being worth 0.25 of the previous one
# E.g. if you have "massively multiplayer" and  "mmorpg" (1 and 1), you get 1 + 0.25 = 1.25
# E.g. if you have "casual", "indie", "cozy" and "moddable" (-0.5, -0.25, -0.5, -0.25), you get -0.5 * 0.25 ^ 0 + -0.5 * 0.25 ^ 1 + -0.25 * 0.25 ^ 2 + -0.25 * 0.25 ^ 3 = -0.64453
df["f2p_tag_score"] = df["tags"].apply(lambda tags: compute_tag_scores(tags))

df.loc[df["monetization_model"] == "f2p", "estimated_ltarpu"] = base_ltarpu_for_estimate * df["f2p_release_years_score"] * df["f2p_tag_score"]
df.loc[df["monetization_model"] == "free", "estimated_ltarpu"] = 0

df["estimated_gross_revenue_boxleiter"] = (df["estimated_owners_boxleiter"] * df["estimated_ltarpu"]).round().astype("Int64")

# Drop the temporary columns
df.drop(columns=["years_since_release", "dislike_ratio", "discount_factor", "f2p_release_years_score", "f2p_tag_score"], inplace=True)

####################################################################
# Data Type Enforcement
####################################################################
# Print the list of all columns
print("Enforcing column types...")
column_type_mapprings = {
    "achievements_count": enforce_int_or_nan_column,
    "average_non_steam_review_score": enforce_float_or_nan_column,
    "average_time_to_beat": enforce_float_or_nan_column,
    "categories": normalize_lists_string_values,
    "controller_support": enforce_string_or_nan_column,
    "developers": normalize_lists_string_values,
    "dlc_count": enforce_int_or_nan_column,
    "early_access": enforce_boolean_or_nan_column,
    "estimated_ltarpu": enforce_float_or_nan_column,
    "estimated_owners_boxleiter": enforce_int_or_nan_column,
    "estimated_gross_revenue_boxleiter": enforce_int_or_nan_column,
    "gamefaqs_difficulty_rating": enforce_string_or_nan_column,
    "genres": normalize_lists_string_values,
    "has_demos": enforce_boolean_or_nan_column,
    "has_drm": enforce_boolean_or_nan_column,
    "languages_supported": normalize_languages_column,
    "languages_with_full_audio": normalize_languages_column,
    "monetization_model": enforce_string_or_nan_column,
    "name": enforce_string_or_nan_column,
    "peak_ccu": enforce_int_or_nan_column,
    "playtime_avg": enforce_float_or_nan_column,
    "playtime_avg_2_weeks": enforce_float_or_nan_column,
    "playtime_median": enforce_float_or_nan_column,
    "playtime_median_2_weeks": enforce_float_or_nan_column,
    "price_latest": enforce_float_or_nan_column,
    "price_original": enforce_float_or_nan_column,
    "publishers": normalize_lists_string_values,
    "recommendations": enforce_int_or_nan_column,
    "release_date": enforce_datetime_column,
    "release_year": enforce_int_or_nan_column,
    "required_age": enforce_int_or_nan_column,
    "runs_on_linux": enforce_boolean_or_nan_column,
    "runs_on_mac": enforce_boolean_or_nan_column,
    "runs_on_steam_deck": enforce_boolean_or_nan_column,
    "runs_on_windows": enforce_boolean_or_nan_column,
    "steam_negative_reviews": enforce_int_or_nan_column,
    "steam_positive_reviews": enforce_int_or_nan_column,
    "steam_positive_review_ratio": enforce_float_or_nan_column,
    "steam_total_reviews": enforce_int_or_nan_column,
    "steam_store_movie_count": enforce_int_or_nan_column,
    "steam_store_screenshot_count": enforce_int_or_nan_column,
    "tags": normalize_lists_string_values,
    "vr_only": enforce_boolean_or_nan_column,
    "vr_supported": enforce_boolean_or_nan_column,
}


# Get tall the columns of the dataset
all_columns = df.columns.tolist()

# For each column_type_mapprings, apply the function to the column and remove it from the list
for column_name, func in column_type_mapprings.items():
    if column_name in all_columns:
        func(df, column_name)
        all_columns.remove(column_name)
    else:
        print(f"❌ Column '{column_name}' not found in the dataset.")

# Raise error if there are any columns left
if all_columns:
    # raise ValueError(f"Unhandled columns (are you missing mappings?): {all_columns}")
    print(f"❌ Unhandled columns (are you missing mappings?): {all_columns}")


####################################################################
# Collapsing pseudo-duplicates
####################################################################
# Many records appear a BUNCH of times for technical reasons, often a single of the enties being the "actual game"
# For example see "Shadow of the Tomb Raider: Definitive Edition"
# We'll collapse these into a single record, keeping the most complete information and highest numerical values

from utils import collapse_pseudo_duplicate_games

df = collapse_pseudo_duplicate_games(df)


####################################################################
# Finalize and save
####################################################################

# Sort columns alphabetically
df = df.reindex(sorted(df.columns), axis=1)

df.index.name = "appid"  # Ensure the index name is correct

# Save results so far to a CSV file
output_filename = "combined_df_preprocessed"
df.to_csv(f"{output_dir}/{output_filename}.csv", index=True)
print(f"Saved cleaned DataFrame to {output_filename}.csv")

# Save also to parquet
df.to_parquet(f"{output_dir}/{output_filename}.parquet")
print(f"Saved cleaned DataFrame to {output_filename}.parquet")


####################################################################
# Special version with far fewer sparse rows
####################################################################
# Drop rows with NA positive reviews
df = df.dropna(subset=["steam_positive_reviews"])

# Drop rows with neither a price_original, price_latest, nor monetization_model
df = df.dropna(subset=["price_original", "price_latest", "monetization_model"], how="all")

# Drop rows without ANY of runs_on_linux, runs_on_mac, runs_on_steam_deck, runs_on_windows
df = df.dropna(subset=["runs_on_linux", "runs_on_mac", "runs_on_steam_deck", "runs_on_windows"], how="all")

# Save results so far to a CSV file
output_filename = "combined_df_preprocessed_dense"
df.to_csv(f"{output_dir}/{output_filename}.csv", index=True)
print(f"Saved cleaned DataFrame to {output_filename}.csv")

# Save also to parquet
df.to_parquet(f"{output_dir}/{output_filename}.parquet")
print(f"Saved cleaned DataFrame to {output_filename}.parquet")

print("Cleaning complete ✅")

subprocess.Popen(f'explorer /select,"{output_dir}\\{output_filename}.parquet"')  # Open the output directory in Windows Explorer
