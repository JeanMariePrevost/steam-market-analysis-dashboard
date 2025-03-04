import json
import random
import pandas as pd
import numpy as np
import ast  # For safely converting string representations of lists
from complex_merge import merge_dataframes_with_mappings, combine_list_unique_values


####################################################################
# Helper functions
####################################################################
def enforce_numerical_column(df, column_name):
    """Enforces a columns as numerical, squashing all non-numeric values to NaN, and validates the column."""
    df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
    is_valid = df[column_name].dropna().apply(lambda x: isinstance(x, (int, float))).all()
    print(f"{column_name} fully numerical: {is_valid} ✅")


def enforce_boolean_column(df, column_name):
    """Enforces a columns as boolean, squashing all non-boolean values to NaN"""
    df[column_name] = df[column_name].apply(lambda x: x if x in [True, False] else np.nan)
    print(f"{column_name} fully boolean ✅")


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
        if pd.isna(value) or value == "" or value.lower() in ["not released", "not_released", "coming soon", "coming_soon", "unknown"]:
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

    df[column_name] = df[column_name].apply(try_parsing_date)

    print(f"{column_name} fully datetime ✅")


def validate_list_or_nan_column(df, column_name):
    """
    Ensures every cell of a column is a list<string> or NaN
    """

    def validate_and_fix(value):
        if pd.isna(value):  # Allow NaN values
            return value
        else:
            try:
                parsed_value = ast.literal_eval(value)  # Convert stringified lists
                if isinstance(parsed_value, list) and all(isinstance(i, str) for i in parsed_value):
                    return parsed_value  # Successfully converted valid list
            except (ValueError, SyntaxError):
                raise TypeError(f"Invalid value in {column_name}: {repr(value)} (type: {type(value).__name__})")

    # Apply the function to enforce correctness
    df[column_name] = df[column_name].apply(validate_and_fix)

    print(f"{column_name} fully list<string> or NaN ✅")


def normalize_lists_strings(df, column_name, force_lowercase=True, remove_duplicate_elements=True, sort_elements=True):
    """
    Normalizes all string lists in a column by optionally forcing lowercase and removing duplicates.
    """

    def normalize_cell(value):
        if isinstance(value, list):
            value_as_list = value
        elif pd.isna(value):
            return pd.NA
        else:
            try:
                value_as_list = ast.literal_eval(value)
                if not isinstance(value_as_list, list):
                    raise ValueError(f"Value is not a list: {repr(value)}")
                if not all(isinstance(x, str) for x in value_as_list):
                    raise ValueError(f"List contains non-string elements: {repr(value)}")
                return value_as_list
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

    df[column_name] = df[column_name].apply(normalize_cell)
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
        elif pd.isna(value):
            return pd.NA
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

        # Empty lists should be NA
        if not normalized_languages or len(normalized_languages) == 0:
            return pd.NA

        # Remove duplicates
        normalized_languages = list(set(normalized_languages))

        normalized_languages.sort()
        return normalized_languages

    df[column_name] = df[column_name].apply(normalize_cell)
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


####################################################################
# Cleaning begins here
####################################################################

df = pd.read_csv(r"F:\OneDrive\MyDocs\Study\TELUQ\Session 8 - Hiver 2025\SCI 1402\Step 1 - Processing\combined_df_step9.csv")

####################################################################
# Assign and validate index
####################################################################
df.set_index("appid", inplace=True)  # Make "appid" the index
df.sort_index(inplace=True)  # Sort by index
print(f"Number of duplicates appids: {df.index.duplicated().sum()}")  # Ensure no duplicate index

####################################################################
# Per column cleaning and validation
####################################################################
# achivements
df.rename(columns={"achievements": "achievements_count"}, inplace=True)
enforce_numerical_column(df, "achievements_count")


# playtime columns (avg, median...)
df.rename(columns={"average_playtime_forever": "playtime_avg"}, inplace=True)
df.rename(columns={"average_playtime_2weeks": "playtime_avg_2_weeks"}, inplace=True)
df.rename(columns={"median_playtime_forever": "playtime_median"}, inplace=True)
df.rename(columns={"median_playtime_2weeks": "playtime_median_2_weeks"}, inplace=True)
enforce_numerical_column(df, "playtime_avg")
enforce_numerical_column(df, "playtime_avg_2_weeks")
enforce_numerical_column(df, "playtime_median")
enforce_numerical_column(df, "playtime_median_2_weeks")

# categories, ensure it's lists or nan
validate_list_or_nan_column(df, "categories")
# DEbug, print head of categories
normalize_lists_strings(df, "categories")

# controller_support, ensure all values one of "none", "partial", "full", otherwise NaN for unknown
df["controller_support"] = df["controller_support"].apply(lambda x: x if x in ["none", "partial", "full"] else np.nan)
print("controller_support fully valid ✅")

# developers, ensure it's lists or nan
validate_list_or_nan_column(df, "developers")
normalize_lists_strings(df, "developers")

# dlc_count, enforce numerical
enforce_numerical_column(df, "dlc_count")

# early_access, ensure all values are either "True", "False", or NaN
enforce_boolean_column(df, "early_access")

# estimated_owners, enforce numerical
enforce_numerical_column(df, "estimated_owners")

# languages stuff
df.rename(columns={"supported_languages": "languages_supported"}, inplace=True)
df.rename(columns={"full_audio_languages": "languages_with_full_audio"}, inplace=True)
validate_list_or_nan_column(df, "languages_supported")
normalize_languages_column(df, "languages_supported")
validate_list_or_nan_column(df, "languages_with_full_audio")
normalize_languages_column(df, "languages_with_full_audio")

# Debug, print all unique values in "gamefaqs_difficulty_rating"
print(df["gamefaqs_difficulty_rating"].unique())

# gamefaqs_difficulty_rating already valid
enforce_numerical_column(df, "gamefaqs_game_length")
enforce_numerical_column(df, "gamefaqs_review_score")

# genres, ensure it's lists or nan
validate_list_or_nan_column(df, "genres")
normalize_lists_strings(df, "genres")

enforce_boolean_column(df, "has_demos")
enforce_boolean_column(df, "has_drm")

# "Time to beat" stuff
# gamefaqs_game_length
# hltb_single
# hltb_complete
# igdb_complete
# igdb_single
add_average_column(df, ["gamefaqs_game_length", "hltb_single", "hltb_complete", "igdb_complete", "igdb_single"], "average_time_to_beat")
df.drop(columns=["gamefaqs_game_length", "hltb_single", "hltb_complete", "igdb_complete", "igdb_single"], inplace=True)
enforce_numerical_column(df, "average_time_to_beat")

# External review scores
# gamefaqs_review_score (out of 5!)
# igdb_review_score (out of 100)
# metacritic_score (out of 100)

# Normalize each rating field to a 0-1 range
df["gamefaqs_review_score"] /= 5
df["igdb_review_score"] /= 100
df["metacritic_score"] /= 100

add_average_column(df, ["gamefaqs_review_score", "igdb_review_score", "metacritic_score"], "average_non_steam_review_score")
df.drop(columns=["gamefaqs_review_score", "igdb_review_score", "metacritic_score"], inplace=True)
enforce_numerical_column(df, "average_non_steam_review_score")

enforce_boolean_column(df, "is_free")

# platforms stuff
df.rename(columns={"linux": "runs_on_linux"}, inplace=True)
df.rename(columns={"mac": "runs_on_mac"}, inplace=True)
df.rename(columns={"windows": "runs_on_windows"}, inplace=True)
enforce_boolean_column(df, "runs_on_linux")
enforce_boolean_column(df, "runs_on_mac")
enforce_boolean_column(df, "runs_on_windows")

# movies and screenshots
df.rename(columns={"screenshots": "steam_store_screenshot_count"}, inplace=True)
enforce_numerical_column(df, "steam_store_screenshot_count")
df.rename(columns={"movies": "steam_store_movie_count"}, inplace=True)
enforce_numerical_column(df, "steam_store_movie_count")

# Steam reviews
df.rename(columns={"negative": "steam_negative_reviews"}, inplace=True)
df.rename(columns={"positive": "steam_positive_reviews"}, inplace=True)
enforce_numerical_column(df, "steam_negative_reviews")
enforce_numerical_column(df, "steam_positive_reviews")

# peak_ccu
enforce_numerical_column(df, "peak_ccu")


# publishers
validate_list_or_nan_column(df, "publishers")
normalize_lists_strings(df, "publishers")

# recommendations
enforce_numerical_column(df, "recommendations")

# release_date stuff (we have YYYY-MM-DD format, "Feb 27, 2018" format, some with time (2020-04-05 00:00:00), and "Not Released", "coming_soon" and "unknown")...
# First, extract "is_released" bool column from "release_date"
df["is_released"] = df["release_date"].apply(lambda x: isinstance(x, str) and x.lower() not in ["not released", "coming soon"])
enforce_datetime_column(df, "release_date")
enforce_boolean_column(df, "is_released")

enforce_numerical_column(df, "required_age")

# steam_deck
df.rename(columns={"steam_deck": "runs_on_steam_deck"}, inplace=True)
enforce_boolean_column(df, "runs_on_steam_deck")

# Drop columns too weak, irrelevant, or sparse
df.drop(columns=["num_reviews_recent", "num_reviews_total", "pct_pos_recent", "pct_pos_total"], inplace=True)

# steam_spy_estimated_owners
enforce_numerical_column(df, "steam_spy_estimated_owners")

# tags
validate_list_or_nan_column(df, "tags")
normalize_lists_strings(df, "tags")

enforce_boolean_column(df, "vr_only")
enforce_boolean_column(df, "vr_supported")


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

# Peak at the columns
print(df.columns)

# Peak at the first 10 "price_latest" and "price_original" values
print(df[["price_latest", "price_original"]].head(10))

# Turn all text (e.g. unlreleased) and empties into NaN
enforce_numerical_column(df, "price_latest")
enforce_numerical_column(df, "price_original")


####################################################################
# Finalize and save
####################################################################

# Sort columns alphabetically
df = df.reindex(sorted(df.columns), axis=1)

# Save results so far to a CSV file
output_file = "combined_df_cleaned.csv"
df.to_csv(output_file)
