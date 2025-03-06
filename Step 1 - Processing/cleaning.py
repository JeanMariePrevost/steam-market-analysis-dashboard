import json
import random
import pandas as pd
import numpy as np
import ast  # For safely converting string representations of lists
from complex_merge import merge_dataframes_with_mappings, combine_list_unique_values


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
    df[column_name] = pd.to_numeric(df[column_name], errors="coerce").astype("Int64")
    print(f"{column_name} fully numerical (integers) ✅")


def enforce_boolean_or_nan_column(df, column_name):
    """Enforces a column as boolean, squashing all non-boolean values to NaN."""
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' does not exist in the DataFrame.")
    df[column_name] = df[column_name].apply(lambda x: x if x in [True, False] else np.nan).astype("boolean")
    print(f"{column_name} fully boolean ✅")


def enforce_string_or_nan_column(df, column_name):
    """Ensures a column is fully string, coercing invalid values to NaN."""
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' does not exist in the DataFrame.")
    df[column_name] = df[column_name].apply(lambda x: x if isinstance(x, str) else np.nan)
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
    df[column_name] = df[column_name].apply(try_parsing_date)

    # Ensure the column is explicitly datetime64[ns]
    df[column_name] = pd.to_datetime(df[column_name], errors="coerce")

    print(f"{column_name} fully datetime ✅")


def enforce_list_column(df, column_name):
    """
    Ensures every cell of a column is a list<string> or [], squashing all invalid values to [].
    """

    def validate_and_fix(value):
        if pd.isna(value) or value == "":  # Correctly type empty lists
            return []
        else:
            try:
                parsed_value = ast.literal_eval(value)  # Convert stringified lists
                if isinstance(parsed_value, list) and all(isinstance(i, str) for i in parsed_value):
                    return parsed_value  # Successfully converted valid list
            except (ValueError, SyntaxError):
                raise TypeError(f"Invalid value in {column_name}: {repr(value)} (type: {type(value).__name__})")

    # Apply the function to enforce correctness
    df[column_name] = df[column_name].apply(validate_and_fix)

    print(f"{column_name} fully list<string> ✅")


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
# Index
####################################################################
df.set_index("appid", inplace=True)  # Make "appid" the index
df.sort_index(inplace=True)  # Sort by index

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
add_average_column(df, ["gamefaqs_game_length", "hltb_single", "hltb_complete", "igdb_complete", "igdb_single"], "average_time_to_beat")
df.drop(columns=["gamefaqs_game_length", "hltb_single", "hltb_complete", "igdb_complete", "igdb_single"], inplace=True)

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
df["is_released"] = df["release_date"].apply(lambda x: isinstance(x, str) and x.lower() not in ["not released", "coming soon"])

# steam_deck
df.rename(columns={"steam_deck": "runs_on_steam_deck"}, inplace=True)

# Drop columns too weak, irrelevant, or sparse
df.drop(columns=["num_reviews_recent", "num_reviews_total", "pct_pos_recent", "pct_pos_total"], inplace=True)


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

# Turn all text (e.g. unlreleased) and empties into NaN
enforce_float_or_nan_column(df, "price_latest")
enforce_float_or_nan_column(df, "price_original")


#######
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
df["tags"] = df["tags"].apply(lambda lst: replace_in_list(lst, "base building", "base-building"))
df["tags"] = df["tags"].apply(lambda lst: replace_in_list(lst, "dystopian ", "dystopian"))
df["tags"] = df["tags"].apply(lambda lst: replace_in_list(lst, "parody ", "parody"))
df["tags"] = df["tags"].apply(lambda lst: replace_in_list(lst, "puzzle-platformer", "puzzle platformer"))
df["tags"] = df["tags"].apply(lambda lst: replace_in_list(lst, "rogue-like", "roguelike"))
df["tags"] = df["tags"].apply(lambda lst: replace_in_list(lst, "rogue-lite", "roguelite"))


####################################################################
# "Usability" columns to introduce
####################################################################
# Introcud a "release_year" column from "release_date"
enforce_datetime_column(df, "release_date")
df["release_year"] = df["release_date"].dt.year


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
    "controller_support": enforce_boolean_or_nan_column,
    "developers": normalize_lists_string_values,
    "dlc_count": enforce_int_or_nan_column,
    "early_access": enforce_boolean_or_nan_column,
    "estimated_owners": enforce_int_or_nan_column,
    "gamefaqs_difficulty_rating": enforce_string_or_nan_column,
    "genres": normalize_lists_string_values,
    "has_demos": enforce_boolean_or_nan_column,
    "has_drm": enforce_boolean_or_nan_column,
    "is_free": enforce_boolean_or_nan_column,
    "is_released": enforce_boolean_or_nan_column,
    "languages_supported": normalize_languages_column,
    "languages_with_full_audio": normalize_languages_column,
    "name": enforce_string_or_nan_column,
    "peak_ccu": enforce_int_or_nan_column,
    "playtime_avg": enforce_float_or_nan_column,
    "playtime_avg_2_weeks": enforce_float_or_nan_column,
    "playtime_median": enforce_float_or_nan_column,
    "playtime_median_2_weeks": enforce_float_or_nan_column,
    "price_latest": enforce_float_or_nan_column,
    "price_original": enforce_float_or_nan_column,
    "publishers": enforce_list_column,
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
    "steam_spy_estimated_owners": enforce_int_or_nan_column,
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

# Raise error if there are any columns left
if all_columns:
    raise ValueError(f"Unhandled columns (are you missing mappings?): {all_columns}")


####################################################################
# Finalize and save
####################################################################

# Sort columns alphabetically
df = df.reindex(sorted(df.columns), axis=1)

# Save results so far to a CSV file
output_filename = "combined_df_cleaned"
df.to_csv(output_filename + ".csv", index=True)
print(f"Saved cleaned DataFrame to {output_filename}.csv")

# Save also to parquet
df.to_parquet(output_filename + ".parquet")
print(f"Saved cleaned DataFrame to {output_filename}.parquet")

df.to_feather(output_filename + ".feather")
print(f"Saved cleaned DataFrame to {output_filename}.feather")

print("Cleaning complete ✅")
