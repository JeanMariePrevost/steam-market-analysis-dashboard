"""
This script processes a JSON file, extracts a sample of random elements,
flattens the nested structures, and saves the result to a CSV file.
I use it to peek into the structure of more complex JSON files and sample data for early analysis.
"""

import json
import random
import pandas as pd


# Load the JSON file
file_path = (
    # r"F:\OneDrive\MyDocs\Study\TELUQ\Session 8 - Hiver 2025\SCI 1402\datasets\fronkongames_steam-games-dataset\games.json"  # Replace with your JSON file path
    # r"F:\OneDrive\MyDocs\Study\TELUQ\Session 8 - Hiver 2025\SCI 1402\datasets\sujaykapadnis_games-on-steam\steamdb.json"  # Replace with your JSON file path
    r"F:\OneDrive\MyDocs\Study\TELUQ\Session 8 - Hiver 2025\SCI 1402\datasets\fronkongames_steam-games-dataset\games.json"  # Replace with your JSON file path
)
with open(file_path, "r", encoding="utf-8") as f:
    data: dict = json.load(f)

# Peek some info about the JSON structure
print(f"The root is a dictionary with {len(data.keys())} keys.")

# DEBUG - Peek the first 5 keys
# print(f"The first 5 keys are: {list(data.keys())[:5]}")


def flatten_dict(source_dict, parent_key="", sep=".", max_depth=1, current_depth=0):
    """
    Recursively flattens a dictionary up to a specified depth.
    Beyond the max_depth, nested structures are represented as strings or summaries. (Otherwise fields like tags unwrap to a ton of columns)

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The base key for nested keys.
        sep (str): Separator for nested keys.
        max_depth (int): Maximum depth to flatten.
        current_depth (int): Current depth of recursion.

    Returns:
        dict: A flattened dictionary.
    """
    items = []
    for k, v in source_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict) and current_depth < max_depth:
            # Recurse into the dictionary if within depth limit
            items.extend(flatten_dict(v, new_key, sep, max_depth, current_depth + 1).items())
        elif isinstance(v, dict):
            # Beyond max depth, represent as a string
            items.append((new_key, f"NestedDict[{len(v)} keys]"))
        elif isinstance(v, list):
            # Summarize lists
            if len(v) > 0 and isinstance(v[0], dict):
                items.append((new_key, f"List[{len(v)}]: {v[:1]}..."))  # Show first element
            else:
                items.append((new_key, f"List[{len(v)}]: {v[:3]}..."))  # Show first 3 elements
        else:
            # Base case: add the key-value pair
            items.append((new_key, v))
    return dict(items)


# # Select up to N random elements
# num_samples = min(10, len(data))
# random_elements = [data[key] for key in random.sample(list(data.keys()), num_samples)]

# # Flatten all sampled elements
# flattened_elements = [flatten_dict(elem, max_depth=0) for elem in random_elements]

# # Convert the flattened elements into a DataFrame
# df = pd.DataFrame(flattened_elements)

# # Write the DataFrame to a CSV file
# csv_file = "sampled_data.csv"
# df.to_csv(csv_file, index=False, encoding="utf-8")
# print(f"CSV file '{csv_file}' created successfully.")

#
#
#
#
#

# Convert the entire dataset to a DataFrame by iterating across all rows manually and printing progress
all_rows = []
for i, key in enumerate(data.keys()):
    flattened_data = flatten_dict(data[key], max_depth=0)
    flattened_data["appid"] = key  # Add the key as a column
    all_rows.append(flattened_data)  # Append dictionary to a list

    if i % 100 == 0:
        print(f"Processed {i}/{len(data)} rows. ({i / len(data) * 100:.2f}%)")

# Convert list of dictionaries into a DataFrame in one step
print("Converting to DataFrame...")
df_all = pd.DataFrame(all_rows)

# Write the DataFrame to a CSV file
print("Writing to CSV file...")
csv_file_all = "json_data.csv"
df_all.to_csv(csv_file_all, index=False, encoding="utf-8")

print(f"CSV file '{csv_file_all}' created successfully.")
