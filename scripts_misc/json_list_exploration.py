"""
This script processes a JSON file, extracts a sample of random elements,
flattens the nested structures, and saves the result to a CSV file.
It mirrors json_dict_exploration.py but for JSON files that are lists instead of dictionaries at the root level.
"""

import json
import random
import pandas as pd

# Load the JSON file
# file_path = r"F:\OneDrive\MyDocs\Study\TELUQ\Session 8 - Hiver 2025\SCI 1402\datasets\sujaykapadnis_games-on-steam\steamdb.json"
file_path = r"F:\OneDrive\MyDocs\Study\TELUQ\Session 8 - Hiver 2025\SCI 1402\datasets\sujaykapadnis_games-on-steam\steamdb.min.json"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)  # Now expecting data to be a list

# Peek some info about the JSON structure
print(f"The root is a list with {len(data)} elements.")

# DEBUG - Peek the first 5 elements' structure
# if len(data) > 0:
#     print(f"First element keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'Not a dictionary'}")


# Function to flatten a dictionary
def flatten_json_list(d, parent_key="", sep=".", max_depth=1, current_depth=0):
    """
    Recursively flattens a dictionary up to a specified depth.
    Beyond the max_depth, nested structures are represented as strings or summaries.
    """
    items = []
    if not isinstance(d, dict):
        return {"value": str(d)}

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict) and current_depth < max_depth:
            # Recurse into the dictionary if within depth limit
            items.extend(flatten_json_list(v, new_key, sep, max_depth, current_depth + 1).items())
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


# Select up to N random elements
num_samples = min(10, len(data))
random_elements = random.sample(data, num_samples)

# Flatten all sampled elements
flattened_elements = [flatten_json_list(elem, max_depth=0) for elem in random_elements]

# Convert the flattened elements into a DataFrame
df = pd.DataFrame(flattened_elements)

# Write the DataFrame to a CSV file
csv_file = "sampled_data.csv"
df.to_csv(csv_file, index=False, encoding="utf-8")

print(f"CSV file '{csv_file}' created successfully.")
