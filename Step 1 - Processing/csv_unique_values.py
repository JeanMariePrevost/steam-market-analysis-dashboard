"""
This script reads a CSV file and counts the number of unique values in a specified column.
It uses the pandas library to handle CSV file operations and provides error handling for common issues such as file not found or column not existing in the CSV file.
It was required to evaluate the coverage of "less structured" data sets, such as reviews and news.

Usage:
    1. Update the `csv_file` variable with the path to your CSV file.
    2. Update the `column_name` variable with the name of the column you want to analyze.
    3. Run the script to see the number of unique values in the specified column.

"""

import pandas as pd


def count_unique_values(csv_file, target_column_name):
    """
    Opens a CSV file and calculates the number of unique values in a specified column.
    """
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        if target_column_name not in df.columns:
            print(f"ERROR - Column '{target_column_name}' not found in the CSV file.")
            # print the list of columns
            print(f"Columns in the CSV file are: {df.columns}")
            return None

        # Calculate the number of unique values
        unique_values = df[target_column_name].nunique()
        return unique_values
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Specify the CSV file and column name
    path_to_csv_file = (
        # r"F:\OneDrive\MyDocs\Study\TELUQ\Session 8 - Hiver 2025\SCI 1402\datasets\paraglondhe_steamreviews\cleaned.csv"
        r"F:\OneDrive\MyDocs\Study\TELUQ\Session 8 - Hiver 2025\SCI 1402\datasets\reviews\andrewmvd_steam-reviews\dataset.csv"
    )
    target_column_name = "app_id"  # HERE - Change this to the column you want to analyze

    print(f"Analyzing unique values in column '{target_column_name}' of the CSV file '{path_to_csv_file}'...")

    # Calculate and print the number of unique values
    unique_count = count_unique_values(path_to_csv_file, target_column_name)
    if unique_count is not None:
        print(f"The column '{target_column_name}' has {unique_count} unique values.")
