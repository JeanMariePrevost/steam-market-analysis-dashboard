# Debug, load inference dataset and print a few infos about it
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Allow importing modules from the parent directory of this script
current_file = Path(__file__).resolve()
parent_dir = current_file.parent.parent
sys.path.insert(0, str(parent_dir))

import utils

inf_df = utils.load_inference_dataset()
print(f"Loaded inference dataset with {len(inf_df)} rows and {len(inf_df.columns)} columns.")
# Has a "steam_total_reviews_bin" column? (the target variable)
print(f"Target variable (steam_total_reviews_bin) has {len(inf_df['steam_total_reviews_bin'].unique())} unique values.")
print(f"Target variable (steam_total_reviews_bin) has {len(inf_df['steam_total_reviews_bin'].value_counts())} unique values after binning:")


## Creating the "baseline" elements for the inference dataset
# First, create the "Series" for the baseline element
baseline_element = pd.Series()

# EVERY SINGLE COLUMNS that starts with a "~" is one-hot, so we make the baseline their "mode"
for col in inf_df.columns:
    if col.startswith("~"):
        baseline_element[col] = inf_df[col].mode()[0]  # [0] is to pick the "first" mode, in case there are multiple modes

# Print number of columns in the baseline element so far
print(f"Baseline element has {len(baseline_element)} columns.")

# Debug, drop those and save the resulting df as "inference_dataset_step_1.csv"
inf_df = inf_df.drop(columns=[col for col in inf_df.columns if col.startswith("~")])
# inf_df.to_csv("feature_engineered_output/inference_dataset_step_1.csv", index=False)
# print("Inference dataset step 1 saved as csv.")


# For each remaining column, print the number of unique values along with its name
for col in inf_df.columns:
    mode = inf_df[col].mode()[0]  # Get the mode of the column
    mean = inf_df[col].mean()  # Get the mean of the column
    median = inf_df[col].median()  # Get the median of the column
    print(f"[{col}] has {len(inf_df[col].unique())} unique values, mode: {mode}, mean: {mean}, median: {median}")

median_columns = [
    "average_time_to_beat",
    "price_original",
]

ordinal = [
    "categories_count",
    "gamefaqs_difficulty_rating",
    "genres_count",
    "languages_supported_count",
    "languages_with_full_audio_count",
    "steam_store_movie_count",
    "steam_store_screenshot_count",
    "tags_count",
    "steam_total_reviews_bin",
]

categorical = [
    "controller_support",
    "early_access",
    "has_demos",
    "has_drm",
    "monetization_model",
    "release_day_of_month",
    "release_month",
    "required_age",
    "runs_on_linux",
    "runs_on_mac",
    "runs_on_steam_deck",
    "runs_on_windows",
    "vr_only",
    "vr_supported",
]

for col in median_columns:
    baseline_element[col] = inf_df[col].median()

for col in ordinal:
    # Rounded median
    baseline_element[col] = round(inf_df[col].median())

for col in categorical:
    baseline_element[col] = inf_df[col].mode()[0]  # [0] is to pick the "first" mode, in case there are multiple modes


# Special cases
# release_year will be 2025 since that,s the current target
baseline_element["release_year"] = 2025

# Drop steam_total_reviews_bin
print(f"Baseline element has {len(baseline_element)} columns.")
baseline_element = baseline_element.drop("steam_total_reviews_bin", errors="ignore")
print(f"Baseline element has {len(baseline_element)} columns.")

# Ensure baseline has the same columns as the inference dataset
inf_df = utils.load_inference_dataset()  # Reload to ensure we have the full df
inf_df = inf_df.drop(columns="steam_total_reviews_bin", errors="ignore")
if len(baseline_element) != len(inf_df.columns):
    print(f"Warning: Baseline element has {len(baseline_element)} columns, but inference dataset has {len(inf_df.columns)} columns.")

for col in inf_df.columns:
    if col not in baseline_element.index:
        baseline_element[col] = 0  # or np.nan, depending on your needs


# Transpose to make it a single row DataFrame
baseline_element = baseline_element.to_frame().T

# Print result
print("Baseline element:")
print(baseline_element)


# Save the baseline element
filename = "baseline_element.parquet"
filepath = os.path.join("output_feature_engineered", filename)
baseline_element.to_parquet(filepath, index=False)  # Save as parquet


# Load and print the baseline element to verify it was saved correctly
loaded_baseline_element = utils.load_baseline_element()
print("Loaded baseline element:")
print(loaded_baseline_element)

# Both are fully equal?
print("Are the loaded and original baseline elements equal?")
print(loaded_baseline_element.equals(baseline_element))


# Test inference to see if it indeed looks like a "baseline" prediction
MODEL_NAME_LGBM_HIGH = "lgbm_high"
MODEL_NAME_LGBM_LOW = "lgbm_low"
MODEL_NAME_XGB = "xgb"


def load_trained_model(model_name):
    models_output_dir = "saved_multiclass_models"
    filename = f"{model_name}_classifier_qwk.joblib"
    filepath = os.path.join(models_output_dir, filename)
    try:
        model = joblib.load(filepath)
        print(f"Model loaded successfully from: {filepath}")
        return model
    except FileNotFoundError:
        print(f"Model file not found: {filepath}. Training new model.")
        return None
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None


model_lgbm_high = load_trained_model(MODEL_NAME_LGBM_HIGH)
model_lgbm_low = load_trained_model(MODEL_NAME_LGBM_LOW)
model_xgb = load_trained_model(MODEL_NAME_XGB)


# Run inference on the baseline element
print(f"Baseline DataFrame shape: {baseline_element.shape}")
print(baseline_element.head())

# Get all soft probs for each model
pred_lgbm_high = model_lgbm_high.predict_proba(baseline_element)
pred_lgbm_low = model_lgbm_low.predict_proba(baseline_element)
pred_xgb = model_xgb.predict_proba(baseline_element)


# Print predictions
print(f"Predictions for {MODEL_NAME_LGBM_HIGH}: {pred_lgbm_high}")
print(f"Predictions for {MODEL_NAME_LGBM_LOW}: {pred_lgbm_low}")
print(f"Predictions for {MODEL_NAME_XGB}: {pred_xgb}")

# Blend
# Blend the predictions using a simple average
predictions = np.array([pred_lgbm_high, pred_lgbm_low, pred_xgb])
average_prediction = np.mean(predictions, axis=0)
print(f"Average prediction: {average_prediction}")


print(f"Sum of average prediction: {np.sum(average_prediction)}")
# Normalize the average prediction to ensure it sums to 1
average_prediction /= np.sum(average_prediction)
print(f"Sum of average prediction: {np.sum(average_prediction)}")


# Show as a histogram
import matplotlib.pyplot as plt

values = average_prediction.flatten()
print(f"Values: {values}")

# Define x-axis labels
bins = [0, 50, 100, 250, 500, 1000, 2500, 5000, np.inf]
categories = []
for i in range(len(bins) - 1):
    if bins[i] == 0:
        categories.append(f"< {bins[i + 1]}")
    elif bins[i + 1] == np.inf:
        categories.append(f">= {bins[i]}")
    else:
        categories.append(f"{bins[i]} - {bins[i + 1]}")
print(f"Categories: {categories}")

# Sanity check, load the inference dataset and build the same plot showing the _count_ of each unique value of steam_total_reviews_bin
inf_df = utils.load_inference_dataset()
print(f"Loaded inference dataset with {len(inf_df)} rows and {len(inf_df.columns)} columns.")
print(f"Target variable (steam_total_reviews_bin) has {len(inf_df['steam_total_reviews_bin'].unique())} unique values.")

proportions = inf_df["steam_total_reviews_bin"].value_counts(normalize=True)
proportions = proportions.sort_index()  # Sort the proportions to match the order of categories
print(f"Actual proportions: {proportions}")


# Create side-by-side subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left subplot: Average predictions probabilities
ax1.bar(categories, values)
ax1.set_xlabel("Category")
ax1.set_ylabel("Probability")
ax1.set_title("Bar Chart of Average Predictions")
ax1.set_xticklabels(categories, rotation=45, ha="right")

# Right subplot: Inference dataset counts (proportions)
ax2.bar(proportions.index.astype(str), proportions.values)
ax2.set_xlabel("Category")
ax2.set_ylabel("Proportion")
ax2.set_title("Bar Chart of Inference Dataset Counts")
ax2.set_xticklabels(proportions.index.astype(str), rotation=45, ha="right")

plt.tight_layout()
plt.show()
