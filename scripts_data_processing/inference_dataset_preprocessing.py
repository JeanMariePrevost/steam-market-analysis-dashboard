"""
Testing the most "lightweight" version of using 3 pre-trained models to perform blended inference.
"""

# Top 10 Averaged Probability Combinations by QWK:
#   AvgProbs_('lgbm_high', 'lgbm_low', 'xgb'): 0.7528
#   AvgProbs_('lgbm_low', 'xgb'): 0.7495
#   AvgProbs_('lgbm_high', 'xgb'): 0.7495
#   AvgProbs_('lgbm_high', 'lgbm_low', 'xgb', 'catboost'): 0.7488
#   AvgProbs_('lgbm_high', 'lgbm_low', 'catboost'): 0.7481
#   AvgProbs_('lgbm_high', 'xgb', 'catboost'): 0.7412
#   AvgProbs_('lgbm_low', 'xgb', 'catboost'): 0.7410
#   AvgProbs_('lgbm_high', 'catboost'): 0.7410
#   AvgProbs_('lgbm_low', 'catboost'): 0.7409
#   AvgProbs_('lgbm_high', 'lgbm_low', 'xgb', 'et'): 0.7380


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


#################################################################
# Creating a new pre-made, ready-to-be-used dataset just for inference
#################################################################
data = utils.load_feature_engineered_dataset_no_na()

# Remove irrelevant, directly-correlated, and "unavailable at prediction time" columns (e.g. you can't have your review score before the game is released)
columns_to_drop = [
    "achievements_count",
    "appid",
    "average_non_steam_review_score",
    "developers",
    "dlc_count",
    "estimated_gross_revenue_boxleiter",
    "estimated_ltarpu",
    "estimated_owners_boxleiter",
    "name",
    "peak_ccu",
    "playtime_avg",
    "playtime_avg_2_weeks",
    "playtime_median",
    "playtime_median_2_weeks",
    "price_latest",
    "publishers",
    "recommendations",
    "steam_negative_reviews",
    "steam_positive_review_ratio",
    "steam_positive_reviews",
    "steam_total_reviews",
    "~tag_masterpiece",  # Remove fully "outcome based" tags
    "~tag_cult classic",  # Remove fully "outcome based" tag
]

######################################################
# Target variable binning (making it categorical)
######################################################
# Since this is a classification task, we need to convert the target to a categorical variable by using binning.
y = data["steam_total_reviews"]  # Extract the target
bins = [-np.inf, 50, 100, 250, 500, 1000, 2500, 5000, np.inf]
labels = range(len(bins) - 1)  # Create labels (or "values") for the bins, e.g. [0, 1, 2, ...]
y_binned = pd.cut(y, bins=bins, labels=labels)
y = pd.Series(y_binned, name="steam_total_reviews_bin")  # Convert the binned target variable to a Series

# Print some debug info on the binning results
print(f"Target variable (steam_total_reviews) has {len(y.unique())} unique values after binning:")
table = pd.DataFrame({"Bin": [f"({bins[i]}, {bins[i+1]}]" for i in range(len(bins) - 1)], "Count": y.value_counts().sort_index().values})
table["Percentage"] = (table["Count"] / table["Count"].sum() * 100).round(2)
print(table.to_string(index=False))

X = data.drop(columns=columns_to_drop, errors="ignore")

# Rebuild a single dataset to be saved
# This is the dataset that will be used for inference
df = pd.concat([X, y], axis=1)
df = df.dropna()  # Drop rows with NaN values

# save as parquet
df.to_parquet("output_feature_engineered/inference_dataset.parquet", index=False)  # We _want_ the index out this time, bbecause appid is _not_ a good independent variable
print("Inference dataset saved as parquet.")

# Save to csv for debugging purposes
df.to_csv("output_feature_engineered/inference_dataset.csv", index=False)  # We _want_ the index out this time, bbecause appid is _not_ a good independent variable
print("Inference dataset saved as CSV.")
