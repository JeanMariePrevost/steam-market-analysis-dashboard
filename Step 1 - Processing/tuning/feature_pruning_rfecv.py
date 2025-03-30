import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

import utils

data = utils.load_feature_engineered_dataset_no_na()
# --- Define features and target ---


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

# Extract the target
y = data["steam_total_reviews"]

# Extract the predictors
X = data.drop(columns=columns_to_drop, errors="ignore")

# --- Initialize a regression model ---
model = LinearRegression()

# --- Setup cross-validation ---
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# --- Initialize RFECV for feature pruning ---
rfecv = RFECV(estimator=model, step=1, cv=cv, scoring="r2", n_jobs=-1, verbose=1)
rfecv.fit(X, y)

# --- Output results ---
print("Optimal number of features:", rfecv.n_features_)
print("Selected features:", X.columns[rfecv.support_].tolist())

# Safety from crashes, save the object to disk using pickle
try:
    import pickle

    with open("rfecv_results.pkl", "wb") as f:
        pickle.dump(rfecv, f)
except ImportError:
    print("Pickle not available, skipping saving the object to disk")
except Exception as e:
    print(f"An error occurred: {e}")

# --- Plot RFECV results ---
# Retrieve mean test scores from cv_results_
scores = rfecv.cv_results_["mean_test_score"]

plt.figure(figsize=(8, 6))
plt.xlabel("Number of Features Selected")
plt.ylabel("Cross-validation Score (R²)")
plt.plot(range(1, len(scores) + 1), scores, marker="o")
plt.title("RFECV Feature Selection")
plt.show()
