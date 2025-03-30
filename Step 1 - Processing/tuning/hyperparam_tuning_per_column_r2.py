"""
This script is like the genric optuna variant, but it iterates through each column of the dataset
and runs a short study not to find the best hyperparameters for the model, but to roughly estimate
the R² score that can be achieved by using only that column as a predictor.
(i.e. the rough "predictive power" of each column)
"""

from datetime import datetime

import numpy as np
import optuna
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

import utils

#################################################################
# Prepare the data
#################################################################
data = utils.load_feature_engineered_dataset_no_na()

# data = data.select_dtypes(include=[np.number])  # Keep only numerical columns

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

# Extract the target
y = data["steam_total_reviews"]

# Scale the target
scaler = StandardScaler()
y = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

# Extract the predictors
X = data.drop(columns=columns_to_drop, errors="ignore")

# global variables to store the results
current_study_mean_r2s = []
last_start_time = None
times_to_train = []


#################################################################
# Functions
#################################################################
def objective(trial):
    params = get_trial_params(trial)

    GREEN = "\033[92m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    # Move the cursor to the start and clear the line.
    print("\r\033[2K", end="")

    print(f"{BOLD}{GREEN}Trial {trial.number} of {max_trials}{RESET}...", end="", flush=True)

    # Get current time to measure how long the training takes
    global last_start_time
    last_start_time = datetime.now()

    # Instantiate the CatBoostRegressor with the suggested hyperparameters
    model = model_class(**params, **fixed_params)

    # Evaluate the model using cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)

    # Calculate an "adjusted R²" to penalize models with high standard deviation (meaning they are not stable)
    # This is a simple way to penalize models that might have benefitted from randomness
    mean_r2 = np.mean(scores)
    std_r2 = np.std(scores)
    # Penalize high variance
    variance_penalty = 0.1 * std_r2
    adjusted_r2 = mean_r2 - variance_penalty  # This will be the "score" as far as Optuna is concerned

    global times_to_train
    times_to_train.append((datetime.now() - last_start_time).total_seconds())

    global current_study_mean_r2s
    current_study_mean_r2s.append(mean_r2)

    # Print "Trial X completed in Y seconds (Mean R²: Z) (current best: W)"
    print(f" completed in {times_to_train[-1]:.2f} seconds (Mean R²: {mean_r2:.4f}) (current best: {max(current_study_mean_r2s):.4f})", end="", flush=True)

    return adjusted_r2


#################################################################
# Study
#################################################################
max_trials = 50  # Length of each "mini study"


def get_trial_params(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": 2,
        "min_child_samples": 2,
        "n_estimators": trial.suggest_int("n_estimators", 2, 40),
    }

    return params


model_class = LGBMRegressor

fixed_params = {
    "verbose": -1,
    "random_seed": 42,
}

# Set Optuna verbosity to WARNING, I use my own print_progress_callback to print the results
optuna.logging.set_verbosity(optuna.logging.WARNING)

#################################################################
# "Per column" loop, we create and run a study _per column_ of the dataset
#################################################################
per_column_r2 = {}

GREEN = "\033[92m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

current_column_index_in_df = 0

for current_studied_column in X.columns:
    print("\n" + "-" * 60)
    current_column_index_in_df += 1
    print(f"\n{CYAN}Starting study for column {current_column_index_in_df} of {len(X.columns)}: {current_studied_column}{RESET}")

    # Reset the results variables
    current_study_mean_r2s = []
    last_start_time = None
    times_to_train = []

    # Create a df with only the current target column
    X_filtered = X[[current_studied_column]]
    X_train, X_valid, y_train, y_valid = train_test_split(X_filtered, y, test_size=0.2, random_state=42)

    study = optuna.create_study(direction="maximize")

    try:
        study.optimize(objective, n_trials=max_trials)
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"An error occurred during the {current_studied_column} study: {e}")
        print("Continuing with the next column...")
        continue

    # Print results for column
    best_mean_r2 = max(current_study_mean_r2s)
    print(f"\nBest result {GREEN}{best_mean_r2:.5f}{RESET}")

    per_column_r2[current_studied_column] = best_mean_r2


#################################################################
# Results
#################################################################
# Sort "per column" results
per_column_r2_sorted = {k: v for k, v in sorted(per_column_r2.items(), key=lambda item: item[1], reverse=True)}

# Print them all in a nice clean manner
print("\n\n\n\n")
print("Top columns by adjusted R² score (slightly penalized for variance):")
for column, r2 in per_column_r2_sorted.items():
    print(f"{column}: {r2:.4f}")
