"""
This script allows you to optimize any model that taks in nothing but a set of hyperparameters and returns a score.
It uses Optuna to optimize the hyperparameters.
It does not support models that require custom logic beyond a search space and model type.
It uses a "
"""

from datetime import datetime

import numpy as np
import optuna
from catboost import CatBoostRegressor
from optuna.trial import TrialState
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

# Extract the predictors
X = data.drop(columns=columns_to_drop, errors="ignore")

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the target variable
scaler = StandardScaler()
y_train = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_valid = scaler.transform(y_valid.values.reshape(-1, 1)).flatten()

# Dict to store trials results by params
trials_results = {}
latest_trial_results = None
last_start_time = None
times_to_train = []


#################################################################
# Functions
#################################################################
def objective(trial):
    params = get_trial_params(trial)
    print(f"Starting trial {trial.number} of {number_of_trials} with params: {params}")

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
    times_to_train.append(datetime.now() - last_start_time)

    # Create a key from the params to store the results
    key = str(params)
    trials_results[key] = {
        "mean_r2": mean_r2,
        "std_r2": std_r2,
        "variance_penalty": variance_penalty,
        "adjusted_r2": adjusted_r2,
        "trial_number": trial.number,
        "time_to_train": times_to_train[-1],
        "trial_object": trial,
    }

    global latest_trial_results
    latest_trial_results = trials_results[key]

    return adjusted_r2


def print_progress_callback(study, trial):
    # Sort the results by adjusted R²
    trials_results_sorted = {k: v for k, v in sorted(trials_results.items(), key=lambda item: item[1]["adjusted_r2"], reverse=True)}
    best_trial_results = list(trials_results_sorted.values())[0]

    # ANSI color codes
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    is_new_best = latest_trial_results["adjusted_r2"] == best_trial_results["adjusted_r2"]
    best_adjusted_r2_tag = f"{BOLD}{GREEN}(NEW BEST){RESET}"

    if not is_new_best:
        best_adjusted_r2_tag = f"{GRAY}(best: {best_trial_results['adjusted_r2']:.3f}){RESET}"

    duration = (datetime.now() - last_start_time).total_seconds()
    median_duration = np.median([t.total_seconds() for t in times_to_train])
    median_penalty = np.median([r["variance_penalty"] for r in trials_results_sorted.values()])
    estimated_seconds_remaining = (number_of_trials - trial.number) * median_duration
    estimated_hours_remaining = estimated_seconds_remaining / 3600

    print(
        f"\n{BOLD}{CYAN}=== Trial {trial.number} of {number_of_trials} ==={RESET}\n"
        f"{YELLOW}  {'Adjusted R²:':20} {latest_trial_results['adjusted_r2']:> 8.3f}   {best_adjusted_r2_tag}\n"
        f"{YELLOW}  {'Mean Raw R²:':20} {latest_trial_results['mean_r2']:> 8.3f}   {GRAY}(best: {best_trial_results['mean_r2']:.3f}){RESET}\n"
        f"{YELLOW}  {'Variance Penalty:':20} {latest_trial_results['variance_penalty']:> 8.3f}   {GRAY}(median: {median_penalty:.3f}){RESET}\n"
        f"{YELLOW}  {'Time to train:':20} {duration:> 7.2f}s   {GRAY}(median: {median_duration:.2f}s){RESET}\n"
        f"\n{GRAY}  ({'Estimated time remaining:':20} {estimated_seconds_remaining:.2f}s, or {estimated_hours_remaining:.2f}h){RESET}\n"
    )

    # Append latest results to a file, using the model as part of the name
    full_latest_result_log_string = f"Score: {latest_trial_results['adjusted_r2']:.3f}, Mean R²: {latest_trial_results['mean_r2']:.3f}, Time to train: {duration:.2f}s, Params: {str(latest_trial_results['trial_object'].params)}\n"
    full_best_result_log_string = (
        f"Score: {best_trial_results['adjusted_r2']:.3f}, Mean R²: {best_trial_results['mean_r2']:.3f}, Time to train: {duration:.2f}s, Params: {str(best_trial_results['trial_object'].params)}\n"
    )

    model_name = model_class.__name__
    with open(f"tuning_results_{model_name}.txt", "a", encoding="utf-8") as f:
        f.write(full_latest_result_log_string)

    print("Best so far:")
    print(full_best_result_log_string)
    print("-" * 60)


#################################################################
# Study
#################################################################
number_of_trials = 500  # Length of study, stored in a vraiable to be able to print as we progress


def get_trial_params(trial):
    return {
        "iterations": trial.suggest_int("iterations", 20, 40),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
        "depth": trial.suggest_int("depth", 2, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 1.0, log=True),
        "loss_function": "RMSE",
        "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 2, 40),
        "subsample": trial.suggest_float("subsample", 0.01, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 120),
    }


model_class = CatBoostRegressor

fixed_params = {
    "verbose": 0,
    "random_seed": 42,
}

# Set Optuna verbosity to WARNING, I use my own print_progress_callback to print the results
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Use Hyperband as the sampler for Optuna
# This speeds up the optimization process by stopping bad trials early (?)
pruner = optuna.pruners.HyperbandPruner()

# Create an Optuna study to minimize/maximize the objective
study = optuna.create_study(direction="maximize", pruner=pruner)

# Optimize the objective over N trials
try:
    study.optimize(objective, n_trials=number_of_trials, callbacks=[print_progress_callback])
except KeyboardInterrupt:
    print("Interrupted by user")


#################################################################
# Results
#################################################################
# Sort results by adjusted R²
trials_results_sorted = {k: v for k, v in sorted(trials_results.items(), key=lambda item: item[1]["adjusted_r2"], reverse=True)}

# Print the 10 best results as adjusted R², mean R2, and params
print("Top 10 best results:")
for i, (params, results) in enumerate(list(trials_results_sorted.items())[:10]):
    print(f"{i+1}. Adjusted R²: {results['adjusted_r2']:.3f}, Mean R²: {results['mean_r2']:.3f} -> Params: {params}")

# Print hyperparameters importance
importances = optuna.importance.get_param_importances(study)
print("Hyperparameters importance:")
print(importances)

print("Visualizing importances:")
fig = optuna.visualization.plot_param_importances(study)
fig.show()

print("Visualizing slice:")
fig = optuna.visualization.plot_slice(study)
fig.show()

print("Visualizing parallel coordinate:")
fig = optuna.visualization.plot_parallel_coordinate(study)
fig.show()

print("Visualizing contour:")
fig = optuna.visualization.plot_contour(study)
fig.show()

print("Visualizing optimization history:")
fig = optuna.visualization.plot_optimization_history(study)
fig.show()
