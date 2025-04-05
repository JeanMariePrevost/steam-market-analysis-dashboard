"""
Same as hyperparam_tuning_generic.py, but transforms the data and seems to get FAR better results.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from optuna.trial import TrialState
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import FunctionTransformer, make_pipeline
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Add the parent directory of the script to sys.path, cause since we moved the script to a new folder, it can't find the utils module
# This is a hack, but it works good enough for a quick and dirty script
sys.path.append(str(Path(__file__).resolve().parent.parent))

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
    "~tag_cult classic",  # Remove fully "outcome based" tags
]

# For slower models or initial exploration, consider using only a small subsample of the data
# Even 0.05 seems to be enough to get a good idea of the model's performance
fraction = 0.05  # Fraction of the data to use for training
if fraction <= 0.1:
    # Very small sample, filter out the outliers, very weakly at the top, heavily at the bottom
    data = data[data["steam_total_reviews"] > 0]  # Remove games with 0-1 reviews
    top_games_to_remove = 25
    data = data.sort_values("steam_total_reviews", ascending=False).iloc[top_games_to_remove:]  # Remove the top 25 games because if they are in the sample, they will skew the results


y = data["steam_total_reviews"]  # Extract the target


#################################
# Testing target scaling
#################################
# StandardScaler, mean=0, std=1, does NOTHING for skewness
# scaler = StandardScaler()
# y = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

# QuantileTransformer to get a normal distribution output, very good for distributing the data points while maintaining the relative distances
# NOT as effective as PowerTransformer with its far more noraml distribution
# target_transformer = QuantileTransformer(output_distribution="normal")
# y = target_transformer.fit_transform(y.values.reshape(-1, 1)).flatten()

# Uniform, seems slightly better than QuantileTransformer with normal
target_transformer = QuantileTransformer(n_quantiles=10000, output_distribution="uniform", random_state=42)
y = target_transformer.fit_transform(y.values.reshape(-1, 1)).flatten()

# Forces the data into a normal distribution much better, but loses the relative distances. Consider comparing error in _original_ scale to truly compare?
# target_transformer = PowerTransformer(method="yeo-johnson")
# y = target_transformer.fit_transform(y.values.reshape(-1, 1)).flatten()

# target_transformer = PowerTransformer(method="box-cox")
# y = target_transformer.fit_transform(y.values.reshape(-1, 1)).flatten()


# Create a pipeline that first applies log1p and then RobustScaler.
# target_transformer = make_pipeline(FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=True), RobustScaler())
# y = target_transformer.fit_transform(y.values.reshape(-1, 1)).flatten()


# skewness = pd.Series(y).skew()
# print(f"Skewness: {skewness:.2f}")

# kurtosis = pd.Series(y).kurtosis()
# print(f"Kurtosis: {kurtosis:.2f}")

# import matplotlib.pyplot as plt

# plt.hist(y, bins=30, edgecolor="k")
# plt.title("Distribution after Transformation")
# plt.xlabel("Transformed Value")
# plt.ylabel("Frequency")
# plt.show()


#################################
# Testing predictors scaling (no gains?)
#################################
X = data.drop(columns=columns_to_drop, errors="ignore")  # Extract the predictors, dropping the irrelevant columns and the target

# All remaining columns ARE numerical, no need to filter anything
# Create a QuantileTransformer to map the numerical values to a uniform [0, 1] distribution.
qt = QuantileTransformer(output_distribution="uniform")
X = qt.fit_transform(X)


y_strata_bins = pd.cut(y, bins=10, labels=False)  # Create strata for stratified sampling to, e.g. not to end up with most of the big hits in either set, or only failed games in the test set


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y_strata_bins)

# # Scale the target variable
# scaler = StandardScaler()
# y_train = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
# y_valid = scaler.transform(y_valid.values.reshape(-1, 1)).flatten()

# Dict to store trials results by params
trials_results = {}
latest_trial_results = None
latest_trial_params = None
last_start_time = None
times_to_train = []
trials_since_last_improvement = 0
last_improvement_amount = 0.0


#################################################################
# Functions
#################################################################
def objective(trial):
    params = get_trial_params(trial)

    GREEN = "\033[92m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    print(f"\n{BOLD}{GREEN}Starting trial {trial.number} of {max_trials}{RESET} with params: {params}")
    print(f"You can safely interrupt at any point with {CYAN}Ctrl+C{RESET} and get the best results so far.")

    # Get current time to measure how long the training takes
    global last_start_time
    last_start_time = datetime.now()

    # Instantiate the CatBoostRegressor with the suggested hyperparameters
    model = model_class(**params)

    # Wrap your regressor in TransformedTargetRegressor to automatically invert the transformation on predictions
    # This is important to get the correct predictions and to ensure R^2 is not directly affected by the transformation
    model_transformed = TransformedTargetRegressor(regressor=model, transformer=target_transformer)

    # Evaluate the model using cross-validation
    scores = cross_val_score(model_transformed, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)

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

    global latest_trial_params
    latest_trial_params = params

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

    global trials_since_last_improvement
    if not is_new_best:
        best_adjusted_r2_tag = f"{GRAY}(best: {best_trial_results['adjusted_r2']:.4f}){RESET}"
        trials_since_last_improvement += 1
    else:
        trials_since_last_improvement = 0
        if trial.number > 1:
            previous_best = list(trials_results_sorted.values())[1]["adjusted_r2"]
        else:
            previous_best = 0.0
        global last_improvement_amount
        last_improvement_amount = latest_trial_results["adjusted_r2"] - previous_best

    duration = (datetime.now() - last_start_time).total_seconds()
    median_duration = np.median([t.total_seconds() for t in times_to_train])
    median_penalty = np.median([r["variance_penalty"] for r in trials_results_sorted.values()])
    estimated_seconds_remaining = (max_trials - trial.number) * median_duration

    hours = int(estimated_seconds_remaining // 3600)
    minutes = int((estimated_seconds_remaining % 3600) // 60)
    formatted_time = f"{hours}h {minutes}m"

    print(
        f"\n{BOLD}{CYAN}=== Trial {trial.number} of {max_trials} ({model_class.__name__}) ==={RESET}\n"
        f"{YELLOW}  {'Adjusted R²:':20} {latest_trial_results['adjusted_r2']:> 8.4f}   {best_adjusted_r2_tag}\n"
        f"{YELLOW}  {'Mean Raw R²:':20} {latest_trial_results['mean_r2']:> 8.4f}   {GRAY}(best: {best_trial_results['mean_r2']:.4f}){RESET}\n"
        f"{YELLOW}  {'Variance Penalty:':20} {latest_trial_results['variance_penalty']:> 8.4f}   {GRAY}(median: {median_penalty:.4f}){RESET}\n"
        f"{YELLOW}  {'Time to train:':20} {duration:> 7.2f}s   {GRAY}(median: {median_duration:.2f}s){RESET} "
        f"{GRAY}  ({'Estimated time remaining:':20} {estimated_seconds_remaining:.2f}s, or {formatted_time}){RESET}\n"
    )

    # Printing progress info like "what and when was the last improvement"
    DIM_GREEN = "\033[2;32m"  # Dim Green
    DIM_CYAN = "\033[2;36m"  # Dim Cyan
    DIM_ORANGE = "\033[2;38;5;208m"  # Dim Orange (using 256-color code 208)
    DIM_RED = "\033[2;31m"  # Dim Red
    RESET = "\033[0m"  # Resets all attributes
    if trials_since_last_improvement < 30:
        print(f"{DIM_GREEN}Last improvement: +{last_improvement_amount:.4f} ({trials_since_last_improvement} trials ago){RESET}")
    elif trials_since_last_improvement < 75:
        print(f"{DIM_CYAN}Last improvement: +{last_improvement_amount:.4f} ({trials_since_last_improvement} trials ago){RESET}")
    elif trials_since_last_improvement < 150:
        print(f"{DIM_ORANGE}Last improvement: +{last_improvement_amount:.4f} ({trials_since_last_improvement} trials ago, consider adjusting the search space?){RESET}")
    else:  # count >= 100
        print(f"{DIM_RED}Last improvement: +{last_improvement_amount:.4f} ({trials_since_last_improvement} trials ago, probably stuck, adjust the search space){RESET}")

    # Append latest results to a file, using the model as part of the name
    global latest_trial_params
    full_latest_result_log_string = (
        f"Score: {latest_trial_results['adjusted_r2']:.4f}, Mean R²: {latest_trial_results['mean_r2']:.4f}, Time to train: {duration:.2f}s, Params: {str(latest_trial_params)}\n"
    )
    full_best_result_log_string = f"Score: {best_trial_results['adjusted_r2']:.4f}, Mean R²: {best_trial_results['mean_r2']:.4f}, Time to train: {duration:.2f}s, Params: {str(latest_trial_params)}\n"

    model_name = model_class.__name__
    with open(f"tuning_results_{model_name}.txt", "a", encoding="utf-8") as f:
        f.write(full_latest_result_log_string)

    print(f"{GREEN}Best so far:{RESET} {full_best_result_log_string}")
    print("-" * 60)


#################################################################
# Study
#################################################################
max_trials = 2500  # Length of study, stored in a variable to be able to print as we progress


def get_trial_params(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 20, 60),
        # "iterations": 45,
        "learning_rate": trial.suggest_float("learning_rate", 0.15, 0.25),
        "depth": trial.suggest_int("depth", 5, 8),
        # "depth": 6,
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 1.0, log=True),
        "loss_function": "RMSE",
        "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 3, 20),
        "subsample": trial.suggest_float("subsample", 0.05, 0.35),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 25, 120),
        "verbose": 0,
    }

    return params


model_class = CatBoostRegressor


# Set Optuna verbosity to WARNING, I use my own print_progress_callback to print the results
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Use Hyperband as the sampler for Optuna
# This speeds up the optimization process by stopping bad trials early (?)
pruner = optuna.pruners.HyperbandPruner()

# Create an Optuna study to minimize/maximize the objective
study = optuna.create_study(direction="maximize", pruner=pruner)


#################################################################
# Control functions
#################################################################


def run_study():
    while True:
        try:
            study.optimize(objective, n_trials=max_trials, callbacks=[print_progress_callback])
            break  # Exit loop if optimization completes without error
        except KeyboardInterrupt:
            print("Interrupted by user")
            break  # Exit loop on manual interruption
        except Exception as e:
            print("An error occurred:", e)
            print("Trying to resume study...")


def show_best_results():
    # Sort results by adjusted R²
    trials_results_sorted = {k: v for k, v in sorted(trials_results.items(), key=lambda item: item[1]["adjusted_r2"], reverse=True)}

    # Print the 10 best results as adjusted R², mean R2, and params
    print("\nTop 10 best results:")
    for i, (params, results) in enumerate(list(trials_results_sorted.items())[:10]):
        print(f"Adjusted R²: {results['adjusted_r2']:.4f}, Mean R²: {results['mean_r2']:.4f}, Time to train: {results['time_to_train'].total_seconds():.2f}s, Params: {params}")

    input("Press Enter to return to main menu...")


def show_hyperparameter_importances():
    # Print hyperparameters importance
    print("Preparing hyperparameters importances...")
    importances = optuna.importance.get_param_importances(study)
    print("Hyperparameters importance:")
    print(importances)

    print("Visualizing importances:")
    fig = optuna.visualization.plot_param_importances(study)
    fig.show()


def show_slice_plot():
    # print("Preparing slice plot...")
    # fig = optuna.visualization.plot_slice(study)
    # fig.show()

    print("Preparing filtered slice plot...")

    # Request user input for exclusion percentiles.
    bottom_input = input("Enter percentiles to exclude from the worst results (0-100, default 0): ").strip()
    top_input = input("Enter percentile to exclude from the best results (0-100, default 0): ").strip()

    # Convert inputs to floats; default to 0 if empty.
    bottom_exclusion = float(bottom_input) if bottom_input else 0.0
    top_exclusion = float(top_input) if top_input else 0.0

    # Filter only completed trials with valid objective values.
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]

    if not completed_trials:
        print("No completed trials with valid objective values found.")
        return

    values = np.array([t.value for t in completed_trials])  # Extract objective values for the completed trials.

    # Determine cutoff values based on the specified exclusion percentiles.
    lower_cutoff = np.percentile(values, bottom_exclusion)
    upper_cutoff = np.percentile(values, 100 - top_exclusion)

    # Select trials within the cutoff range.
    filtered_trials = [t for t in completed_trials if lower_cutoff <= t.value <= upper_cutoff]

    # Create a temporary study object to hold the filtered trials.
    filtered_study = optuna.create_study(direction=study.direction)
    for trial in filtered_trials:
        filtered_study.add_trial(trial)

    # Generate and show the slice plot.
    fig = optuna.visualization.plot_slice(filtered_study)
    fig.show()


def show_history_plot():
    print("Preparing history plot...")
    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()


# Always start with running the study to avoid running the script without any results
run_study()

while True:
    banner_width = 60
    inner_width = banner_width - 2
    print("\n" * 2)
    print("╔" + "═" * inner_width + "╗")
    print("║" + "MAIN MENU".center(inner_width) + "║")
    print("║" + ("(" + model_class.__name__ + ")").center(inner_width) + "║")
    print("╚" + "═" * inner_width + "╝")
    print("1. Run/Resume study")
    print("2. Show best results")
    print("3. Show slice plot")
    print("4. Show history plot")
    print("5. Show hyperparameter importances")
    print("6. Exit")
    choice = input("")

    if choice == "1":
        run_study()
    elif choice == "2":
        show_best_results()
    elif choice == "3":
        show_slice_plot()
    elif choice == "4":
        show_history_plot()
    elif choice == "5":
        show_hyperparameter_importances()
    elif choice == "6":
        choice = input("Are you sure you want to exit? (y/n)")
        if choice.lower() == "y":
            break  # Exit the loop
    else:
        print("Invalid choice. Please try again.")
        continue


print("Script finished.")
