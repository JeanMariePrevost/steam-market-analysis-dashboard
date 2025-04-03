"""
Same as hyperparam_tuning_generic.py, but the task is to classify games "chances of success" basically.
"""

import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, cohen_kappa_score, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import QuantileTransformer

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
# Even 0.05 seems to be enough to get a good idea of the model's "rough" performance for initial exploration
fraction = 1.0  # Fraction of the data to use for training
if fraction < 1.0:
    # Create a stratified sample of the data
    fraction_strata_bins = pd.cut(data["steam_total_reviews"], bins=10, labels=False)
    sample = data.groupby(fraction_strata_bins).apply(lambda x: x.sample(frac=fraction))
    sample = sample.reset_index(drop=True)
    print("\033[91m\n" + f"WARNING:\nUsing only {fraction} of the dataset \nfor faster training; results may not generalize fully.\n\033[0m")
    print(f"Number of rows in the sample: {len(sample)}")
    data = sample

y = data["steam_total_reviews"]  # Extract the target

# Since this is a classification task, we need to convert the target to a categorical variable
# First, we define the bins cuttoffs for the target variable
# bins = [-np.inf, 50, 300, 1000, 2500, 10000, np.inf]
bins = [-np.inf, 50, 100, 250, 500, 1000, 2500, 5000, np.inf]
labels = range(len(bins) - 1)  # Create labels for the bins, e.g. [0, 1, 2, ...]
y_binned = pd.cut(y, bins=bins, labels=labels)
y = pd.Series(y_binned, name="steam_total_reviews")  # Convert the binned target variable to a Series

#########################
# Print some debug info on the binning results
print(f"Target variable (steam_total_reviews) has {len(y.unique())} unique values after binning:")
bin_bounds = [f"({bins[i]}, {bins[i+1]}]" for i in range(len(bins) - 1)]  # Create human-readable bin labels
bin_counts = y.value_counts().sort_index()  # Count the elements in each bin and sort by bin order

# Build a DataFrame for the summary table
table = pd.DataFrame(
    {
        "Bin": bin_bounds,
        "Count": bin_counts.values,
    }
)
table["Percentage"] = (table["Count"] / table["Count"].sum() * 100).round(2)

print(table.to_string(index=False))  # Print the table
#########################


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
# target_transformer = QuantileTransformer(n_quantiles=10000, output_distribution="uniform", random_state=42)
# y = target_transformer.fit_transform(y.values.reshape(-1, 1)).flatten()

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


# Dict to store trials results by params, just "utility" stuff
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
    # Get the hyperparameters for this trial.
    if trial.number == 0:
        # If this is the first trial, use the starting trial params
        try:
            params = get_first_trial_params(trial)
        except Exception as e:
            print(f"Error getting first trial params: {e}")
            print("Using default params instead.")
            params = get_trial_params(trial)
    else:
        # Otherwise, use the regular params
        params = get_trial_params(trial)

    GREEN = "\033[92m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    print(f"\n{BOLD}{GREEN}Starting trial {trial.number} of {max_trials}{RESET} with params: {params}")
    print(f"You can safely interrupt at any point with {CYAN}Ctrl+C{RESET} and get the best results so far.")

    # Mark the start time for training this trial.
    global last_start_time
    last_start_time = datetime.now()

    model = model_class(**params)

    # print(f"Training {model_class.__name__} , trial {trial.number}...")
    # scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_weighted", n_jobs=-1)  #  New weighted F1 score
    # weighted_f1 = np.mean(scores)

    # Trying a custom score that accounts for orinality, meaning large mistakes are penalized more
    def weighted_kappa(y_true, y_pred):
        return cohen_kappa_score(y_true, y_pred, weights="quadratic")

    custom_scorer = make_scorer(weighted_kappa)

    scores = cross_val_score(model, X_train, y_train, cv=5, scoring=custom_scorer, n_jobs=-1)
    mean_score = np.mean(scores)

    # Track training time for this trial.
    global times_to_train
    times_to_train.append(datetime.now() - last_start_time)

    # Store the results for later inspection.
    key = str(params)
    trials_results[key] = {
        "trial_score": mean_score,
        "trial_number": trial.number,
        "time_to_train": times_to_train[-1],
        "trial_object": trial,
    }

    global latest_trial_results
    latest_trial_results = trials_results[key]

    global latest_trial_params
    latest_trial_params = params

    # Debug, print full report
    # model.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=0) # We need to re-fit the model to get the predictions, because cross_val_score doesn't return them
    # print(classification_report(y_valid, model.predict(X_valid), zero_division=0))

    # Since higher accuracy is better, we return the adjusted_acc as our objective.
    return mean_score


def print_progress_callback(study, trial):
    # Sort the results by score
    trials_results_sorted = {k: v for k, v in sorted(trials_results.items(), key=lambda item: item[1]["trial_score"], reverse=True)}
    best_trial_results = list(trials_results_sorted.values())[0]

    # ANSI color codes
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    is_new_best = latest_trial_results["trial_score"] == best_trial_results["trial_score"]
    best_adjusted_acc_tag = f"{BOLD}{GREEN}(NEW BEST){RESET}"

    global trials_since_last_improvement
    if not is_new_best:
        best_adjusted_acc_tag = f"{GRAY}(best: {best_trial_results['trial_score']:.4f}){RESET}"
        trials_since_last_improvement += 1
    else:
        trials_since_last_improvement = 0
        if trial.number > 1:
            previous_best = list(trials_results_sorted.values())[1]["trial_score"]
        else:
            previous_best = 0.0
        global last_improvement_amount
        last_improvement_amount = latest_trial_results["trial_score"] - previous_best

    duration = (datetime.now() - last_start_time).total_seconds()
    median_duration = np.median([t.total_seconds() for t in times_to_train])
    # median_penalty = np.median([r["variance_penalty"] for r in trials_results_sorted.values()])
    estimated_seconds_remaining = (max_trials - trial.number) * median_duration

    hours = int(estimated_seconds_remaining // 3600)
    minutes = int((estimated_seconds_remaining % 3600) // 60)
    formatted_time = f"{hours}h {minutes}m"

    print(
        f"\n{BOLD}{CYAN}=== Trial {trial.number} of {max_trials} ({model_class.__name__}) ==={RESET}\n"
        f"{YELLOW}  {'Score:':20} {latest_trial_results['trial_score']:> 8.4f}   {best_adjusted_acc_tag}\n"
        # f"{YELLOW}  {'Mean Acc:':20} {latest_trial_results['mean_accuracy']:> 8.4f}   {GRAY}(best: {best_trial_results['mean_accuracy']:.4f}){RESET}\n"
        # f"{YELLOW}  {'Variance Penalty:':20} {latest_trial_results['variance_penalty']:> 8.4f}   {GRAY}(median: {median_penalty:.4f}){RESET}\n"
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
        f"Score: {latest_trial_results['trial_score']:.4f}, Time to train: {duration:.2f}s, Params: {str(latest_trial_params)}\n"
        # f"Score: {latest_trial_results['trial_score']:.4f}, Mean Acc: {latest_trial_results['mean_accuracy']:.4f}, Time to train: {duration:.2f}s, Params: {str(latest_trial_params)}\n"
    )
    full_best_result_log_string = (
        f"Score: {best_trial_results['trial_score']:.4f}, Time to train: {duration:.2f}s, Params: {str(latest_trial_params)}\n"
        # f"Score: {best_trial_results['trial_score']:.4f}, Mean Acc: {best_trial_results['mean_accuracy']:.4f}, Time to train: {duration:.2f}s, Params: {str(latest_trial_params)}\n"
    )

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
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 0.9),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
        "boosting_type": "gbdt",
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 0.9),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 100.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 1.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 1e-3, 0.5, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "num_leaves": trial.suggest_int("num_leaves", 4, 64),
        "objective": "multiclass",
        "subsample_for_bin": trial.suggest_int("subsample_for_bin", 20000, 150000),
        "verbosity": -1,
    }

    # LGBMClassifier, FULL search space for afterwards
    # params = {
    #     "verbosity": -1,
    #     "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),  # Alias: subsample
    #     "bagging_freq": trial.suggest_int("bagging_freq", 0, 10),  # Alias: subsample_freq
    #     "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "rf"]),
    #     "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),  # Specific to Classifier
    #     "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),  # Alias: colsample_bytree - Decreased lower bound
    #     "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),  # Alias: reg_alpha
    #     "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 100.0, log=True),  # Alias: reg_lambda - Increased upper bound
    #     "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
    #     "max_depth": trial.suggest_int("max_depth", 3, 20),
    #     "min_child_samples": trial.suggest_int("min_child_samples", 1, 100),  # Decreased lower bound
    #     "min_child_weight": trial.suggest_float("min_child_weight", 1e-5, 10.0, log=True),  # Added based on Regressor params
    #     "min_gain_to_split": trial.suggest_float("min_gain_to_split", 1e-8, 1.0, log=True),  # Added based on Regressor params (alias: min_split_gain)
    #     "n_estimators": trial.suggest_int("n_estimators", 50, 3000),
    #     "num_leaves": trial.suggest_int("num_leaves", 2, 256),
    #     "subsample_for_bin": trial.suggest_int("subsample_for_bin", 20000, 300000),
    # }
    return params


def get_first_trial_params(trial):
    """
    Here you can define a "starting trial" to be used for the study.
    E.g. to "resume" from a best known setup.
    Simple return get_trial_params() if you want to use the default params.
    """
    # return get_trial_params(trial)
    starting_trial_params = {
        "bagging_fraction": 0.8243982460922371,
        "bagging_freq": 1,
        "boosting_type": "gbdt",
        "class_weight": "balanced",
        "feature_fraction": 0.5071762634392396,
        "lambda_l1": 0.07291150498780326,
        "lambda_l2": 3.1114924911217766e-07,
        "learning_rate": 0.09817278409944286,
        "max_depth": 10,
        "min_child_samples": 79,
        "min_child_weight": 0.8506250802309826,
        "min_gain_to_split": 0.0021260826590470336,
        "n_estimators": 488,
        "num_leaves": 57,
        "subsample_for_bin": 134204,
        "verbosity": -1,
    }
    return starting_trial_params


# Switch from CatBoostRegressor to CatBoostClassifier
model_class = LGBMClassifier


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
    # Sort results by adjusted score
    trials_results_sorted = {k: v for k, v in sorted(trials_results.items(), key=lambda item: item[1]["trial_score"], reverse=True)}

    # Print the 10 best results
    print("\nTop 10 best results:")
    for i, (params, results) in enumerate(list(trials_results_sorted.items())[:10]):
        print(f"Score: {results['trial_score']:.4f}, Time to train: {results['time_to_train'].total_seconds():.2f}s, Params: {params}")
        # print(f"Adjusted Acc: {results['trial_score']:.4f}, Mean Acc: {results['mean_accuracy']:.4f}, Time to train: {results['time_to_train'].total_seconds():.2f}s, Params: {params}")

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
