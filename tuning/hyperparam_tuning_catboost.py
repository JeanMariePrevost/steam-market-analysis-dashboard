"""
This script uses optuna to perform hyperparameter tuning for a CatBoost model.
"""

from datetime import datetime

import numpy as np
import optuna
from catboost import CatBoostRegressor
from optuna.trial import TrialState
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

import utils

# Load a sample dataset (California housing in this case)
data = utils.load_feature_engineered_dataset_no_na()

# data = data.select_dtypes(include=[np.number])  # Keep only numerical columns for this model


# Remove directly correlated columns that wouldn't be available at prediction time
columns_to_drop = [
    "estimated_gross_revenue_boxleiter",
    "estimated_ltarpu",
    "estimated_owners_boxleiter",
    "name",
    "peak_ccu",
    "recommendations",
    "steam_negative_reviews",
    "steam_positive_review_ratio",
    "steam_positive_reviews",
    "steam_total_reviews",
    "publishers",
    "developers",
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

number_of_trials = 20


def objective(trial):
    # Define the hyperparameter search space

    # Narrower still search space
    params = {
        # "iterations": trial.suggest_int("iterations", 30, 60),
        "iterations": 45,
        "learning_rate": trial.suggest_float("learning_rate", 0.15, 0.25),
        # "depth": trial.suggest_int("depth", 5, 8),
        "depth": 6,
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 1.0, log=True),
        "loss_function": "RMSE",
        "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 3, 20),
        "subsample": trial.suggest_float("subsample", 0.05, 0.35),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 25, 120),
    }

    # Narrowed search space
    # params = {
    #     "iterations": trial.suggest_int("iterations", 10, 60),
    #     "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.2, log=True),
    #     "depth": trial.suggest_int("depth", 1, 10),
    #     "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
    #     "loss_function": "RMSE",
    #     # "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 2, 20),
    #     "subsample": trial.suggest_float("subsample", 0.05, 1.0),
    #     "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
    #     "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    # }

    # Wide original search space
    # params = {
    #     "iterations": trial.suggest_int("iterations", 10, 400),
    #     "learning_rate": trial.suggest_float("learning_rate", 0.0001, 1.0, log=True),
    #     "depth": trial.suggest_int("depth", 1, 10),
    #     "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
    #     "loss_function": "RMSE",
    #     "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 2, 20),
    #     "subsample": trial.suggest_float("subsample", 0.05, 1.0),
    #     "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
    #     "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    # }

    print(f"Starting trial {trial.number} of {number_of_trials} with params: {params}")

    # Get current time to measure how long the training takes
    time_before = datetime.now()

    # Instantiate the CatBoostRegressor with the suggested hyperparameters
    model = CatBoostRegressor(**params, verbose=0, random_seed=42)

    # Use 5-fold cross-validation to evaluate R² directly.
    # Higher R² is better.
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
    mean_r2 = np.mean(scores)

    print(f"Time taken: {datetime.now() - time_before}")

    return mean_r2


def print_progress_callback(study, trial):
    latest_r2 = trial.value  # R² from the current trial
    best_r2 = study.best_value

    if trial.state == TrialState.PRUNED:
        print(f"Trial {trial.number} was PRUNED (early stopping applied).")

    if latest_r2 == best_r2:
        print(f"Trial {trial.number} completed with R²: {latest_r2:.3f} (NEW BEST)")
    else:
        print(f"Trial {trial.number} completed with R²: {latest_r2:.3f}")

    # Append latest restuls to a file, using the model as part of thename
    with open("tuning_results_catboost.txt", "a", encoding="utf-8") as f:
        f.write(f"{trial.params}\t R²: {latest_r2:.3f}\n")

    print(f"Best R² so far: {best_r2:.3f} -> Params: {study.best_params}")
    print("-" * 60)


# Use Hyperband as the sampler for Optuna
# This speeds up the optimization process by stopping bad trials early
pruner = optuna.pruners.HyperbandPruner()


# Create an Optuna study to minimize/maximize the objective
study = optuna.create_study(direction="maximize", pruner=pruner)

# Optimize the objective over 50 trials while printing progress
study.optimize(objective, n_trials=number_of_trials, callbacks=[print_progress_callback])

print("Best hyperparameters found:")
print(study.best_params)

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
