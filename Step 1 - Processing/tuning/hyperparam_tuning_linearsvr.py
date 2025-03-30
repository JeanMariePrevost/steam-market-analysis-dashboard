"""
This script uses optuna to perform hyperparameter tuning for an Extra Trees model.
"""

from datetime import datetime

import numpy as np
import optuna
from optuna.trial import TrialState
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR

import utils

# Load a sample dataset (California housing in this case)
data = utils.load_feature_engineered_dataset_no_na()

data = data.select_dtypes(include=[np.number])  # Keep only numerical columns for this model


# Remove directly correlated columns that wouldn't be available at prediction time
columns_to_drop = [
    "estimated_gross_revenue_boxleiter",
    "estimated_ltarpu",
    "estimated_owners_boxleiter",
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

# Store all results in a dictionary as their parameters, with value R^2
results = {}


def objective(trial):
    # Define the hyperparameter search space
    params = {
        "C": trial.suggest_loguniform("C", 1e-3, 1e1),  # Lower range for C to reduce complexity
        "epsilon": trial.suggest_loguniform("epsilon", 1e-2, 1e-1),  # Larger epsilon for fewer updates
        "tol": trial.suggest_loguniform("tol", 1e-3, 1e-2),  # Higher tolerance for faster convergence
        "loss": trial.suggest_categorical("loss", ["epsilon_insensitive"]),  # Avoid squared loss (slower)
        "max_iter": trial.suggest_int("max_iter", 500, 5000),  # Lower max iterations to force early stopping
    }

    # if params["kernel"] in ["poly", "rbf", "sigmoid"]:
    #     params["gamma"] = trial.suggest_loguniform("gamma", 1e-4, 1e1)

    # if params["kernel"] == "poly":
    #     params["degree"] = trial.suggest_int("degree", 2, 5)

    print(f"Starting trial {trial.number} with params: {params}")

    # Instantiate the SVR model with the suggested hyperparameters
    model = LinearSVR(**params)

    # Get current time to measure how long the training takes
    time_before = datetime.now()

    # Use 5-fold cross-validation to evaluate R² directly.
    # Higher R² is better.
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=4)
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

    # Create a string key of the parameters for the results dictionary
    params_string = ", ".join(f"{key}: {value:.3f}" if isinstance(value, (float, int)) else f"{key}: {value}" for key, value in trial.params.items())
    # Store the results in the dictionary
    results[params_string] = latest_r2

    # Append latest restuls to a file, using the model as part of thename
    with open("tuning_results_LinearSVR.txt", "a", encoding="utf-8") as f:
        f.write(f"R²: {latest_r2:.3f} - {trial.params}\n")

    print(f"Best R² so far: {best_r2:.3f} -> Params: {study.best_params}")
    print("-" * 60)


# Use Hyperband as the sampler for Optuna
# This speeds up the optimization process by stopping bad trials early
pruner = optuna.pruners.HyperbandPruner()


# Create an Optuna study to minimize/maximize the objective
study = optuna.create_study(direction="maximize", pruner=pruner)

# Optimize the objective over 50 trials while printing progress
study.optimize(objective, n_trials=200, callbacks=[print_progress_callback])

print("Best hyperparameters found:")
print(study.best_params)


# Print the top 10 results
results_sorted = sorted(results.items(), key=lambda x: x[1], reverse=True)
print("Top 10 results:")
for i, (params, r2) in enumerate(results_sorted[:10], start=1):
    print(f"#{i} - R²: {r2:.3f}, Params: {params}")
