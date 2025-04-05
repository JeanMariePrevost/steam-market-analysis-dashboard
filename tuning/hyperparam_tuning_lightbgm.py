"""
This script uses optuna to perform hyperparameter tuning for a LightGBM model.
"""

import numpy as np
import optuna
from lightgbm import LGBMRegressor
from optuna.trial import TrialState
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

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
        "n_estimators": trial.suggest_int("n_estimators", 20, 400),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.99, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 5, 150),
        "max_depth": trial.suggest_int("max_depth", 2, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }

    # Instantiate the CatBoostRegressor with the suggested hyperparameters
    model = LGBMRegressor(**params, verbose=-1, random_seed=42)

    # Use 5-fold cross-validation to evaluate R² directly.
    # Higher R² is better.
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
    mean_r2 = np.mean(scores)

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
    params_string = ", ".join(f"{key}: {value:.3f}" for key, value in trial.params.items())
    # Store the results in the dictionary
    results[params_string] = latest_r2

    # Append latest restuls to a file, using the model as part of thename
    with open("tuning_results_LightGBM.txt", "a", encoding="utf-8") as f:
        f.write(f"R²: {latest_r2:.3f} - {trial.params}\n")

    print(f"Best R² so far: {best_r2:.3f} -> Params: {study.best_params}")
    print("-" * 60)


# Use Hyperband as the sampler for Optuna
# This speeds up the optimization process by stopping bad trials early
pruner = optuna.pruners.HyperbandPruner()


# Create an Optuna study to minimize/maximize the objective
study = optuna.create_study(direction="maximize", pruner=pruner)

# Optimize the objective over 50 trials while printing progress
study.optimize(objective, n_trials=100, callbacks=[print_progress_callback])

print("Best hyperparameters found:")
print(study.best_params)


# Print the top 10 results
results_sorted = sorted(results.items(), key=lambda x: x[1], reverse=True)
print("Top 10 results:")
for i, (params, r2) in enumerate(results_sorted[:10], start=1):
    print(f"#{i} - R²: {r2:.3f}, Params: {params}")
