import itertools
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from optuna.trial import TrialState
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import cohen_kappa_score, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

# Add the parent directory of the script to sys.path, cause since we moved the script to a new folder, it can't find the utils module
# This is a hack, but it works good enough for a quick and dirty script
sys.path.append(str(Path(__file__).resolve().parent.parent))

import utils

#################################################################
# Prepare the data
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

# seed = 42  # Set a random state for reproducibility
seed = random.randint(0, 1000)  # Or uncomment this to use a random seed
print(f"Random seed: {seed}")

######################################################
# Stratified sampling (optional)
######################################################
# If you want to use only a fraction of the data for faster testing, set the fraction here.
# Even 0.05 works surprisingly well for very rough initial exploration
fraction = 1.0  # Fraction of the data to use for training
if fraction < 1.0:
    # Create a stratified sample of the data
    fraction_strata_bins = pd.cut(data["steam_total_reviews"], bins=10, labels=False)
    sample = data.groupby(fraction_strata_bins).apply(lambda x: x.sample(frac=fraction, random_state=seed))
    sample = sample.reset_index(drop=True)
    print("\033[91m\n" + f"WARNING:\nUsing only {fraction} of the dataset \nfor faster training; results may not generalize fully.\n\033[0m")
    print(f"Number of rows in the sample: {len(sample)}")
    data = sample

######################################################
# Target variable binning (making it categorical)
######################################################
# Since this is a classification task, we need to convert the target to a categorical variable by using binning.
y = data["steam_total_reviews"]  # Extract the target
bins = [-np.inf, 50, 100, 250, 500, 1000, 2500, 5000, np.inf]
labels = range(len(bins) - 1)  # Create labels (or "values") for the bins, e.g. [0, 1, 2, ...]
y_binned = pd.cut(y, bins=bins, labels=labels)
y = pd.Series(y_binned, name="steam_total_reviews")  # Convert the binned target variable to a Series

# Print some debug info on the binning results
print(f"Target variable (steam_total_reviews) has {len(y.unique())} unique values after binning:")
table = pd.DataFrame({"Bin": [f"({bins[i]}, {bins[i+1]}]" for i in range(len(bins) - 1)], "Count": y.value_counts().sort_index().values})
table["Percentage"] = (table["Count"] / table["Count"].sum() * 100).round(2)
print(table.to_string(index=False))


################################################################
# Building the splits
################################################################
X = data.drop(columns=columns_to_drop, errors="ignore")  # Extract the predictors, dropping the irrelevant columns and the target

# All remaining columns ARE numerical, no need to filter anything
# Create a QuantileTransformer to map the numerical values to a uniform [0, 1] distribution.
qt = QuantileTransformer(output_distribution="uniform")
X = qt.fit_transform(X)

y_strata_bins = pd.cut(y, bins=10, labels=False)  # Create strata for stratified sampling to, e.g. not to end up with most of the big hits in either set, or only failed games in the test set
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y_strata_bins)


################################################################
# Loading already trained models (optional)
################################################################
base_models = {
    "lgbm_high": None,
    "lgbm_low": None,
    "xgb": None,
    "catboost": None,
    "et": None,
    "rf": None,
}

base_model_already_trained = {
    "lgbm_high": False,
    "lgbm_low": False,
    "xgb": False,
    "catboost": False,
    "et": False,
    "rf": False,
}

models_output_dir = "saved_multiclass_models"
os.makedirs(models_output_dir, exist_ok=True)  # Create directory if it doesn't exist

load_if_exists = True  # True to load existing saved models instead of training new ones
if load_if_exists:
    print("Loading existing models...")
    for name, model in base_models.items():
        filename = f"{name}_classifier_qwk.joblib"
        filepath = os.path.join(models_output_dir, filename)
        try:
            base_models[name] = joblib.load(filepath)
            base_model_already_trained[name] = True
            print(f"Model loaded successfully from: {filepath}")
        except FileNotFoundError:
            print(f"Model file not found: {filepath}. Training new model.")
            # If the model file is not found, it will be trained below
            continue
        except Exception as e:
            print(f"Error loading model {name}: {e}")


################################################################
# Defining the models (if not loaded)
################################################################
if base_models["lgbm_high"] is None:
    base_models["lgbm_high"] = LGBMClassifier(
        bagging_fraction=0.6439393943221187,
        bagging_freq=7,
        boosting_type="gbdt",
        class_weight="balanced",
        feature_fraction=0.6782418960781161,
        lambda_l1=3.476560126339504e-05,  # Often referred to as reg_alpha
        lambda_l2=0.07079376172417023,  # Often referred to as reg_lambda
        learning_rate=0.01874515225414181,
        max_depth=7,
        min_child_samples=26,
        min_child_weight=0.2798897634943844,
        min_gain_to_split=0.009328865858032935,  # Often referred to as min_split_gain
        n_estimators=388,
        num_leaves=39,
        subsample_for_bin=93363,
        verbosity=-1,
    )

if base_models["lgbm_low"] is None:
    base_models["lgbm_low"] = LGBMClassifier(
        n_estimators=78,
        max_depth=8,
        learning_rate=0.09208534448153478,
        num_leaves=57,
        lambda_l1=0.1527468630392078,
        lambda_l2=0.8460604647916626,
        feature_fraction=0.33167223288930514,
        bagging_fraction=0.7947658996006763,
        bagging_freq=7,
        min_child_samples=38,
        min_child_weight=0.05402111206488199,
        min_gain_to_split=0.23753727898491814,
        subsample_for_bin=69736,
        boosting_type="gbdt",
        class_weight="balanced",
        verbosity=-1,
        random_state=42,
        n_jobs=-1,
    )

if base_models["xgb"] is None:
    base_models["xgb"] = XGBClassifier(
        objective="multi:softprob",
        n_estimators=401,
        learning_rate=0.232972588655614,
        max_depth=9,
        min_child_weight=0.002276550854036569,
        gamma=0.25913862611573124,
        subsample=0.831371456318946,
        colsample_bytree=0.5907077778760849,
        colsample_bylevel=0.3907152100115424,
        colsample_bynode=0.855466789883967,
        reg_alpha=9.0288636468865e-08,
        reg_lambda=4.480055837935949,
        scale_pos_weight=0.5974198051782446,
        max_delta_step=8,
        tree_method="approx",
        verbosity=0,
    )

if base_models["catboost"] is None:
    base_models["catboost"] = CatBoostClassifier(
        bagging_temperature=0.035005651609606184,
        border_count=33,
        colsample_bylevel=0.522,
        depth=8,
        early_stopping_rounds=26,
        iterations=197,
        l2_leaf_reg=1.29,
        learning_rate=0.2713710861150384,
        loss_function="MultiClass",
        min_data_in_leaf=73,
        random_strength=0.831717139271116,
    )

if base_models["et"] is None:
    base_models["et"] = ExtraTreesClassifier(
        n_estimators=771,
        criterion="gini",
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features=0.2703238113965761,
        bootstrap=False,
        ccp_alpha=4.3089533122468846e-05,
        random_state=42,
        n_jobs=-1,
    )

if base_models["rf"] is None:
    base_models["rf"] = RandomForestClassifier(
        n_estimators=142,
        criterion="gini",
        max_depth=11,
        min_samples_split=7,
        min_samples_leaf=2,
        min_weight_fraction_leaf=9.92810819852466e-05,
        max_features=0.3,
        max_leaf_nodes=143,
        bootstrap=True,
        max_samples=0.855,
        ccp_alpha=0.000151,
        random_state=42,
        n_jobs=-1,
    )


################################################################
# Training, evaluating and saving the models
################################################################
scores = {}

print("Starting Model Training and Evaluation using Quadratic Weighted Kappa (QWK)")

# Train, evaluate and save each model
for name, model in base_models.items():
    if base_model_already_trained[name]:
        print(f"Model {name} loaded from an already trained state. Skipping training.")
        continue

    time_before_training = datetime.now()
    print(f"\nTraining {name} model...")
    model.fit(X_train, y_train)
    print("Training completed.")
    print(f"Training took: {(datetime.now() - time_before_training).total_seconds()} seconds")

    # Evaluate the model using QWK
    y_pred = model.predict(X_valid)

    # Calculate QWK score
    score = cohen_kappa_score(y_valid, y_pred, weights="quadratic")
    scores[name] = score
    print(f"{name} Quadratic Weighted Kappa (QWK) score on validation set: {score:.4f}")

    # Save the trained model
    filename = f"{name}_classifier_qwk.joblib"
    filepath = os.path.join(models_output_dir, filename)
    try:
        joblib.dump(model, filepath)
        print(f"Model saved successfully to: {filepath}")
    except Exception as e:
        print(f"Error saving model {name}: {e}")

# Print the final evaluation summary
print("\nEvaluation Summary (Quadratic Weighted Kappa)")
# Sort scores highest to lowest for better readability
sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
for name, score in sorted_scores.items():
    print(f"{name}: {score:.4f}")

# Calculate and print mean score
if scores:  # Avoid error if scores dict is empty
    mean_score = np.mean(list(scores.values()))
    print(f"\nMean QWK score: {mean_score:.4f}")
else:
    print("\nNo models were evaluated.")

print(f"\nModels saved in directory: {models_output_dir}")


#########################################################################################################
# Blended models tests
#########################################################################################################
print("\n\n\n")
print('Starting the "blended" model test...')

print("\n--- Evaluating Probability Averaging Blends ---")

# 1. Generate probability distributions from each base model
#    Store them for reuse. Also calculate individual QWK for reference.
model_probas = {}
model_qwk = []
print("Individual Model Performance (QWK based on predict()):")
for name, model in base_models.items():
    # Get probabilities (shape: n_samples, n_classes)
    probas = model.predict_proba(X_valid)
    model_probas[name] = probas

    # Get single best class prediction for individual QWK score
    pred_int = np.argmax(probas, axis=1)
    score = cohen_kappa_score(y_valid, pred_int, weights="quadratic")
    model_qwk.append(score)
    print(f"  {name}: {score:.4f}")

# Stack probability arrays for easier aggregation
# Resulting shape: (n_models, n_samples, n_classes)
all_probas_array = np.array(list(model_probas.values()))

# 2. Compute and evaluate the probability average blend of ALL models
print('\n"Complete Blend" Performance (QWK):')

# Average probabilities across models for each sample and class
# Shape after mean: (n_samples, n_classes)
avg_probas_all = np.mean(all_probas_array, axis=0)


# Determine the final predicted class by finding the class with the highest average probability
blend_pred_all = np.argmax(avg_probas_all, axis=1)

# Calculate QWK for the blended predictions
qwk_blend_all = cohen_kappa_score(y_valid, blend_pred_all, weights="quadratic")
print(f"  All Models Averaged Probs: {qwk_blend_all:.4f}")


# 2.1. Simple Average Blend
avg_probas = np.mean(all_probas_array, axis=0)
blend_pred_avg = np.argmax(avg_probas, axis=1)
qwk_avg = cohen_kappa_score(y_valid, blend_pred_avg, weights="quadratic")
print(f"Average Probabilities: {qwk_avg:.4f}")

# 2.2. Weighted Average Blend (using individual model QWK scores)
weighted_probas = np.average(all_probas_array, axis=0, weights=model_qwk)
blend_pred_weighted = np.argmax(weighted_probas, axis=1)
qwk_weighted = cohen_kappa_score(y_valid, blend_pred_weighted, weights="quadratic")
print(f"Weighted Average Probabilities: {qwk_weighted:.4f}")

# 2.3. Median Blend
median_probas = np.median(all_probas_array, axis=0)
blend_pred_median = np.argmax(median_probas, axis=1)
qwk_median = cohen_kappa_score(y_valid, blend_pred_median, weights="quadratic")
print(f"Median Probabilities: {qwk_median:.4f}")

# 2.4. Percentile Blends (25th and 75th)
perc25_probas = np.percentile(all_probas_array, 25, axis=0)
blend_pred_perc25 = np.argmax(perc25_probas, axis=1)
qwk_perc25 = cohen_kappa_score(y_valid, blend_pred_perc25, weights="quadratic")
print(f"25th Percentile Probabilities: {qwk_perc25:.4f}")

perc75_probas = np.percentile(all_probas_array, 75, axis=0)
blend_pred_perc75 = np.argmax(perc75_probas, axis=1)
qwk_perc75 = cohen_kappa_score(y_valid, blend_pred_perc75, weights="quadratic")
print(f"75th Percentile Probabilities: {qwk_perc75:.4f}")

# 2.5. Geometric Mean Blend
eps = 1e-12  # Small value to avoid log(0)
n_models = all_probas_array.shape[0]  # Number of models
geo_probas = np.exp(np.mean(np.log(all_probas_array + eps), axis=0))
blend_pred_geo = np.argmax(geo_probas, axis=1)
qwk_geo = cohen_kappa_score(y_valid, blend_pred_geo, weights="quadratic")
print(f"Geometric Mean Probabilities: {qwk_geo:.4f}")

# 2.6. Harmonic Mean Blend
harm_probas = n_models / np.sum(1 / (all_probas_array + eps), axis=0)
blend_pred_harm = np.argmax(harm_probas, axis=1)
qwk_harm = cohen_kappa_score(y_valid, blend_pred_harm, weights="quadratic")
print(f"Harmonic Mean Probabilities: {qwk_harm:.4f}")

# 2.7. Trimmed Mean Blend (trimming 10% extremes from each end)
trim_count = int(np.floor(0.1 * n_models))
if trim_count * 2 < n_models:
    sorted_probas = np.sort(all_probas_array, axis=0)
    trimmed_probas = np.mean(sorted_probas[trim_count : n_models - trim_count, :, :], axis=0)
else:
    trimmed_probas = np.mean(all_probas_array, axis=0)
blend_pred_trim = np.argmax(trimmed_probas, axis=1)
qwk_trim = cohen_kappa_score(y_valid, blend_pred_trim, weights="quadratic")
print(f"Trimmed Mean Probabilities: {qwk_trim:.4f}")


# 3. Evaluate blends of model combinations by averaging probabilities
print("\n--- Evaluating Averaged Probability Blends for Model Combinations (2 to N-1 models) ---")

combo_scores = {}
num_models = len(base_models)
for r in range(2, num_models):
    # Iterate through specific combinations of model names
    for combo_names in itertools.combinations(base_models.keys(), r):
        # Get probability arrays for the current combination
        combo_probas_list = [model_probas[name] for name in combo_names]
        combo_probas_array = np.array(combo_probas_list)  # Shape: (r, n_samples, n_classes)

        # Average probabilities for the combination
        avg_probas_combo = np.mean(combo_probas_array, axis=0)  # Shape: (n_samples, n_classes)

        # Determine the final predicted class from the combo's averaged probabilities
        blend_pred_combo = np.argmax(avg_probas_combo, axis=1)

        # Calculate QWK for the combination blend
        qwk_combo = cohen_kappa_score(y_valid, blend_pred_combo, weights="quadratic")

        combo_key = f"AvgProbs_{combo_names}"
        combo_scores[combo_key] = qwk_combo

sorted_combos = sorted(combo_scores.items(), key=lambda item: item[1], reverse=True)
print("\nTop 10 Averaged Probability Combinations by QWK:")
for combo_desc, score in sorted_combos[:10]:
    print(f"  {combo_desc}: {score:.4f}")

# Reprint the random state for clarity
print(f"\nRandom seed used: {seed}")

print("Process completed. Exiting.")
