import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import utils

# Page configuration & custom CSS
st.set_page_config(page_title="Overview of Releases & Trends")
utils.display_streamlit_custom_navigation()
st.markdown(
    """
    <style>
    .stMainBlockContainer {
        max-width: 1000px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Page Title & Description
st.title("Project Viability Estimator")
st.write("This page provides a tool to estimate the viability of your project based on historical data from Steam.")
st.write("These are only projected estimations, and should not be used as a definitive measure of success.")
st.info(
    """Limitations:
    
Predictions from multiple machine learning models are blended to achieve
a more stable and accurate (QWK ≈ 0.7507) estimated number of ***steam reviews***,
used as a direct ***proxy*** for the number of players.

The models rely on historical data and may not predict unknown combinations of features accurately.
"""
)


# Sidebar Filter: Release Year Range
st.sidebar.title("Project Details")

# Multiline text are in sidebar
st.sidebar.markdown(
    """
    **Instructions:**
    - Enter a list of feature-value pairs in the format `feature1=value1`
    - Each pair should be on a separate line
    - Every missing feature defaults to the baseline value
    - Hit ctrl+enter to submit
    """
)
input_text = st.sidebar.text_area(
    "Features",
    placeholder="""e.g.
    average_time_to_beat=20.0
    controller_support=0
    languages_supported_count=3
    monetization_model=2
    release_month=7
    runs_on_linux=1
    runs_on_mac=1
    runs_on_windows=1
    runs_on_steam_deck=1
    ~tag_tactical=1
    ~tag_rpg=1
    ~tag_indie=1
    """,
    height=450,
)

suggestion_score_cutoff = st.sidebar.number_input("Suggestion score cutoff", min_value=0.0, value=0.025, step=0.001, format="%.3f")


# Debug
# st.write("Input:")
# st.write(input_text)

# Extracting features from the input text
input_lines = input_text.strip().split("\n")
user_input_features = {}
for line in input_lines:
    if "=" in line:
        feature, value = line.split("=")
        feature = feature.strip()
        value = value.strip()
        if feature.startswith("~"):
            if value == "1" or value == "0":
                user_input_features[feature] = 1 if value == "1" else 0  # Convert to binary
            else:
                st.error(f"Value for '{feature}' must be either 0 or 1. Got: '{value}' \n\n Defaulting to baseline.")
        else:
            try:
                user_input_features[feature] = float(value)  # Convert to float
            except ValueError:
                st.error(f"Value for '{feature}' must be a valid number. Got: '{value}' \n\n Defaulting to baseline.")


# st.write("Extracted Features:")
# st.write(user_input_features)

# Build element from baseline
input_element = utils.load_baseline_element()
for feature, value in user_input_features.items():
    if feature in input_element:
        input_element[feature] = value
    else:
        closest_match = utils.find_closest_string_in_list(feature, input_element.keys())
        if closest_match:
            st.warning(f"Feature '{feature}' not recognized and will be skipped. Did you mean '{closest_match}'?")
        else:
            st.warning(f"Feature '{feature}' not recognized and will be skipped. No close match found.")
# st.write("Baseline Element:")
# st.write(input_element)


#######################################################################
# Loading models
#######################################################################
MODEL_NAME_LGBM_HIGH = "lgbm_high"
MODEL_NAME_LGBM_LOW = "lgbm_low"
MODEL_NAME_XGB = "xgb"

with st.spinner("Loading models...", show_time=True):

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

    if model_lgbm_high is None:
        st.error("Error loading the 'lgbm_high' model. Please check the corresponding model file.")
        st.stop()
    if model_lgbm_low is None:
        st.error("Error loading the 'lgbm_low' model. Please check the corresponding model file.")
        st.stop()
    if model_xgb is None:
        st.error("Error loading the 'xgb' model. Please check the corresponding model file.")
        st.stop()


#######################################################################
# Inference
#######################################################################
with st.spinner("Running inference...", show_time=True):
    lgbm_high_preds = model_lgbm_high.predict_proba(input_element)
    lgbm_low_preds = model_lgbm_low.predict_proba(input_element)
    xgb_preds = model_xgb.predict_proba(input_element)

    # st.write("Predictions:")
    # st.write(f"Predictions for {MODEL_NAME_LGBM_HIGH}: {lgbm_high_preds}")
    # st.write(f"Predictions for {MODEL_NAME_LGBM_LOW}: {lgbm_low_preds}")
    # st.write(f"Predictions for {MODEL_NAME_XGB}: {xgb_preds}")

    # Calculate the blended prediction
    blended_prediction = (lgbm_high_preds + lgbm_low_preds + xgb_preds) / 3
    # st.write(f"Blended Prediction: {blended_prediction}")


# Debug, test inference times individually
import time

before = time.time()
time_lgmb_high_preds = model_lgbm_high.predict_proba(input_element)
print(f"Time taken for {MODEL_NAME_LGBM_HIGH}: {time.time() - before} seconds")
before = time.time()
time_lgmb_low_preds = model_lgbm_low.predict_proba(input_element)
print(f"Time taken for {MODEL_NAME_LGBM_LOW}: {time.time() - before} seconds")
before = time.time()
time_xgb_preds = model_xgb.predict_proba(input_element)
print(f"Time taken for {MODEL_NAME_XGB}: {time.time() - before} seconds")


import math


def calculate_estimated_owners_boxleiter(steam_total_reviews: int, release_year: int, price_original: float) -> int:
    """
    Calculate the estimated number of owners using Boxleiter's method for a single game input.
    """

    # Determine temporary scale factor: 1.0 by default, 0.75 if total reviews > 100000.
    temp_scale_factor = 0.75 if steam_total_reviews > 100000 else 1.0

    # Determine review inflation factor based on the release year.
    if release_year <= 2013:
        temp_review_inflation_factor = 79 / 80
    elif release_year == 2014:
        temp_review_inflation_factor = 72 / 80
    elif release_year == 2015:
        temp_review_inflation_factor = 62 / 80
    elif release_year == 2016:
        temp_review_inflation_factor = 52 / 80
    elif release_year == 2017:
        temp_review_inflation_factor = 43 / 80
    elif release_year == 2018:
        temp_review_inflation_factor = 38 / 80
    elif release_year == 2019:
        temp_review_inflation_factor = 36 / 80
    elif release_year > 2020:
        temp_review_inflation_factor = 31 / 80
    else:
        temp_review_inflation_factor = 1.0

    boxleiter_number = 80  # Boxleiter's multiplier constant.
    temp_commitment_bias_mult = 3 + 2 * math.exp(-0.2 * price_original) - 2  # Compute the commitment bias multiplier based on the original price.
    estimated_owners = steam_total_reviews * boxleiter_number * temp_scale_factor * temp_review_inflation_factor * temp_commitment_bias_mult

    return int(round(estimated_owners))


#######################################################################
# Plotting
#######################################################################
with st.spinner("Creating bar chart...", show_time=True):
    # Define x-axis labels
    bins = [0, 50, 100, 250, 500, 1000, 2500, 5000, np.inf]
    bins_using_user_count = []
    for i in range(len(bins) - 1):
        value_as_owners = calculate_estimated_owners_boxleiter(bins[i], input_element["release_year"][0], input_element["price_original"][0])
        nicely_rounded = utils.step_round(value_as_owners)
        bins_using_user_count.append(nicely_rounded)
    bins = bins_using_user_count
    bins.append(np.inf)  # Append infinity to the last bin for open-ended range
    print(f"Bins: {bins}")
    categories = []
    for i in range(len(bins) - 1):
        if bins[i] == 0:
            categories.append(f"< {bins[i + 1]/1000:.0f}k")
        elif bins[i + 1] == np.inf:
            categories.append(f">= {bins[i]/1000:.0f}k")
        else:
            categories.append(f"{bins[i]/1000:.0f}k - {bins[i + 1]/1000:.0f}k")
    print(f"Categories: {categories}")

    # Normalize the blended prediction to ensure it sums to 1
    probabilities = blended_prediction.flatten()
    probabilities /= np.sum(probabilities)

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(categories, probabilities, color="skyblue")

    # Customize the chart
    ax.set_title("Estimated Probabilities of Playerbase Size", pad=20)
    ax.set_xlabel("Estimated Playerbase (Boxleiter, rounded)")
    ax.set_ylabel("Probability")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Add value labels on top of each bar
    for i, prob in enumerate(probabilities):
        ax.text(i, prob, f"{prob:.3f}", ha="center", va="bottom")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Display in Streamlit
    st.pyplot(fig)


#######################################################################
# Suggestions
#######################################################################


def get_score_for_prediction(prediction: list):
    """Calculate an arbitrary "score" for a set of predictions."""
    score = 0
    for i, prob in enumerate(prediction.flatten()):
        score += prob * i

    return score


st.write("### Suggestions")
progress_bar = st.progress(0, "Loading models...")
with progress_bar:

    scores = {}

    def evaluate_feature_change_score(feature_name, value):
        """Test a specific change by modifying the input element and running inference to identify the most impactful changes."""
        test_name = f"{feature_name}_to_{value}"
        if scores.get(test_name) is not None:
            return scores[test_name]  # If the test has already been run, return the score
        else:
            # print(f"Running test for {test_name}...")
            modified_element = input_element.copy()  # Create a copy of the input element to modify
            modified_element.loc[0, feature_name] = value  # Modify it for the test

            # Get predictions
            preds = []
            preds.append(model_lgbm_high.predict_proba(modified_element))
            preds.append(model_lgbm_low.predict_proba(modified_element))
            # preds.append(model_xgb.predict_proba(modified_element)) # XGBoost is like 25x slower, so maybe don't use it here

            blended_prediction = np.mean(preds, axis=0)
            blended_prediction /= np.sum(blended_prediction)  # Normalize the blended prediction to ensure it sums to 1

            return get_score_for_prediction(blended_prediction.flatten())

    tests_queue = []  # Stores tests to run as tuples (feature, value)

    # First, test "removing" user-input features by setting them to baseline values
    baseline_element = utils.load_baseline_element()
    for feature, value in user_input_features.items():
        # If the feature's value is not the baseline, test it
        if feature in baseline_element and input_element[feature][0] != value:
            tests_queue.append((feature, baseline_element[feature][0]))

    # Test flipping every feature that starts with "~" (i.e., binary features)
    for feature, value in input_element.iloc[0].items():
        if feature.startswith("~"):
            # flip the feature from user input
            value = 1 if value == 0 else 0
            tests_queue.append((feature, value))

    hard_coded_tests = {
        "average_time_to_beat": [0, 5, 10, 20, 50, 100],
        "categories_count": [0, 1, 2, 3, 4, 5, 10, 15, 20],
        "controller_support": [-1, 0, 1],
        "early_access": [-1, 0, 1],
        "gamefaqs_difficulty_rating": [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        "genres_count": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "has_demos": [-1, 0, 1],
        "languages_supported_count": [0, 1, 2, 3, 5, 10, 20],
        "languages_with_full_audio_count": [0, 1, 2, 3, 5, 10, 20],
        "monetization_model": [-1, 0, 1, 2],
        "price_original": [0.99, 1.99, 2.99, 4.99, 9.99, 14.99, 19.99, 24.99, 29.99, 49.99, 59.99, 79.99],
        "release_day_of_month": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
        "release_month": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "required_age": [0, 12, 13, 16, 18],
        "runs_on_linux": [-1, 0, 1],
        "runs_on_mac": [-1, 0, 1],
        "runs_on_steam_deck": [-1, 0, 1],
        "runs_on_windows": [-1, 0, 1],
        "steam_store_movie_count": [0, 1, 2, 3, 4, 5, 10, 20],
        "steam_store_screenshot_count": [0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50],
        "tags_count": [0, 1, 2, 3, 4, 5, 10, 20, 30, 37],
        "vr_only": [-1, 0, 1],
        "vr_supported": [-1, 0, 1],
    }

    # Add hard-coded tests
    for feature, values in hard_coded_tests.items():
        for value in values:
            tests_queue.append((feature, value))

    # Debug - Keep only the first 50 test to speed testing
    # tests_queue = tests_queue[:50]

    # Run all tests in the queue
    count = 0
    for feature, value in tests_queue:
        count += 1
        progress_bar.progress(count / len(tests_queue), f"Analyzing interactions... {count}/{len(tests_queue)}...")
        score = evaluate_feature_change_score(feature, value)
        scores[f"{feature}_to_{value}"] = score

    # Get a baseline score using the exact same estimators, by passing a feature and its value in the input element as is
    user_input_baseline_score = evaluate_feature_change_score("average_time_to_beat", input_element["average_time_to_beat"][0])
    print(f"Baseline score: {user_input_baseline_score}")

    # Make scores relative to the baseline score
    for score in scores:
        scores[score] = scores[score] - user_input_baseline_score

progress_bar.empty()
print("Scores:")
# Sort the scores dictionary by value in descending order
sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

# Debug - Print each label + score, one per line
for label, score in sorted_scores.items():
    print(f"{label}: {score}")


# Display the top N suggestions
suggestions_to_show = 9999
st.write("This table displays the changes most likely to impact the estimated playerbase.")
st.write("The higher the score, the stronger the predicted positive impact.")

# Build a simple dataframe to display the suggestions
suggestions_df = pd.DataFrame(sorted_scores.items(), columns=["Feature Change", "Score"])

# Filter out suggestions with absolute score below the cutoff
count_before_filter = len(suggestions_df)
suggestions_df = suggestions_df[suggestions_df["Score"].abs() > suggestion_score_cutoff]
suggestions_df = suggestions_df.sort_values(by="Score", ascending=False)
st.write(f"Showing {len(suggestions_df)} suggestions ({count_before_filter - len(suggestions_df)} hidden for scoring below the cutoff)")

suggestions_df["Score"] = suggestions_df["Score"].apply(lambda x: f"{x:.3f}")
suggestions_df["Feature Change"] = suggestions_df["Feature Change"].apply(lambda x: x.replace("_to_", " = "))
# suggestions_df["Feature Change"] = suggestions_df["Feature Change"].apply(lambda x: x.replace("_", " "))


st.dataframe(suggestions_df.head(suggestions_to_show), use_container_width=True, hide_index=True)
