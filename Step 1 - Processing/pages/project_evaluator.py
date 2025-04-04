import os

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.ticker import AutoMinorLocator

import utils
from utils import load_main_dataset

# Page configuration & custom CSS
st.set_page_config(page_title="Overview of Releases & Trends")
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
st.write("Multiple machine learning models predictions are blended to achieve a more stable estimated number of steam reviews, used as a direct proxy for the number of players.")


# Sidebar Filter: Release Year Range
st.sidebar.title("Input")
# min_year = int(df["release_year"].min())
# max_year = int(df["release_year"].max())
# selected_year_range = st.sidebar.slider("Select Release Year Range", min_year, max_year, (2007, max_year))

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
    "Game Name",
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


# Debug
st.write("Input:")
st.write(input_text)

# Extracting features from the input text
input_lines = input_text.strip().split("\n")
features = {}
for line in input_lines:
    if "=" in line:
        feature, value = line.split("=")
        feature = feature.strip()
        value = value.strip()
        if feature.startswith("~"):
            if value == "1" or value == "0":
                features[feature] = 1 if value == "1" else 0  # Convert to binary
            else:
                st.error(f"Value for '{feature}' must be either 0 or 1. Got: '{value}' \n\n Defaulting to baseline.")
        else:
            try:
                features[feature] = float(value)  # Convert to float
            except ValueError:
                st.error(f"Value for '{feature}' must be a valid number. Got: '{value}' \n\n Defaulting to baseline.")


st.write("Extracted Features:")
st.write(features)

# Build element from baseline
input_element = utils.load_baseline_element()
for feature, value in features.items():
    if feature in input_element:
        input_element[feature] = value
    else:
        closest_match = utils.find_closest_string_in_list(feature, input_element.keys())
        if closest_match:
            st.warning(f"Feature '{feature}' not recognized and will be skipped. Did you mean '{closest_match}'?")
        else:
            st.warning(f"Feature '{feature}' not recognized and will be skipped. No close match found.")
st.write("Baseline Element:")
st.write(input_element)


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


# Run inference
with st.spinner("Running inference...", show_time=True):
    lgbm_high_preds = model_lgbm_high.predict_proba(input_element)
    lgbm_low_preds = model_lgbm_low.predict_proba(input_element)
    xgb_preds = model_xgb.predict_proba(input_element)

    st.write("Predictions:")
    st.write(f"Predictions for {MODEL_NAME_LGBM_HIGH}: {lgbm_high_preds}")
    st.write(f"Predictions for {MODEL_NAME_LGBM_LOW}: {lgbm_low_preds}")
    st.write(f"Predictions for {MODEL_NAME_XGB}: {xgb_preds}")

    # Calculate the blended prediction
    blended_prediction = (lgbm_high_preds + lgbm_low_preds + xgb_preds) / 3
    st.write(f"Blended Prediction: {blended_prediction}")

# Print a nice bar chart with the blended prediction
with st.spinner("Creating bar chart...", show_time=True):
    # Define x-axis labels
    bins = [0, 50, 100, 250, 500, 1000, 2500, 5000, np.inf]
    categories = []
    for i in range(len(bins) - 1):
        if bins[i] == 0:
            categories.append(f"< {bins[i + 1]}")
        elif bins[i + 1] == np.inf:
            categories.append(f">= {bins[i]}")
        else:
            categories.append(f"{bins[i]} - {bins[i + 1]}")
    print(f"Categories: {categories}")

    # Normalize the blended prediction to ensure it sums to 1
    probabilities = blended_prediction.flatten()
    probabilities /= np.sum(probabilities)

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(categories, probabilities, color="skyblue")

    # Customize the chart
    ax.set_title("Probability Distribution Across Categories", pad=20)
    ax.set_xlabel("Estimated Number of Reviews")
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
