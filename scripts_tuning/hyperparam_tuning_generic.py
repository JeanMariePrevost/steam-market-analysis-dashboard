"""
This script allows you to optimize any model that taks in nothing but a set of hyperparameters and returns a score.
It uses Optuna to optimize the hyperparameters.
It does not support models that require custom logic beyond a search space and model type.
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
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
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


#################################################################
# Testing with WAY more tags to drop from not having any predictive power
#################################################################
low_relevance_columns = [
    "~tag_fantasy",
    "~tag_puzzle",
    "~category_vr supported",
    "~tag_looter shooter",
    "~category_remote play together",
    "~tag_horses",
    "~lang_audio_ukrainian",
    "~tag_assassin",
    "~lang_audio_thai",
    "~tag_beautiful",
    "~tag_minimalist",
    "~tag_2d platformer",
    "~lang_audio_turkish",
    "~tag_party game",
    "~tag_historical",
    "~tag_world war ii",
    "~tag_memes",
    "~tag_western",
    "~tag_indie",
    "~tag_logic",
    "~category_family sharing",
    "~tag_arcade",
    "~tag_retro",
    "~tag_hidden object",
    "steam_store_screenshot_count",
    "~tag_automobile sim",
    "~tag_fighting",
    "~tag_linear",
    "~tag_shoot 'em up",
    "~tag_aliens",
    "~tag_lore-rich",
    "~tag_loot",
    "~tag_extraction shooter",
    "~tag_colorful",
    "~tag_space",
    "~tag_hunting",
    "~tag_moba",
    "~tag_physics",
    "~genre_early access",
    "~tag_blood",
    "~tag_stylized",
    "~tag_crime",
    "~tag_dragons",
    "~tag_management",
    "~tag_visual novel",
    "~tag_controller",
    "~tag_heist",
    "~tag_old school",
    "~tag_economy",
    "~category_tracked controller support",
    "~tag_destruction",
    "~tag_ninja",
    "~tag_fast-paced",
    "~tag_interactive fiction",
    "~tag_abstract",
    "~tag_pixel graphics",
    "~tag_text-based",
    "~tag_3d",
    "~tag_2d",
    "~tag_city builder",
    "~tag_kickstarter",
    "~tag_silent protagonist",
    "~tag_swordplay",
    "~tag_dark",
    "~tag_split screen",
    "~tag_clicker",
    "~tag_rts",
    "~tag_grid-based movement",
    "~tag_based on a novel",
    "~tag_co-op campaign",
    "~tag_games workshop",
    "~tag_top-down shooter",
    "~lang_audio_norwegian",
    "~tag_rpgmaker",
    "~tag_vr",
    "~lang_audio_bulgarian",
    "~tag_point & click",
    "~lang_bangla",
    "~tag_cute",
    "~lang_urdu",
    "~tag_driving",
    "~lang_estonian",
    "~lang_uzbek",
    "~tag_education",
    "~lang_albanian",
    "~lang_bosnian",
    "~lang_galician",
    "~lang_azerbaijani",
    "~tag_superhero",
    "~lang_gujarati",
    "~lang_icelandic",
    "~lang_tamil",
    "~lang_mongolian",
    "~lang_macedonian",
    "~lang_luxembourgish",
    "~lang_zulu",
    "~lang_turkmen",
    "~lang_tajik",
    "~lang_swahili",
    "~lang_telugu",
    "~tag_colony sim",
    "~lang_tatar",
    "~lang_kyrgyz",
    "~tag_medieval",
    "~lang_marathi",
    "~lang_nepali",
    "~lang_hausa",
    "~lang_maori",
    "~lang_igbo",
    "~lang_yoruba",
    "~lang_malayalam",
    "~lang_kinyarwanda",
    "~lang_maltese",
    "~lang_scots",
    "~lang_uyghur",
    "~lang_sinhala",
    "~lang_kannada",
    "~lang_khmer",
    "~lang_assamese",
    "~lang_slovenian",
    "~lang_odia",
    "~lang_xhosa",
    "~lang_serbian",
    "~lang_hebrew",
    "~lang_quechua",
    "~lang_konkani",
    "~lang_amharic",
    "~lang_sindhi",
    "~lang_sotho",
    "~tag_turn-based",
    "~tag_local multiplayer",
    "~lang_tigrinya",
    "~lang_dari",
    "~lang_cherokee",
    "~lang_valencian",
    "~lang_wolof",
    "~lang_sorani",
    "~lang_tswana",
    "~lang_armenian",
    "~tag_nudity",
    "~tag_top-down",
    "~tag_crowdfunded",
    "~lang_irish",
    "~lang_welsh",
    "~tag_match 3",
    "~lang_afrikaans",
    "~tag_time management",
    "~tag_score attack",
    "~lang_audio_malay",
    "~tag_diplomacy",
    "~tag_collectathon",
    "~lang_filipino",
    "~lang_basque",
    "~tag_grand strategy",
    "~tag_conversation",
    "~lang_audio_serbian",
    "~lang_audio_bangla",
    "~lang_malay",
    "~lang_audio_gujarati",
    "~lang_audio_georgian",
    "~lang_audio_urdu",
    "~lang_audio_zulu",
    "~lang_audio_bosnian",
    "~lang_audio_albanian",
    "~lang_audio_latvian",
    "~lang_audio_uzbek",
    "~lang_audio_basque",
    "~lang_audio_estonian",
    "~lang_audio_azerbaijani",
    "~lang_audio_galician",
    "~lang_audio_tamil",
    "~lang_audio_mongolian",
    "~lang_audio_macedonian",
    "~lang_audio_khmer",
    "~lang_audio_welsh",
    "~lang_audio_igbo",
    "~lang_audio_kannada",
    "~lang_audio_belarusian",
    "~lang_audio_marathi",
    "~lang_audio_scots",
    "~lang_audio_telugu",
    "~lang_audio_odia",
    "~lang_audio_sinhala",
    "~lang_audio_amharic",
    "~lang_audio_uyghur",
    "~lang_audio_turkmen",
    "~lang_audio_hausa",
    "~lang_audio_armenian",
    "~lang_audio_tatar",
    "~lang_audio_tajik",
    "~lang_audio_nepali",
    "~lang_audio_sotho",
    "~lang_audio_malayalam",
    "~lang_audio_maltese",
    "~lang_audio_valencian",
    "~lang_audio_swahili",
    "~lang_audio_yoruba",
    "~lang_audio_luxembourgish",
    "~lang_audio_kinyarwanda",
    "~lang_audio_xhosa",
    "~lang_audio_tswana",
    "~lang_audio_wolof",
    "~lang_audio_cherokee",
    "~lang_audio_tigrinya",
    "~lang_audio_maori",
    "~lang_audio_assamese",
    "~lang_audio_sorani",
    "~lang_audio_konkani",
    "~lang_audio_dari",
    "~lang_audio_sindhi",
    "~lang_audio_irish",
    "~lang_audio_kazakh",
    "~lang_audio_quechua",
    "~lang_audio_kyrgyz",
    "~lang_audio_icelandic",
    "~lang_audio_hebrew",
    "~lang_audio_afrikaans",
    "~lang_audio_slovenian",
    "~lang_audio_filipino",
    "~lang_audio_romanian",
    "~tag_mystery dungeon",
    "~tag_isometric",
    "~tag_star wars",
    "~tag_tabletop",
    "~lang_audio_persian",
    "~tag_nonlinear",
    "~tag_animation & modeling",
    "~tag_artificial intelligence",
    "~tag_nature",
    "~tag_episodic",
    "~tag_lara croft",
    "~tag_solitaire",
    "~lang_vietnamese",
    "~tag_puzzle platformer",
    "~lang_audio_greek",
    "~tag_arena shooter",
    "~tag_bullet hell",
    "~lang_audio_croatian",
    "~tag_warhammer 40k",
    "~tag_runner",
    "~genre_violent",
    "~tag_psychedelic",
    "~category_steamvr collectibles",
    "~tag_precision platformer",
    "~tag_1990's",
    "~lang_latvian",
    "~tag_illuminati",
    "~category_shared/split screen co-op",
    "~tag_experimental",
    "~tag_modern",
    "~tag_vr only",
    "~tag_surreal",
    "~lang_audio_indonesian",
    "~tag_underground",
    "~tag_idler",
    "~tag_sokoban",
    "~tag_flight",
    "~lang_hindi",
    "~lang_belarusian",
    "~tag_6dof",
    "~tag_vampire",
    "~tag_resource management",
    "~tag_hand-drawn",
    "~lang_croatian",
    "~lang_kazakh",
    "~tag_addictive",
    "~tag_cats",
    "~tag_gothic",
    "~lang_audio_lithuanian",
    "~tag_relaxing",
    "~tag_side scroller",
    "~lang_audio_hindi",
    "~tag_turn-based tactics",
    "~tag_action-adventure",
    "~genre_education",
    "~genre_gore",
    "~tag_otome",
    "~tag_beat 'em up",
    "~tag_cinematic",
    "~tag_immersive sim",
    "~tag_automation",
    "~tag_board game",
    "~tag_trivia",
    "~tag_word game",
    "~tag_reboot",
    "~tag_mystery",
    "~tag_on-rails shooter",
    "~tag_drama",
    "~tag_steampunk",
    "~tag_lgbtq+",
    "~tag_walking simulator",
    "~tag_video production",
    "~tag_1980s",
    "~tag_minigames",
    "~tag_choose your own adventure",
    "~tag_philosophical",
    "~lang_audio_vietnamese",
    "release_month",
    "~tag_conspiracy",
    "~tag_touch-friendly",
    "~tag_3d vision",
    "~genre_video production",
    "~genre_software training",
    "~lang_georgian",
    "~lang_audio_czech",
    "~lang_slovak",
    "~lang_lithuanian",
    "~tag_traditional roguelike",
    "~tag_female protagonist",
    "release_day_of_month",
    "~lang_audio_arabic",
    "~tag_spelling",
    "~tag_typing",
    "~tag_action rts",
    "~tag_card game",
    "~tag_nostalgia",
    "~tag_3d fighter",
    "~tag_time manipulation",
    "~tag_real time tactics",
    "~tag_farming sim",
    "~tag_mouse only",
    "~tag_bullet time",
    "~tag_cozy",
    "~tag_immersive",
    "~tag_racing",
    "~tag_music-based procedural generation",
    "~tag_benchmark",
    "~genre_sexual content",
    "~genre_nudity",
    "~tag_software training",
    "~tag_experience",
    "~tag_lovecraftian",
    "~tag_dynamic narration",
    "~tag_metroidvania",
    "~tag_asymmetric vr",
    "~tag_snow",
    "~tag_musou",
    "~tag_philisophical",
    "~tag_time attack",
    "~genre_game development",
    "~tag_america",
    "~tag_outbreak sim",
    "~tag_wholesome",
    "~tag_combat racing",
    "~lang_audio_slovak",
    "~tag_boxing",
    "~tag_baseball",
    "~tag_twin stick shooter",
    "~tag_2.5d",
    "~tag_parody",
    "~genre_strategy",
    "~tag_programming",
    "~tag_bikes",
    "~tag_sequel",
    "~tag_lego",
    "~tag_gambling",
    "~tag_emotional",
    "~tag_asynchronous multiplayer",
    "~tag_electronic music",
    "~tag_360 video",
    "~tag_werewolves",
    "~tag_tennis",
    "~tag_farming",
    "~tag_pinball",
    "~tag_dark humor",
    "~tag_dinosaurs",
    "~tag_pool",
    "~tag_web publishing",
    "~tag_dark comedy",
    "~tag_transportation",
    "~tag_dating sim",
    "~tag_mars",
    "~tag_turn-based strategy",
    "~tag_hockey",
    "~tag_martial arts",
    "~tag_narrative",
    "~tag_gamemaker",
    "~tag_dog",
    "~tag_instrumental music",
    "~tag_auto battler",
    "~tag_quick-time events",
    "~category_vr support",
    "~tag_faith",
    "~tag_roguevania",
    "~genre_web publishing",
    "~tag_card battler",
    "~tag_8-bit music",
    "~tag_god game",
    "~tag_motocross",
    "~tag_cycling",
    "~tag_medical sim",
    "~tag_wargame",
    "~tag_noir",
    "~tag_boomer shooter",
    "~tag_ambient",
    "~tag_unforgiving",
    "~tag_cricket",
    "~tag_world war i",
    "~genre_accounting",
    "~tag_action roguelike",
    "~tag_underwater",
    "~tag_anime",
    "~tag_hentai",
    "~tag_feature film",
    "~tag_political",
    "~tag_soundtrack",
    "runs_on_windows",
    "~tag_agriculture",
    "~genre_racing",
    "~tag_rock music",
    "~tag_electronic",
    "~tag_sports",
    "~tag_lemmings",
    "~tag_jump scare",
    "~tag_wrestling",
    "~tag_documentary",
    "~tag_audio production",
    "~tag_mechs",
    "~tag_rugby",
    "~tag_cooking",
    "~tag_strategy rpg",
    "~tag_sexual content",
    "~tag_spaceships",
    "~tag_steam machine",
    "~tag_football (american)",
    "~tag_volleyball",
    "~tag_job simulator",
    "~tag_offroad",
    "~tag_mahjong",
    "~tag_movie",
    "~tag_fox",
    "~genre_audio production",
    "~tag_shop keeper",
    "~tag_coding",
    "~tag_time travel",
    "~tag_submarine",
    "~tag_voice control",
    "~tag_hobby sim",
    "~genre_sports",
    "~tag_intentionally awkward controls",
    "~tag_short",
    "~tag_birds",
    "~tag_well-written",
    "~tag_villain protagonist",
    "~tag_vehicular combat",
    "~category_mods (require hl2)",
    "~tag_golf",
    "~tag_real-time",
    "~tag_gun customization",
    "controller_support",
    "~category_mods",
    "~genre_360 video",
    "~genre_documentary",
    "~genre_episodic",
    "~genre_movie",
    "~genre_short",
    "~genre_tutorial",
    "~tag_dwarf",
    "~tag_elf",
    "~tag_snooker",
    "~tag_tile-matching",
    "~tag_naval combat",
    "~tag_cold war",
    "~tag_boss rush",
    "~tag_alternate history",
    "~tag_skateboarding",
    "~tag_fmv",
    "~tag_skating",
    "~tag_politics",
    "~tag_bmx",
    "~tag_satire",
    "~tag_political sim",
    "~tag_voxel",
    "~tag_vikings",
    "~tag_sniper",
    "~tag_level editor",
    "~tag_transhumanism",
    "~tag_party-based rpg",
    "~tag_foreign",
    "~tag_roguelike deckbuilder",
    "~tag_party",
    "~tag_cartoon",
    "~tag_snowboarding",
    "~tag_dungeons & dragons",
    "~tag_fishing",
    "~tag_skiing",
    "~tag_atv",
    "~tag_mini golf",
    "~tag_motorbike",
    "~tag_inventory management",
    "~tag_escape room",
    "~lang_audio_danish",
    "~category_steam leaderboards",
    "~tag_character action game",
    "~tag_jet",
    "~tag_mythology",
    "~tag_capitalism",
    "~tag_comic book",
    "~tag_roguelite",
    "~tag_deckbuilding",
    "~tag_naval",
    "~tag_combat",
    "~tag_hardware",
    "~tag_roguelike",
    "~tag_life sim",
    "~tag_basketball",
    "~tag_trains",
    "~tag_cyberpunk",
    "~tag_futuristic",
    "~tag_dystopian",
    "~tag_multiple endings",
    "~tag_spectacle fighter",
    "~lang_audio_dutch",
    "~tag_hacking",
    "~tag_creature collector",
    "~tag_narration",
    "~tag_trading card game",
    "~tag_rome",
    "~tag_jrpg",
    "~tag_sailing",
    "~tag_football (soccer)",
    "~tag_mining",
    "~tag_turn-based combat",
    "~tag_bowling",
    "~tag_tactical rpg",
    "~tag_music",
    "~tag_hex grid",
    "~tag_tutorial",
    "~tag_robots",
    "~tag_magic",
    "~genre_simulation",
    "~tag_dungeon crawler",
    "~tag_procedural generation",
    "~tag_3d platformer",
    "~tag_space sim",
    "~tag_choices matter",
    "~tag_rhythm",
    "~tag_2d fighter",
    "~tag_crpg",
    "~tag_investigation",
    "~tag_design & illustration",
    "~tag_chess",
    "~tag_software",
    "~tag_utilities",
    "~lang_indonesian",
    "~lang_persian",
    "~tag_tower defense",
    "~lang_audio_hungarian",
    "~tag_cartoony",
    "~tag_nsfw",
    "~lang_audio_finnish",
    "~tag_family friendly",
    "~tag_perma death",
    "~tag_archery",
    "~tag_romance",
    "~category_steam turn notifications",
    "~tag_football",
    "~tag_4x",
    "~genre_utilities",
    "~tag_tanks",
    "~tag_platformer",
    "~tag_pirates",
    "~tag_detective",
    "~tag_psychological",
    "~tag_demons",
    "~tag_game development",
    "~tag_social deduction",
    "~lang_catalan",
    "~tag_soccer",
    "~category_shared/split screen pvp",
    "~category_shared/split screen",
    "~genre_design & illustration",
    "~tag_science",
    "~genre_animation & modeling",
    "~lang_audio_swedish",
    "~tag_4 player local",
    "~tag_thriller",
    "~lang_audio_catalan",
    "~tag_photo editing",
    "~genre_photo editing",
    "~tag_gaming",
    "~tag_supernatural",
    "~genre_adventure",
]

# data = data.drop(columns=low_relevance_columns, errors="ignore")


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

# For slower models or initial exploration, consider using only a small subsample of the data
# Even 0.05 seems to be enough to get a good idea of the model's performance
fraction = 0.05  # Fraction of the data to use for training
if fraction <= 0.1:
    # Very small sample, filter out the outliers, very weakly at the top, heavily at the bottom
    data = data[data["steam_total_reviews"] > 1]  # Remove games with 0-1 reviews
    top_games_to_remove = 25
    data = data.sort_values("steam_total_reviews", ascending=False).iloc[top_games_to_remove:]  # Remove the top 25 games because if they are in the sample, they will skew the results


y = data["steam_total_reviews"]  # Extract the target
X = data.drop(columns=columns_to_drop, errors="ignore")  # Extract the predictors, dropping the irrelevant columns and the target

if fraction < 1.0:
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.sample(frac=fraction, random_state=42)
    X_valid = X_valid.sample(frac=fraction, random_state=42)
    y_train = y_train.loc[X_train.index]
    y_valid = y_valid.loc[X_valid.index]
    # Print a warning banner in red
    print("\033[91m\n" + f"WARNING:\nUsing only {fraction} of the dataset \nfor faster training; results may not generalize fully.\n\033[0m")
    print(f"Number of rows in the sample: {len(X_train)}")
else:
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the target variable
scaler = StandardScaler()
y_train = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_valid = scaler.transform(y_valid.values.reshape(-1, 1)).flatten()

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
    params = {}

    # Solver selection remains exploratory.
    params["solver"] = "adam"

    # # Network architecture
    n_layers = trial.suggest_int("n_layers", 2, 3)
    layer1_units = trial.suggest_int("layer1_units", 32, 256)
    layer2_units = trial.suggest_int("layer2_units", 8, layer1_units)  # Layer 2 must be less than or equal to layer 1

    # For a three-layer network, enforce that the third layer is at most "K" and not more than layer2_units.
    if n_layers == 3:
        layer3_units_high = min(16, layer2_units)  # Layer 3 must be less than or equal to layer 2 and has a maximum
        layer3_units = trial.suggest_int("layer3_units", 4, layer3_units_high)
        params["hidden_layer_sizes"] = (layer1_units, layer2_units, layer3_units)
    else:
        params["hidden_layer_sizes"] = (layer1_units, layer2_units)

    # Debug, testing for training speed
    # params["hidden_layer_sizes"] = (50, 50, 4)  # ~10s @330 rows
    # params["hidden_layer_sizes"] = (100, 50, 4)  # ~15s @330 rows
    # params["hidden_layer_sizes"] = (100, 100, 4)  # ~18s @330 rows
    # params["hidden_layer_sizes"] = (200, 100, 4)  # ~37s @330 rows
    # params["hidden_layer_sizes"] = (250, 250, 4)  # ~53S @330 rows

    # Other hyperparameters
    params["activation"] = "relu"
    params["alpha"] = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
    # params["batch_size"] = trial.suggest_int("batch_size", 4, 256, log=True)
    params["beta_1"] = trial.suggest_float("beta_1", 0.85, 0.99)
    params["beta_2"] = trial.suggest_float("beta_2", 0.9, 0.9999, log=True)
    params["epsilon"] = trial.suggest_float("epsilon", 1e-9, 1e-7, log=True)
    params["learning_rate_init"] = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)
    params["learning_rate"] = "adaptive"
    params["max_iter"] = 1000
    params["shuffle"] = True

    # Early stopping
    params["early_stopping"] = True  # No worries if False, it's a performance hit, but it's worth it and isn't exponentially slower
    params["n_iter_no_change"] = 100
    params["tol"] = 1e-5  # No worries since worst case it no early stopping

    # Misc
    params["verbose"] = False
    params["warm_start"] = False

    return params


fixed_params = {
    # "verbosity": 0,  # Suppress XGBRegressor output
    # "verbose": -1,  # Suppress LGBMRegressor output
}

# Try SVR, ElasticNet, and a decently simple MLPRegressor before moving to stack
model_class = MLPRegressor


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
