import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from optuna.trial import TrialState
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error, r2_score
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
fraction = 1.0  # Fraction of the data to use for training
if fraction <= 0.1:
    print("\033[91m\n" + f"WARNING:\nUsing a very small fraction ({fraction}) of the dataset \nfor faster training; training data will exclude outliers and may not generalize well.\n\033[0m")
    # Very small sample, filter out the outliers, very weakly at the top, heavily at the bottom
    data = data[data["steam_total_reviews"] > 1]  # Remove games with 0-1 reviews
    top_games_to_remove = 25
    data = data.sort_values("steam_total_reviews", ascending=False).iloc[top_games_to_remove:]  # Remove the top 25 games because if they are in the sample, they will skew the results


y = data["steam_total_reviews"]  # Extract the target
X = data.drop(columns=columns_to_drop, errors="ignore")  # Extract the predictors, dropping the irrelevant columns and the target

# seed = 42  # Set a random state for reproducibility
seed = random.randint(0, 1000)  # Or uncomment this to use a random seed
print(f"Random seed: {seed}")

if fraction < 1.0:
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_train = X_train.sample(frac=fraction, random_state=seed)
    X_valid = X_valid.sample(frac=fraction, random_state=seed)
    y_train = y_train.loc[X_train.index]
    y_valid = y_valid.loc[X_valid.index]
    # Print a warning banner in red
    print("\033[91m\n" + f"WARNING:\nUsing only {fraction} of the dataset \nfor faster training; results may not generalize fully.\n\033[0m")
    print(f"Number of rows in the sample: {len(X_train)}")
else:
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=seed)

# Scale the target variable
scaler = StandardScaler()
y_train = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_valid = scaler.transform(y_valid.values.reshape(-1, 1)).flatten()


# Define base models with your tuned hyperparameters
mlp_reg = MLPRegressor(
    solver="adam",
    hidden_layer_sizes=(216, 162),
    activation="relu",
    alpha=0.02841699292881125,
    batch_size=136,
    learning_rate="adaptive",
    learning_rate_init=0.005974385093746269,
    max_iter=50,
    shuffle=True,
    tol=1.3074343950880321e-07,
    verbose=True,
    warm_start=False,
    beta_1=0.8107951547310408,
    beta_2=0.8989192729472247,
    epsilon=1.0242032157822251e-07,
    early_stopping=False,
    n_iter_no_change=39,
)

extra_trees = ExtraTreesRegressor(
    n_estimators=467,
    criterion="friedman_mse",
    max_depth=34,
    min_samples_split=69,
    max_features=0.25392231481556493,
    min_impurity_decrease=4.9985173365604005,
    bootstrap=False,
    ccp_alpha=7.84125187827584e-05,
)

dt_regressor = DecisionTreeRegressor(
    criterion="friedman_mse",
    splitter="best",
    max_depth=46,
    min_samples_split=51,
    min_samples_leaf=75,
    max_features=0.9587016970482302,
    min_impurity_decrease=2.5864712856205774,
    ccp_alpha=6.23553680715953e-05,
)

elastic_net = ElasticNet(
    alpha=0.060894035886975076,
    l1_ratio=0.7560695882925308,
)

rf_regressor = RandomForestRegressor(
    ccp_alpha=0.000151,
    criterion="friedman_mse",
    max_depth=11,
    max_features=0.3,
    max_samples=0.855,
)


xgb_regressor_pruned = XGBRegressor(
    colsample_bytree=0.4799,
    eval_metric="rmsle",
    # feature_selector="greedy",
    gamma=1.826,
    grow_policy="lossguide",
    learning_rate=0.024960677618899953,
    max_delta_step=9,
    max_depth=9,
    max_leaves=112,
    min_child_weight=6,
    n_estimators=165,
    num_parallel_tree=10,
    reg_alpha=2.5,
    reg_lambda=0.02,
    subsample=0.8155,
    tree_method="approx",
    # top_k=32,
)

xgb_regressor_limited = XGBRegressor(
    colsample_bytree=0.25968154894274487,
    eval_metric="rmsle",
    # feature_selector="greedy",
    gamma=0.1855174849750825,
    grow_policy="lossguide",
    learning_rate=0.8267257948740029,
    max_delta_step=1,
    max_depth=6,
    max_leaves=13,
    min_child_weight=6,
    n_estimators=38,
    num_parallel_tree=21,
    reg_alpha=18.940284499594384,
    reg_lambda=2.2458027527360963e-05,
    subsample=0.9873311844726601,
    tree_method="hist",
)


lgbm_regressor = LGBMRegressor(
    boosting_type="gbdt",
    colsample_bytree=0.2970113752542446,
    importance_type="gain",
    learning_rate=0.019064828567768227,
    max_depth=13,
    min_child_samples=2,
    min_child_weight=0.08,
    min_split_gain=0.0013416702983401459,
    n_estimators=833,
    num_leaves=114,
    reg_alpha=0.0020191308023190433,
    reg_lambda=29.692567967373314,
    subsample=0.5931035850505068,
    subsample_for_bin=22518,
    subsample_freq=4,
    verbosity=-1,
)


base_models = {
    # "mlp": mlp_reg, # VERY slow, talking 1+ hours to train, every other model is like seconds to 1-3 minutes
    "xgb_pruned": xgb_regressor_pruned,
    "xgb_limited": xgb_regressor_limited,
    "lgbm": lgbm_regressor,
    "rf": rf_regressor,
    "et": extra_trees,
    "dt": dt_regressor,
    "elastic_net": elastic_net,
}

# Train and evaluate each model as a sanity check
scores = {}
for name, model in base_models.items():
    print(f"Training {name} model...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    score = r2_score(y_valid, y_pred)
    scores[name] = score
    print(f"{name} R² score on validation set: {score:.4f}\n")

# Reprint the scores
print("Scores:")
for name, score in scores.items():
    print(f"{name}: {score:.4f}")

# print mean score
mean_score = np.mean(list(scores.values()))
print(f"\nMean R² score: {mean_score:.4f}")


# Qui


#########################################################################################################
# Blended model quick test
#########################################################################################################
print("\n\n\n")
print('Starting the "blended" model test...')
# We only use the "best" few models here
blend_base_models = {
    "xgb_pruned": xgb_regressor_pruned,
    "xgb_limited": xgb_regressor_limited,
    "lgbm": lgbm_regressor,
    "rf": rf_regressor,
    "et": extra_trees,
    "dt": dt_regressor,
    "elastic_net": elastic_net,
}

# 1. Generate predictions from each model on X_valid
predictions = []
for name, model in blend_base_models.items():
    pred = model.predict(X_valid)
    predictions.append(pred)
    # mse = mean_squared_error(y_valid, pred)
    # print(f"Model {name} MSE: {mse}")
    r2 = r2_score(y_valid, pred)
    print(f"Model {name} R² score: {r2:.3f}")

# 2. Compute the simple average of predictions
#    This will average across the predictions from each model.
blend_pred_mean = np.mean(predictions, axis=0)
blend_pred_median = np.median(predictions, axis=0)
blend_pred_max = np.max(predictions, axis=0)

# 3. Calculate R² score for each blended prediction
mean_blend_r2 = r2_score(y_valid, blend_pred_mean)
median_blend_r2 = r2_score(y_valid, blend_pred_median)
max_blend_r2 = r2_score(y_valid, blend_pred_max)
percentile_60th_blend_r2 = r2_score(y_valid, np.percentile(predictions, 60, axis=0))
percentile_40th_blend_r2 = r2_score(y_valid, np.percentile(predictions, 40, axis=0))
print(f"Mean blended model R² score: {mean_blend_r2:.3f}")
print(f"Median blended model R² score: {median_blend_r2:.3f}")
print(f"Max blended model R² score: {max_blend_r2:.3f}")
print(f"75th percentile blended model R² score: {percentile_60th_blend_r2:.3f}")
print(f"25th percentile blended model R² score: {percentile_40th_blend_r2:.3f}")


# Testing the different _combinations_
import itertools

print("\nTesting combinations of 2 to n-1 base models:")
# Here, n is the number of base models.
# We test for combinations of size 2 up to n-1 (thus not including the ensemble of all models,
# which we already computed above).
combos_and_scores = {}
for r in range(2, len(blend_base_models)):
    for combo in itertools.combinations(blend_base_models.keys(), r):
        # Generate predictions for the current combination of models
        combo_predictions = [blend_base_models[name].predict(X_valid) for name in combo]
        # Mean-based blend
        combo_mean_pred = np.mean(combo_predictions, axis=0)
        mean_r2 = r2_score(y_valid, combo_mean_pred)
        combos_and_scores[f"mean_{combo}"] = mean_r2

        # Median-based blend
        combo_median_pred = np.median(combo_predictions, axis=0)
        median_r2 = r2_score(y_valid, combo_median_pred)
        combos_and_scores[f"median_{combo}"] = median_r2

# Sort the combinations by their R² score:
sorted_combos = sorted(combos_and_scores.items(), key=lambda x: x[1], reverse=True)
print("\nTop 10 combinations of models by R² score:")
for combo, score in sorted_combos[:10]:
    print(f"{combo}: {score:.4f}")

# Reprint the random state and fraction for clarity
print(f"\nRandom seed used: {seed}")
print(f"Fraction of data used: {fraction}")

print("Process completed. Exiting.")


#########################################################################################################
# Stacking model test
#########################################################################################################


# print("\n\n\n")
# print("Starting the stacked model setup and training...")

# # Define the meta-learner
# meta_model = Ridge()

# # Turn the dict into a list of tuples
# estimators = [(name, model) for name, model in base_models.items()]

# # Set up the stacking regressor
# stacked_reg = StackingRegressor(estimators=estimators, final_estimator=meta_model, cv=5)  # Adjust the cross-validation strategy as needed

# # Train the stacked ensemble
# stacked_reg.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = stacked_reg.predict(X_valid)

# # Evaluate the model
# print(f"Stacked model results:")
# mse = mean_squared_error(y_valid, y_pred)
# print(f"Mean Squared Error: {mse:.3f}")

# # Calculate R² score
# r2 = r2_score(y_valid, y_pred)
# print(f"R² score: {r2:.3f}")

# # Reprinte the belnded model results and the individual model results, R^2 only
# print("\n")
# print("Blended model results:")
# print(f"Blended Model MSE: {blend_mse}")
# print(f"Blended Model R² score: {blend_r2:.3f}")
# print("\n")
# print("Individual model R² score:")  # using "scores" variable from earlier
# for name, score in scores.items():
#     print(f"{name}: {score:.4f}")


# print("Process completed.")
