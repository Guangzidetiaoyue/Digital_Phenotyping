"""
Configuration for Module 2: Longitudinal Trajectory Analysis

All paths, parameters, and constants for s07鈥搒10.
Fully aligned with config_module1.py 鈥?same absolute paths, same disorder
names, same subreddit sets, same random seed.

Directory layout (on NAS):
  /path/to/project/
  鈹溾攢鈹€ cohort_v1/
  鈹?  鈹溾攢鈹€ silver_v2/data/              鈫?Raw posts (partitioned)
  鈹?  鈹溾攢鈹€ intermediate_results/
  鈹?  鈹?  鈹斺攢鈹€ step_3.4_psm_matched.parquet
  鈹?  鈹斺攢鈹€ final_longitudinal_data_user_meta.parquet
  鈹溾攢鈹€ module1_results/
  鈹?  鈹溾攢鈹€ features/                    鈫?s01/s02 output
  鈹?  鈹?  鈹溾攢鈹€ user_features_raw.parquet
  鈹?  鈹?  鈹溾攢鈹€ user_features_matched.parquet
  鈹?  鈹?  鈹溾攢鈹€ user_features_standardized.parquet
  鈹?  鈹?  鈹溾攢鈹€ feature_metadata.json
  鈹?  鈹?  鈹溾攢鈹€ feature_selection.json
  鈹?  鈹?  鈹溾攢鈹€ imputation_params.json
  鈹?  鈹?  鈹溾攢鈹€ winsorize_params.json
  鈹?  鈹?  鈹斺攢鈹€ robust_scaler_params.json
  鈹?  鈹斺攢鈹€ statistics/                  鈫?s03/s04/s05/s06 output
  鈹?      鈹溾攢鈹€ effect_sizes_full.csv
  鈹?      鈹溾攢鈹€ effect_size_matrix_clean.csv
  鈹?      鈹溾攢鈹€ factor_loadings_features.csv
  鈹?      鈹溾攢鈹€ factor_scores_disorders.csv
  鈹?      鈹溾攢鈹€ transdiagnostic_features.csv
  鈹?      鈹斺攢鈹€ ...
  鈹斺攢鈹€ module2_results/                 鈫?NEW: Module 2 output
      鈹溾攢鈹€ trajectories/                鈫?s07 output
      鈹?  鈹溾攢鈹€ user_trajectories.parquet
      鈹?  鈹斺攢鈹€ trajectory_metadata.json
      鈹溾攢鈹€ statistics/                  鈫?s08/s09/s10 output
      鈹斺攢鈹€ figures/
"""

import os

# =====================================================================
#  ABSOLUTE PATHS 鈥?aligned with config_module1.py
# =====================================================================

# ---- Lab NAS root for this project ----
PROJECT_ROOT = "/path/to/project"

# ---- Module 1 directories (READ-ONLY for Module 2) ----
SILVER_DATA_DIR  = os.path.join(PROJECT_ROOT, "cohort_v1", "silver_v2", "data") #"data"
M1_OUTPUT_DIR    = os.path.join(PROJECT_ROOT, "module1_results")
FEATURES_DIR     = os.path.join(M1_OUTPUT_DIR, "features")
M1_STATS_DIR     = os.path.join(M1_OUTPUT_DIR, "statistics")
FIGURES_DIR      = os.path.join(M1_OUTPUT_DIR, "figures")

# ---- Module 1 key input files (from preprocessing) ----
PSM_MATCHED_PATH = os.path.join(
    PROJECT_ROOT, "cohort_v1", "intermediate_results",
    "step_3.4_psm_matched.parquet"
)
USER_META_PATH = os.path.join(
    PROJECT_ROOT, "cohort_v1",
    "final_longitudinal_data_user_meta.parquet"
)
REGISTRY_PATH = os.path.join(
    PROJECT_ROOT, "cohort_v1", "silver_v2", "_registry",
    "user_lookup.parquet"
)

# ---- Module 2 directories (WRITE) ----
M2_OUTPUT_DIR    = os.path.join(PROJECT_ROOT, "module2_results")
TRAJECTORY_DIR   = os.path.join(M2_OUTPUT_DIR, "trajectories")
STATS_DIR        = os.path.join(M2_OUTPUT_DIR, "statistics")
M2_FIGURES_DIR   = os.path.join(M2_OUTPUT_DIR, "figures")
TEMP_DIR         = "/tmp/module2_temp"

# Ensure output directories exist at import time
for _d in [M2_OUTPUT_DIR, TRAJECTORY_DIR, STATS_DIR, M2_FIGURES_DIR, TEMP_DIR]:
    os.makedirs(_d, exist_ok=True)

# =====================================================================
#  EXTERNAL RESOURCES (shared with Module 1 鈥?same absolute paths)
# =====================================================================

RESOURCES_DIR        = "/path/to/code/data"
LIWC_DICTIONARY_PATH = os.path.join(RESOURCES_DIR, "LIWC2015_Dictionary.dic")
NRC_LEXICON_PATH     = os.path.join(RESOURCES_DIR, "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")
USE_LIWC             = os.path.exists(LIWC_DICTIONARY_PATH)

ROBERTA_MODEL_NAME   = "/path/to/code/roberta_model/base_best_roberta-base"

# =====================================================================
#  MODULE 1 OUTPUT FILE REFERENCES
# =====================================================================

# s01/s02 feature files
RAW_FEATURES_FILE          = "user_features_raw.parquet"
MATCHED_FEATURES_FILE      = "user_features_matched.parquet"
STANDARDIZED_FEATURES_FILE = "user_features_standardized.parquet"
FEATURE_METADATA_FILE      = "feature_metadata.json"
FEATURE_SELECTION_FILE     = "feature_selection.json"
QUALITY_REPORT_FILE        = "quality_report.json"

# s02 preprocessing params
IMPUTATION_PARAMS_FILE     = "imputation_params.json"
WINSORIZE_PARAMS_FILE      = "winsorize_params.json"
SCALER_PARAMS_FILE         = "robust_scaler_params.json"
IMPUTATION_FLAGS_FILE      = "imputation_flags.parquet"

# s05 factor analysis outputs
FACTOR_LOADINGS_PATH       = os.path.join(M1_STATS_DIR, "factor_loadings_features.csv")
FACTOR_SCORES_PATH         = os.path.join(M1_STATS_DIR, "factor_scores_disorders.csv")

# s05 transdiagnostic classification
TRANSDIAGNOSTIC_FEATURES_PATH = os.path.join(M1_STATS_DIR, "transdiagnostic_features.csv")

# =====================================================================
#  MODULE 2 OUTPUT FILES
# =====================================================================

# s07
TRAJECTORY_OUTPUT_FILE   = "user_trajectories.parquet"
TRAJECTORY_METADATA_FILE = "trajectory_metadata.json"

# =====================================================================
#  DISORDER DEFINITIONS 鈥?MUST match config_module1.py EXACTLY
# =====================================================================

DISORDER_GROUPS = [
    "depression", "bipolar", "anxiety", "ocd", "ptsd",
    "bpd", "eating_disorders", "schizophrenia", "adhd", "autism"
]

DISORDER_DISPLAY_NAMES = {
    "depression":       "Depression",
    "bipolar":          "Bipolar Disorder",
    "anxiety":          "Anxiety Disorders",
    "ocd":              "OCD",
    "ptsd":             "PTSD",
    "bpd":              "Borderline PD",
    "eating_disorders": "Eating Disorders",
    "schizophrenia":    "Schizophrenia",
    "adhd":             "ADHD",
    "autism":           "Autism Spectrum"
}

# HiTOP spectra 鈥?MUST match config_module1.py structure
HITOP_SPECTRA = {
    "Internalizing":         ["depression", "anxiety", "eating_disorders"],
    "Thought Disorder":      ["schizophrenia", "bipolar"],
    "Externalizing":         ["adhd"],
    "Detachment/Personality": ["bpd", "ptsd", "ocd"],
    "Neurodevelopmental":    ["autism"]
}

HITOP_COLORS = {
    "Internalizing":          "#E74C3C",
    "Thought Disorder":       "#9B59B6",
    "Externalizing":          "#F39C12",
    "Detachment/Personality": "#3498DB",
    "Neurodevelopmental":     "#2ECC71"
}

def get_spectrum(disorder):
    """Get HiTOP spectrum for a disorder."""
    for spectrum, disorders in HITOP_SPECTRA.items():
        if disorder in disorders:
            return spectrum
    return "Unknown"

# =====================================================================
#  MENTAL HEALTH SUBREDDITS 鈥?MUST match config_module1.py EXACTLY
# =====================================================================

MENTAL_HEALTH_SUBREDDITS = {
    "depression", "anxiety", "bipolar", "bipolarreddit", "bpd",
    "ptsd", "ocd", "adhd", "autism", "aspergers",
    "suicidewatch", "selfharm", "eating_disorder", "edanonymous",
    "mentalhealth", "therapy", "psychiatry", "schizophrenia",
    "socialanxiety", "panicparty", "agoraphobia", "cptsd",
    "dpdr", "dissociation", "anxietydepression", "getting_over_it",
    "stopselfharm", "madeofstyrofoam", "arfid",
    "bulimia", "anorexianervosa", "bingeeatingdisorder",
    "depression_help", "depressed", "bipolar2", "bipolar1",
    "anxietysupport", "ocd_support", "ptsdsupport",
    "adhd_anxiety", "autism_support", "aspergirls", "autisticadults"
}

SUBREDDIT_CATEGORIES = {
    "support": MENTAL_HEALTH_SUBREDDITS | {
        "offmychest", "trueoffmychest", "confessions", "vent",
        "relationship_advice", "survivorsofabuse", "raisedbynarcissists",
        "divorce", "grief", "widowers", "cancer", "chronicpain",
        "emotional_support", "caregiversupport", "griefsupport",
        "addiction", "alcoholicsanonymous", "stopdrinking", "leaves",
        "narcissisticabuse", "toxicrelationships", "domesticviolence"
    },
    "entertainment": {
        "funny", "gaming", "movies", "television", "music",
        "memes", "dankmemes", "jokes", "aww", "pics",
        "gifs", "videos", "comics", "anime", "manga",
        "gamingcirclejerk", "pcmasterrace", "netflix", "disney",
        "starwars", "marvelstudios", "harrypotter", "lotr"
    },
    "information": {
        "science", "technology", "worldnews", "news", "politics",
        "askscience", "explainlikeimfive", "todayilearned",
        "askhistorians", "space", "futurology",
        "datascience", "programming", "learnprogramming", "statistics",
        "machinelearning", "artificial", "philosophy", "economics"
    },
    "social": {
        "askreddit", "casualconversation", "makefriendshere",
        "socialskills", "introvert", "amitheasshole",
        "needafriend", "r4r", "penpals", "chat", "meetnewpeoplehere"
    }
}

# =====================================================================
#  SELECTION ARTIFACT FEATURES 鈥?MUST match Module 1 exclusion
# =====================================================================

SELECTION_ARTIFACT_FEATURES = {
    "mental_health_sub_ratio",
    "support_sub_ratio",
    "subreddit_entropy",
    "subreddit_herfindahl",
    "primary_subreddit_ratio",
}

# =====================================================================
#  COMPUTATION 鈥?aligned with config_module1.py
# =====================================================================

NUM_CORES  = 12           # Same as Module 1
BATCH_SIZE = 50000         # Same as Module 1
RANDOM_SEED = 2025         # Same as Module 1

# =====================================================================
#  S07 鈥?TRAJECTORY CONSTRUCTION PARAMETERS
# =====================================================================

# ---- Window configuration ----
WINDOW_SIZE_DAYS = 90      # 3-month observation window
WINDOW_STEP_DAYS = 30      # 1-month step 鈫?60-day overlap

# ---- Per-window quality thresholds ----
# Lower than full-period thresholds because each window is only 3 months
MIN_POSTS_PER_WINDOW  = 5
MIN_WORDS_PER_WINDOW  = 50

# ---- Per-user coverage requirements ----
MIN_PRE_WINDOWS  = 2       # 鈮? reliable pre-t0 windows  (鈮? months)
MIN_POST_WINDOWS = 2       # 鈮? reliable post-t0 windows (鈮? months)
MIN_TOTAL_RELIABLE_WINDOWS = 4

# ---- Trajectory range (relative to t0) ----
TRAJECTORY_PRE_DAYS  = 720  # 2 years before t0
TRAJECTORY_POST_DAYS = 720  # 2 years after t0

PRODROME_WINDOW_DAYS = 365

# ---- Baseline windows ACTUALLY EXECUTED (canonical; code uses these values) ----
# p-factor z-score baseline = mean over t0-360 .. t0-270 days (s07/s09).
PFACTOR_BASELINE_WINDOW = (-360, -270)
# ---- p-factor severity construction ----
PFACTOR_WEIGHT_SOURCE = "module1_factor1"   # or "equal_weight", "user_level_fa"

# ---- Linguistic extraction (inherited from Module 1) ----
MATTR_MAX_WORDS          = 500    # Same as Module 1
MIN_POSTS_FOR_FEATURES   = 10     # Full-period threshold (reference)
MIN_WORDS_FOR_LINGUISTIC = 100    # Same as Module 1
MIN_POSTS_FOR_TEMPORAL   = 15     # Same as Module 1

# =====================================================================
#  S08 鈥?TRAJECTORY MORPHOLOGY PARAMETERS
# =====================================================================

# ---- PELT changepoint detection ----
PELT_MIN_SEGMENT_LENGTH  = 3
PELT_PENALTY_MULTIPLIER  = 2.0    # None 鈫?auto BIC

# ---- Event study ----
# NOTE (2026-09-07, freeze audit): these two constants are INFORMATIONAL/UNUSED. The executed event-study reference in s08_trajectory_morphology.py is window_index -8..-4 (pre-t0 windows; centers approx t0-195..-75 d), i.e., NOT -360..-180 days. Do not report -360..-180 as the baseline.
EVENT_STUDY_BASELINE_START = -360  # Days from t0  [UNUSED - see note]
EVENT_STUDY_BASELINE_END   = -180  # Days from t0  [UNUSED - see note]
# ACTUALLY-EXECUTED event-study reference window (window indices; s08/s10).
EVENT_STUDY_REF_WINDOW = (-8, -4)

# ---- Multi-variable changepoint consensus ----
CONSENSUS_TOLERANCE_DAYS = 30

# ---- Parallel trends test ----
PARALLEL_TRENDS_ALPHA     = 0.05
PARALLEL_TRENDS_MAX_PRE_R = 0.3

# =====================================================================
#  S09 鈥?TRAJECTORY HETEROGENEITY PARAMETERS
# =====================================================================

# ---- GMM subtype discovery (values ACTUALLY EXECUTED in s09) ----
GMM_MAX_COMPONENTS    = 5
GMM_MIN_PER_COMPONENT = 50
GMM_COVARIANCE_TYPE   = "full"
GMM_N_INIT            = 10      # final model (s09)
GMM_MAX_ITER          = 300     # final model (s09)
GMM_REG_COVAR         = 1e-3    # executed (was documented 1e-6; s09 uses 1e-3)
GMM_SELECTION_N_INIT     = 5    # model-selection loop (s09)
GMM_SELECTION_MAX_ITER   = 200  # model-selection loop (s09)
GMM_BOOTSTRAP_REG_COVAR  = 1e-4 # stability bootstrap (s09, diag covariance)

# ---- Subtype naming thresholds ----
SUBTYPE_STEP_HEIGHT_THRESHOLD = 0.5
SUBTYPE_SLOPE_THRESHOLD      = 0.005
SUBTYPE_VOLATILITY_THRESHOLD = 0.3
SUBTYPE_STABLE_MAX_STEP      = 0.3

# ---- Subtype predictor analysis ----
PREDICTOR_CV_FOLDS   = 3
PREDICTOR_MIN_SAMPLE = 50

# =====================================================================
#  S10 鈥?CAUSAL ATTRIBUTION PARAMETERS
# =====================================================================

# ---- DID estimation ----
DID_ALPHA = 0.05

# 鍦?config_module2.py 涓慨鏀癸細
DID_POST_START = int(WINDOW_SIZE_DAYS / 2)  # 浠?30 鏀逛负 45锛屼笌 s07 phase="transition" 瀵归綈
DID_PRE_END   = -int(WINDOW_SIZE_DAYS / 2)  # 浠?-30 鏀逛负 -45

# ---- Placebo t0 test (values ACTUALLY EXECUTED in s10) ----
PLACEBO_N_TESTS     = 100    # executed (s10 calls with 100; was documented 500)
PLACEBO_MIN_SHIFT   = 90     # shift drawn in [MIN_SHIFT, SHIFT_RANGE] days
PLACEBO_SHIFT_RANGE = 360    # executed (drawn from 90..360 d, random sign)
PLACEBO_SEED_OFFSET = 100    # executed seed = RANDOM_SEED + 100 (was documented 1000)

# ---- Dose-response ----
DOSE_RESPONSE_MIN_SAMPLE = 30

# ---- Subtype-stratified DID ----
STRATIFIED_DID_MIN_SUBTYPE = 30

# =====================================================================
#  STATISTICAL TESTING 鈥?aligned with config_module1.py
# =====================================================================

ALPHA      = 0.05
FDR_METHOD = "fdr_bh"

SMALL_EFFECT  = 0.2
MEDIUM_EFFECT = 0.5
LARGE_EFFECT  = 0.8

BOOTSTRAP_N          = 10000   # Same as Module 1
BOOTSTRAP_CI         = 95
BOOTSTRAP_MIN_SAMPLE = 20

CV_FOLDS    = 5                 # Same as Module 1
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# =====================================================================
#  CROSS-MODULE VALIDATION
# =====================================================================

MIN_PATIENTS_PER_DISORDER  = 30
MIN_CONTROLS_FOR_DID       = 50
MIN_USERS_FOR_USER_FA      = 500

MNAR_IMPUTATION_QUANTILE = 0.90  # 娌夐粯鏈熼€€缂╄祴鍊硷紝P90 姣?P75 鏇寸鍚堜复搴婃瀬绔〃鐜?# =====================================================================
#  LOGGING & QUALITY
# =====================================================================

LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

TRAJECTORY_QUALITY_REPORT = "trajectory_quality_report.json"
MORPHOLOGY_REPORT         = "morphology_report.json"
HETEROGENEITY_REPORT      = "heterogeneity_report.json"
CAUSAL_REPORT             = "causal_report.json"

# =====================================================================
#  DERIVED CONSTANTS (do not modify)
# =====================================================================

WINDOW_OVERLAP_DAYS = WINDOW_SIZE_DAYS - WINDOW_STEP_DAYS

N_WINDOWS_PER_USER = (
    (TRAJECTORY_PRE_DAYS + TRAJECTORY_POST_DAYS - WINDOW_SIZE_DAYS)
    // WINDOW_STEP_DAYS + 1
)

T0_WINDOW_INDEX = TRAJECTORY_PRE_DAYS // WINDOW_STEP_DAYS
