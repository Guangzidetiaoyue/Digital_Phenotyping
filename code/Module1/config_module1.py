"""
Module 1 Configuration: Digital Phenotyping Atlas
Nature/Science Protocol — All parameters centralized for reproducibility
v1.1: Added MIN_POSTS_FOR_TEMPORAL, unified output naming, source policy
"""

import os
from pathlib import Path

# ======================== PATHS ========================
SILVER_DATA_DIR = "/path/to/project/cohort_v1/silver_v2/data"
USER_META_PATH = "/path/to/project/cohort_v1/final_longitudinal_data_user_meta.parquet"
PSM_MATCHED_PATH = "/path/to/project/cohort_v1/intermediate_results/step_3.4_psm_matched.parquet"
REGISTRY_PATH = "/path/to/project/cohort_v1/silver_v2/_registry/user_lookup.parquet"

# Output directories
MODULE1_OUTPUT_DIR = "/path/to/project/module1_results"
FEATURES_DIR = os.path.join(MODULE1_OUTPUT_DIR, "features")
STATS_DIR = os.path.join(MODULE1_OUTPUT_DIR, "statistics")
FIGURES_DIR = os.path.join(MODULE1_OUTPUT_DIR, "figures_lyc")
TABLES_DIR = os.path.join(MODULE1_OUTPUT_DIR, "tables")
TEMP_DIR = "/tmp/module1_temp"

for d in [MODULE1_OUTPUT_DIR, FEATURES_DIR, STATS_DIR, FIGURES_DIR, TABLES_DIR, TEMP_DIR]:
    os.makedirs(d, exist_ok=True)

# [FIX] Unified output file names for Step 02 compatibility
RAW_FEATURES_FILE = "user_features_raw.parquet"
FEATURE_METADATA_FILE = "feature_metadata.json"
QUALITY_REPORT_FILE = "quality_report.json"

# ======================== ANALYSIS PARAMETERS ========================

DISORDER_GROUPS = [
    "depression", "bipolar", "anxiety", "ocd", "ptsd",
    "bpd", "eating_disorders", "schizophrenia", "adhd", "autism"
]

DISORDER_DISPLAY_NAMES = {
    "depression": "Depression",
    "bipolar": "Bipolar Disorder",
    "anxiety": "Anxiety Disorders",
    "ocd": "OCD",
    "ptsd": "PTSD",
    "bpd": "Borderline PD",
    "eating_disorders": "Eating Disorders",
    "schizophrenia": "Schizophrenia",
    "adhd": "ADHD",
    "autism": "Autism Spectrum"
}

HITOP_SPECTRA = {
    "Internalizing": ["depression", "anxiety", "eating_disorders"],
    "Thought Disorder": ["schizophrenia", "bipolar"],
    "Externalizing": ["adhd"],
    "Detachment/Personality": ["bpd", "ptsd", "ocd"],
    "Neurodevelopmental": ["autism"]
}

HITOP_COLORS = {
    "Internalizing": "#E74C3C",
    "Thought Disorder": "#9B59B6",
    "Externalizing": "#F39C12",
    "Detachment/Personality": "#3498DB",
    "Neurodevelopmental": "#2ECC71"
}

def get_spectrum(disorder):
    for spectrum, disorders in HITOP_SPECTRA.items():
        if disorder in disorders:
            return spectrum
    return "Unknown"

# ======================== FEATURE CONFIGURATION ========================

USE_LIWC = True
LIWC_DICTIONARY_PATH = "/path/to/code/data/LIWC2015_Dictionary.dic"#"data/LIWC2015_Dictionary.dic"

# [FIX] Source consistency policy
# "strict": If primary source fails for ANY post, abort
# "permissive": Fall back to next tier per-post (may cause inconsistency)
SOURCE_POLICY = "strict"

LIWC_CATEGORIES_TO_USE = [
    "posemo", "negemo", "anx", "anger", "sad",
    "insight", "cause", "tentat", "certain", "discrep", "differ",
    "i", "we", "you", "shehe", "they", "ipron",
    "focuspast", "focuspresent", "focusfuture",
    "social", "family", "friend",
    "body", "health", "sexual", "ingest",
    "affiliation", "achiev", "power", "reward", "risk",
    "death", "relig", "swear", "netspeak"
]

USE_LIWC_FALLBACK = True

SENTIMENT_MODEL = "vader"
ROBERTA_MODEL_NAME = "/path/to/code/roberta_model/base_best_roberta-base"  #"roberta_model/base_best_roberta-base"

NRC_LEXICON_PATH = "/path/to/code/data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"  #"data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

MENTAL_HEALTH_SUBREDDITS = {
    "depression", "anxiety", "bipolar", "bipolarreddit", "bpd",
    "ptsd", "ocd", "adhd", "autism", "aspergers",
    "suicidewatch", "selfharm", "eating_disorder", "edanonymous",
    "mentalhealth", "therapy", "psychiatry", "schizophrenia",
    "socialanxiety", "panicparty", "agoraphobia", "cptsd",
    "dpdr", "dissociation", "anxietydepression", "getting_over_it",
    "stopselfharm", "madeofstyrofoam", "arfid",
    "bulimia", "anorexianervosa", "bingeeatingdisorder",
    # [FIX] Added common variants
    "depression_help", "depressed", "bipolar2", "bipolar1",
    "anxietysupport", "ocd_support", "ptsdsupport",
    "adhd_anxiety", "autism_support", "aspergirls", "autisticadults"
}

SUBREDDIT_CATEGORIES = {
    "support": MENTAL_HEALTH_SUBREDDITS | {
        "offmychest", "trueoffmychest", "confessions", "vent",
        "relationship_advice", "survivorsofabuse", "raisedbynarcissists",
        "divorce", "grief", "widowers", "cancer", "chronicpain",
        # [FIX] Added more support subs
        "emotional_support", "caregiversupport", "griefsupport",
        "addiction", "alcoholicsanonymous", "stopdrinking", "leaves",
        "narcissisticabuse", "toxicrelationships", "domesticviolence"
    },
    "entertainment": {
        "funny", "gaming", "movies", "television", "music",
        "memes", "dankmemes", "jokes", "aww", "pics",
        "gifs", "videos", "comics", "anime", "manga",
        # [FIX] Added more entertainment
        "gamingcirclejerk", "pcmasterrace", "netflix", "disney",
        "starwars", "marvelstudios", "harrypotter", "lotr"
    },
    "information": {
        "science", "technology", "worldnews", "news", "politics",
        "askscience", "explainlikeimfive", "todayilearned",
        "askhistorians", "space", "futurology",
        # [FIX] Added more info subs
        "datascience", "programming", "learnprogramming", "statistics",
        "machinelearning", "artificial", "philosophy", "economics"
    },
    "social": {
        "askreddit", "casualconversation", "makefriendshere",
        "socialskills", "introvert", "amitheasshole",
        # [FIX] Added more social subs
        "needafriend", "r4r", "penpals", "chat", "meetnewpeoplehere"
    }
}

# ======================== PROCESSING ========================
NUM_CORES = 16
BATCH_SIZE = 50000
RANDOM_SEED = 2025
MIN_POSTS_FOR_FEATURES = 10
MIN_WORDS_FOR_LINGUISTIC = 100
# [FIX] Separate threshold for temporal features (need more posts for stable rhythm)
MIN_POSTS_FOR_TEMPORAL = 15
# [FIX] Max words for MATTR computation (avoid slowdown on very long posts)
MATTR_MAX_WORDS = 500

# ======================== STATISTICAL PARAMETERS ========================
ALPHA = 0.05
FDR_METHOD = "fdr_bh"
BOOTSTRAP_N = 10000
CV_FOLDS = 5
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SMALL_EFFECT = 0.2
MEDIUM_EFFECT = 0.5
LARGE_EFFECT = 0.8