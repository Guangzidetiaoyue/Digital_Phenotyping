# config.py - Configuration for Longitudinal Mental Health Phenotyping on Reddit
# Optimized for Nature/Science/PNAS standards
# Date: 2025-12-25
# Updated: Expanded disease ontology (DSM-5/ICD-11) for transdiagnostic analysis

import re
from datetime import datetime

# ======================== 1. LOGGING & METADATA ========================
LOG_LEVEL = "INFO"
LOG_FILE = "reddit_cohort_construction.log"

# Metadata for reproducibility and ethical review
METADATA = {
    "project_name": "Longitudinal Digital Phenotyping (2006-2022)",
    "dataset_version": "Pushshift_2023_Dump", 
    "code_version": "v2.0_Nature_Protocol",
    "irb_status": "Exempt (Public Data / Unlinked)", 
    "privacy_protocol": "SHA-256 + Salted Hash + PII Redaction",
    "run_date": datetime.now().strftime("%Y-%m-%d"),
    "disease_ontology_version": "DSM-5/ICD-11 (Transdiagnostic Expansion)"
}

# ======================== INPUT DATA LAYOUT ========================
# Define the subdirectory structure of your source data
DATA_LAYOUT = {
    "root_dir_name": "data",       # The parent folder name (informational)
    "comments_dir": "comments",    # Folder containing RC_*.zst
    "submissions_dir": "submissions" # Folder containing RS_*.zst
}

# ======================== PSM & SCIENTIFIC RIGOR ========================
PSM_CONFIG = {
    "ratio": 5,                # 1 Patient : 5 Controls
    "caliper": 0.005,           # Tight matching (0.05 * SD of Propensity Score) 0.05
    "seed": 2025,
    "replace": False,          # Sampling without replacement (higher quality)
    
    # Validation Thresholds
    "max_smd": 0.1,            # Standardized Mean Difference must be < 0.1
    "min_common_support": 0.1,  # Min overlap in propensity scores
    "min_pre_exposure_days": 30
}

# Socio-Demographic Proxies (for matching)
PROXY_COVARIATES = [
    "circadian_peak_hour",      # Proxy for Timezone/Geography
    "readability_grade",        # Proxy for Education/Age
    "subreddit_entropy",        # Proxy for Interests
    "post_frequency",           # Behavior
    "account_tenure_at_t0"      # Experience
]

# Bot Filtering (Scientific Indicators)
BOT_THRESHOLDS = {
    "min_vocab_density": 0.3,   # MSTTR or simple TTR on sample
    "max_posts_per_day": 60,    # Cap for human possibility
    "min_account_age": 30       # Days
}

# ======================== 2. DISEASE ONTOLOGY (Social Media Adapted) ========================
# Includes DSM-5 clinical terms AND common Reddit vernacular/abbreviations.
# Logic: These keywords are matched against the text captured by INCLUSION_REGEX.

MENTAL_DISORDERS = {
    "depression": [
        # Clinical
        "major depressive disorder", "mdd", "clinical depression", 
        "dysthymia", "persistent depressive disorder", 
        # Colloquial / Common
        "depression", "major depression", "severe depression",
        "postpartum depression", "seasonal affective disorder"
    ],
    
    "bipolar": [
        # Clinical
        "bipolar disorder", "bipolar i", "bipolar ii", "cyclothymia", 
        # Colloquial / Abbreviations
        "bipolar", "bipolar 1", "bipolar 2", "bd i", "bd ii", "bd 1", "bd 2",
        "manic depression", "manic depressive", # Dated but still used by older users
        "rapid cycling"
    ],
    
    "anxiety": [
        # Clinical
        "generalized anxiety disorder", "social anxiety disorder", "panic disorder", 
        "agoraphobia",
        # Colloquial / Abbreviations
        "anxiety", "gad", "social anxiety", "social phobia", 
        "severe anxiety", "general anxiety"
    ],
    
    "ocd": [
        # Clinical
        "obsessive compulsive disorder", 
        # Colloquial
        "ocd", "obsessive-compulsive", "obsessive compulsive"
    ],
    
    "ptsd": [
        # Clinical
        "post-traumatic stress disorder", 
        # Colloquial / Abbreviations
        "ptsd", "cptsd", "c-ptsd", "complex ptsd", 
        "combat ptsd", "complex trauma"
    ],
    
    "bpd": [
        # Clinical
        "borderline personality disorder", 
        # Colloquial / Abbreviations
        "bpd", "borderline", "borderline personality",
        "eupd"  # "Emotionally Unstable Personality Disorder" (Common in UK/Europe)
    ],
    
    "eating_disorders": [
        # Clinical
        "anorexia nervosa", "bulimia nervosa", "binge eating disorder",
        "avoidant restrictive food intake disorder",
        # Colloquial / Abbreviations
        "anorexia", "bulimia", "eating disorder", "ednos", "arfid",
        "binge eating", "compulsive eating"
    ],
    
    "schizophrenia": [
        # Clinical
        "schizophrenia", "schizoaffective disorder", "paranoid schizophrenia",
        # Colloquial
        "schizoaffective", "psychosis" # Note: Psychosis is a symptom, but often used as a diagnosis proxy
    ],
    
    "adhd": [
        # Clinical
        "attention deficit hyperactivity disorder", 
        # Colloquial / Abbreviations
        "adhd", "add", "attention deficit",
        "attention deficit disorder" # Dated (DSM-IV) but heavily used by adults diagnosed as kids
    ],
    
    "autism": [
        # Clinical
        "autism spectrum disorder", 
        # Colloquial / Abbreviations
        "autism", "asd", "aspergers", "asperger's", "asperger syndrome"
        # Note: Aspergers is removed from DSM-5 but widely used in self-identification
    ]
}
# ======================== SUICIDE & RISK MARKERS ========================
# NOTE: These are NOT used for the "Diagnosis" regex. 
# These are used to tag the SEVERITY of the extracted patients.
SUICIDE_KEYWORDS = [
    "suicide", "suicidal", "kill myself", "end my life", "want to die", 
    "suicide attempt", "overdose", "jumped off", "hung myself"
]

# Clinical concept: Non-Suicidal Self-Injury (NSSI)
SELF_HARM_KEYWORDS = [
    "self harm", "self-harm", "cutting", "cut myself", "burning myself", 
    "hit myself", "hurt myself"
]

# ======================== 3. HIGH-PRECISION EXTRACTION REGEX ========================
# Scientific Note: We prioritize Precision over Recall to minimize false positives.
# Pattern avoids capturing newlines to prevent matching whole paragraphs.
# Captures 3-50 characters after "diagnosed with" to get the full disorder name.
INCLUSION_REGEX = re.compile(
    r"I (?:was|have been|am|got) (?:officially |clinically |formally |professionally )?diagnosed with ([a-zA-Z0-9\s\-\']{3,50})",
    re.IGNORECASE
)

RETROSPECTIVE_REGEX = re.compile(
    r"(?:\b(years?|months?|weeks?)\s+ago\b)|"    # "3 years ago"
    r"(?:\bin\s+(?:19|20)\d{2}\b)|"               # "in 1999"
    r"(?:\bwhen I was\b)|"                        # "when I was a kid"
    r"(?:\back in\b)|"                            # "back in high school"
    r"(?:\bsince\s+(?:19|20)\d{2}\b)",            # "since 2012"
    re.IGNORECASE
)
# Strict Exclusions:
# REMOVED: "i think", "maybe" (These cause False Negatives; handled by SpaCy dependency parsing instead)
# KEPT: Third-party references (mom, dad, friend) and self-diagnosis declarations.
EXCLUSION_KEYWORDS = [
    "self-diagnosed", "self diagnosis", 
    "my mom", "my dad", "my friend", "my partner", "my wife", "my husband",
    "my brother", "my sister", "my family", "my son", "my daughter", "my bf", "my gf",
    "not diagnosed", "never diagnosed", "mislabeled as", "misdiagnosed with"
]

# ======================== 4. COHORT FILTERS (SCIENTIFIC RIGOR) ========================
BOT_THRESHOLDS = {
    # Raised to 60 to capture manic episodes (patients often post frequently during mania)
    "max_posts_per_day": 60,         
    
    # LIWC standard for automated text detection
    "min_text_diversity": 0.2,       
    
    # CHANGED from 365 -> 30. 
    # Scientific Reason: Removing users <1 year introduces "Survivorship Bias".
    # We must include users who join specifically to discuss their acute diagnosis.
    "min_account_age_days": 30,      
    
    # Stricter check for spam bots
    "max_duplicate_text_ratio": 0.75 ,
    "burner_bot_posts_per_day": 10
}

# ======================== 5. CONTROL GROUP SAMPLING ========================
# Necessary for "Case-Control" study design required by top journals.
CONTROL_CONFIG = {
    "enabled": True,
    # Probability to select a user as a "Candidate Control" if they are NOT a patient.
    # 0.001 ≈ 1 in 1000 users. Given Reddit's size, this yields a massive pool for matching.
    "sampling_rate": 0.001,  
    "min_history_posts": 5   # Candidate must have at least some history
}

# ======================== 6. PROCESSING & I/O PARAMETERS ========================
PROCESSING_CONFIG = {
    # Performance: Larger batches reduce I/O overhead
    "batch_size": 200000,            
    
    # Parallelization: -1 uses all available logical cores minus 1
    "cpu_cores": 12,                
    
    # Security: Salt for SHA-256 hashing to prevent dictionary attacks on User IDs
    # CHANGE THIS to a secure random string in production!
    "salt": "NATURE_PAPER_2025_SECURE_SALT_v9X", 
    
    # Paths
    "output_dir": "/data1/qizhixiao/path/to/reddit/emotion_disorder_1225/cohort_v1",#"/path/to/project/cohort_v1",  #"./data/cohort_v1",
    
    # File Formats:
    # Parquet is 10-50x faster than CSV for read operations on this scale.
    "master_index_file": "master_patient_index.parquet",  
    "final_cohort_file": "final_longitudinal_data.parquet",
    "metadata_file": "metadata.json"
}
