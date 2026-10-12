# phase1_finder.py - Phase 1: Patient Identification + Control Sampling
# Optimized for Nature/Science Protocol (Sensitivity Analysis + Comorbidity)
# {'name': 't1_c031zca', 'score_hidden': False, 'link_id': 't3_679o0', 'archived': True, 'subreddit': 'politics', 'ups': 0, 'body': "did they tell homeless O'Reilly is hoarding crack and booze?", 'controversiality': 0, 'subreddit_id': 't5_2cneq', 'score': 0, 'edited': False, 'distinguished': None, 'author_flair_css_class': None, 'author': 'louis_xiv42', 'gilded': 0, 'created_utc': '1201824002', 'author_flair_text': None, 'retrieved_on': 1425826296, 'parent_id': 't3_679o0', 'downs': 0, 'id': 'c031zca'} comment

# {'distinguished': None, 'edited': False, 'score': 1, 'subreddit_id': 't5_mouw', 'body': "As other pointed out, their theory won't help us getting more oil, as we pump up oil at least a million times faster than it is produced by nature.\n\nThat said, if their theory where right, there would be oil to be found a lot more places than it is today, including places where there is provably no oil. I think they would have a hard time explaining why there is no oil in many pourous rock formations covered with a inpenetrable rock sedementary or igneous. A lot of these has no oil, but if their theory was right, most of those would be filled with oil and gas. The current theory that the oil originates from organic matter of shallow seas, explains this quite well.", 'controversiality': 0, 'ups': 1, 'subreddit': 'science', 'archived': True, 'link_id': 't3_67el2', 'score_hidden': False, 'name': 't1_c032hcb', 'id': 'c032hcb', 'downs': 0, 'parent_id': 't3_67el2', 'retrieved_on': 1425826573, 'author_flair_text': None, 'created_utc': '1201967105', 'gilded': 0, 'author': 'alephnil', 'author_flair_css_class': None} submissions
import os
import glob
import random
import pandas as pd
from multiprocessing import Pool
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

# Import utilities and final config
from utils import setup_logger, read_zst_file, get_cpu_cores
from config import (
    INCLUSION_REGEX, EXCLUSION_KEYWORDS, MENTAL_DISORDERS,
    RETROSPECTIVE_REGEX, PROCESSING_CONFIG, CONTROL_CONFIG, DATA_LAYOUT 
)

# Initialize logger
logger = setup_logger()

# Global SpaCy (worker initialization)
nlp = None
def init_worker():
    """Initialize SpaCy for dependency parsing (per worker process)"""
    global nlp
    if nlp is None:
        import spacy
        # CRITICAL: Disable NER/Vectors for speed, keep Parser for syntax checks
        nlp = spacy.load("en_core_web_sm", disable=["ner", "vectors", "textcat"])
        # logger.debug("SpaCy initialized for Phase 1 worker")

# ======================== TEXT NORMALIZATION ========================
def get_content_from_obj(obj, file_type):
    """
    Extracts text based on file type.
    RS (Submission): Title + Selftext (Highest value for diagnosis)
    RC (Comment): Body
    """
    if file_type == "RS":
        title = obj.get("title", "")
        selftext = obj.get("selftext", "")
        if selftext in ["[deleted]", "[removed]", None]:
            selftext = ""
        # Combine title and text to catch: Title="Diagnosed today", Text="Details..."
        return f"{title}\n{selftext}".strip()
    else:
        # RC
        body = obj.get("body", "")
        return body.strip()

# ======================== DIAGNOSIS VALIDATION (MULTI-STAGE) ========================
def validate_diagnosis(text, extracted_term):
    """
    Validates a diagnosis using a 4-Stage Filter.
    Returns: (is_valid: bool, disorder: str, is_retrospective: bool)
    """
    # STAGE 1: Ontology Mapping
    matched_disorder = None
    term_lower = extracted_term.lower().strip()
    for disorder_key, keywords in MENTAL_DISORDERS.items():
        if any(k in term_lower for k in keywords):
            matched_disorder = disorder_key
            break
    if not matched_disorder:
        return False, None, None

    # STAGE 2: Keyword Exclusion
    text_lower = text.lower()
    if any(ex in text_lower for ex in EXCLUSION_KEYWORDS):
        return False, None, None

    # STAGE 3: Temporal Tagging (Sensitivity Analysis)
    # We do NOT reject retrospective reports; we tag them.
    is_retrospective = bool(RETROSPECTIVE_REGEX.search(text))

    # STAGE 4: Syntactic Dependency Parsing
    try:
        # Limit text length to optimize NLP speed
        doc = nlp(text[:2000] if len(text) > 2000 else text)
        
        for token in doc:
            if token.lemma_ == "diagnose" and token.pos_ == "VERB":
                # Check negation ("I was NOT diagnosed")
                if any(child.dep_ == "neg" for child in token.children):
                    return False, None, None
                
                # Verify Subject is First-Person ("I" or "Me")
                subjects = [
                    c.text.lower() for c in token.children 
                    if c.dep_ in ("nsubj", "nsubjpass", "dobj")
                ]
                if "i" in subjects or "me" in subjects:
                    return True, matched_disorder, is_retrospective
    except Exception:
        # Fallback: If NLP crashes on weird text, rely on Regex+Exclusions (Pass)
        # In a strict setting, you might return False here.
        pass
    
    return False, None, None

# ======================== SINGLE FILE SCAN ========================
def scan_single_file(zst_path):
    """
    Scans a single ZST file for Patients and Candidate Controls.
    Handles both RC (Comments) and RS (Submissions).
    """
    init_worker()
    file_name = os.path.basename(zst_path)
    
    # Determine File Type
    if file_name.startswith("RS_"):
        file_type = "RS"
    elif file_name.startswith("RC_"):
        file_type = "RC"
    else:
        logger.warning(f"Skipping unknown file type: {file_name}")
        return {}

    # Local Storage
    # We store a LIST of diagnoses to handle multiple hits per file
    local_patient_history = defaultdict(list)
    local_controls = {}
    user_post_count = defaultdict(int) 
    
    # Control Config
    sampling_rate = CONTROL_CONFIG["sampling_rate"] if CONTROL_CONFIG["enabled"] else 0
    min_posts = CONTROL_CONFIG["min_history_posts"]

    try:
        # Stream file (Single Pass)
        stream = read_zst_file(zst_path)
        
        for obj in tqdm(stream, desc=f"Scanning {file_name}", unit=" objs"):
            author = obj.get("author", "[deleted]")
            if author == "[deleted]": continue

            # Extract Text
            text_content = get_content_from_obj(obj, file_type)
            if not text_content or text_content in ["[deleted]", "[removed]"]:
                continue
            
            # Update Activity Counter (for Control filtering)
            user_post_count[author] += 1
            created_utc = int(obj.get("created_utc", 0))

            # --- PATH A: PATIENT FINDING ---
            # We scan EVERY post for diagnosis, even if user is already found, 
            # to capture the EARLIEST one (t0) or multiple disorders.
            match = INCLUSION_REGEX.search(text_content)
            if match:
                is_valid, disorder, is_retro = validate_diagnosis(text_content, match.group(1))
                
                if is_valid:
                    local_patient_history[author].append({
                        "ts": created_utc,
                        "disorder": disorder,
                        "post_id": obj.get("id"),
                        "source": file_name,
                        "is_retrospective": is_retro
                    })
                    # If this user was a control candidate, upgrade them to Patient
                    if author in local_controls:
                        del local_controls[author]
                    continue

            # --- PATH B: CONTROL SAMPLING ---
            if CONTROL_CONFIG["enabled"]:
                # Only if not a patient and not already sampled
                if author not in local_patient_history and author not in local_controls:
                    # Probabilistic Sampling
                    if random.random() < sampling_rate:
                        # Activity Check
                        if user_post_count[author] >= min_posts:
                            local_controls[author] = {
                                "group": "control_candidate",
                                "t0_ts": created_utc, # Pseudo-t0
                                "source_file": file_name
                            }

        # Format output for Reduce step
        formatted_patients = {}
        for user, diagnoses in local_patient_history.items():
            formatted_patients[user] = {
                "group": "patient",
                "diagnoses": diagnoses # List of valid diagnoses
            }
            
        return {**formatted_patients, **local_controls}

    except Exception as e:
        logger.error(f"Scan failed for {file_name}: {e}")
        return {}

# ======================== MASTER INDEX BUILDER (REDUCE STEP) ========================
def build_master_index(zst_root_dir):
    """
    Builds index from organized data structure:
    zst_root_dir/
      ├── comments/RC_*.zst
      └── submissions/RS_*.zst
    """
    # Construct paths using the config layout
    comments_path = os.path.join(zst_root_dir, DATA_LAYOUT["comments_dir"], "RC_*.zst")
    submissions_path = os.path.join(zst_root_dir, DATA_LAYOUT["submissions_dir"], "RS_*.zst")
    
    # Glob both paths
    comment_files = glob.glob(comments_path)
    submission_files = glob.glob(submissions_path)
    
    # Validation
    if not comment_files and not submission_files:
        logger.error(f"No ZST files found!")
        logger.error(f"Checked Comments: {comments_path}")
        logger.error(f"Checked Submissions: {submissions_path}")
        return pd.DataFrame()
        
    all_files = comment_files + submission_files
    all_files.sort() # Sorts roughly by year/month if naming standard holds
    
    logger.info(f"Found {len(comment_files)} Comment files and {len(submission_files)} Submission files.")
    
    # 2. Parallel Map
    cores = get_cpu_cores()
    with Pool(cores, initializer=init_worker) as pool:
        results = pool.map(scan_single_file, all_files)
    
    # 3. Reduce (Merge Logic)
    logger.info("Merging results and resolving comorbidities...")
    
    # Intermediate storage
    merged_users = defaultdict(lambda: {"control_entry": None, "diagnoses": []})
    
    for file_result in results:
        for user, info in file_result.items():
            if info["group"] == "patient":
                # Collect ALL diagnoses from all months
                merged_users[user]["diagnoses"].extend(info["diagnoses"])
            elif info["group"] == "control_candidate":
                # Collect earliest control appearance
                curr = merged_users[user]["control_entry"]
                if not curr or info["t0_ts"] < curr["t0_ts"]:
                    merged_users[user]["control_entry"] = info

    # 4. Final Cohort Construction
    final_registry = {}
    
    for user, data in merged_users.items():
        # RULE: Patient status overrides Control
        if data["diagnoses"]:
            # Sort diagnoses by time to find EARLIEST ONSET
            sorted_diagnoses = sorted(data["diagnoses"], key=lambda x: x["ts"])
            first_diagnosis = sorted_diagnoses[0]
            
            # Analyze Comorbidity
            unique_disorders = set(d["disorder"] for d in sorted_diagnoses)
            is_comorbid = len(unique_disorders) > 1
            
            # Flag Retrospective
            # If the T0 diagnosis is retrospective, flag the user
            is_retro = first_diagnosis["is_retrospective"]
            
            final_registry[user] = {
                "group": "patient",
                "t0_ts": first_diagnosis["ts"],
                "t0_date": datetime.utcfromtimestamp(first_diagnosis["ts"]).strftime("%Y-%m-%d"),
                "disorder": first_diagnosis["disorder"], # The T0 disorder
                "diagnosis_post_id": first_diagnosis["post_id"],
                "is_comorbid": is_comorbid,
                "is_retrospective": is_retro,
                "all_disorders": ",".join(sorted(unique_disorders)),
                "source_file": first_diagnosis["source"]
            }
            
        elif data["control_entry"]:
            ctrl = data["control_entry"]
            final_registry[user] = {
                "group": "control_candidate",
                "t0_ts": ctrl["t0_ts"],
                "t0_date": "PENDING", # Matched in Post-Processing
                "disorder": "control",
                "diagnosis_post_id": None,
                "is_comorbid": False,
                "is_retrospective": False,
                "all_disorders": "",
                "source_file": ctrl["source_file"]
            }

    # 5. Output
    master_df = pd.DataFrame.from_dict(final_registry, orient="index")
    master_df.reset_index(inplace=True)
    master_df.rename(columns={"index": "user_id"}, inplace=True)
    
    # Stats Logging
    n_pat = len(master_df[master_df["group"] == "patient"])
    n_ctrl = len(master_df[master_df["group"] == "control_candidate"])
    n_comorbid = master_df["is_comorbid"].sum()
    n_retro = master_df["is_retrospective"].sum()
    
    logger.info(f"Phase 1 Complete.")
    logger.info(f"Patients: {n_pat} (Comorbid: {n_comorbid}, Retrospective: {n_retro})")
    logger.info(f"Candidate Controls: {n_ctrl}")
    
    output_path = os.path.join(PROCESSING_CONFIG["output_dir"], PROCESSING_CONFIG["master_index_file"])
    master_df.to_parquet(output_path, compression="snappy", index=False)
    logger.info(f"Master index saved to {output_path}")
    
    return master_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 1: Patient Finder (RS+RC)")
    parser.add_argument("--zst_dir", required=True, help="Path to ZST files")
    args = parser.parse_args()
    
    build_master_index(args.zst_dir)