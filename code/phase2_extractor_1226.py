# phase2_extractor.py - Phase 2: Multi-Modal History Extraction
# Optimized for Nature/Science: Text + Network + Behavior + Chronobiology

import os
import glob
import json
import logging
import traceback
import pandas as pd
import pyarrow.parquet as pq
from multiprocessing import Pool, Manager
from datetime import datetime
from tqdm import tqdm

# Import utilities and final config
from utils import (
    setup_logger, read_zst_file, clean_text, hash_user_id,
    get_cpu_cores, save_metadata, init_nlp_tools
)
from config import (
    PROCESSING_CONFIG, SUICIDE_KEYWORDS, SELF_HARM_KEYWORDS, DATA_LAYOUT
)

# Initialize logger
logger = setup_logger()

# ======================== 1. ROBUST VALIDATION UTILS ========================

def is_valid_parquet(file_path):
    """
    Scientifically rigorous check: 
    Not just size > 0, but can PyArrow actually read the footer?
    """
    if not os.path.exists(file_path):
        return False
    try:
        # Read metadata only (fast)
        pq.read_metadata(file_path)
        return True
    except Exception:
        return False
    
# Global PII tools (worker initialization)
analyzer = None
anonymizer = None

def init_pii_worker():
    """Initialize PII redaction tools (per worker process)"""
    global analyzer, anonymizer
    if analyzer is None or anonymizer is None:
        _, analyzer, anonymizer = init_nlp_tools()

# ======================== SINGLE FILE EXTRACTION ========================
def extract_single_file_safe(args):
    """
    Wrapper to ensure NO exception crashes the main process.
    Returns a status dict: {'file': ..., 'status': ..., 'count': ..., 'error': ...}
    """
    zst_path, target_users, output_shard_dir = args
    file_name = os.path.basename(zst_path)
    shard_path = os.path.join(output_shard_dir, f"{file_name}.parquet")
    
    # Initialize PII tools
    try:
        init_pii_worker()
    except Exception as e:
        return {'file': file_name, 'status': 'FAILED', 'error': f"Init failed: {str(e)}"}

    # Processing Logic
    try:
        is_submission = file_name.startswith("RS_")
        records = []
        
        # Stream Read
        stream = read_zst_file(zst_path)
        
        for obj in stream:
            author = obj.get("author")
            if author not in target_users:
                continue
            
            # --- EXTRACT LOGIC (Condensed for brevity, same as previous logic) ---
            # ... (Insert the Content/Network/Behavior extraction logic here) ...
            #                 if selftext in ["[deleted]", "[removed]", None]: selftext = ""  raw_text = f"{title}\n{selftext}".strip()
            # (Copying the core logic from previous version for context consistency)
            # === 1. CONTENT & NETWORK NORMALIZATION ===
            if is_submission:
                # Submissions
                title = obj.get("title", "")
                selftext = obj.get("selftext", "")
                if selftext in ["[deleted]", "[removed]", None]: selftext = ""
                
                raw_text = f"{title}\n{selftext}".strip()
                data_type = "submission"
                
                # Network: Root Node
                parent_id = None
                link_id = obj.get("id")
                # Normalize link_id to t3_ format
                if link_id and not link_id.startswith("t3_"): 
                    link_id = f"t3_{link_id}"
                
                is_top_level = True
                interaction_type = "root_story"
                
                # Submission specific metadata
                domain = obj.get("domain", "")
                url = obj.get("url", "")
                num_comments = obj.get("num_comments", 0)
                controversiality = 0 # Not present in submissions usually
                
            else:
                # Comments
                raw_text = obj.get("body", "").strip()
                data_type = "comment"
                
                # Network: Child Node
                parent_id = obj.get("parent_id", "")
                link_id = obj.get("link_id", "")
                
                # Determine Interaction Depth
                if parent_id.startswith("t3_") or parent_id == link_id:
                    is_top_level = True
                    interaction_type = "direct_reply_to_op"
                else:
                    is_top_level = False
                    interaction_type = "nested_discussion"
                
                # Comment specific metadata
                domain = ""
                url = ""
                num_comments = 0
                controversiality = obj.get("controversiality", 0)

            # Skip empty content
            if not raw_text or raw_text in ["[deleted]", "[removed]"]:
                continue
            
            # === 2. METADATA & TIMELINE ===
            user_info = target_users[author]
            post_ts = int(obj.get("created_utc", 0))
            post_id = obj.get("id", "")
            
            if user_info["group"] == "patient":
                t0_date = user_info["t0_date"]
                is_diag = (post_id == user_info.get("diagnosis_post_id"))
                rel_days = round((post_ts - user_info["t0_ts"]) / 86400, 2)
            else:
                t0_date = "PENDING"
                is_diag = False
                rel_days = 0

            # === 3. BEHAVIORAL INDICATORS (Science Rigor) ===
            
            # Editing behavior (Perfectionism/Anxiety marker)
            # 'edited' can be False or a timestamp
            is_edited = obj.get("edited") is not False
            
            # Risk behavior (Mania/Impulsivity marker)
            is_nsfw = obj.get("over_18", False)
            
            # Media Sharing (Rich content)
            has_media = bool(obj.get("media") or (url and url.endswith(('.jpg', '.png', '.gif', '.mp4'))))

            # Identity Claims (Flair)
            flair_text = obj.get("author_flair_text") or ""

            # === 4. NLP & RISK FLAGS ===
            txt_low = raw_text.lower()
            flag_suicide = any(k in txt_low for k in SUICIDE_KEYWORDS)
            flag_self_harm = any(k in txt_low for k in SELF_HARM_KEYWORDS)

            # PII Cleaning
            clean_text_content = clean_text(raw_text, analyzer, anonymizer)

            # === 5. FINAL RECORD CONSTRUCTION ===
            records.append({
                "user_id_hash": hash_user_id(author),
                
                # Temporal (Chronobiology)
                "post_timestamp": post_ts,
                "post_datetime": datetime.utcfromtimestamp(post_ts).strftime("%Y-%m-%d %H:%M:%S"),
                "relative_days": rel_days,
                
                # Network (Social Isolation Analysis)
                "data_type": data_type,
                "interaction_type": interaction_type,
                "parent_id": parent_id,
                "link_id": link_id,
                "is_top_level": is_top_level,
                "num_comments": num_comments,
                "subreddit": obj.get("subreddit", ""),
                
                # Social Reward (Reinforcement Learning)
                "score": obj.get("score", 0),
                "controversiality": controversiality,
                "gilded": obj.get("gilded", 0),
                
                # Behavioral (Impulsivity/Anxiety)
                "is_edited": is_edited,
                "is_nsfw": is_nsfw,
                "has_media": has_media,
                "domain": domain,
                "author_flair": flair_text,
                
                # Content & Clinical
                "clean_text": clean_text_content,
                "flag_suicide": flag_suicide,
                "flag_self_harm": flag_self_harm,
                "group": user_info["group"],
                "disorder_group": user_info["disorder"],
                "t0_date": t0_date,
                "is_diagnosis_post": is_diag,
                "is_retrospective": user_info.get("is_retrospective", False),
                "post_id": post_id
            })

        # === RESULT HANDLING ===
        if records:
            df = pd.DataFrame(records)
            df.to_parquet(shard_path, compression="snappy", index=False)
            return {'file': file_name, 'status': 'COMPLETED', 'count': len(records), 'path': shard_path}
        else:
            # CLEANLINESS: Do NOT write empty parquet files.
            # We return status 'EMPTY' so the manifest knows we checked it.
            return {'file': file_name, 'status': 'EMPTY', 'count': 0}

    except Exception as e:
        # SAFETY: Capture any crash (ZST corruption, Memory, etc.)
        err_msg = traceback.format_exc()
        return {'file': file_name, 'status': 'FAILED', 'error': str(e)} # Don't send full stack to main log to keep it clean

# ======================== 3. MANIFEST MANAGER ========================

class ManifestManager:
    """Tracks state of every file (Success/Empty/Failed)"""
    def __init__(self, output_dir):
        self.path = os.path.join(output_dir, "processing_manifest.jsonl")
        self.state = {}
        self.load()
    
    def load(self):
        if os.path.exists(self.path):
            with open(self.path, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        self.state[entry['file']] = entry
                    except: pass
    
    def update(self, result):
        """Append update to log (Crash-resilient append-only)"""
        entry = {
            "file": result['file'],
            "status": result['status'],
            "count": result.get('count', 0),
            "timestamp": datetime.now().isoformat(),
            "error": result.get('error', None)
        }
        self.state[result['file']] = entry
        
        with open(self.path, 'a') as f:
            f.write(json.dumps(entry) + "\n")

    def should_process(self, file_path, shard_dir):
        file_name = os.path.basename(file_path)
        if file_name not in self.state:
            return True # Never seen
        
        entry = self.state[file_name]
        
        # If previously marked EMPTY or COMPLETED, verify validity
        if entry['status'] == 'COMPLETED':
            # Verification: Does the output file actually exist and valid?
            expected_shard = os.path.join(shard_dir, f"{file_name}.parquet")
            if is_valid_parquet(expected_shard):
                return False # Really done
            else:
                return True # Corrupt/Missing, re-process
        
        elif entry['status'] == 'EMPTY':
            return False # Known empty, skip
            
        elif entry['status'] == 'FAILED':
            return True # Retry failures
            
        return True

# ======================== BATCH EXECUTION ========================
def extract_user_history(zst_root_dir, master_index_path):
    """
    Main Driver for Phase 2.
    Scans Comments and Submissions folders defined in config.
    Returns: Path to the directory containing shards.
    """
    # 1. Load Master Index
    if not os.path.exists(master_index_path):
        logger.error("Master index missing.")
        return None
    
    master_df = pd.read_parquet(master_index_path)
    target_users = master_df.set_index("user_id").to_dict("index")
    logger.info(f"Target Users Loaded: {len(target_users):,}")

    # 2. Setup Shard Directory
    output_dir = PROCESSING_CONFIG["output_dir"]
    shard_dir = os.path.join(output_dir, "temp_shards")
    os.makedirs(shard_dir, exist_ok=True)

    manifest = ManifestManager(output_dir)
    
    # 3. Gather Files (Using DATA_LAYOUT from config)
    comments_path = os.path.join(zst_root_dir, DATA_LAYOUT["comments_dir"], "RC_*.zst")
    submissions_path = os.path.join(zst_root_dir, DATA_LAYOUT["submissions_dir"], "RS_*.zst")
    
    files_comments = glob.glob(comments_path)
    files_submissions = glob.glob(submissions_path)
    all_files = files_comments + files_submissions
    all_files.sort()

    worklist = []
    skipped_stats = {"COMPLETED": 0, "EMPTY": 0}
    
    for f in all_files:
        if manifest.should_process(f, shard_dir):
            worklist.append((f, target_users, shard_dir))
        else:
            status = manifest.state[os.path.basename(f)]['status']
            skipped_stats[status] = skipped_stats.get(status, 0) + 1
            
    logger.info(f"Total Files: {len(all_files)}")
    logger.info(f"Skipped (Done): {skipped_stats.get('COMPLETED', 0)}")
    logger.info(f"Skipped (Empty): {skipped_stats.get('EMPTY', 0)}")
    logger.info(f"To Process: {len(worklist)}")
    
    if not worklist:
        return shard_dir

    # 5. Parallel Execution (Using imap for real-time monitoring)
    cores = get_cpu_cores()
    success_count = 0
    fail_count = 0
    empty_count = 0
    
    
    # Use imap to handle results as they come in (Real-time Manifest Update)
    with Pool(cores, initializer=init_pii_worker, maxtasksperchild=5) as pool:
        # Use tqdm to track progress
        for result in tqdm(pool.imap_unordered(extract_single_file_safe, worklist), total=len(worklist)):
            
            # Update Manifest immediately
            manifest.update(result)
            
            if result['status'] == 'COMPLETED':
                success_count += 1
            elif result['status'] == 'EMPTY':
                empty_count += 1
            elif result['status'] == 'FAILED':
                fail_count += 1
                logger.error(f"Failed {result['file']}: {result.get('error')}")

    # 6. Final Report
    logger.info("=== BATCH COMPLETE ===")
    logger.info(f"New Shards: {success_count}")
    logger.info(f"Empty Files: {empty_count} (Clean - No files written)")
    logger.info(f"Failures: {fail_count}")
    
    if fail_count > 0:
        logger.warning(f"Warning: {fail_count} files failed. Check processing_manifest.jsonl for details.")
        
    return shard_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 2: Multi-Modal Extraction")
    parser.add_argument("--zst_root_dir", required=True, help="Root dir containing comments/ and submissions/")
    parser.add_argument("--master_index", required=True, help="Path to master_patient_index.parquet")
    args = parser.parse_args()
    
    extract_user_history(args.zst_root_dir, args.master_index)