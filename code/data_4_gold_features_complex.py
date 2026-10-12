import os
import glob
import json
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.stats import entropy
from tqdm import tqdm

# CONFIG
SILVER_DIR = "/path/to/project/cohort_v1/silver_v2/data"
GOLD_ROOT = "/path/to/project/cohort_v1/gold_v2"
POST_FEAT_DIR = os.path.join(GOLD_ROOT, "post_level_features")
WEEKLY_DIR = os.path.join(GOLD_ROOT, "weekly_aggregates")
TAG_DIR = os.path.join(GOLD_ROOT, "dimension_tables")
PROGRESS_FILE = os.path.join(GOLD_ROOT, "partition_progress.json")

# Ensure Dirs
for d in [POST_FEAT_DIR, WEEKLY_DIR, TAG_DIR]: os.makedirs(d, exist_ok=True)

analyzer = SentimentIntensityAnalyzer()

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f: return json.load(f)
    return {}

def save_progress(prog):
    with open(PROGRESS_FILE, "w") as f: json.dump(prog, f)

# MENTOR REQUIREMENT: Burstiness (B) and Memory (M)
def calculate_temporal_dynamics(timestamps):
    """
    Returns (burstiness, memory_coefficient).
    Timestamps must be sorted unix seconds.
    """
    n = len(timestamps)
    if n < 3: 
        return np.nan, np.nan
    
    # Inter-arrival times
    tau = np.diff(timestamps) # [t1-t0, t2-t1, ...]
    
    # 1. Burstiness (Goh & Barabasi, 2008)
    # Range -1 (Regular) to 1 (Bursty)
    mean_tau = np.mean(tau)
    std_tau = np.std(tau)
    
    if (mean_tau + std_tau) == 0:
        B = 0
    else:
        B = (std_tau - mean_tau) / (std_tau + mean_tau)
        
    # 2. Memory Coefficient (M)
    # Correlation between consecutive intervals (tau_i, tau_i+1)
    tau_1 = tau[:-1]
    tau_2 = tau[1:]
    
    if len(tau_1) < 2 or np.std(tau_1) == 0 or np.std(tau_2) == 0:
        M = 0
    else:
        # Pearson correlation
        M = np.corrcoef(tau_1, tau_2)[0, 1]
        
    return B, M

def process_partition(file_path):
    part_id = os.path.basename(file_path).replace("part_", "").replace(".parquet", "")
    df = pd.read_parquet(file_path)
    
    # 1. Dimension Table (Tags)
    user_tags = df[["user_id_hash", "group", "is_retrospective", "t0_date"]].drop_duplicates("user_id_hash")
    user_tags.to_parquet(os.path.join(TAG_DIR, f"tags_{part_id}.parquet"))
    
    # 2. Level 1 Features (Redundancy)
    # Compute Sentiment Per Post
    df["sentiment"] = df["clean_text"].fillna("").apply(
        lambda x: analyzer.polarity_scores(str(x))['compound']
    )
    # Save granular
    l1_cols = ["user_id_hash", "post_timestamp", "sentiment", "subreddit"]
    df[l1_cols].to_parquet(os.path.join(POST_FEAT_DIR, f"posts_{part_id}.parquet"))
    
    # 3. Level 2 Features (Weekly Aggregates)
    df["dt"] = pd.to_datetime(df["post_timestamp"], unit='s')
    weekly_rows = []
    
    for uid, sub in df.groupby("user_id_hash"):
        sub = sub.set_index("dt").sort_index()
        weeks = sub.resample("W")
        
        for w_start, w_frame in weeks:
            w_end = w_start + pd.Timedelta(days=6, hours=23, minutes=59)
            
            # --- Handling Silence ---
            if w_frame.empty:
                weekly_rows.append({
                    "user_id_hash": uid,
                    "week_start": w_start,
                    "week_end": w_end,
                    "post_count": 0,
                    # MENTOR REQUIREMENT: Explicit NaN for silence
                    "sentiment_mean": np.nan,
                    "sentiment_std": np.nan, 
                    "entropy": np.nan,
                    "burstiness": np.nan,
                    "memory_coef": np.nan,
                    "active_subreddits": []
                })
                continue
            
            # --- Handling Activity ---
            # A. Physics (Sentiment)
            sents = w_frame["sentiment"].values
            s_mean = np.mean(sents)
            # MENTOR REQUIREMENT: Std needs >1 point
            s_std = np.std(sents) if len(sents) > 1 else np.nan
            
            # B. Complex Systems (Temporal)
            ts = w_frame["post_timestamp"].values
            B, M = calculate_temporal_dynamics(ts)
            
            # C. Cognition (Entropy)
            all_text = " ".join(w_frame["clean_text"].fillna("").astype(str).tolist()).split()
            if len(all_text) > 0:
                ent = entropy(pd.Series(all_text).value_counts())
            else:
                ent = np.nan
                
            # D. Ecology (Subreddits for Hull)
            subs = w_frame["subreddit"].dropna().unique().tolist()
            
            weekly_rows.append({
                "user_id_hash": uid,
                "week_start": w_start,
                "week_end": w_end,
                "post_count": len(w_frame),
                "sentiment_mean": s_mean,
                "sentiment_std": s_std,
                "entropy": ent,
                "burstiness": B,
                "memory_coef": M,
                "active_subreddits": subs
            })
            
    # Save Weekly
    if weekly_rows:
        pd.DataFrame(weekly_rows).to_parquet(os.path.join(WEEKLY_DIR, f"weekly_{part_id}.parquet"))

if __name__ == "__main__":
    progress = load_progress()
    files = sorted(glob.glob(os.path.join(SILVER_DIR, "*.parquet")))
    
    for f in tqdm(files, desc="Gold Extraction"):
        fname = os.path.basename(f)
        if progress.get(fname) == "DONE": continue
            
        try:
            process_partition(f)
            progress[fname] = "DONE"
            save_progress(progress)
        except Exception as e:
            print(f"Error {fname}: {e}")
            progress[fname] = "FAILED"
            save_progress(progress)