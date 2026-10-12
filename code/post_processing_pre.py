# post_processing.py - Scientific Rigor Edition
# Implements Incidence Density Sampling, SMD Validation, and Socio-Demographic Proxies

import os
import glob
import gc, random
import json
import logging
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import textstat
from datetime import datetime
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, PowerTransformer
from scipy.stats import entropy
from math import pi, atan2
# Imports
from utils import setup_logger, save_metadata
from config import PROCESSING_CONFIG, PSM_CONFIG, BOT_THRESHOLDS

RANDOM_SEED = PSM_CONFIG.get("seed", 2025) 
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

logger = setup_logger()

def calculate_fast_ari(text):
    """
    Calculates Automated Readability Index (ARI).
    Performance: ~50x faster than Flesch-Kincaid (No syllable counting).
    """
    if not text: return 0.0
    
    # 1. Count Characters (Alphanumeric only ideally, but len is close enough for proxy)
    num_chars = len(text.replace(" ", ""))
    
    # 2. Count Words (Spaces + 1)
    # Using count is faster than len(split())
    num_words = text.count(" ") + 1
    
    # 3. Count Sentences (Punctuation)
    # Fast scan for terminators
    num_sentences = text.count(".") + text.count("!") + text.count("?")
    if num_sentences == 0: num_sentences = 1
    
    # 4. Calculate ARI
    # Formula: 4.71 * (chars/words) + 0.5 * (words/sentences) - 21.43
    ari_score = (4.71 * (num_chars / num_words)) + \
                (0.5 * (num_words / num_sentences)) - 21.43
                
    # Clamp to reasonable grade levels (0-20)
    return max(0, min(ari_score, 20))

# ======================== 1. MEMORY-SAFE FEATURE EXTRACTION ========================

def calculate_user_proxies(text_list, timestamps, subreddits):
    """
    Calculates socio-demographic proxies with Circadian Regularity.
    """
    if not text_list or not timestamps:
        return None

    # --- A. CIRCADIAN RHYTHM (Timezone + Regularity) ---
    # 1. Convert timestamps to continuous hours (0-23.99)
    # Adding minutes ensures precision
    hours = np.array([
        datetime.utcfromtimestamp(ts).hour + 
        datetime.utcfromtimestamp(ts).minute / 60.0 
        for ts in timestamps
    ])
    
    # 2. Convert to Radians (0 to 2π)
    radians = hours * (2 * pi / 24)
    
    # 3. Calculate Vector Components
    avg_sin = np.mean(np.sin(radians))
    avg_cos = np.mean(np.cos(radians))
    
    # 4. Calculate Mean Angle (Timezone)
    avg_angle = atan2(avg_sin, avg_cos)
    if avg_angle < 0: 
        avg_angle += 2 * pi
    circadian_mean_hour = avg_angle * (24 / (2 * pi))
    
    # 5. Calculate Resultant Vector Length (REGULARITY)
    # Formula: R = sqrt(mean_sin^2 + mean_cos^2)
    # Range: 0.0 (Chaos) to 1.0 (Clockwork Precision)
    circadian_regularity = np.sqrt(avg_sin**2 + avg_cos**2)

    # --- B. TEXT SAMPLING ---
    TARGET_SIZE = 50000
    sample_pool = list(text_list)
    random.shuffle(sample_pool)
    
    sampled_text_parts = []
    current_len = 0
    for text in sample_pool:
        clean = text.strip()
        if clean and clean[-1] not in ".?!": 
            clean += "."
        sampled_text_parts.append(clean)
        current_len += len(clean)
        if current_len >= TARGET_SIZE: 
            break
    full_sample_text = " ".join(sampled_text_parts)

    # --- C. EDUCATION & VOCAB ---
    try:
        readability = calculate_fast_ari(full_sample_text)
    except:
        readability = 0.0

    words = full_sample_text.split()
    vocab_density = len(set(words)) / len(words) if len(words) > 10 else 0.0

    # --- D. INTEREST ENTROPY ---
    sub_entropy = entropy(list(Counter(subreddits).values()), base=2) if subreddits else 0.0

    return {
        "readability_grade": round(readability, 2),
        "vocab_density": round(vocab_density, 3),
        "subreddit_entropy": round(sub_entropy, 3),
        
        # New Time Variables
        "circadian_mean_hour": round(circadian_mean_hour, 2), # The "When"
        "circadian_regularity": round(circadian_regularity, 4) # The "Stability"
    }


def extract_features_mapreduce(shard_dir):
    """
    Iterates through shards to aggregate user stats.
    MEMORY SAFETY: Caps text buffer per user to prevent OOM.
    """
    logger.info("Step 1: Feature Extraction (Map-Reduce)...")
    
    parquet_files = glob.glob(os.path.join(shard_dir, "*.parquet"))
    parquet_files.sort()
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {shard_dir}")

    user_stats = {}
    
    # MAP PHASE
    for i, f in enumerate(parquet_files):
        try:
            if i % 10 == 0:
                logger.info(f"Scanning shard {i}/{len(parquet_files)}...")
                gc.collect()

            # Read only necessary columns
            table = pq.read_table(f, columns=[
                "user_id_hash", "group", "t0_date", "post_timestamp", 
                "clean_text", "subreddit", "is_retrospective"
            ])
            df = table.to_pandas()
            
            # Pre-calculate simple text stats vectorized (for Bot Filtering)
            df['char_len'] = df['clean_text'].str.len()
            df['word_count'] = df['clean_text'].str.count(" ") + 1
            
            for user, sub_df in df.groupby("user_id_hash"):
                if user not in user_stats:
                    user_stats[user] = {
                        "group": sub_df["group"].iloc[0],
                        "real_t0_date": sub_df["t0_date"].iloc[0],
                        "timestamps": [],
                        "texts": [], # Will be capped
                        "subreddits": [],
                        "text_len_buffer": 0,
                        "total_chars": 0,
                        "total_words": 0,
                        "is_retrospective": sub_df["is_retrospective"].any()
                    }
                
                # Metadata (Always keep full history)
                user_stats[user]["timestamps"].extend(sub_df["post_timestamp"].tolist())
                user_stats[user]["subreddits"].extend(sub_df["subreddit"].tolist())
                
                # Accumulate totals for Bot Filtering
                user_stats[user]["total_chars"] += sub_df['char_len'].sum()
                user_stats[user]["total_words"] += sub_df['word_count'].sum()
                
                # OPTIMIZATION: Buffer Text for NLP
                # Only keep ~15k chars. This allows ARI calc but prevents RAM explosion.
                if user_stats[user]["text_len_buffer"] < 50000:
                    texts = sub_df["clean_text"].tolist()
                    for t in texts:
                        user_stats[user]["texts"].append(t)
                        user_stats[user]["text_len_buffer"] += len(t)
                        if user_stats[user]["text_len_buffer"] >= 50000: break
            
            del df, table
            
        except Exception as e:
            logger.warning(f"Error reading shard {f}: {e}")
            continue

    # REDUCE PHASE
    logger.info(f"Aggregating stats for {len(user_stats)} users...")
    final_features = []
    
    for user, data in user_stats.items():
        if not data["timestamps"]: 
            continue
        
        # Calculate Proxies (NLP + Chronobiology)
        proxies = calculate_user_proxies(data["texts"], data["timestamps"], data["subreddits"])
        if not proxies:
            continue

        timestamps = sorted(data["timestamps"])
        first_post = timestamps[0]
        last_post = timestamps[-1]
        tenure_days = max(1, (last_post - first_post) / 86400)
        total_posts = len(timestamps)
        
        # Calculate Avg Word Length (Bot Detection)
        avg_word_len = data["total_chars"] / max(1, data["total_words"])

        row = {
            "user_id_hash": user,
            "group": data["group"],
            "real_t0_date": data["real_t0_date"],
            "first_post_ts": first_post,
            "last_post_ts": last_post,
            "tenure_days": tenure_days,
            "total_posts": total_posts,
            "posts_per_day": total_posts / max(1, tenure_days),
            "avg_word_len": avg_word_len,
            "is_retrospective": data["is_retrospective"]
        }
        row.update(proxies)
        final_features.append(row)
        
    del user_stats
    gc.collect()
    
    final_df = pd.DataFrame(final_features)
    final_df.sort_values("user_id_hash",inplace=True)
    final_df.reset_index(drop=True, inplace=True)
    return final_df

# ======================== 2. BOT FILTERING (ROBUST) ========================

def filter_bots_scientific(feature_df):
    """
    Multi-Dimensional Bot Veto (CONSORT-Compliant Logging).
    Filters bots via 4 orthogonal criteria: Speed/Quality/Diversity/Burner.
    Logs exclusion rates for CONSORT diagram (required for top journals).
    
    Args:
        feature_df (pd.DataFrame): User-level feature dataframe with columns:
            - posts_per_day: Avg daily posts per user
            - avg_word_len: Average word length of user's text
            - vocab_density: Type-Token Ratio (lexical diversity)
            - tenure_days: Account age in days
        
    Returns:
        pd.DataFrame: Cleaned dataframe with non-bot users
    """
    logger.info("Step 2: Scientific Bot Filtering...")
    n_total = len(feature_df)
    
    feature_df = feature_df.copy()
    feature_df["tenure_days"] = feature_df["tenure_days"].clip(lower=1)
    feature_df["avg_word_len"] = feature_df["avg_word_len"].fillna(2.5)
    feature_df["vocab_density"] = feature_df["vocab_density"].fillna(0.0)
    feature_df["posts_per_day"] = feature_df["posts_per_day"].clip(upper=1000)
    # 1. SPEED VETO (Superhuman Posting Frequency)
    mask_speed = feature_df["posts_per_day"] <= BOT_THRESHOLDS["max_posts_per_day"]
    
    # 2. QUALITY VETO (Spam/Gibberish - Short Words)
    # Avg word length < 2.5 implies "lol", "k", "yes" spam
    mask_quality = feature_df["avg_word_len"] >= 2.5
    
    # 3. DIVERSITY VETO (Repetitive Bots - Low Lexical Diversity)
    # TTR < 0.2 means recycling same words
    mask_diversity = feature_df["vocab_density"] >= BOT_THRESHOLDS["min_text_diversity"]
    
    # 4. BURNER VETO (New Account + High Volume)
    # Use config values instead of hardcode (REPRODUCIBILITY!)
    mask_burner = ~(
        (feature_df["tenure_days"] < BOT_THRESHOLDS["min_account_age_days"]) &
        (feature_df["posts_per_day"] > BOT_THRESHOLDS["burner_bot_posts_per_day"])
    )
    
    clean_df = feature_df[mask_speed & mask_quality & mask_diversity & mask_burner].copy()
    clean_df = clean_df.reset_index(drop=True)
    
    # ======================== CONSORT-Compliant Logging (顶刊要求) ========================
    # Calculate exclusion counts (absolute + percentage)
    n_speed = (~mask_speed).sum()
    n_qual = (~mask_quality).sum()
    n_div = (~mask_diversity).sum()
    n_burn = (~mask_burner).sum()
    n_total_removed = n_total - len(clean_df)
    
    # Log with percentages (critical for CONSORT diagram)
    logger.info(f"--- BOT EXCLUSION (CONSORT) ---")
    logger.info(f"1. Speed (> {BOT_THRESHOLDS['max_posts_per_day']}/day): {n_speed} ({n_speed/n_total:.1%})")
    logger.info(f"2. Quality (Avg Word Len < 2.5): {n_qual} ({n_qual/n_total:.1%})")
    logger.info(f"3. Diversity (TTR < {BOT_THRESHOLDS['min_text_diversity']}): {n_div} ({n_div/n_total:.1%})")
    logger.info(f"4. Burner (Age < {BOT_THRESHOLDS['min_account_age_days']}d + High Freq): {n_burn} ({n_burn/n_total:.1%})")
    logger.info(f"--------------------------------")
    logger.info(f"TOTAL REMOVED: {n_total_removed} ({n_total_removed/n_total:.1%})")
    logger.info(f"REMAINING NON-BOT USERS: {len(clean_df)} ({len(clean_df)/n_total:.1%})")
    
    return clean_df

# ======================== 3. PSM WITH CALENDAR ANCHORING ========================

def assign_shadow_t0(clean_df):
    """
    Scientific Shadow T0 Assignment: Risk-Set / Incidence Density Sampling.
    
    Ensures:
    1. Temporal Validity: Control was actually active at the assigned T0.
    2. Calendar Alignment: Control T0s follow the same distribution as Patient T0s.
    3. Tenure Alignment: Control has sufficient history prior to T0.
    """
    logger.info("Step 3: Assigning Shadow T0s (Risk-Set Sampling)...")
    clean_df.sort_values("user_id_hash")
    
    # Separate Groups
    patients = clean_df[clean_df["group"] == "patient"].copy()
    controls = clean_df[clean_df["group"] == "control_candidate"].copy()
    
    # 1. Build the Patient Probability Distribution (The "Risk Set")
    # We want controls to have T0s distributed exactly like Patients
    valid_patient_t0s = pd.to_datetime(patients["real_t0_date"], errors='coerce').dropna()
    patient_t0_timestamps = valid_patient_t0s.astype('int64') // 10**9
    
    if len(patient_t0_timestamps) == 0:
        logger.error("No valid patient T0 dates found!")
        return clean_df

    # 2. Vectorized Rejection Sampling
    # We try to assign a valid date. If the date is outside the user's lifespan, we retry.
    # We do 3 passes. This covers >99% of valid controls without looping 1M times.
    
    controls["shadow_t0_ts"] = np.nan
    remaining_indices = controls.index
    
    # Minimum required history (e.g., 30 days) to avoid "Day 1" T0s
    MIN_PRE_EXPOSURE = PSM_CONFIG["min_pre_exposure_days"] * 86400 
    local_rng = np.random.RandomState(RANDOM_SEED)
    
    for attempt in range(1, 6):
        num_needed = len(remaining_indices)
        if num_needed == 0: break
        
        # Draw random dates from Patient Distribution
        draws = local_rng.choice(patient_t0_timestamps, size=num_needed)
        
        # Get bounds for these specific users
        subset = controls.loc[remaining_indices]
        starts = subset["first_post_ts"].values
        ends = subset["last_post_ts"].values
        
        # 3. Check Scientific Constraints
        # Constraint A: Must be after user joined + buffer (Pre-Exposure check)
        # Constraint B: Must be before user stopped posting (Censoring check)
        is_valid = (draws > (starts + MIN_PRE_EXPOSURE)) & (draws < ends)
        
        # Assign valid draws
        valid_mask = pd.Series(is_valid, index=remaining_indices)
        valid_indices = valid_mask[valid_mask].index
        
        controls.loc[valid_indices, "shadow_t0_ts"] = draws[is_valid]
        
        # Update remaining list
        remaining_indices = valid_mask[~valid_mask].index
        
        logger.info(f"  Attempt {attempt}: Assigned {len(valid_indices)} T0s. Remaining: {len(remaining_indices)}")

    # 4. Drop Controls that couldn't be matched to the timeline
    # (e.g., A control active only in 2010 cannot cover a 2020 diagnosis distribution)
    valid_controls = controls.dropna(subset=["shadow_t0_ts"]).copy()
    valid_controls["shadow_t0_ts"] = valid_controls["shadow_t0_ts"].astype(int)
    
    dropped_count = len(controls) - len(valid_controls)
    if dropped_count > 0:
        logger.warning(f"  Dropped {dropped_count} controls due to temporal mismatch (Calendar Misalignment).")

    # 5. Unify Dataframes
    patients = patients.copy()
    patients["shadow_t0_ts"] = pd.to_datetime(patients["real_t0_date"]).astype('int64') // 10**9
    
    result_df = pd.concat([patients, valid_controls])
    
    # 6. Scientific Validation Log
    # Prove that the distributions match
    p_mean = datetime.utcfromtimestamp(patients["shadow_t0_ts"].mean()).year
    c_mean = datetime.utcfromtimestamp(valid_controls["shadow_t0_ts"].mean()).year
    logger.info(f"  Validation: Patient Mean Year = {p_mean}, Control Mean Year = {c_mean}")
    
    return result_df

def calculate_smd(df, cols):
    """Calculate Standardized Mean Difference (SMD) between patient and control groups"""
    smd_dict = {}
    
    patients = df[df["group"] == "patient"]
    controls = df[df["group"] == "control_candidate"]
    
    if len(patients) == 0 or len(controls) == 0:
        logger.warning("Cannot calculate SMD: Zero samples for patients or controls")
        return {col: np.nan for col in cols}
    
    for col in cols:
        if col not in df.columns:
            logger.warning(f"Column {col} does not exist, skipping SMD calculation")
            smd_dict[col] = np.nan
            continue
            
        # Remove NaN values
        p_vals = patients[col].dropna()
        c_vals = controls[col].dropna()
        
        if len(p_vals) == 0 or len(c_vals) == 0:
            smd_dict[col] = np.nan
            continue
        
        mean_p = p_vals.mean()
        mean_c = c_vals.mean()
        var_p = p_vals.var(ddof=1)  # Sample variance (degrees of freedom = 1)
        var_c = c_vals.var(ddof=1)
        
        # Pooled standard deviation
        pooled_sd = np.sqrt((var_p + var_c) / 2)
        
        if pooled_sd > 0:
            smd = abs(mean_p - mean_c) / pooled_sd
        else:
            smd = 0  # Both groups have zero variance (identical distributions)
        
        smd_dict[col] = smd
    
    return smd_dict

def run_psm_matching(df_with_t0):
    """
    Execute hybrid matching (exact stratification + Mahalanobis distance propensity score matching).
    Design objective: Maximize covariate balance (SMD < 0.1) while preserving sample size.
    
    Parameters:
        df_with_t0 (pd.DataFrame): DataFrame containing user features and assigned shadow_t0_ts.
        
    Returns:
        matched_df (pd.DataFrame): DataFrame containing successfully matched patients and control users.
    """
    logger.info("Step 4: Hybrid Exact+PS Matching (Final Optimized)...")
    
    # ==========================================
    # 1. Data Preparation and Deterministic Sorting
    # ==========================================
    df = df_with_t0.sort_values("user_id_hash").reset_index(drop=True).copy()
    
    # ==========================================
    # 2. Core Feature Engineering (for Matching)
    # ==========================================
    # A. Calculate raw tenure and fix potential negative/zero values
    df["raw_tenure"] = (df["shadow_t0_ts"] - df["first_post_ts"]) / 86400
    # Critical fix: Ensure tenure is positive (minimal positive value for log1p compatibility)
    min_positive_tenure = 1e-3  # 0.001 days (~1.4 minutes)
    df.loc[df["raw_tenure"] < min_positive_tenure, "raw_tenure"] = min_positive_tenure
    
    # B. Gentle Winsorization (instead of aggressive clipping)
    cols_to_winsorize = ["posts_per_day", "raw_tenure", "subreddit_entropy", "readability_grade"]
    for col in cols_to_winsorize:
        if col in df.columns:
            q_low = df[col].quantile(0.01)
            q_high = df[col].quantile(0.99)
            df[f"{col}_win"] = df[col].clip(lower=q_low, upper=q_high)
        else:
            logger.warning(f"Column {col} does not exist, skipping winsorization.")
    
    # C. Log transformation for skewed variables
    # Note: circadian_* variables are typically cyclic and retain linear scale
    df["match_posts"] = np.log1p(df["posts_per_day_win"])
    df["match_tenure"] = np.log1p(df["raw_tenure_win"])
    df["match_entropy"] = np.log1p(df["subreddit_entropy_win"])
    df["match_readability"] = df["readability_grade_win"]  # Winsorized, retain linear scale
    
    # Circadian features used directly (already circular statistics)
    df["match_circadian_hour"] = df["circadian_mean_hour"]
    df["match_circadian_reg"] = df["circadian_regularity"]
    
    # Define final feature set for modeling
    match_features = [
        "match_readability",
        "match_circadian_hour",
        "match_circadian_reg", 
        "match_entropy",
        "match_posts",
        "match_tenure"
    ]
    
    # Drop missing values
    model_data = df.dropna(subset=match_features).copy()
    logger.info(f"Modeling data: {len(model_data)} rows (Patients: {sum(model_data['group']=='patient')})")
    
    # ==========================================
    # 3. Data-Driven: Identify Stubborn Variables for Exact Stratification
    # ==========================================
    # Strategy: Calculate SMD on full unmatched sample, select top 1-2 most imbalanced variables
    def rank_variables_by_imbalance(data, feature_cols):
        """Rank variables by Standardized Mean Difference (SMD) to identify most imbalanced features"""
        smd_scores = {}
        patients = data[data['group'] == 'patient']
        controls = data[data['group'] == 'control_candidate']
        
        for feat in feature_cols:
            # Calculate SMD
            mean_t = patients[feat].mean()
            mean_c = controls[feat].mean()
            var_t = patients[feat].var()
            var_c = controls[feat].var()
            n_t, n_c = len(patients), len(controls)
            
            pooled_sd = np.sqrt(((n_t-1)*var_t + (n_c-1)*var_c) / (n_t + n_c - 2))
            if pooled_sd > 0:
                smd = abs(mean_t - mean_c) / pooled_sd
            else:
                smd = 0
            smd_scores[feat] = smd
            
        # Sort by SMD descending
        ranked = sorted(smd_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked
    
    # Identify most imbalanced variables
    ranked_vars = rank_variables_by_imbalance(model_data, match_features)
    logger.info("Variable imbalance ranking (pre-matching):")
    for var, smd in ranked_vars[:3]:  # Show top 3
        logger.info(f"  {var}: SMD = {smd:.3f}")
    
    # Select top imbalanced variables for exact stratification (typically 1-2)
    n_strata_vars = min(2, len([v for v, s in ranked_vars if s > 0.15]))
    exact_match_vars = [ranked_vars[i][0] for i in range(n_strata_vars)]
    
    if exact_match_vars:
        logger.info(f"Variables selected for exact stratification: {exact_match_vars}")
    else:
        logger.info("No significantly imbalanced variables found, using propensity score matching only.")
        exact_match_vars = []
    
    # Create stratification IDs
    if exact_match_vars:
        # Tertile binning for each selected variable (balance sample size and stratification effect)
        for var in exact_match_vars:
            try:
                model_data[f'stratum_{var}'] = pd.qcut(
                    model_data[var], 
                    q=3, 
                    labels=False, 
                    duplicates='drop'
                )
            except Exception as e:
                logger.warning(f"Stratification failed for variable {var}: {e}. Using equal-width binning.")
                model_data[f'stratum_{var}'] = pd.cut(
                    model_data[var], 
                    bins=3, 
                    labels=False
                )
        
        # Combine stratification variables (if multiple selected)
        if len(exact_match_vars) == 1:
            model_data['exact_stratum'] = model_data[f'stratum_{exact_match_vars[0]}'].astype(str)
        else:
            model_data['exact_stratum'] = (
                model_data[f'stratum_{exact_match_vars[0]}'].astype(str) + "_" +
                model_data[f'stratum_{exact_match_vars[1]}'].astype(str)
            )
    else:
        # No exact stratification needed - all samples in single stratum
        model_data['exact_stratum'] = "all"
    
    logger.info(f"Created {model_data['exact_stratum'].nunique()} exact strata.")
    
    # ==========================================
    # 4. Calculate Global Propensity Scores
    # ==========================================
    # Standardization
    scaler = StandardScaler()
    X_raw = scaler.fit_transform(model_data[match_features])
    
    # Yeo-Johnson transformation for skewed distributions
    pt = PowerTransformer(method='yeo-johnson')
    X_transformed = pt.fit_transform(X_raw)
    
    # Logistic regression fitting
    lr = LogisticRegression(
        solver='lbfgs',
        max_iter=2000,
        C=0.1,  # L2 regularization strength
        random_state=PSM_CONFIG["seed"],
        n_jobs=-1  # Use all CPU cores
    )
    y = (model_data["group"] == "patient").astype(int)
    lr.fit(X_transformed, y)
    
    # Use logit scores (decision function) for matching (more stable at extremes than probabilities)
    model_data["ps_logit"] = lr.decision_function(X_transformed)
    
    # ==========================================
    # 5. Intra-Stratum Matching (Core of Hybrid Strategy)
    # ==========================================
    matched_records = []
    global_matched_control_ids = set()
    
    # Statistical metrics
    strata_stats = []
    
    # Iterate over each exact stratum
    for stratum_id, stratum_df in model_data.groupby('exact_stratum'):
        patients = stratum_df[stratum_df['group'] == 'patient']
        controls = stratum_df[stratum_df['group'] == 'control_candidate']
        
        # Skip strata with insufficient samples
        if len(patients) == 0 or len(controls) < PSM_CONFIG["ratio"]:
            logger.debug(f"Skipping stratum {stratum_id}: {len(patients)} patients, {len(controls)} candidate controls")
            continue
        
        # Calculate propensity score distribution for dynamic caliper
        stratum_ps_std = stratum_df["ps_logit"].std()
        if pd.isna(stratum_ps_std) or stratum_ps_std == 0:
            stratum_ps_std = 0.1  # Default value
        
        # Dynamic caliper: 0.2 standard deviations (relaxed slightly with exact stratification)
        stratum_caliper = PSM_CONFIG["caliper"] * stratum_ps_std
        
        # Nearest neighbor matching (using propensity score logit)
        nn = NearestNeighbors(
            n_neighbors=min(PSM_CONFIG["ratio"] * 3, len(controls)),  # Find extra candidates
            radius=stratum_caliper,
            algorithm='ball_tree'  # Efficient for 1D data
        )
        nn.fit(controls[["ps_logit"]])
        
        distances, indices = nn.kneighbors(patients[["ps_logit"]])
        
        # Match controls for each patient in the stratum
        for i, (dists, idxs) in enumerate(zip(distances, indices)):
            # Filter controls within caliper
            valid_matches = [(dist, idx) for dist, idx in zip(dists, idxs) if dist <= stratum_caliper]
            
            if not valid_matches:
                continue  # No suitable controls for this patient in stratum
            
            # Add patient
            patient_record = patients.iloc[[i]].copy()
            matched_records.append(patient_record)
            
            # Add controls (without replacement)
            matches_added = 0
            for dist, idx in sorted(valid_matches):  # Sort by distance, pick closest first
                if matches_added >= PSM_CONFIG["ratio"]:
                    break
                
                control_id = controls.iloc[idx]["user_id_hash"]
                
                # Check if control already matched globally (no replacement)
                if not PSM_CONFIG["replace"] and control_id in global_matched_control_ids:
                    continue
                
                control_record = controls.iloc[[idx]].copy()
                matched_records.append(control_record)
                global_matched_control_ids.add(control_id)
                matches_added += 1
            
            # Record stratum matching stats
            strata_stats.append({
                'stratum': stratum_id,
                'patient_id': patient_record['user_id_hash'].iloc[0],
                'controls_matched': matches_added
            })
    
    # ==========================================
    # 6. Combine Results and Basic Validation
    # ==========================================
    if not matched_records:
        logger.error("Error: No matches found! Please check data or relax matching criteria.")
        return pd.DataFrame()
    
    matched_df = pd.concat(matched_records, ignore_index=True)
    matched_df = matched_df.drop_duplicates(subset="user_id_hash")
    
    # Match outcome statistics
    n_matched_patients = (matched_df['group'] == 'patient').sum()
    n_matched_controls = (matched_df['group'] == 'control_candidate').sum()
    
    logger.info(f"Matching complete: {n_matched_patients} patients, {n_matched_controls} controls")
    logger.info(f"Patient match rate: {n_matched_patients / len(model_data[model_data['group']=='patient']):.1%}")
    
    # ==========================================
    # 7. Comprehensive Validation (Multi-Dimensional)
    # ==========================================
    logger.info("=" * 50)
    logger.info("Comprehensive Matching Quality Validation")
    logger.info("=" * 50)
    
    # 7.1 Validate SMD (raw variables)
    raw_validation_cols = [
        "readability_grade", "circadian_mean_hour", "circadian_regularity",
        "subreddit_entropy", "posts_per_day", "raw_tenure"
    ]
    
    smd_raw = calculate_smd(matched_df, raw_validation_cols)
    
    logger.info("1. Standardized Mean Difference (SMD) - Raw Variables:")
    logger.info("   (Target: All variables < 0.1, ideal < 0.05)")
    
    all_smd_pass = True
    for var, smd_val in smd_raw.items():
        status = "✅" if smd_val < 0.1 else "⚠️" if smd_val < 0.2 else "❌"
        logger.info(f"   {var:25s}: {smd_val:6.4f} {status}")
        if smd_val >= 0.1:
            all_smd_pass = False
    
    # 7.2 Validate SMD (transformed matching variables)
    logger.info("\n2. Covariate Balance in Matched Variable Space (Transformed):")
    smd_transformed = calculate_smd(matched_df, match_features)
    
    for var, smd_val in smd_transformed.items():
        status = "✅" if smd_val < 0.1 else "⚠️" if smd_val < 0.2 else "❌"
        logger.info(f"   {var:25s}: {smd_val:6.4f} {status}")
    
    # 7.3 Propensity score distribution overlap validation
    logger.info("\n3. Propensity Score Distribution Overlap Test:")
    ps_patients = matched_df[matched_df['group']=='patient']['ps_logit']
    ps_controls = matched_df[matched_df['group']=='control_candidate']['ps_logit']
    
    # Kolmogorov-Smirnov test: Test if two distributions are identical
    ks_stat, ks_p = stats.ks_2samp(ps_patients, ps_controls)
    logger.info(f"   Kolmogorov-Smirnov Test: D = {ks_stat:.4f}, p = {ks_p:.4e}")
    logger.info(f"   {'✅ Distributions not significantly different' if ks_p > 0.05 else '⚠️ Distributions may differ'}")
    
    # 7.4 Matching ratio check
    logger.info("\n4. Matching Ratio and Sample Size:")
    logger.info(f"   Patient:Control match ratio ≈ 1:{n_matched_controls/n_matched_patients:.2f}")
    
    # Check if patients have insufficient matched controls
    if n_matched_controls / n_matched_patients < PSM_CONFIG["ratio"] * 0.5:
        logger.warning("   ⚠️ Average match ratio below 50% of target - may impact statistical power.")
    else:
        logger.info("   ✅ Matching ratio meets expectations.")
    
    # ==========================================
    # 8. Final Assessment and Recommendations
    # ==========================================
    logger.info("\n" + "=" * 50)
    logger.info("Final Assessment")
    logger.info("=" * 50)
    
    if all_smd_pass:
        logger.info("🎉 SUCCESS: All key covariates achieve good balance (SMD < 0.1)!")
        logger.info("   Matched cohort is ready for subsequent analysis.")
    else:
        logger.warning("⚠️  WARNING: Residual imbalance exists for some covariates (SMD ≥ 0.1).")
        logger.warning("   Recommendations:")
        
        # Diagnose problematic variables
        problem_vars = [var for var, smd in smd_raw.items() if smd >= 0.1]
        logger.warning(f"   1. Primary problematic variables: {problem_vars}")
        
        if len(problem_vars) == 1 and problem_vars[0] == "readability_grade":
            logger.warning("   2. 'readability_grade' may be a disease phenotype rather than a confounder.")
            logger.warning("      Consider controlling for this variable in subsequent regression models.")
        else:
            logger.warning("   2. Consider the following adjustments:")
            logger.warning("      - Increase control group sampling rate in Phase 1")
            logger.warning("      - Implement finer stratification for residual imbalanced variables")
            logger.warning("      - Include these variables as covariates in the final analysis model")
    
    # Save matching metadata for post-analysis
    match_metadata = {
        'n_patients': n_matched_patients,
        'n_controls': n_matched_controls,
        'match_ratio': n_matched_controls / n_matched_patients,
        'smd_summary': {k: float(v) for k, v in smd_raw.items()},  # Convert to native Python types
        'exact_match_vars_used': exact_match_vars,
        'n_strata': model_data['exact_stratum'].nunique(),
        'match_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save metadata to file
    metadata_path = os.path.join(PROCESSING_CONFIG["output_dir"], "matching_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(match_metadata, f, indent=2, default=str)
    
    logger.info(f"Matching metadata saved to: {metadata_path}")
    
    return matched_df



# ======================== 4. DATASET RECONSTRUCTION ========================

def build_final_dataset(matched_meta_df, raw_shard_dir, output_path):
    """
    Filters the massive raw Parquet shards to keep ONLY the matched users.
    Assigns the correct T0 to controls.
    """
    logger.info("Step 4: Reconstructing full dataset for matched users...")
    
    # Create lookup map for T0s
    # Controls now have a 'shadow_t0_ts' that we must enforce as their analytical T0
    user_map = matched_meta_df.set_index("user_id_hash")[["shadow_t0_ts", "group"]].to_dict("index")
    valid_users = set(user_map.keys())
    
    # Output writer (Append mode)
    # We use PyArrow dataset writer or manual chunking
    writer = None
    schema = None
    
    parquet_files = glob.glob(os.path.join(raw_shard_dir, "*.parquet"))
    total_rows = 0
    
    for f in parquet_files:
        try:
            # Read everything, filter in memory (files are usually manageable chunks)
            df = pd.read_parquet(f)
            
            # Filter
            filtered = df[df["user_id_hash"].isin(valid_users)].copy()
            if filtered.empty: continue
            
            # Update T0 and Relative Days for Controls
            # Vectorized Map
            filtered["assigned_t0"] = filtered["user_id_hash"].map(lambda x: user_map[x]["shadow_t0_ts"])
            
            # Recalculate relative days based on the ASSIGNED Shadow T0
            filtered["relative_days"] = (filtered["post_timestamp"] - filtered["assigned_t0"]) / 86400
            
            # Write
            table = pa.Table.from_pandas(filtered)
            
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(output_path, schema, compression='snappy')
            
            # Check for schema consistency (float vs int issues) before writing
            # PyArrow handles this better if we cast, but let's assume pandas handled types
            writer.write_table(table)
            total_rows += len(filtered)
            
        except Exception as e:
            logger.error(f"Error processing shard {f} for final output: {e}")
            
    if writer:
        writer.close()
        
    logger.info(f"Final Matched Cohort Saved: {output_path} ({total_rows:,} rows)")

# ======================== MAIN ========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_dir", required=True, help="Directory of Phase 2 shards")
    parser.add_argument("--output_file", required=True, help="Path for final matched parquet")
    args = parser.parse_args()
    
    # 1. Extract Features
    features = extract_features_mapreduce(args.shard_dir)
    
    # 2. Filter Bots
    clean_features = filter_bots_scientific(features)
    
    # 3. Match
    t0_assigned = assign_shadow_t0(clean_features)
    matched_meta = run_psm_matching(t0_assigned)
    
    # 4. Build Final Data
    build_final_dataset(matched_meta, args.shard_dir, args.output_file)