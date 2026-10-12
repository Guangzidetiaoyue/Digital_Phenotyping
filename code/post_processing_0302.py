import os
import glob
import gc
import json
import shutil
import random
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
from math import pi, atan2
from multiprocessing import Pool
from tqdm import tqdm
import hashlib
from config import PROCESSING_CONFIG

WORK_DIR = "/path/to/code" #"/data1/qizhixiao/user/temp_data"
os.makedirs(WORK_DIR, exist_ok=True)
PROCESSING_CONFIG["local_temp_dir"] = "/tmp/reddit_final_build"
PROCESSING_CONFIG["nas_upload_md5_check"] = True

try:
    from utils import setup_logger, clean_text_heavy, init_nlp_tools
    from config import PROCESSING_CONFIG, PSM_CONFIG, BOT_THRESHOLDS
except ImportError:
    import sys; sys.exit(1)


logger = setup_logger()
# Workers only need these for final dataset build, not for stats extraction
nlp, analyzer, anonymizer = None, None, None

def init_post_worker():
    global nlp, analyzer, anonymizer
    # For Phase 1 (Stats), we DO NOT load Spacy/Presidio to save 500MB+ per core
    # We only load them if explicitly needed later
    pass

def init_final_build_worker():
    """Only load heavy NLP models for the final text redaction step"""
    global nlp, analyzer, anonymizer
    try:
        nlp, analyzer, anonymizer = init_nlp_tools()
    except: pass


def init_local_temp():
    """初始化本地临时目录，清理残留文件"""
    os.makedirs(PROCESSING_CONFIG["local_temp_dir"], exist_ok=True)
    # 清理7天前的残留文件（避免磁盘占满）
    for f in glob.glob(os.path.join(PROCESSING_CONFIG["local_temp_dir"], "*.parquet")):
        if os.path.getmtime(f) < (pd.Timestamp.now().timestamp() - 7*86400):
            os.remove(f)
            
def get_file_md5(file_path, chunk_size=4096):
    """计算文件MD5（用于校验完整性）"""
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()

def validate_parquet_file(file_path):
    """校验Parquet文件完整性（读取元数据+最后10行）"""
    try:
        pf = pq.ParquetFile(file_path)
        # 校验元数据
        if not pf.metadata:
            return False
        # 读取最后10行（验证数据页完整性）
        if pf.num_row_groups > 0:
            last_rg = pf.read_row_group(pf.num_row_groups - 1, columns=pf.schema.names[:2])
            if last_rg.num_rows == 0 and pf.num_row_groups > 1:
                return False
        return True
    except Exception as e:
        logger.error(f"文件{file_path}校验失败: {e}")
        return False
  
def upload_to_nas(local_file, nas_file):
    """
    终极修复版：纯手动字节流复制文件内容，无任何权限/元数据操作
    核心：仅读写文件内容，不调用chmod/copystat/copymode等任何权限相关函数
    """
    try:
        # 1. 先删除NAS上的旧文件（仅尝试删除，失败也不终止，避免旧文件干扰）
        if os.path.exists(nas_file):
            try:
                # 不修改权限，直接尝试删除（NAS若允许删除则删，不允许则跳过）
                os.remove(nas_file)
            except Exception as e:
                logger.warning(f"跳过删除NAS旧文件（权限限制）: {nas_file}, 错误: {e}")
        
        # 2. 纯手动按块复制文件内容（核心：仅读写字节，无任何额外操作）
        chunk_size = 1024 * 1024 * 10  # 10MB/块（平衡速度和内存占用）
        total_size = os.path.getsize(local_file)
        copied_size = 0
        
        # 以二进制只读模式打开本地文件，二进制写入模式打开NAS文件
        with open(local_file, 'rb') as f_in, open(nas_file, 'wb') as f_out:
            while True:
                # 读取本地文件块
                chunk = f_in.read(chunk_size)
                if not chunk:
                    break
                # 写入NAS文件块
                f_out.write(chunk)
                copied_size += len(chunk)
        
        # 校验：确保内容完整复制（字节数一致）
        if copied_size != total_size:
            # 清理不完整的NAS文件
            if os.path.exists(nas_file):
                os.remove(nas_file)
            raise Exception(f"文件内容复制不完整：本地{total_size}字节 vs NAS{copied_size}字节")
        
        logger.info(f"纯手动复制内容完成：{local_file} → {nas_file}（{total_size/1024/1024:.2f}MB）")
        
        # 3. 校验文件完整性（仅读取内容，不修改任何属性）
        # MD5校验（仅读取内容，无权限操作）
        if PROCESSING_CONFIG["nas_upload_md5_check"]:
            local_md5 = get_file_md5(local_file)
            nas_md5 = get_file_md5(nas_file)
            if local_md5 != nas_md5:
                os.remove(nas_file)
                raise Exception(f"MD5不一致：本地{local_md5} vs NAS{nas_md5}")
        
        # Parquet格式校验（仅读取元数据，不修改属性）
        if not validate_parquet_file(nas_file):
            os.remove(nas_file)
            raise Exception("NAS文件Parquet格式损坏（内容不完整）")
        
        logger.info(f"上传成功：{nas_file}")
        return True
    
    except Exception as e:
        logger.error(f"上传NAS失败 {local_file} → {nas_file}: {e}")
        # 清理不完整的NAS文件（仅尝试，失败也不报错）
        if os.path.exists(nas_file):
            try:
                os.remove(nas_file)
            except:
                pass
        return False

# ======================== 1. STATS EXTRACTION (MEMORY SAFE) ========================

def process_shard_stats_only(parquet_file):
    """
    [MAPPER] Scans one shard. Calculates vectors/counts LOCALLY.
    CRITICAL: Does NOT return text. Returns numeric aggregates only.
    """
    try:
        # Open file handler
        pf = pq.ParquetFile(parquet_file, pre_buffer=True) # pre_buffer optimizes NAS reads
        
        # Determine columns
        cols = ["user_id_hash", "group", "t0_date", "post_timestamp", 
                "clean_text", "subreddit", "is_retrospective"]
        has_dtype = "data_type" in pf.schema.names
        if has_dtype: cols.append("data_type")
        
        results = {} # Aggregate locally within this file first
        is_rs_file = os.path.basename(parquet_file).startswith("RS")

        # ITERATE BATCHES (Crucial for OOM Prevention)
        # 50k rows is roughly 50-100MB of RAM per chunk
        for batch in pf.iter_batches(batch_size=50000, columns=cols):
            df = batch.to_pandas()
            if df.empty: continue

            # --- Vectorized Ops on CHUNK ---
            # FillNA to avoid crashes
            txt = df["clean_text"].fillna("")
            
            # Lengths & Counts
            n_ch = txt.str.len().astype(np.int32)
            # Count spaces + 1 for words
            n_w = txt.str.count(" ").astype(np.int32) + 1
            # Count punctuation for sentences
            n_sn = txt.str.count(r'[.!?]').astype(np.int32) + 1
            
            # Local Vocab Density (Approximation for RAM safety)
            # We calculate density per post/shard. 
            def get_density(t):
                if not isinstance(t, str) or len(t) < 50:
                    return 0.0
                words = t.split()
                if not words:
                    return 0.0
                return len(set(words)) / len(words)
            
            # We weight this by word count later
            local_densities = txt.apply(get_density).values
            weighted_vd = local_densities * n_w

            # 4. Circadian Math
            timestamps = df["post_timestamp"].values
            hours = (timestamps % 86400) / 3600.0
            rads = hours * (2 * pi / 24)
            sin_t = np.sin(rads)
            cos_t = np.cos(rads)

            # --- Aggregate per User in this Chunk ---
            # We group by user inside the chunk
            # Note: Users might be split across chunks, but we merge `results` dict
            groups = df.groupby("user_id_hash")
            
            for user, idx in groups.indices.items():
                # Get scalar values using first index
                first_idx = idx[0]
                
                if user not in results:
                    results[user] = {
                        "u": user,
                        "g": df["group"].iloc[first_idx],
                        "t0": df["t0_date"].iloc[first_idx],
                        "ret": bool(df["is_retrospective"].iloc[first_idx]),
                        "n_p": 0, "n_s": 0,
                        "n_ch": 0, "n_w": 0, "n_sn": 0,
                        "s_sin": 0.0, "s_cos": 0.0, "w_vd": 0.0,
                        "min_t": float('inf'), "max_t": 0,
                        "sub": Counter()
                    }
                
                entry = results[user]
                
                # Slicing numpy arrays is fast (Views)
                entry["n_p"] += len(idx)
                
                if has_dtype:
                    entry["n_s"] += (df["data_type"].iloc[idx] == "submission").sum()
                else:
                    entry["n_s"] += len(idx) if is_rs_file else 0

                entry["n_ch"] += n_ch[idx].sum()
                entry["n_w"] += n_w[idx].sum()
                entry["n_sn"] += n_sn[idx].sum()
                entry["s_sin"] += sin_t[idx].sum()
                entry["s_cos"] += cos_t[idx].sum()
                entry["w_vd"] += weighted_vd[idx].sum()
                
                # Time bounds
                ts_chunk = timestamps[idx]
                min_ts = ts_chunk.min()
                max_ts = ts_chunk.max()
                if min_ts < entry["min_t"]: entry["min_t"] = min_ts
                if max_ts > entry["max_t"]: entry["max_t"] = max_ts
                
                # Subreddits (Counter update)
                entry["sub"].update(df["subreddit"].iloc[idx])

            # Explicitly delete chunk to free RAM before next iteration
            del df, txt, n_ch, n_w, n_sn, rads
                      
        return list(results.values())
    
    except Exception as e:
        logger.warning(f"Shard error {parquet_file}: {e}")
        return []

def aggregate_batch_and_flush(batch_files, batch_id, temp_dir):
    """
    [REDUCER 1] Processes a batch of files, aggregates in RAM, flushes to Disk.
    This prevents RAM from growing indefinitely.
    """
    # 1. Parallel Map
    batch_results = []
    with Pool(PROCESSING_CONFIG["cpu_cores"], initializer=init_post_worker) as pool:
        # imap_unordered is memory friendly
        itr = pool.imap_unordered(process_shard_stats_only, batch_files)
        for res in itr:
            batch_results.extend(res)
    
    if not batch_results: return
    
    # 2. Aggregation in RAM (Compact)
    user_agg = {}
    
    for r in batch_results:
        u = r["u"]
        if u not in user_agg:
            user_agg[u] = {
                "u": u, "g": r["g"], "t0": r["t0"], "ret": r["ret"],
                "n_p": 0, "n_s": 0, "n_ch": 0, "n_w": 0, "n_sn": 0,
                "s_sin": 0.0, "s_cos": 0.0, "w_vd": 0.0,
                "min_t": float('inf'), "max_t": 0,
                "sub": Counter()
            }
        
        entry = user_agg[u]
        entry["n_p"] += r["n_p"]
        entry["n_s"] += r["n_s"]
        entry["n_ch"] += r["n_ch"]
        entry["n_w"] += r["n_w"]
        entry["n_sn"] += r["n_sn"]
        entry["s_sin"] += r["s_sin"]
        entry["s_cos"] += r["s_cos"]
        entry["w_vd"] += r["w_vd"]
        if r["min_t"] < entry["min_t"]: entry["min_t"] = r["min_t"]
        if r["max_t"] > entry["max_t"]: entry["max_t"] = r["max_t"]
        entry["sub"].update(r["sub"])

    # 3. Serialize & Flatten for Parquet
    # We convert the Counter to a JSON string.
    # This removes the Python Object overhead during storage.
    flat_rows = []
    for u, v in user_agg.items():
        # Keep only top 200 subs per batch to prevent massive JSON strings
        top_subs = dict(v["sub"].most_common(200))
        v["sub_json"] = json.dumps(top_subs)
        del v["sub"] # Free RAM
        flat_rows.append(v)
    
    # 4. Flush to Parquet
    out_path = os.path.join(temp_dir, f"temp_stats_{batch_id}.parquet")
    pd.DataFrame(flat_rows).to_parquet(out_path, index=False, compression='snappy')
    
    # 5. Explicit Cleanup
    del batch_results, user_agg, flat_rows
    gc.collect()

def final_merge_stats(temp_dir):
    """
    [REDUCER 2] Merges the intermediate partial stats files.
    """
    logger.info("Merging intermediate stats files...")
    temp_files = glob.glob(os.path.join(temp_dir, "*.parquet"))
    
    global_agg = {}
    
    for f in tqdm(temp_files, desc="Merging Batches"):
        df = pd.read_parquet(f)
        for _, row in df.iterrows():
            u = row["u"]
            if u not in global_agg:
                global_agg[u] = {
                    "u": u, "g": row["g"], "t0": row["t0"], "ret": row["ret"],
                    "n_p": 0, "n_s": 0, "n_ch": 0, "n_w": 0, "n_sn": 0,
                    "s_sin": 0.0, "s_cos": 0.0, "w_vd": 0.0,
                    "min_t": float('inf'), "max_t": 0,
                    "sub": Counter()
                }
            
            entry = global_agg[u]
            entry["n_p"] += row["n_p"]
            entry["n_s"] += row["n_s"]
            entry["n_ch"] += row["n_ch"]
            entry["n_w"] += row["n_w"]
            entry["n_sn"] += row["n_sn"]
            entry["s_sin"] += row["s_sin"]
            entry["s_cos"] += row["s_cos"]
            entry["w_vd"] += row["w_vd"]
            
            if row["min_t"] < entry["min_t"]: entry["min_t"] = row["min_t"]
            if row["max_t"] > entry["max_t"]: entry["max_t"] = row["max_t"]
            
            if row["sub_json"]:
                entry["sub"].update(json.loads(row["sub_json"]))
    
    # Calculate Final Features
    final_features = []
    logger.info("Calculating derived features...")
    
    for u, v in global_agg.items():
        # Tenure
        tenure = max(0.1, (v["max_t"] - v["min_t"]) / 86400)
        
        # 1. Readability (Global Avg)
        avg_cpw = v["n_ch"] / max(1, v["n_w"])
        avg_wps = v["n_w"] / max(1, v["n_sn"])
        ari = (4.71 * avg_cpw) + (0.5 * avg_wps) - 21.43
        
        # 2. Vocab Density (Weighted Avg)
        vocab_density = v["w_vd"] / max(1, v["n_w"])
        
        # 3. Circadian (Vector Avg)
        n = max(1, v["n_p"])
        avg_sin = v["s_sin"] / n
        avg_cos = v["s_cos"] / n
        reg = np.sqrt(avg_sin**2 + avg_cos**2)
        angle = atan2(avg_sin, avg_cos)
        if angle < 0: angle += 2 * pi
        hour = angle * (24 / (2*pi))
        
        # 4. Entropy
        sub_ent = entropy(list(v["sub"].values()), base=2)
        
        # 5. Bot Proxy
        avg_word_len = v["n_ch"] / max(1, v["n_w"])
        
        final_features.append({
            "user_id_hash": u,
            "group": v["g"],
            "real_t0_date": v["t0"],
            "is_retrospective": v["ret"],
            
            "posts_per_day": round(v["n_p"] / tenure, 3),
            "tenure_days": round(tenure, 1),
            "submission_ratio": round(v["n_s"] / n, 3),
            "readability_grade": round(max(0, min(ari, 20)), 2),
            "vocab_density": round(vocab_density, 3),
            "subreddit_entropy": round(sub_ent, 3),
            "circadian_mean_hour": round(hour, 2),
            "circadian_regularity": round(reg, 4),
            
            "avg_word_len": round(avg_word_len, 2),
            "total_posts": v["n_p"],
            "first_post_ts": v["min_t"],
            "last_post_ts": v["max_t"]
        })
        
    return pd.DataFrame(final_features)

def extract_features_disk_backed(shard_dir):
    """
    Orchestrates the Batched Map-Reduce.
    """
    temp_dir = os.path.join(WORK_DIR, "temp_stats_agg")
    if os.path.exists(temp_dir): 
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    files = sorted(glob.glob(os.path.join(shard_dir, "*.parquet")))
    if not files: 
        raise ValueError("No shards found.")
    
    # Process in batches of 50 files
    # 50 files * 30GB uncompressed ~ high load, but we only store stats.
    BATCH_SIZE = 24 
    total_batches = (len(files) + BATCH_SIZE - 1) // BATCH_SIZE
    
    logger.info(f"Extracting features from {len(files)} shards in {total_batches} batches...")
    
    for i in range(total_batches):
        batch = files[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        logger.info(f"Processing Batch {i+1}/{total_batches}...")
        
        if not os.path.exists(os.path.join(temp_dir, f"temp_stats_{i}.parquet")):
            aggregate_batch_and_flush(batch, i, temp_dir)
            
        gc.collect() # Force RAM release
        
    # Final Merge
    df = final_merge_stats(temp_dir)
    
    # Cleanup
    # shutil.rmtree(temp_dir)
    return df

# ======================== 2. BOT FILTERING (UNCHANGED) ========================

def filter_bots_scientific(feature_df):
    logger.info("Step 2: Scientific Bot Filtering...")
    df = feature_df.copy().fillna(0)
    
    mask_speed = df["posts_per_day"] <= BOT_THRESHOLDS["max_posts_per_day"]
    mask_quality = df["avg_word_len"] >= 2.5
    mask_diversity = df["vocab_density"] >= BOT_THRESHOLDS["min_text_diversity"]
    
    mask_burner = ~(
        (df["tenure_days"] < BOT_THRESHOLDS["min_account_age_days"]) &
        (df["posts_per_day"] > BOT_THRESHOLDS["burner_bot_posts_per_day"])
    )
    
    clean_df = df[mask_speed & mask_quality & mask_diversity & mask_burner].copy()
    logger.info(f"Bots removed: {len(df) - len(clean_df)} | Remaining: {len(clean_df)}")
    return clean_df.reset_index(drop=True)

# ======================== 3. PSM MATCHING (UNCHANGED) ========================

def assign_shadow_t0(clean_df):
    logger.info("Step 3: Assigning Shadow T0...")
    pats = clean_df[clean_df["group"] == "patient"]
    if pats.empty: return clean_df
    
    pat_t0_dist = pd.to_datetime(pats["real_t0_date"]).astype(np.int64) // 10**9
    
    controls = clean_df[clean_df["group"] == "control_candidate"].copy()
    # Random sampling from patient distribution
    controls["shadow_t0_ts"] = np.random.choice(pat_t0_dist.values, size=len(controls))
    
    # Check Validity (Pre/Post history)
    min_pre = PSM_CONFIG["min_pre_exposure_days"] * 86400
    min_post = 30 * 86400 
    valid_mask = (
        (controls["shadow_t0_ts"] > (controls["first_post_ts"] + min_pre)) &
        (controls["shadow_t0_ts"] < (controls["last_post_ts"] - min_post))
    )
    
    valid_controls = controls[valid_mask].copy()
    
    # Patients get their real T0
    pats = pats.copy()
    pats["shadow_t0_ts"] = pat_t0_dist
    
    return pd.concat([pats, valid_controls], ignore_index=True)

def run_psm_matching(df):
    logger.info("Step 4: Propensity Score Matching...")
    df = df.copy()
    df["log_posts"] = np.log1p(df["posts_per_day"])
    df["log_tenure"] = np.log1p(df["tenure_days"])
    
    # Feature set
    features = ["log_posts", "log_tenure", "readability_grade", 
                "subreddit_entropy", "submission_ratio"]
    
    data = df.dropna(subset=features)
    
    # 1. Propensity Score
    scaler = StandardScaler()
    X = scaler.fit_transform(data[features])
    y = (data["group"] == "patient").astype(int)
    
    lr = LogisticRegression(n_jobs=-1, max_iter=1000)
    lr.fit(X, y)
    data["ps_logit"] = lr.decision_function(X)
    
    # 2. Matching
    pats = data[data["group"] == "patient"]
    ctrls = data[data["group"] == "control_candidate"]
    
    nbrs = NearestNeighbors(n_neighbors=PSM_CONFIG["ratio"], algorithm='ball_tree')
    nbrs.fit(ctrls[["ps_logit"]])
    
    caliper = PSM_CONFIG["caliper"] * data["ps_logit"].std()
    dists, idxs = nbrs.kneighbors(pats[["ps_logit"]])
    
    matched_dfs = []
    used_ctrls = set()
    
    for i in range(len(pats)):
        matched_dfs.append(pats.iloc[[i]])
        cnt = 0
        for d, idx in zip(dists[i], idxs[i]):
            if d > caliper: continue
            real_idx = ctrls.index[idx]
            if not PSM_CONFIG["replace"] and real_idx in used_ctrls: continue
            matched_dfs.append(ctrls.iloc[[idx]])
            used_ctrls.add(real_idx)
            cnt += 1
            if cnt >= PSM_CONFIG["ratio"]: break
            
    matched_df = pd.concat(matched_dfs)
    logger.info(f"Matched Total: {len(matched_df)}")
    return matched_df

# ======================== 4. FINAL BUILD (MEMORY SAFE) ========================

def process_final_build_chunk(args):
    """
    Reads huge Parquet file in chunks, filters, redacts, and writes chunks sequentially.
    Never holds full file in RAM.
    """
    input_file, valid_map, output_file = args
    original_filename = os.path.basename(input_file)
    local_input_temp = os.path.join(
        PROCESSING_CONFIG["local_temp_dir"],
        f"input_{original_filename}.tmp.{os.getpid()}.parquet"
    )
    local_output_temp = os.path.join(
        PROCESSING_CONFIG["local_temp_dir"],
        f"output_{original_filename}.tmp.{os.getpid()}.parquet"
    )
    
    try:
        # Pre-buffer for NAS speedup
        shutil.copy2(input_file, local_input_temp)
        nas_file_size = os.path.getsize(input_file)
        local_file_size = os.path.getsize(local_input_temp)
        if nas_file_size != local_file_size:
            raise Exception(f"输入文件拷贝大小不一致：NAS({nas_file_size}B) vs 本地({local_file_size}B)")
        # 校验Parquet完整性（避免拷贝的文件本身已损坏）
        if not validate_parquet_file(local_input_temp):
            raise Exception("本地拷贝的输入文件Parquet格式损坏")
        logger.info(f"输入文件本地拷贝并校验完成: {local_input_temp}")
        
        if os.path.exists(output_file) and validate_parquet_file(output_file):
            logger.info(f"NAS文件已存在且完整，跳过: {output_file}")
            return True
        
        
        pf = pq.ParquetFile(local_input_temp, pre_buffer=True)
        # schema = pf.schema.to_arrow_schema()
       
        # We need to construct a new schema because we add cols
        # It's safer to let pandas infer, but we use ParquetWriter
        writer = None
        target_schema = pa.schema([
            ("user_id_hash", pa.string()),
            ("post_timestamp", pa.int64()),
            ("post_datetime", pa.string()),
            ("relative_days", pa.float64()),
            ("data_type", pa.string()),
            ("interaction_type", pa.string()),
            ("parent_id", pa.string()),
            ("link_id", pa.string()),
            ("is_top_level", pa.bool_()),
            ("num_comments", pa.int64()),
            ("subreddit", pa.string()),
            ("score", pa.float64()),
            ("controversiality", pa.int64()),
            ("gilded", pa.int64()),
            ("is_edited", pa.bool_()),
            ("is_nsfw", pa.bool_()),
            ("has_media", pa.bool_()),
            ("domain", pa.string()),
            ("author_flair", pa.string()),
            ("clean_text", pa.string()),
            ("flag_suicide", pa.bool_()),
            ("flag_self_harm", pa.bool_()),
            ("group", pa.string()),
            ("disorder_group", pa.string()),
            ("t0_date", pa.string()),
            ("is_diagnosis_post", pa.bool_()),
            ("is_retrospective", pa.bool_()),
            ("post_id", pa.string()),
            ("assigned_t0", pa.int64()),
            ("__index_level_0__", pa.int64())
        ])
        # Iterate chunks
        for batch in pf.iter_batches(batch_size=PROCESSING_CONFIG["batch_size"]):
            df = batch.to_pandas()
            
            # 1. Filter
            df = df[df["user_id_hash"].isin(valid_map.keys())].copy()
            if df.empty: 
                continue
            
            # 2. Add Metadata
            df["assigned_t0"] = df["user_id_hash"].map(valid_map)
            df["relative_days"] = (df["post_timestamp"] - df["assigned_t0"]) / 86400
            df["score"] = df["score"].fillna(0).astype(np.float64)
            # 3. Redact (Expensive)
            if analyzer and anonymizer:
                df["clean_text"] = df["clean_text"].fillna("").apply(lambda x: clean_text_heavy(x, analyzer, anonymizer))
            
            # 4. Stream Write
            table = pa.Table.from_pandas(df, schema=target_schema, safe=False)
            
            if writer is None:
                writer = pq.ParquetWriter(
                    local_output_temp,
                    target_schema,
                    compression='snappy'
                )
            writer.write_table(table)
            
        if writer:
            writer.close()
            # 校验本地文件完整性
            if not validate_parquet_file(local_output_temp):
                raise Exception("本地文件校验失败，可能损坏")
            logger.info(f"本地文件处理完成并校验通过: {local_output_temp}")
        
        # 5. 上传到NAS
        if not upload_to_nas(local_output_temp, output_file):
            raise Exception("上传NAS失败")
        
        return True

    except Exception as e:
        logger.error(f"处理失败 {input_file}: {e}")
        # 异常时清理所有不完整文件
        for temp_file in [local_input_temp, local_output_temp]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        if os.path.exists(output_file):
            os.remove(output_file)
        return False
    
    finally:
        # 最终清理所有本地临时文件（无论成功/失败）
        for temp_file in [local_input_temp, local_output_temp]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"清理临时文件失败 {temp_file}: {e}")


def build_final_dataset(matched_df, raw_dir, out_file):
    logger.info("Step 5: Building Final Dataset...")
    init_local_temp()
    # Map: User -> T0
    user_map = matched_df.set_index("user_id_hash")["shadow_t0_ts"].to_dict()
    
    out_dir = out_file.replace(".parquet", "_dataset")
    os.makedirs(out_dir, exist_ok=True)
    files_finished = os.listdir(out_dir)
    
    files_all = sorted(glob.glob(os.path.join(raw_dir, "*.parquet")))
    files = []
    for fi in files_all:
        f_temp = fi.split('/')[-1]
        if f_temp not in files_finished:
            files.append(fi)
    work_args = [(f, user_map, os.path.join(out_dir, os.path.basename(f))) for f in files]
    
    # Use fewer cores for this step because PII redaction is CPU/RAM intensive
    safe_cores = max(4, PROCESSING_CONFIG["cpu_cores"] // 2)
    
    with Pool(safe_cores, initializer=init_final_build_worker) as pool:
        results = list(tqdm(pool.imap_unordered(process_final_build_chunk, work_args), total=len(work_args)))
    
    # 结果统计
    success_count = sum(results)
    fail_count = len(results) - success_count
    logger.info(f"处理完成 - 成功: {success_count}, 失败: {fail_count}")
    if fail_count > 0:
        logger.warning(f"有{fail_count}个文件处理失败，请检查日志")
    
    # 最终校验NAS目录下的文件
    nas_files = glob.glob(os.path.join(out_dir, "*.parquet"))
    corrupt_nas_files = [f for f in nas_files if not validate_parquet_file(f)]
    if corrupt_nas_files:
        logger.error(f"NAS上发现{len(corrupt_nas_files)}个损坏文件: {corrupt_nas_files}")
    else:
        logger.info(f"Final Dataset Location: {out_dir} (所有文件校验通过)")
    
    logger.info(f"Final Dataset Location: {out_dir}")

# ======================== MAIN ========================

if __name__ == "__main__":
    # upload_to_nas("/tmp/reddit_final_build/output_RC_2016-10.zst.parquet.tmp.1784190.parquet","/path/to/project/cohort_v1/final_longitudinal_data_dataset/RC_2015-08.zst.parquet")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_dir", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()
    
    
    # Work directory for intermediate files
    # WORK_DIR = "/data1/qizhixiao/user/temp_data"#os.path.dirname(args.output_file)
    
    # 1. Extract Features (Disk-Backed)
    features = extract_features_disk_backed(args.shard_dir)
    
    # 2. Filter
    clean_meta = filter_bots_scientific(features)
    
    # 3. Match
    t0_assigned = assign_shadow_t0(clean_meta)
    matched_meta = run_psm_matching(t0_assigned)
    
    # Save the index (Lightweight)
    matched_meta.to_parquet(args.output_file, index=False)
    
    # 4. Build Final Data (Heavy)
    build_final_dataset(matched_meta, args.shard_dir, args.output_file)