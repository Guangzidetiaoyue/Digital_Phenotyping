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
from scipy.optimize import minimize
from math import pi, atan2
from multiprocessing import Pool
from tqdm import tqdm
import hashlib

WORK_DIR = "/path/to/code"
os.makedirs(WORK_DIR, exist_ok=True)

try:
    from utils import setup_logger, clean_text_heavy, init_nlp_tools
    from config import PROCESSING_CONFIG, BOT_THRESHOLDS
except ImportError:
    import sys; sys.exit(1)

# ======================== PSM_CONFIG (完整定义，避免缺键) ========================
# 如果 config.py 中已有 PSM_CONFIG，可以从那里导入后用 update 合并
# 这里给出完整的默认值，确保所有键都存在

try:
    from config import PSM_CONFIG as _PSM_CONFIG_BASE
except ImportError:
    _PSM_CONFIG_BASE = {}

PSM_CONFIG = {
    # --- 匹配架构 ---
    "strategy": "stratified_ps",

    # --- 分层设置 ---
    "exact_strata_vars": ["tenure_quartile"],
    "n_tenure_quantiles": 4,

    # --- PS模型 ---
    "ps_features": [
        "log_posts_per_day",
        "log_tenure",
        "readability_grade",
        "subreddit_entropy",
        "submission_ratio",
        "circadian_mean_hour_sin",
        "circadian_mean_hour_cos",
        "circadian_regularity",
        "vocab_density",
        "log_total_posts",
        "active_days_ratio",
    ],

    # --- 匹配参数 ---
    "ratio": 5,
    "caliper_sd_multiplier": 0.2,
    "replace": False,

    # --- 质量门槛 ---
    "max_smd": 0.1,
    "max_mean_smd": 0.05,
    "min_common_support": 0.05,
    "min_matched_ratio": 3.0,

    # --- 敏感性分析 ---
    "run_iptw": True,
    "run_entropy_balancing": True,
    "rosenbaum_gamma_range": [1.0, 1.5, 2.0, 2.5, 3.0],

    # --- 最低数据要求 ---
    "min_pre_exposure_days": 30,
    "min_post_exposure_days": 30,
    "min_total_posts": 5,
}
# 用 config.py 中的值覆盖默认值（如果有的话）
PSM_CONFIG.update(_PSM_CONFIG_BASE)

PROCESSING_CONFIG["local_temp_dir"] = "/tmp/reddit_final_build"
PROCESSING_CONFIG["nas_upload_md5_check"] = True

logger = setup_logger()
nlp, analyzer, anonymizer = None, None, None


def init_post_worker():
    global nlp, analyzer, anonymizer
    pass


def init_final_build_worker():
    """Only load heavy NLP models for the final text redaction step"""
    global nlp, analyzer, anonymizer
    try:
        nlp, analyzer, anonymizer = init_nlp_tools()
    except:
        pass


def init_local_temp():
    """初始化本地临时目录，清理残留文件"""
    os.makedirs(PROCESSING_CONFIG["local_temp_dir"], exist_ok=True)
    for f in glob.glob(os.path.join(PROCESSING_CONFIG["local_temp_dir"], "*.parquet")):
        if os.path.getmtime(f) < (pd.Timestamp.now().timestamp() - 7 * 86400):
            os.remove(f)


def get_file_md5(file_path, chunk_size=4096):
    """计算文件MD5（用于校验完整性）"""
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def validate_parquet_file(file_path):
    """校验Parquet文件完整性"""
    try:
        pf = pq.ParquetFile(file_path)
        if not pf.metadata:
            return False
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
    极速复制：仅复制内容，忽略元数据，无MD5校验。
    适用于中间过程文件（Phase 2 shards）。
    """
    try:
        # 如果目标文件存在，尝试删除（避免覆盖权限问题）
        if os.path.exists(nas_file):
            try:
                os.remove(nas_file)
            except OSError:
                pass # 如果删除失败，copyfile会覆盖，或者报错也无所谓

        # 核心优化：使用内核级拷贝，忽略权限元数据
        shutil.copyfile(local_file, nas_file)

        # 仅做大小校验（极快，不读内容）
        local_size = os.path.getsize(local_file)
        nas_size = os.path.getsize(nas_file)
        
        if local_size != nas_size:
            # 大小不一致，极为罕见，视为失败
            if os.path.exists(nas_file): os.remove(nas_file)
            raise IOError(f"Size mismatch: Local {local_size} vs NAS {nas_size}")

        # 可选：仅校验 Parquet 文件头尾完整性（比 MD5 快非常多）
        # 如果你对数据完整性极度敏感，可以保留这个，否则建议注释掉
        # if not validate_parquet_file(nas_file):
        #     raise Exception("Parquet validation failed")

        # 复制成功后删除本地文件（完成 Move 操作）
        os.remove(local_file)
        return True

    except Exception as e:
        logger.error(f"Fast upload failed {local_file} -> {nas_file}: {e}")
        return False

# def upload_to_nas(local_file, nas_file):
#     """纯手动字节流复制文件内容"""
#     try:
#         if os.path.exists(nas_file):
#             try:
#                 os.remove(nas_file)
#             except Exception as e:
#                 logger.warning(f"跳过删除NAS旧文件: {nas_file}, 错误: {e}")

#         chunk_size = 1024 * 1024 * 10
#         total_size = os.path.getsize(local_file)
#         copied_size = 0

#         with open(local_file, 'rb') as f_in, open(nas_file, 'wb') as f_out:
#             while True:
#                 chunk = f_in.read(chunk_size)
#                 if not chunk:
#                     break
#                 f_out.write(chunk)
#                 copied_size += len(chunk)

#         if copied_size != total_size:
#             if os.path.exists(nas_file):
#                 os.remove(nas_file)
#             raise Exception(f"文件内容复制不完整：本地{total_size}字节 vs NAS{copied_size}字节")

#         logger.info(f"复制完成：{local_file} → {nas_file}（{total_size / 1024 / 1024:.2f}MB）")

#         if PROCESSING_CONFIG["nas_upload_md5_check"]:
#             local_md5 = get_file_md5(local_file)
#             nas_md5 = get_file_md5(nas_file)
#             if local_md5 != nas_md5:
#                 os.remove(nas_file)
#                 raise Exception(f"MD5不一致：本地{local_md5} vs NAS{nas_md5}")

#         if not validate_parquet_file(nas_file):
#             os.remove(nas_file)
#             raise Exception("NAS文件Parquet格式损坏")

#         logger.info(f"上传成功：{nas_file}")
#         return True

#     except Exception as e:
#         logger.error(f"上传NAS失败 {local_file} → {nas_file}: {e}")
#         if os.path.exists(nas_file):
#             try:
#                 os.remove(nas_file)
#             except:
#                 pass
#         return False


# ======================== 1. STATS EXTRACTION (MEMORY SAFE) ========================

def process_shard_stats_only(parquet_file):
    """
    [MAPPER] 增强版：额外收集 unique active days 用于匹配协变量。
    每个shard内部按user聚合，返回聚合结果列表。
    """
    try:
        pf = pq.ParquetFile(parquet_file, pre_buffer=True)

        cols = ["user_id_hash", "group", "t0_date", "post_timestamp",
                "clean_text", "subreddit", "is_retrospective"]
        has_dtype = "data_type" in pf.schema.names
        if has_dtype:
            cols.append("data_type")

        results = {}
        is_rs_file = os.path.basename(parquet_file).startswith("RS")

        for batch in pf.iter_batches(batch_size=50000, columns=cols):
            df = batch.to_pandas()
            if df.empty:
                continue

            txt = df["clean_text"].fillna("")

            n_ch = txt.str.len().astype(np.int32)
            n_w = txt.str.count(" ").astype(np.int32) + 1
            n_sn = txt.str.count(r'[.!?]').astype(np.int32) + 1

            def get_density(t):
                if not isinstance(t, str) or len(t) < 50:
                    return 0.0
                words = t.split()
                if not words:
                    return 0.0
                return len(set(words)) / len(words)

            local_densities = txt.apply(get_density).values
            weighted_vd = local_densities * n_w

            timestamps = df["post_timestamp"].values
            hours = (timestamps % 86400) / 3600.0
            rads = hours * (2 * pi / 24)
            sin_t = np.sin(rads)
            cos_t = np.cos(rads)

            # 计算每条帖子的"天"标识（用于unique active days）
            day_ids = (timestamps // 86400).astype(np.int32)

            groups = df.groupby("user_id_hash")

            for user, idx in groups.indices.items():
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
                        "sub": Counter(),
                        "active_days": set()
                    }

                entry = results[user]
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

                ts_chunk = timestamps[idx]
                min_ts = ts_chunk.min()
                max_ts = ts_chunk.max()
                if min_ts < entry["min_t"]:
                    entry["min_t"] = min_ts
                if max_ts > entry["max_t"]:
                    entry["max_t"] = max_ts

                entry["sub"].update(df["subreddit"].iloc[idx])
                entry["active_days"].update(day_ids[idx].tolist())

            del df, txt, n_ch, n_w, n_sn, rads, day_ids

        return list(results.values())

    except Exception as e:
        logger.warning(f"Shard error {parquet_file}: {e}")
        return []


def aggregate_batch_and_flush(batch_files, batch_id, temp_dir):
    """
    [REDUCER 1] 处理一批文件，在RAM中聚合后写入磁盘。
    正确处理 active_days set 的跨shard合并。
    """
    batch_results = []
    with Pool(PROCESSING_CONFIG["cpu_cores"], initializer=init_post_worker) as pool:
        itr = pool.imap_unordered(process_shard_stats_only, batch_files)
        for res in itr:
            batch_results.extend(res)

    if not batch_results:
        return

    user_agg = {}

    for r in batch_results:
        u = r["u"]
        if u not in user_agg:
            user_agg[u] = {
                "u": u, "g": r["g"], "t0": r["t0"], "ret": r["ret"],
                "n_p": 0, "n_s": 0, "n_ch": 0, "n_w": 0, "n_sn": 0,
                "s_sin": 0.0, "s_cos": 0.0, "w_vd": 0.0,
                "min_t": float('inf'), "max_t": 0,
                "sub": Counter(),
                "active_days": set()
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
        if r["min_t"] < entry["min_t"]:
            entry["min_t"] = r["min_t"]
        if r["max_t"] > entry["max_t"]:
            entry["max_t"] = r["max_t"]
        entry["sub"].update(r["sub"])
        entry["active_days"].update(r.get("active_days", set()))

    flat_rows = []
    for u, v in user_agg.items():
        top_subs = dict(v["sub"].most_common(200))
        v["sub_json"] = json.dumps(top_subs)
        v["n_unique_days"] = len(v["active_days"])
        del v["sub"]
        del v["active_days"]
        flat_rows.append(v)

    out_path = os.path.join(temp_dir, f"temp_stats_{batch_id}.parquet")
    pd.DataFrame(flat_rows).to_parquet(out_path, index=False, compression='snappy')

    del batch_results, user_agg, flat_rows
    gc.collect()


def final_merge_stats(temp_dir):
    """
    [REDUCER 2] 合并中间统计文件。
    
    关键修复：n_unique_days 跨批次用 sum（因为数据按月分文件，
    不同batch的活跃天几乎不重叠）。
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
                    "sub": Counter(),
                    "n_unique_days": 0
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

            if row["min_t"] < entry["min_t"]:
                entry["min_t"] = row["min_t"]
            if row["max_t"] > entry["max_t"]:
                entry["max_t"] = row["max_t"]

            if row.get("sub_json"):
                entry["sub"].update(json.loads(row["sub_json"]))

            # 修复：跨批次用 sum（数据按月分文件，不同batch天ID不重叠）
            # 但需要 cap 到 tenure_days，避免极端情况下超过100%
            entry["n_unique_days"] += row.get("n_unique_days", 0)

    # Calculate Final Features
    final_features = []
    logger.info("Calculating derived features...")

    for u, v in global_agg.items():
        tenure = max(0.1, (v["max_t"] - v["min_t"]) / 86400)

        # 1. Readability
        avg_cpw = v["n_ch"] / max(1, v["n_w"])
        avg_wps = v["n_w"] / max(1, v["n_sn"])
        ari = (4.71 * avg_cpw) + (0.5 * avg_wps) - 21.43

        # 2. Vocab Density
        vocab_density = v["w_vd"] / max(1, v["n_w"])

        # 3. Circadian
        n = max(1, v["n_p"])
        avg_sin = v["s_sin"] / n
        avg_cos = v["s_cos"] / n
        reg = np.sqrt(avg_sin ** 2 + avg_cos ** 2)
        angle = atan2(avg_sin, avg_cos)
        if angle < 0:
            angle += 2 * pi
        hour = angle * (24 / (2 * pi))

        # 4. Entropy
        sub_ent = entropy(list(v["sub"].values()), base=2)

        # 5. Bot Proxy
        avg_word_len = v["n_ch"] / max(1, v["n_w"])

        # 6. 活跃天数比例（cap到1.0）
        active_days_ratio = min(1.0, v["n_unique_days"] / max(1, tenure))

        final_features.append({
            "user_id_hash": u,
            "group": v["g"],
            "real_t0_date": v["t0"],
            "is_retrospective": v["ret"],

            # 原有特征
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
            "last_post_ts": v["max_t"],

            # 新增特征（用于增强PSM）
            "log_total_posts": round(np.log1p(v["n_p"]), 3),
            "active_days_ratio": round(active_days_ratio, 3),
            "circadian_mean_hour_sin": round(np.sin(angle), 4),
            "circadian_mean_hour_cos": round(np.cos(angle), 4),
            "n_unique_days": v["n_unique_days"],
        })

    return pd.DataFrame(final_features)


def extract_features_disk_backed(shard_dir):
    """Orchestrates the Batched Map-Reduce."""
    temp_dir = os.path.join(WORK_DIR, "temp_stats_agg")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(shard_dir, "*.parquet")))
    if not files:
        raise ValueError("No shards found.")

    BATCH_SIZE = 24
    total_batches = (len(files) + BATCH_SIZE - 1) // BATCH_SIZE

    logger.info(f"Extracting features from {len(files)} shards in {total_batches} batches...")

    for i in range(total_batches):
        batch = files[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        logger.info(f"Processing Batch {i + 1}/{total_batches}...")

        if not os.path.exists(os.path.join(temp_dir, f"temp_stats_{i}.parquet")):
            aggregate_batch_and_flush(batch, i, temp_dir)

        gc.collect()

    df = final_merge_stats(temp_dir)
    return df


# ======================== 2. BOT FILTERING ========================

def filter_bots_scientific(feature_df):
    """增强版Bot过滤：增加最小帖子数要求"""
    logger.info("Step 2: Scientific Bot Filtering (Enhanced)...")
    df = feature_df.copy().fillna(0)

    n_before = len(df)

    mask_speed = df["posts_per_day"] <= BOT_THRESHOLDS["max_posts_per_day"]
    mask_quality = df["avg_word_len"] >= 2.5
    mask_diversity = df["vocab_density"] >= BOT_THRESHOLDS["min_text_diversity"]

    mask_burner = ~(
        (df["tenure_days"] < BOT_THRESHOLDS["min_account_age_days"]) &
        (df["posts_per_day"] > BOT_THRESHOLDS["burner_bot_posts_per_day"])
    )

    mask_min_posts = df["total_posts"] >= PSM_CONFIG.get("min_total_posts", 5)

    clean_df = df[
        mask_speed & mask_quality & mask_diversity & mask_burner & mask_min_posts
    ].copy()

    n_after = len(clean_df)
    n_removed = n_before - n_after

    n_pat_before = (df["group"] == "patient").sum()
    n_pat_after = (clean_df["group"] == "patient").sum()
    n_ctrl_before = (df["group"] != "patient").sum()
    n_ctrl_after = (clean_df["group"] != "patient").sum()

    logger.info(f"  过滤前: {n_before} (患者={n_pat_before}, 对照={n_ctrl_before})")
    logger.info(f"  过滤后: {n_after} (患者={n_pat_after}, 对照={n_ctrl_after})")
    logger.info(f"  移除: {n_removed} ({n_removed / max(1, n_before) * 100:.1f}%)")
    logger.info(f"  各条件移除数:")
    logger.info(f"    速度过滤: {(~mask_speed).sum()}")
    logger.info(f"    质量过滤: {(~mask_quality).sum()}")
    logger.info(f"    多样性过滤: {(~mask_diversity).sum()}")
    logger.info(f"    小号过滤: {(~mask_burner).sum()}")
    logger.info(f"    最小帖子数: {(~mask_min_posts).sum()}")

    return clean_df.reset_index(drop=True)


# ======================== 3. SHADOW T0 ========================

def assign_shadow_t0(clean_df):
    """
    改进版 Shadow T0 分配：
    修复：用过滤后的患者T0分布来抽样对照的shadow T0
    """
    logger.info("Step 3: Assigning Shadow T0 (Enhanced)...")

    pats = clean_df[clean_df["group"] == "patient"].copy()
    controls = clean_df[clean_df["group"] == "control_candidate"].copy()

    if pats.empty:
        return clean_df

    # --- 患者端处理 ---
    pat_t0_ts = pd.to_datetime(pats["real_t0_date"]).astype(np.int64) // 10 ** 9
    pats["shadow_t0_ts"] = pat_t0_ts.values

    min_pre = PSM_CONFIG["min_pre_exposure_days"] * 86400
    min_post = PSM_CONFIG["min_post_exposure_days"] * 86400

    pat_valid = (
        (pats["shadow_t0_ts"] > (pats["first_post_ts"] + min_pre)) &
        (pats["shadow_t0_ts"] < (pats["last_post_ts"] - min_post)) &
        (pats["total_posts"] >= PSM_CONFIG["min_total_posts"])
    )
    pats = pats[pat_valid].copy()
    logger.info(f"  患者经T0有效性筛选: {pat_valid.sum()}/{len(pat_valid)}")

        # --- 修复：用过滤后的患者T0分布来抽样 ---
    valid_pat_t0_values = pats["shadow_t0_ts"].values
    if len(valid_pat_t0_values) == 0:
        logger.warning("  没有有效患者，返回空DataFrame")
        return pd.DataFrame()

    # --- 对照端处理 ---
    MAX_TRIES = 5
    controls["shadow_t0_ts"] = np.nan

    for attempt in range(MAX_TRIES):
        needs_t0 = controls["shadow_t0_ts"].isna()
        if not needs_t0.any():
            break

        n_need = needs_t0.sum()
        # 修复：从过滤后的患者T0分布中抽样（而非过滤前）
        sampled_t0 = np.random.choice(valid_pat_t0_values, size=n_need)
        controls.loc[needs_t0, "shadow_t0_ts"] = sampled_t0

        # 校验有效性
        invalid = (
            (controls["shadow_t0_ts"] <= (controls["first_post_ts"] + min_pre)) |
            (controls["shadow_t0_ts"] >= (controls["last_post_ts"] - min_post))
        )
        controls.loc[invalid & needs_t0, "shadow_t0_ts"] = np.nan

        n_valid_this_round = needs_t0.sum() - controls["shadow_t0_ts"].isna().sum()
        logger.info(f"    T0分配尝试 {attempt + 1}/{MAX_TRIES}: 新增有效 {n_valid_this_round}")

    # 最终筛选
    controls = controls.dropna(subset=["shadow_t0_ts"])
    controls = controls[controls["total_posts"] >= PSM_CONFIG["min_total_posts"]]
    controls["shadow_t0_ts"] = controls["shadow_t0_ts"].astype(np.int64)

    logger.info(f"  有效对照: {len(controls)}")
    logger.info(f"  有效患者: {len(pats)}")
    logger.info(f"  患者:对照 = 1:{len(controls) / max(1, len(pats)):.1f}")

    return pd.concat([pats, controls], ignore_index=True)


# ======================== 4. PSM MATCHING ========================

def calculate_smd(treated, control, variables):
    """计算标准化均值差（Standardized Mean Difference）"""
    results = {}
    for var in variables:
        t_mean = treated[var].mean()
        c_mean = control[var].mean()
        t_var = treated[var].var()
        c_var = control[var].var()
        pooled_sd = np.sqrt((t_var + c_var) / 2)
        if pooled_sd < 1e-10:
            results[var] = 0.0
        else:
            results[var] = (t_mean - c_mean) / pooled_sd
    return results


def calculate_iptw_weights(df, ps_features, treatment_col="is_patient"):
    """
    计算IPTW权重（用于三角验证和双重稳健分析）
    返回：df 增加 iptw_weight 列, ess
    """
    df = df.copy()
    X = df[ps_features].values
    y = df[treatment_col].astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=2000, C=1.0, penalty='l2')
    lr.fit(X_scaled, y)
    ps = lr.predict_proba(X_scaled)[:, 1]

    # Trim极端PS（stabilize weights）
    ps = np.clip(ps, 0.01, 0.99)

    # ATT权重：treated gets 1, control gets ps/(1-ps)
    df["ps_score"] = ps
    df["iptw_att_weight"] = np.where(
        y == 1,
        1.0,
        ps / (1 - ps)
    )

    # 归一化权重（使对照组权重之和 = 对照组人数）
    ctrl_mask = y == 0
    if ctrl_mask.sum() > 0:
        w_sum = df.loc[ctrl_mask, "iptw_att_weight"].sum()
        df.loc[ctrl_mask, "iptw_att_weight"] *= ctrl_mask.sum() / w_sum

    # ESS计算
    ctrl_weights = df.loc[ctrl_mask, "iptw_att_weight"].values
    ess = (ctrl_weights.sum() ** 2) / (ctrl_weights ** 2).sum() if len(ctrl_weights) > 0 else 0

    c_stat = lr.score(X_scaled, y)

    logger.info(f"  IPTW: C-statistic={c_stat:.4f}, ESS={ess:.0f}/{ctrl_mask.sum()}, "
                f"Weight range=[{ctrl_weights.min():.4f}, {ctrl_weights.max():.4f}]")

    return df, ess


def run_psm_matching(df):
    """
    重构版匹配函数：
    1. 特征工程增强
    2. Tenure分位数精确分层
    3. 层内PS计算+Caliper匹配
    4. 层内+全局平衡性校验
    5. 并行IPTW计算（用于三角验证）
    
    修复：ps_logit 列的访问时序问题
    """
    logger.info("=" * 70)
    logger.info("Step 4: Enhanced Stratified PSM Matching")
    logger.info("=" * 70)

    df = df.copy()

    # ========================
    # 4.1 特征工程
    # ========================
    df["is_patient"] = (df["group"] == "patient").astype(int)
    df["log_posts_per_day"] = np.log1p(df["posts_per_day"])
    df["log_tenure"] = np.log1p(df["tenure_days"])

    # log_total_posts 可能已在特征提取阶段计算
    if "log_total_posts" not in df.columns:
        df["log_total_posts"] = np.log1p(df["total_posts"])

    # 昼夜节律sin/cos
    if "circadian_mean_hour_sin" not in df.columns:
        rads = df["circadian_mean_hour"] * (2 * pi / 24)
        df["circadian_mean_hour_sin"] = np.sin(rads)
        df["circadian_mean_hour_cos"] = np.cos(rads)

    # 活跃天数比例
    if "active_days_ratio" not in df.columns:
        df["active_days_ratio"] = np.minimum(
            1.0, df["total_posts"] / np.maximum(1, df["tenure_days"])
        )

    # PS特征集
    ps_features = list(PSM_CONFIG["ps_features"])  # 复制，避免修改原配置

    # 检查所有特征存在
    missing_features = [f for f in ps_features if f not in df.columns]
    if missing_features:
        logger.warning(f"  缺失特征（将从PS模型中移除）: {missing_features}")
        ps_features = [f for f in ps_features if f in df.columns]

    # 移除包含NaN的行
    n_before_na = len(df)
    df = df.dropna(subset=ps_features).copy()
    n_after_na = len(df)
    if n_before_na != n_after_na:
        logger.info(f"  移除含NaN行: {n_before_na - n_after_na}")

    logger.info(f"  PS特征 ({len(ps_features)}): {ps_features}")
    logger.info(f"  数据量: 患者={df['is_patient'].sum()}, "
                f"对照={(df['is_patient'] == 0).sum()}")

    # ========================
    # 4.2 Tenure精确分层
    # ========================
    n_q = PSM_CONFIG["n_tenure_quantiles"]

    # 以患者的分布为基准计算分位数边界
    patient_tenure = df.loc[df["is_patient"] == 1, "tenure_days"]
    quantile_edges = [0] + [patient_tenure.quantile(q) for q in np.linspace(1 / n_q, 1, n_q)]
    quantile_edges[-1] = df["tenure_days"].max() + 1

    # 去重边界
    quantile_edges = sorted(set(quantile_edges))
    if len(quantile_edges) < 2:
        quantile_edges = [0, df["tenure_days"].max() + 1]

    df["tenure_quartile"] = pd.cut(
        df["tenure_days"],
        bins=quantile_edges,
        labels=False,
        include_lowest=True
    )

    logger.info(f"  Tenure分层边界: {[round(e, 1) for e in quantile_edges]}")
    for q in sorted(df["tenure_quartile"].dropna().unique()):
        q_data = df[df["tenure_quartile"] == q]
        n_pat = (q_data["is_patient"] == 1).sum()
        n_ctrl = (q_data["is_patient"] == 0).sum()
        logger.info(f"    Q{int(q)}: 患者={n_pat}, 对照={n_ctrl}, "
                     f"比例=1:{n_ctrl / max(1, n_pat):.1f}")

    # ========================
    # 4.3 层内PS计算 + Caliper匹配
    # ========================
    matched_dfs = []
    stratum_reports = []

    # 平衡性检查变量（原始尺度）
    balance_vars = [
        "posts_per_day", "tenure_days", "readability_grade",
        "subreddit_entropy", "submission_ratio", "vocab_density",
        "circadian_regularity"
    ]
    balance_vars = [v for v in balance_vars if v in df.columns]

    for q_val in sorted(df["tenure_quartile"].dropna().unique()):
        stratum = df[df["tenure_quartile"] == q_val].copy()

        logger.info(f"\n  --- Stratum Q{int(q_val)} ---")

        n_pats_in_stratum = (stratum["is_patient"] == 1).sum()
        n_ctrls_in_stratum = (stratum["is_patient"] == 0).sum()
        logger.info(f"  患者: {n_pats_in_stratum}, 对照候选: {n_ctrls_in_stratum}")

        if n_pats_in_stratum == 0:
            logger.warning(f"  跳过：无患者")
            continue

        if n_ctrls_in_stratum < n_pats_in_stratum:
            logger.warning(f"  跳过PS匹配（对照不足），保留患者用于IPTW")
            matched_dfs.append(stratum[stratum["is_patient"] == 1])
            continue

        # 层内标准化 + PS计算
        scaler = StandardScaler()
        X_s = scaler.fit_transform(stratum[ps_features])
        y_s = stratum["is_patient"].values

        lr = LogisticRegression(max_iter=2000, C=1.0, penalty='l2', n_jobs=-1)
        lr.fit(X_s, y_s)

        # 修复：ps_logit 加到 stratum 上，然后再分切片
        stratum["ps_logit"] = lr.decision_function(X_s)

        # 修复：在 ps_logit 赋值之后才切片
        pats_s = stratum[stratum["is_patient"] == 1]
        ctrls_s = stratum[stratum["is_patient"] == 0]

        # 层内Caliper
        ps_std = stratum["ps_logit"].std()
        caliper = PSM_CONFIG["caliper_sd_multiplier"] * ps_std
        logger.info(f"  PS logit std={ps_std:.4f}, caliper={caliper:.4f}")

        # 共同支撑域检查
        ps_min_pat = pats_s["ps_logit"].min()
        ps_max_pat = pats_s["ps_logit"].max()
        ps_min_ctrl = ctrls_s["ps_logit"].min()
        ps_max_ctrl = ctrls_s["ps_logit"].max()

        overlap_min = max(ps_min_pat, ps_min_ctrl)
        overlap_max = min(ps_max_pat, ps_max_ctrl)

        logger.info(f"  PS范围: 患者=[{ps_min_pat:.4f}, {ps_max_pat:.4f}], "
                     f"对照=[{ps_min_ctrl:.4f}, {ps_max_ctrl:.4f}]")
        logger.info(f"  共同支撑域: [{overlap_min:.4f}, {overlap_max:.4f}]")

        if overlap_max <= overlap_min:
            logger.warning(f"  共同支撑域为空，跳过此层PS匹配，保留患者")
            matched_dfs.append(pats_s)
            continue

        # 截断到共同支撑域（加一个caliper的缓冲）
        in_support = (
            (stratum["ps_logit"] >= overlap_min - caliper) &
            (stratum["ps_logit"] <= overlap_max + caliper)
        )
        stratum_trimmed = stratum[in_support].copy()
        pats_t = stratum_trimmed[stratum_trimmed["is_patient"] == 1]
        ctrls_t = stratum_trimmed[stratum_trimmed["is_patient"] == 0]

        n_pat_trimmed = len(pats_t)
        n_ctrl_trimmed = len(ctrls_t)
        logger.info(f"  共同支撑域后: 患者={n_pat_trimmed}, 对照={n_ctrl_trimmed}")

        if n_ctrl_trimmed < n_pat_trimmed:
            logger.warning(f"  共同支撑域内对照不足，保留患者")
            matched_dfs.append(pats_t)
            continue

        # 最近邻匹配（层内，基于ps_logit的1D距离）
        target_ratio = PSM_CONFIG["ratio"]
        n_neighbors_search = min(target_ratio * 2, n_ctrl_trimmed)

        nbrs = NearestNeighbors(
            n_neighbors=n_neighbors_search,
            algorithm='ball_tree',
            metric='euclidean'
        )
        nbrs.fit(ctrls_t[["ps_logit"]].values)
        dists, idxs = nbrs.kneighbors(pats_t[["ps_logit"]].values)

        # 收集匹配的对照索引
        used_ctrls = set()
        matched_ctrl_indices = []
        match_counts = []

        for i in range(len(pats_t)):
            cnt = 0
            for j in range(len(dists[i])):
                d = dists[i][j]
                idx = idxs[i][j]
                if d > caliper:
                    continue
                real_idx = ctrls_t.index[idx]
                if not PSM_CONFIG["replace"] and real_idx in used_ctrls:
                    continue
                matched_ctrl_indices.append(real_idx)
                used_ctrls.add(real_idx)
                cnt += 1
                if cnt >= target_ratio:
                    break
            match_counts.append(cnt)

        # 组装层内匹配结果
        layer_df = pd.concat([
            pats_t,
            stratum_trimmed.loc[matched_ctrl_indices]
        ], ignore_index=True)

        avg_ratio = np.mean(match_counts) if match_counts else 0
        zero_match = sum(1 for c in match_counts if c == 0)

        logger.info(f"  匹配结果: 平均比例=1:{avg_ratio:.2f}, 零匹配患者={zero_match}")

        # 层内平衡性校验
        layer_pats = layer_df[layer_df["is_patient"] == 1]
        layer_ctrls = layer_df[layer_df["is_patient"] == 0]

        if len(layer_ctrls) > 0:
            smd_dict = calculate_smd(layer_pats, layer_ctrls, balance_vars)

            layer_max_smd = max(abs(v) for v in smd_dict.values())
            layer_mean_smd = np.mean([abs(v) for v in smd_dict.values()])

            stratum_report = {
                "stratum": f"Q{int(q_val)}",
                "n_patient": len(layer_pats),
                "n_control": len(layer_ctrls),
                "avg_ratio": round(avg_ratio, 2),
                "max_smd": round(layer_max_smd, 4),
                "mean_smd": round(layer_mean_smd, 4),
                "smd_details": {k: round(v, 4) for k, v in smd_dict.items()},
                "pass": layer_max_smd < PSM_CONFIG["max_smd"]
            }
            stratum_reports.append(stratum_report)

            for var, smd in smd_dict.items():
                status = "✓" if abs(smd) < 0.1 else ("~" if abs(smd) < 0.2 else "✗")
                logger.info(f"    {var:25s}: SMD = {smd:.4f} {status}")
        else:
            logger.warning(f"  层内无匹配对照")

        matched_dfs.append(layer_df)

    # ========================
    # 4.4 合并所有层 + 全局平衡性校验
    # ========================
    if not matched_dfs:
        raise ValueError("匹配完全失败：没有任何层产出匹配结果")

    matched_df = pd.concat(matched_dfs, ignore_index=True)

    # 去重
    matched_df = matched_df.drop_duplicates(subset=["user_id_hash"], keep="first")

    global_pats = matched_df[matched_df["is_patient"] == 1]
    global_ctrls = matched_df[matched_df["is_patient"] == 0]

    logger.info(f"\n{'=' * 70}")
    logger.info(f"全局匹配结果:")
    logger.info(f"  患者: {len(global_pats)}")
    logger.info(f"  对照: {len(global_ctrls)}")
    logger.info(f"  比例: 1:{len(global_ctrls) / max(1, len(global_pats)):.2f}")

    global_smd = calculate_smd(global_pats, global_ctrls, balance_vars)

    logger.info(f"\n  全局平衡性:")
    all_pass = True
    for var, smd in global_smd.items():
        status = "✓" if abs(smd) < 0.1 else ("~" if abs(smd) < 0.2 else "✗")
        logger.info(f"    {var:25s}: SMD = {smd:.4f} {status}")
        if abs(smd) >= PSM_CONFIG["max_smd"]:
            all_pass = False

    mean_smd = np.mean([abs(v) for v in global_smd.values()])
    max_smd = max(abs(v) for v in global_smd.values())
    logger.info(f"  平均SMD: {mean_smd:.4f}")
    logger.info(f"  最大SMD: {max_smd:.4f}")
    logger.info(f"  全部通过(<{PSM_CONFIG['max_smd']}): {'✓ YES' if all_pass else '✗ NO'}")

    # ========================
    # 4.5 IPTW权重计算（三角验证用）
    # ========================
    iptw_mean_smd = None
    iptw_max_smd = None
    ess = None
    dr_adjust_vars = []

    if PSM_CONFIG.get("run_iptw", False):
        logger.info(f"\n  计算IPTW权重（用于三角验证）...")
        matched_df, ess = calculate_iptw_weights(
            matched_df, ps_features, treatment_col="is_patient"
        )

        # IPTW加权后的平衡性
        iptw_smd = {}
        for var in balance_vars:
            t_vals = matched_df.loc[matched_df["is_patient"] == 1, var]
            c_vals = matched_df.loc[matched_df["is_patient"] == 0, var]
            c_weights = matched_df.loc[matched_df["is_patient"] == 0, "iptw_att_weight"]

            t_mean = t_vals.mean()
            c_mean = np.average(c_vals, weights=c_weights)
            t_var = t_vals.var()
            c_var = np.average((c_vals - c_mean) ** 2, weights=c_weights)
            pooled_sd = np.sqrt((t_var + c_var) / 2)
            iptw_smd[var] = (t_mean - c_mean) / pooled_sd if pooled_sd > 1e-10 else 0.0

        logger.info(f"\n  IPTW加权后平衡性:")
        for var, smd in iptw_smd.items():
            status = "✓" if abs(smd) < 0.1 else ("~" if abs(smd) < 0.2 else "✗")
            logger.info(f"    {var:25s}: SMD = {smd:.4f} {status}")

        iptw_mean_smd = np.mean([abs(v) for v in iptw_smd.values()])
        iptw_max_smd = max(abs(v) for v in iptw_smd.values())
        logger.info(f"  IPTW平均SMD: {iptw_mean_smd:.4f}")
        logger.info(f"  IPTW最大SMD: {iptw_max_smd:.4f}")
        logger.info(f"  IPTW ESS: {ess:.0f}/{len(global_ctrls)}")

        dr_adjust_vars = [var for var, smd in iptw_smd.items() if abs(smd) >= 0.1]
        if dr_adjust_vars:
            logger.info(f"  双重稳健需额外调整的变量: {dr_adjust_vars}")
        else:
            logger.info(f"  IPTW已完全平衡，无需额外回归调整 ✓")

    # ========================
    # 4.6 Entropy Balancing（第三种权重方案）
    # ========================
    if PSM_CONFIG.get("run_entropy_balancing", False):
        logger.info(f"\n  计算Entropy Balancing权重...")
        try:
            matched_df = entropy_balancing(matched_df, ps_features, balance_vars)
        except Exception as e:
            logger.warning(f"  Entropy Balancing失败（非致命）: {e}")

    # ========================
    # 4.7 保存诊断报告
    # ========================
    report = {
        "timestamp": datetime.now().isoformat(),
        "global": {
            "n_patient": int(len(global_pats)),
            "n_control": int(len(global_ctrls)),
            "ratio": round(len(global_ctrls) / max(1, len(global_pats)), 2),
            "mean_smd": round(mean_smd, 4),
            "max_smd": round(max_smd, 4),
            "all_pass": all_pass,
            "smd_details": {k: round(v, 4) for k, v in global_smd.items()}
        },
        "strata": stratum_reports,
        "ps_features_used": ps_features,
        "config": {k: v for k, v in PSM_CONFIG.items() if not callable(v)}
    }

    if PSM_CONFIG.get("run_iptw", False) and iptw_mean_smd is not None:
        report["iptw"] = {
            "mean_smd": round(iptw_mean_smd, 4),
            "max_smd": round(iptw_max_smd, 4),
            "ess": round(ess, 0),
            "dr_adjust_vars": dr_adjust_vars,
            "smd_details": {k: round(v, 4) for k, v in iptw_smd.items()}
        }

    report_path = os.path.join(WORK_DIR, "psm_matching_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"\n  匹配诊断报告已保存: {report_path}")

    # ========================
    # 4.8 质量门控
    # ========================
    if max_smd >= 0.2:
        worst_var = max(global_smd, key=lambda k: abs(global_smd[k]))
        logger.warning(
            f"  ⚠️ 严重警告：全局最大SMD={max_smd:.4f} >= 0.2，"
            f"变量={worst_var}。建议检查分层策略或扩展协变量集。"
        )
    elif not all_pass:
        logger.warning(
            f"  ⚠️ 警告：部分变量SMD>{PSM_CONFIG['max_smd']}，"
            f"建议在下游分析中使用IPTW或双重稳健策略。"
        )
    else:
        logger.info(f"  ✅ 匹配质量通过所有检查！")

    # 保存双重稳健配置
    dr_config = {
        "primary_analysis": "stratified_psm",
        "psm_balanced": all_pass,
        "iptw_weights_available": PSM_CONFIG.get("run_iptw", False),
        "residual_imbalance_vars": [
            var for var, smd in global_smd.items() if abs(smd) >= 0.1
        ],
        "recommended_strategy": (
            "unadjusted" if all_pass
            else "iptw_weighted_regression" if PSM_CONFIG.get("run_iptw", False)
            else "regression_adjusted"
        ),
        "regression_covariates": [
            var for var, smd in global_smd.items() if abs(smd) >= 0.05
        ]
    }

    dr_config_path = os.path.join(WORK_DIR, "doubly_robust_config.json")
    with open(dr_config_path, "w") as f:
        json.dump(dr_config, f, indent=2)
    logger.info(f"  双重稳健配置已保存: {dr_config_path}")

    return matched_df


def entropy_balancing(df, ps_features, balance_vars, max_iter=500, tol=1e-6):
    """
    Entropy Balancing：自动求解使对照组加权矩（均值）与患者组完全一致的权重。
    参考：Hainmueller (2012)
    """
    treated = df[df["is_patient"] == 1]
    control = df[df["is_patient"] == 0]

    target_means = treated[balance_vars].mean().values
    C = control[balance_vars].values
    n_c = len(control)

    lambda_init = np.zeros(len(balance_vars))

    def objective(lam):
        exp_term = np.exp(np.clip(C @ lam, -500, 500))  # 防溢出
        log_sum = np.log(exp_term.sum())
        return log_sum - target_means @ lam

    def gradient(lam):
        exp_term = np.exp(np.clip(C @ lam, -500, 500))
        weights = exp_term / exp_term.sum()
        weighted_means = C.T @ weights
        return weighted_means - target_means

    result = minimize(
        objective, lambda_init, jac=gradient,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol}
    )

    if not result.success:
        logger.warning(f"  Entropy Balancing未完全收敛: {result.message}")

    exp_term = np.exp(np.clip(C @ result.x, -500, 500))
    raw_weights = exp_term / exp_term.sum() * n_c

    df = df.copy()
    df["eb_weight"] = 1.0
    df.loc[control.index, "eb_weight"] = raw_weights

    # 验证平衡性
    logger.info(f"  Entropy Balancing加权后平衡性:")
    for i, var in enumerate(balance_vars):
        t_mean = target_means[i]
        c_mean = np.average(control[var].values, weights=raw_weights)
        t_var = treated[var].var()
        c_var = np.average((control[var].values - c_mean) ** 2, weights=raw_weights)
        pooled_sd = np.sqrt((t_var + c_var) / 2)
        smd = (t_mean - c_mean) / pooled_sd if pooled_sd > 1e-10 else 0.0
        status = "✓" if abs(smd) < 0.1 else ("~" if abs(smd) < 0.2 else "✗")
        logger.info(f"    {var:25s}: SMD = {smd:.4f} {status}")

    logger.info(f"  EB权重范围: [{raw_weights.min():.4f}, {raw_weights.max():.4f}]")
    logger.info(f"  EB权重均值: {raw_weights.mean():.4f}")
    ess_eb = (raw_weights.sum() ** 2) / (raw_weights ** 2).sum()
    logger.info(f"  EB ESS: {ess_eb:.0f}/{n_c}")

    return df


# ======================== 5. FINAL BUILD ========================

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
        if not validate_parquet_file(local_input_temp):
            raise Exception("本地拷贝的输入文件Parquet格式损坏")
        logger.info(f"输入文件本地拷贝并校验完成: {local_input_temp}")

        if os.path.exists(output_file) and validate_parquet_file(output_file):
            logger.info(f"NAS文件已存在且完整，跳过: {output_file}")
            return True

        pf = pq.ParquetFile(local_input_temp, pre_buffer=True)

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
                df["clean_text"] = df["clean_text"].fillna("").apply(
                    lambda x: clean_text_heavy(x, analyzer, anonymizer)
                )

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
            if not validate_parquet_file(local_output_temp):
                raise Exception("本地文件校验失败，可能损坏")
            logger.info(f"本地文件处理完成并校验通过: {local_output_temp}")

        # 5. 上传到NAS
        if os.path.exists(local_output_temp):
            if not upload_to_nas(local_output_temp, output_file):
                raise Exception("上传NAS失败")
        else:
            logger.info(f"无匹配数据，跳过: {input_file}")

        return True

    except Exception as e:
        logger.error(f"处理失败 {input_file}: {e}")
        for temp_file in [local_input_temp, local_output_temp]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
            except:
                pass
        return False

    finally:
        for temp_file in [local_input_temp, local_output_temp]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"清理临时文件失败 {temp_file}: {e}")


def build_final_dataset(matched_df, raw_dir, out_file):
    """
    增强版：将匹配权重（IPTW/EB）嵌入最终数据集元数据
    """
    logger.info("Step 5: Building Final Dataset...")
    init_local_temp()

    # 保存用户级元数据（轻量，供下游分析直接加载）
    weight_cols = ["shadow_t0_ts"]
    for col in ["iptw_att_weight", "eb_weight", "ps_score", "tenure_quartile"]:
        if col in matched_df.columns:
            weight_cols.append(col)

    meta_cols = ["user_id_hash", "group", "is_patient"] + weight_cols
    meta_cols = [c for c in meta_cols if c in matched_df.columns]
    user_meta = matched_df[meta_cols].copy()
    meta_path = out_file + '_user_meta.parquet' #out_file.replace(".parquet", "_user_meta.parquet")
    user_meta.to_parquet(meta_path, index=False)
    logger.info(f"  用户元数据已保存: {meta_path}")

    # 构建user_map（仅传T0给final build，权重在元数据中）
    user_map = matched_df.set_index("user_id_hash")["shadow_t0_ts"].to_dict()

    out_dir = out_file.replace(".parquet", "_dataset")
    os.makedirs(out_dir, exist_ok=True)
    files_finished = set(os.listdir(out_dir))

    files_all = sorted(glob.glob(os.path.join(raw_dir, "*.parquet")))
    files = [fi for fi in files_all if os.path.basename(fi) not in files_finished]
    work_args = [(f, user_map, os.path.join(out_dir, os.path.basename(f))) for f in files]

    safe_cores = max(4, PROCESSING_CONFIG["cpu_cores"] // 2)

    logger.info(f"  待处理文件: {len(files)}/{len(files_all)} (已完成: {len(files_finished)})")

    with Pool(safe_cores, initializer=init_final_build_worker) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_final_build_chunk, work_args),
            total=len(work_args)
        ))

    success_count = sum(results)
    fail_count = len(results) - success_count
    logger.info(f"处理完成 - 成功: {success_count}, 失败: {fail_count}")

    if fail_count > 0:
        logger.warning(f"有{fail_count}个文件处理失败，请检查日志")

    # 最终校验
    nas_files = glob.glob(os.path.join(out_dir, "*.parquet"))
    corrupt_nas_files = [f for f in nas_files if not validate_parquet_file(f)]
    if corrupt_nas_files:
        logger.error(f"NAS上发现{len(corrupt_nas_files)}个损坏文件: {corrupt_nas_files}")
    else:
        logger.info(f"Final Dataset Location: {out_dir} (所有文件校验通过)")


# ======================== MAIN ========================

if __name__ == "__main__":
    output_dir = PROCESSING_CONFIG["output_dir"]
    master_index_path = os.path.join(output_dir, PROCESSING_CONFIG["master_index_file"])
    final_cohort_path = os.path.join(output_dir, PROCESSING_CONFIG["final_cohort_file"])
    import argparse

    parser = argparse.ArgumentParser(
        description="Reddit Mental Health Cohort Post-Processing Pipeline"
    )
    parser.add_argument("--shard_dir", default="/path/to/reddit/emotion_disorder_1225/cohort_v1/temp_shards" ,help="Phase 2输出的shard目录")
    parser.add_argument("--output_file", default="/path/to/project/cohort_v1/intermediate_results/final_longitudinal_data.parquet", help="输出路径（.parquet）")
    parser.add_argument("--skip_final_build", action="store_true",
                        help="跳过最终数据集构建（仅做匹配）")
    parser.add_argument("--resume_from", type=str, default=None,
                        choices=["features", "bot_filter", "shadow_t0", "psm", "final_build"],
                        help="从指定步骤开始重新计算（该步骤及之后全部重跑）")
    args = parser.parse_args()

    # ========================
    # 中间结果路径定义
    # ========================
    intermediate_dir = os.path.join(output_dir, "intermediate_results")
    os.makedirs(intermediate_dir, exist_ok=True)

    features_path = os.path.join(intermediate_dir, "step_3.1_features_extracted.parquet")
    bot_filtered_path = os.path.join(intermediate_dir, "step_3.2_bot_filtered.parquet")
    shadow_t0_path = os.path.join(intermediate_dir, "step_3.3_shadow_t0_assigned.parquet")
    psm_matched_path = os.path.join(intermediate_dir, "step_3.4_psm_matched.parquet")

    # resume_from 逻辑：
    # None → 每步优先加载缓存，无缓存才重算
    # "features" → 从features开始全部重跑
    # "bot_filter" → features用缓存，bot_filter及之后重跑
    # 依此类推
    STEP_ORDER = ["features", "bot_filter", "shadow_t0", "psm", "final_build"]

    def should_recompute(step_name):
        """判断某步是否需要重新计算"""
        if args.resume_from is None:
            return False  # 默认：优先用缓存
        step_idx = STEP_ORDER.index(step_name)
        resume_idx = STEP_ORDER.index(args.resume_from)
        return step_idx >= resume_idx

    def check_and_load(path, step_name):
        """检查中间结果是否存在且有效"""
        if should_recompute(step_name):
            logger.info(f"  [重算] {step_name}: --resume_from={args.resume_from}")
            return None
        if os.path.exists(path) and validate_parquet_file(path):
            logger.info(f"  [跳过] {step_name}: 加载已有结果 {path}")
            return pd.read_parquet(path)
        return None

    def save_intermediate(df, path, step_name):
        """保存中间结果"""
        df.to_parquet(path, index=False, compression='snappy')
        logger.info(f"  [保存] {step_name}: {path} ({len(df)} rows)")

    # ========================
    # Pipeline Start
    # ========================
    logger.info("=" * 70)
    logger.info("PIPELINE START")
    logger.info(f"  Shard目录: {args.shard_dir}")
    logger.info(f"  输出文件: {args.output_file}")
    logger.info(f"  工作目录: {WORK_DIR}")
    logger.info(f"  resume_from: {args.resume_from}")
    logger.info("=" * 70)

    # ========================
    # Step 1: 特征提取
    # ========================
    features = check_and_load(features_path, "features")

    if features is None:
        logger.info("\n[Step 3.1] Extracting Features (Disk-Backed Map-Reduce)...")
        features = extract_features_disk_backed(args.shard_dir)
        save_intermediate(features, features_path, "Step 3.1 Features")
        logger.info(f"  特征提取完成: {len(features)} 用户")
        logger.info(f"    患者: {(features['group'] == 'patient').sum()}")
        logger.info(f"    对照候选: {(features['group'] != 'patient').sum()}")

    gc.collect()

    # ========================
    # Step 2: Bot过滤
    # ========================
    clean_meta = check_and_load(bot_filtered_path, "bot_filter")

    if clean_meta is None:
        logger.info("\n[Step 3.2] Scientific Bot Filtering...")
        clean_meta = filter_bots_scientific(features)
        save_intermediate(clean_meta, bot_filtered_path, "Step 3.2 Bot Filter")

    del features
    gc.collect()

    # ========================
    # Step 3: Shadow T0分配
    # ========================
    t0_assigned = check_and_load(shadow_t0_path, "shadow_t0")

    if t0_assigned is None:
        logger.info("\n[Step 3.3] Assigning Shadow T0 (Enhanced)...")
        t0_assigned = assign_shadow_t0(clean_meta)
        save_intermediate(t0_assigned, shadow_t0_path, "Step 3.3 Shadow T0")
        logger.info(f"  T0分配后: {len(t0_assigned)} 用户")
        logger.info(f"    患者: {(t0_assigned['group'] == 'patient').sum()}")
        logger.info(f"    有效对照: {(t0_assigned['group'] != 'patient').sum()}")

    del clean_meta
    gc.collect()

    # ========================
    # Step 4: PSM匹配
    # ========================
    matched_meta = check_and_load(psm_matched_path, "psm")

    if matched_meta is None:
        logger.info("\n[Step 3.4] Enhanced Stratified PSM Matching...")
        matched_meta = run_psm_matching(t0_assigned)
        save_intermediate(matched_meta, psm_matched_path, "Step 3.4 PSM")

    del t0_assigned
    gc.collect()

    # ========================
    # 保存匹配索引
    # ========================
    logger.info(f"\n保存匹配索引: {args.output_file}")
    matched_meta.to_parquet(args.output_file, index=False)

    n_pat = (matched_meta["is_patient"] == 1).sum() if "is_patient" in matched_meta.columns \
        else (matched_meta["group"] == "patient").sum()
    n_ctrl = len(matched_meta) - n_pat
    logger.info(f"  最终队列: 患者={n_pat}, 对照={n_ctrl}, 比例=1:{n_ctrl / max(1, n_pat):.2f}")

    if "iptw_att_weight" in matched_meta.columns:
        logger.info(f"  IPTW权重: ✓ 已计算")
    if "eb_weight" in matched_meta.columns:
        logger.info(f"  Entropy Balancing权重: ✓ 已计算")
    if "tenure_quartile" in matched_meta.columns:
        logger.info(f"  Tenure分层: ✓ (Q0-Q{matched_meta['tenure_quartile'].max():.0f})")

    # ========================
    # Step 5: 最终数据集构建
    # ========================
    if args.skip_final_build and not should_recompute("final_build"):
        logger.info("\n[Step 3.5] 跳过最终数据集构建 (--skip_final_build)")
    else:
        logger.info("\n[Step 3.5] Building Final Dataset...")
        build_final_dataset(matched_meta, args.shard_dir, args.output_file)

    # ========================
    # 完成摘要
    # ========================
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  匹配索引: {args.output_file}")
    logger.info(f"  中间结果: {intermediate_dir}/")
    logger.info(f"  匹配报告: {os.path.join(WORK_DIR, 'psm_matching_report.json')}")
    logger.info(f"  双重稳健配置: {os.path.join(WORK_DIR, 'doubly_robust_config.json')}")

    if not args.skip_final_build:
        out_dir = args.output_file.replace(".parquet", "_dataset")
        meta_path = args.output_file.replace(".parquet", "_user_meta.parquet")
        logger.info(f"  最终数据集: {out_dir}/")
        logger.info(f"  用户元数据: {meta_path}")

    logger.info("\n  下游分析建议:")

    report_path = os.path.join(WORK_DIR, "psm_matching_report.json")
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)

        if report["global"]["all_pass"]:
            logger.info("    ✓ PSM平衡性全部通过，可直接进行未加权分析")
            logger.info("    ✓ 建议同时报告IPTW加权结果作为敏感性分析")
        else:
            residual_vars = report["global"].get("smd_details", {})
            imbalanced = [k for k, v in residual_vars.items() if abs(v) >= 0.1]
            logger.info(f"    ⚠ PSM匹配后残余不平衡变量: {imbalanced}")
            logger.info(f"    → 主分析建议使用IPTW加权回归")
            logger.info(f"    → 将 {imbalanced} 作为回归协变量（双重稳健）")
            logger.info(f"    → 敏感性分析: 仅PSM / 仅IPTW / 仅回归调整 / Entropy Balancing")

    logger.info("\n  后续步骤:")
    logger.info("    1. 检查 psm_matching_report.json 中各层平衡性")
    logger.info("    2. 运行 psm_validation.py 生成平衡性可视化")
    logger.info("    3. 随机抽样1000条诊断文本进行人工标注验证")
    logger.info("    4. 启动 Module 1 (数字表型特征提取) 和 Module 2 (时序轨迹分析)")
    logger.info("=" * 70)


