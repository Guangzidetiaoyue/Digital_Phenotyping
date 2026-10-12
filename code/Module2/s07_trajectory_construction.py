"""
Step 07: Longitudinal Trajectory Construction (v1.2 - Patched)

Module 2 的数据基础层。回溯 Silver Data，按 t0 对齐的滑动窗口
重新提取帖子级特征，构建用户×窗口的纵向特征矩阵。

关键设计决策 (Updated):
  1. 90天窗口 / 30天步长 → 分辨率与稳定性的平衡
  2. 窗口特征提取复用 s01 的 extract_linguistic_features_post()
  3. p-factor severity score 使用 Module 1 Factor 1 载荷加权，方向由载荷符号决定
  4. 严格排除前驱期的基线定义 (t0 前 360-270 天)
  5. 对照组 shadow_t0 对齐后同样构建轨迹
  6. [FIX v1.2] 修复轨迹平滑索引错位、特征名匹配脆弱、社交退缩降级等致命/严重Bug

输出: user_trajectories.parquet (长表: 用户×窗口)
"""
import pyarrow.parquet as pq
import os,sys
import gc
import json
import logging
import warnings
import numpy as np
import pandas as pd
from math import pi
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from datetime import datetime, timedelta, timezone
from scipy.stats import norm
from config_module2 import (
    SILVER_DATA_DIR, FEATURES_DIR, TRAJECTORY_DIR,
    PSM_MATCHED_PATH, USER_META_PATH,
    NUM_CORES, BATCH_SIZE, RANDOM_SEED,
    WINDOW_SIZE_DAYS, WINDOW_STEP_DAYS,
    MIN_POSTS_PER_WINDOW, MIN_PRE_WINDOWS, MIN_POST_WINDOWS,
    MIN_WORDS_PER_WINDOW, MATTR_MAX_WORDS,
    NRC_LEXICON_PATH, USE_LIWC, LIWC_DICTIONARY_PATH,
    FACTOR_LOADINGS_PATH,
    TRAJECTORY_OUTPUT_FILE, TRAJECTORY_METADATA_FILE,
)

# ✅ 修复：添加 project/ 目录，Python 才能找到 Module1 包
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)  # 上一级 = project/
sys.path.insert(0, project_dir)

RAW_FEATURES_FILE="user_features_raw.parquet"

os.makedirs(TRAJECTORY_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(TRAJECTORY_DIR, "trajectory_construction.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)
np.random.seed(RANDOM_SEED)


# ======================== CONFIGURATION ========================

TRAJECTORY_PRE_DAYS = 720
TRAJECTORY_POST_DAYS = 720

WINDOW_OVERLAP = WINDOW_SIZE_DAYS - WINDOW_STEP_DAYS

PFACTOR_WEIGHT_SOURCE = "module1_factor1"

MIN_TOTAL_RELIABLE_WINDOWS = 4

# [REVIEW] 增加高质量窗口的阈值，用于后续分析分层
MIN_POSTS_HIGH_QUALITY = 10
MIN_WORDS_HIGH_QUALITY = 1000
# [FIX v1.4] MNAR 插值分位数配置
# 沉默可能代表极端退缩，P90 比 P75 更符合临床极端表现
MNAR_IMPUTATION_QUANTILE = 0.90  
# ================= 修复位置：模块级配置 =================
# 替换 TRAJECTORY_FEATURE_SUBSET 列表


from collections import Counter, defaultdict
from scipy.stats import entropy, kurtosis

# 新增的行为/时间/社交特征常量列表，用于后续 NaN 填充
BEHAVIORAL_FEATURES = [
    'circadian_mean_hour', 'circadian_concentration', 'night_activity_ratio', 
    'weekend_activity_ratio', 'social_jetlag_hours', 'sleep_disruption_index',
    'posting_regularity_cv', 'posting_burstiness', 'interval_kurtosis', 
    'interval_median_hours', 'interval_mean_hours', 'hourly_entropy', 
    'weekly_entropy', 'temporal_diversity_index', 'tenure_days', 'n_posts_total', 
    'n_active_days', 'active_day_ratio', 'posts_per_day', 'posts_per_active_day', 
    'max_gap_days', 'mean_gap_days', 'gap_regularity_cv', 'submission_ratio', 
    'n_submissions', 'n_comments', 'score_mean', 'score_median', 'score_std', 
    'score_max', 'score_min', 'score_range', 'neg_score_ratio', 
    'high_engagement_ratio', 'avg_comments_received', 'total_comments_received', 
    'discussion_initiated_ratio', 'top_level_comment_ratio', 'avg_comment_depth', 
    'controversy_rate', 'gilded_rate', 'subreddit_entropy', 'subreddit_herfindahl', 
    'n_unique_subreddits', 'mental_health_sub_ratio', 'support_sub_ratio', 
    'entertainment_sub_ratio', 'information_sub_ratio', 'social_sub_ratio', 
    'other_sub_ratio', 'posts_per_day_log', 'posts_per_active_day_log', 
    'interval_median_hours_log', 'max_gap_days_log', 'tenure_days_log', 
    'n_posts_total_log', 'score_mean_log_signed', 'ari_std', 'post_length_mean', 
    'post_length_std', 'post_length_cv', 'self_focus_variability', 
    'social_focus_variability', 'vader_compound_last_mean', 'url_usage_rate'
]
# 显式定义每个特征的聚合方式
FEATURE_AGGREGATION_MAP = {
    # 计数类 → 比率 (per 100 words)
    "anxiety_score": "rate",
    "anger_score": "rate",
    "sadness_score": "rate",
    "positive_score": "rate",
    "negative_score": "rate",
    "first_person_score": "rate",
    "second_person_score": "rate",
    "insight_score": "rate",
    "causal_score": "rate",
    "tentative_score": "rate",
    "certainty_score": "rate",
    "death_score": "rate",
    "past_focus_score": "rate",
    "present_focus_score": "rate",
    "future_focus_score": "rate",
    "question_ratio": "mean",
    "exclamation_ratio": "mean",
    "netspeak_count": "rate",
    "filler_count": "rate",
    
    # 连续值类 → 加权均值
    "vader_compound": "mean",
    "vader_pos": "mean",
    "vader_neg": "mean",
    "vader_compound_min": "mean",
    "self_focus_ratio": "mean",
    "cognitive_flexibility": "mean",
    "temporal_balance": "mean",
    "mattr_score": "mean",
    "ari_score": "mean",
    "negative_net_score": "mean",
    
    "impersonal_score": "rate",      # 补充：非人称代词
    "social_focus_ratio": "mean",    # 补充：社会聚焦指数
    "third_person_score": "rate",    # 补充：第三人称
    "discrepancy_score": "rate",     # 补充：差异认知
    "inhibition_score": "rate",      # 补充：抑制词
    "swear_score": "rate",           # 补充：脏话
    "assent_score": "rate",          # 补充：赞同词
    "nonfluency_score": "rate",      # 补充：不流利词
}

# 自动生成输出特征名列表（用于NaN填充和P-factor计算）
def get_output_feature_names():
    names = []
    for feat, agg in FEATURE_AGGREGATION_MAP.items():
        names.append(f"{feat}_{agg}")
    return names

TRAJECTORY_FEATURE_SUBSET = get_output_feature_names()


# ======================== P-FACTOR NAME BRIDGE ========================
# 桥接 s01 用户级聚合名 (Factor Loadings来源) 与 s07 窗口级聚合名 (轨迹特征)
USER_TO_TRAJECTORY_NAME_MAP = {
    # 情感类
    "anxiety_rate": "anxiety_score_rate",
    "anger_rate": "anger_score_rate",
    "sadness_rate": "sadness_score_rate",
    "positive_affect_rate": "positive_score_rate",
    "negative_affect_rate": "negative_score_rate",
    
    # 认知类
    "insight_rate": "insight_score_rate",
    "causal_thinking_rate": "causal_score_rate",
    "certainty_rate": "certainty_score_rate",
    "tentative_language_rate": "tentative_score_rate",
    "discrepancy_rate": "discrepancy_score_rate",
    "inhibition_rate": "inhibition_score_rate",
    
    # 代词类
    "first_person_rate": "first_person_score_rate",
    "second_person_rate": "second_person_score_rate",
    "third_person_rate": "third_person_score_rate",
    "impersonal_pronoun_rate": "impersonal_score_rate",
    
    # 时间定向
    "past_focus_rate": "past_focus_score_rate",
    "present_focus_rate": "present_focus_score_rate",
    "future_focus_rate": "future_focus_score_rate",
    
    # 风险与其他
    "death_mortality_rate": "death_score_rate",
    "swearing_rate": "swear_score_rate",
    "assent_rate": "assent_score_rate",
    "nonfluency_rate": "nonfluency_score_rate",
    "inhibition_marker_rate": "inhibition_score_rate",
    
    # 行为标记
    "netspeak_usage_rate": "netspeak_count_rate",
    "filler_word_rate": "filler_count_rate",
    
    # 连续值 (s01用户级 _mean 对应 s07窗口级 _mean)
    "vader_compound_mean": "vader_compound_mean",
    "vader_compound_std": "vader_between_post_std", # s01用户级跨帖标准差 vs s07窗口级
    "net_emotion_tone": "negative_net_score_mean",
    "self_focus_index": "self_focus_ratio_mean",
    "cognitive_flexibility": "cognitive_flexibility_mean",
    "temporal_balance": "temporal_balance_mean",
    "ari_mean": "ari_score_mean",
    "mattr_vocabulary_diversity": "mattr_score_mean",
        # 补充缺失映射
    "social_focus_index": "social_focus_ratio_mean",
    "vader_positive_mean": "vader_pos_mean",
    "vader_negative_mean": "vader_neg_mean",
    "vader_within_post_volatility": "vader_within_post_volatility",
    "emotion_inertia": "emotion_inertia_window",
    "question_ratio": "question_ratio_mean",
    "exclamation_ratio": "exclamation_ratio_mean",
    "vader_compound_min_mean": "vader_compound_min_mean",
    "vader_compound_last_mean": "vader_compound_last_mean",  # 如果窗口级也计算了
    
     # 补充同名映射
    "url_usage_rate": "url_usage_rate",
    "ari_std": "ari_std",
    "post_length_mean": "post_length_mean",
    "post_length_std": "post_length_std",
    "post_length_cv": "post_length_cv",
    "self_focus_variability": "self_focus_variability",
    "social_focus_variability": "social_focus_variability",
    "circadian_mean_hour": "circadian_mean_hour",
    "circadian_concentration": "circadian_concentration",
    "night_activity_ratio": "night_activity_ratio",
    "weekend_activity_ratio": "weekend_activity_ratio",
    "social_jetlag_hours": "social_jetlag_hours",
    "sleep_disruption_index": "sleep_disruption_index",
    "posting_regularity_cv": "posting_regularity_cv",
    "posting_burstiness": "posting_burstiness",
    "interval_kurtosis": "interval_kurtosis",
    "interval_median_hours": "interval_median_hours",
    "interval_mean_hours": "interval_mean_hours",
    "hourly_entropy": "hourly_entropy",
    "weekly_entropy": "weekly_entropy",
    "temporal_diversity_index": "temporal_diversity_index",
    "tenure_days": "tenure_days",
    "n_posts_total": "n_posts_total",
    "n_active_days": "n_active_days",
    "active_day_ratio": "active_day_ratio",
    "posts_per_day": "posts_per_day",
    "posts_per_active_day": "posts_per_active_day",
    "max_gap_days": "max_gap_days",
    "mean_gap_days": "mean_gap_days",
    "gap_regularity_cv": "gap_regularity_cv",
    "submission_ratio": "submission_ratio",
    "n_submissions": "n_submissions",
    "n_comments": "n_comments",
    "score_mean": "score_mean",
    "score_median": "score_median",
    "score_std": "score_std",
    "score_max": "score_max",
    "score_min": "score_min",
    "score_range": "score_range",
    "neg_score_ratio": "neg_score_ratio",
    "high_engagement_ratio": "high_engagement_ratio",
    "avg_comments_received": "avg_comments_received",
    "total_comments_received": "total_comments_received",
    "discussion_initiated_ratio": "discussion_initiated_ratio",
    "top_level_comment_ratio": "top_level_comment_ratio",
    "avg_comment_depth": "avg_comment_depth",
    "controversy_rate": "controversy_rate",
    "gilded_rate": "gilded_rate",
    "subreddit_entropy": "subreddit_entropy",
    "subreddit_herfindahl": "subreddit_herfindahl",
    "n_unique_subreddits": "n_unique_subreddits",
    "mental_health_sub_ratio": "mental_health_sub_ratio",
    "support_sub_ratio": "support_sub_ratio",
    "entertainment_sub_ratio": "entertainment_sub_ratio",
    "information_sub_ratio": "information_sub_ratio",
    "social_sub_ratio": "social_sub_ratio",
    "other_sub_ratio": "other_sub_ratio",
    "posts_per_day_log": "posts_per_day_log",
    "posts_per_active_day_log": "posts_per_active_day_log",
    "interval_median_hours_log": "interval_median_hours_log",
    "max_gap_days_log": "max_gap_days_log",
    "tenure_days_log": "tenure_days_log",
    "n_posts_total_log": "n_posts_total_log",
    "score_mean_log_signed": "score_mean_log_signed"

}

def translate_pfactor_name(loading_name: str) -> Optional[str]:
    """将 Factor Loading 中的用户级特征名翻译为 s07 窗口级特征名"""
    if loading_name in USER_TO_TRAJECTORY_NAME_MAP:
        return USER_TO_TRAJECTORY_NAME_MAP[loading_name]
    # 兜底：直接尝试原名
    return loading_name

# ======================== IMPORT FROM MODULE 1 ========================
# import sys
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Module1.s01_feature_extraction import (
    extract_linguistic_features_post,
    extract_temporal_features_post,
    init_worker, get_liwc, get_nrc, get_vader,
    _dict_manager,
    RE_WORDS, RE_URL, RE_SENTENCES,
    calc_mattr, safe_divide,
)


# ======================== T0 MAPPING LOADER ========================

def load_t0_mapping() -> pd.DataFrame:
    """
    v2.1: 所有路径返回前统一补全 disorder_group
    """
    logger.info("Loading t0 mapping (v2.1 - multi-source with Silver augmentation)...")
    
    t0_df = None
    
    # ================================================================
    # 来源 ①: PSM 匹配文件
    # ================================================================
    if os.path.exists(PSM_MATCHED_PATH):
        logger.info(f"  Source 1: PSM matched file → {PSM_MATCHED_PATH}")
        try:
            psm_df = pd.read_parquet(PSM_MATCHED_PATH)
            
            # 确定列名
            uid_col = None
            for candidate in ["user_id_hash", "user_id", "uid", "id"]:
                if candidate in psm_df.columns:
                    uid_col = candidate
                    break
            
            group_col = None
            for candidate in ["is_patient", "group"]:
                if candidate in psm_df.columns:
                    group_col = candidate
                    break
            
            # --- 路径 A: shadow_t0_ts ---
            if uid_col and "shadow_t0_ts" in psm_df.columns:
                read_cols = [uid_col, "shadow_t0_ts"]
                # ✅ 如果 PSM 已有 disorder_group，直接读取
                if "disorder_group" in psm_df.columns:
                    read_cols.append("disorder_group")
                
                t0_df = psm_df[read_cols].copy()
                t0_df = t0_df.rename(columns={
                    uid_col: "user_id_hash",
                    "shadow_t0_ts": "t0_timestamp"
                })
                
                # ✅ 先规范化时间戳
                t0_df["t0_timestamp"] = pd.to_numeric(t0_df["t0_timestamp"], errors="coerce")
                t0_df.loc[t0_df["t0_timestamp"] > 1e12, "t0_timestamp"] /= 1000.0
                
                # ✅ 先赋值 group
                if group_col == "is_patient":
                    t0_df["group"] = np.where(
                        psm_df.loc[t0_df.index, group_col] == 1, "patient", "control"
                    )
                elif group_col == "group":
                    t0_df["group"] = np.where(
                        psm_df.loc[t0_df.index, group_col].isin(["patient"]), "patient", "control"
                    )
                
                # ✅ 处理 disorder_group：优先用 PSM 已有的，否则设 unknown
                if "disorder_group" in t0_df.columns:
                    # PSM 文件已有 disorder_group（经 quick_fix 补全）
                    # 确保对照统一为 "control"
                    t0_df.loc[t0_df["group"] == "control", "disorder_group"] = "control"
                    n_known = (t0_df["disorder_group"] != "unknown").sum()
                    n_pat = (t0_df["group"] == "patient").sum()
                    logger.info(f"    PSM disorder_group: {n_known}/{n_pat} patients known")
                else:
                    t0_df["disorder_group"] = np.where(
                        t0_df["group"] == "patient", "unknown", "control"
                    )
                
                t0_df["t0_source"] = "psm_shadow_t0_ts"
                
                # ✅ 再过滤
                t0_df = t0_df[
                    t0_df["t0_timestamp"].notna() &
                    (t0_df["t0_timestamp"] > 1e9) &
                    (t0_df["t0_timestamp"] < 2e9)
                ].copy()
                t0_df = t0_df.reset_index(drop=True)
                
                if len(t0_df) > 0:
                    _save_pair_info_from_psm(psm_df, uid_col, group_col)
                    # ⚠️ 不要在这里 return！需要先补全 disorder_group
            
            # --- 路径 B: real_t0_date ---
            elif uid_col and "real_t0_date" in psm_df.columns:
                disorder_col = None
                for candidate in ["disorder_group", "disorder"]:
                    if candidate in psm_df.columns:
                        disorder_col = candidate
                        break
                
                t0_df = _construct_t0_from_psm_real_dates(
                    psm_df, uid_col, group_col, disorder_col
                )
                
        except Exception as e:
            logger.warning(f"  PSM file load failed: {e}")
    
    # ================================================================
    # 来源 ②: user_meta 文件
    # ================================================================
    if t0_df is None and os.path.exists(USER_META_PATH):
        logger.info(f"  Source 2: User meta file → {USER_META_PATH}")
        try:
            user_meta = pd.read_parquet(USER_META_PATH)
            
            if "shadow_t0_ts" in user_meta.columns:
                t0_df = user_meta[["user_id_hash", "shadow_t0_ts"]].copy()
                t0_df = t0_df.rename(columns={"shadow_t0_ts": "t0_timestamp"})
                
                # ✅ 先规范化
                t0_df["t0_timestamp"] = pd.to_numeric(
                    t0_df["t0_timestamp"], errors="coerce"
                )
                t0_df.loc[t0_df["t0_timestamp"] > 1e12, "t0_timestamp"] /= 1000.0
                
                # 先赋值 group
                group_col = "group" if "group" in user_meta.columns else "is_patient"
                if group_col == "is_patient":
                    t0_df["group"] = np.where(
                        user_meta.loc[t0_df.index, group_col] == 1,
                        "patient", "control"
                    )
                else:
                    t0_df["group"] = np.where(
                        user_meta.loc[t0_df.index, group_col].isin(["patient"]),
                        "patient", "control"
                    )
                
                t0_df["disorder_group"] = np.where(
                    t0_df["group"] == "patient", "unknown", "control"
                )
                t0_df["t0_source"] = "user_meta_shadow_t0_ts"
                
                # 再过滤
                t0_df = t0_df[
                    t0_df["t0_timestamp"].notna() &
                    (t0_df["t0_timestamp"] > 1e9) &
                    (t0_df["t0_timestamp"] < 2e9)
                ].copy()
                t0_df = t0_df.reset_index(drop=True)
                
        except Exception as e:
            logger.warning(f"  User meta load failed: {e}")
    
    # ================================================================
    # 来源 ③: Silver 数据的 assigned_t0
    # ================================================================
    if t0_df is None:
        logger.info("  Source 3: Scanning Silver data for assigned_t0...")
        t0_df = _load_t0_from_silver_data()
    
    # ================================================================
    # 来源 ④: 特征文件的 t0_date
    # ================================================================
    if t0_df is None:
        logger.info("  Source 4 (fallback): Parsing t0_date from features...")
        t0_df = _load_t0_from_features_fallback()
    
    # ================================================================
    # ✅ 统一补全 disorder_group（所有路径汇聚于此）
    # ================================================================
    if t0_df is not None and len(t0_df) > 0:
        t0_df = _augment_disorder_from_silver(t0_df)
        
        # 最终统计
        n_valid = t0_df["t0_timestamp"].notna().sum()
        n_pat = (t0_df["group"] == "patient").sum()
        n_ctrl = (t0_df["group"] == "control").sum()
        n_disorder_known = sum(t0_df["disorder_group"] != "unknown")
        logger.info(f"  ✓ Final t0 mapping: {n_valid} valid users "
                   f"(patient={n_pat}, control={n_ctrl})")
        logger.info(f"    disorder_group known: {n_disorder_known}/{n_pat} patients")
    
    return t0_df


def _augment_disorder_from_silver(t0_df: pd.DataFrame) -> pd.DataFrame:
    n_unknown = (t0_df["disorder_group"] == "unknown").sum()
    n_patient = (t0_df["group"] == "patient").sum()
    
    if n_unknown == 0 or n_patient == 0:
        return t0_df
    
    target_uids = set(
        t0_df.loc[
            (t0_df["group"] == "patient") & 
            (t0_df["disorder_group"] == "unknown"),
            "user_id_hash"
        ]
    )
    
    if not target_uids:
        return t0_df
    
    disorder_map = {}
    
    # ========== 优先源1: s01 特征文件（极快、最完整）==========
    s01_candidates = [
        os.path.join(FEATURES_DIR, "user_features_raw.parquet"),
        os.path.join(FEATURES_DIR, RAW_FEATURES_FILE),
    ]
    for feat_path in s01_candidates:
        if not os.path.exists(feat_path):
            continue
        try:
            available = pq.ParquetFile(feat_path).schema.names
            if "disorder_group" not in available:
                continue
            df = pd.read_parquet(feat_path, columns=["user_id_hash", "disorder_group"])
            valid = df[
                df["user_id_hash"].isin(target_uids) & 
                df["disorder_group"].notna() &
                ~df["disorder_group"].astype(str).isin(
                    {"control", "control_candidate", "unknown", "nan", "None", ""}
                )
            ].drop_duplicates(subset=["user_id_hash"])
            disorder_map.update(dict(zip(valid["user_id_hash"], valid["disorder_group"].astype(str))))
            del df, valid
            logger.info(f"  [augment] s01 features recovered: {len(disorder_map)}/{len(target_uids)}")
            break
        except Exception as e:
            logger.warning(f"  [augment] s01 features failed: {e}")
    
    # ========== 补充源2: Silver 分区扫描 ==========
    remaining = target_uids - set(disorder_map.keys())
    if remaining and os.path.exists(SILVER_DATA_DIR):
        silver_map = _scan_silver_for_disorder_map(remaining)
        disorder_map.update(silver_map)
        logger.info(f"  [augment] After Silver scan: {len(disorder_map)}/{len(target_uids)}")
    
    # 赋值
    if disorder_map:
        # [FIX v1.7] 必须先合并 t0_df 已有的 disorder_group（如 PSM 文件提供）
        # 再覆盖，否则 map(disorder_map) 会把不在 disorder_map 中的已知患者
        # 映射为 NaN → fillna("unknown")，清空 PSM 的 12.7 万患者 disorder_group
        # （本 bug 曾导致患者 known 从 127198 骤降至 755）。
        existing = (
            t0_df[["user_id_hash", "disorder_group"]]
            .drop_duplicates("user_id_hash")
            .set_index("user_id_hash")["disorder_group"]
            .replace({"unknown": np.nan})  # unknown 视作缺失，允许被 disorder_map 覆盖
        )
        combined = existing.combine_first(pd.Series(disorder_map))
        t0_df["disorder_group"] = np.where(
            t0_df["group"] == "patient",
            t0_df["user_id_hash"].map(combined).fillna("unknown"),
            "control"
        )
    
    return t0_df

def _scan_silver_for_disorder_map(
    target_uids: set,
    max_partitions: int = None,
    max_files_per_partition: int = 20,
) -> Dict[str, str]:
    """
    从 Silver Data 中恢复 disorder_group 映射。
    
    两级策略:
      1. 优先使用 _registry/user_lookup.parquet 定向扫描（极快）
      2. 注册表不可用时，全分区扫描（较慢但完整）
    
    参数:
        target_uids: 需要恢复 disorder_group 的用户ID集合
        max_partitions: 最多扫描多少个分区（None=不限制）
        max_files_per_partition: 每个分区最多扫描多少个文件
    
    返回:
        {user_id_hash: disorder_group} 映射字典
        仅包含成功恢复的用户
    """
    import pyarrow.parquet as pq
    import glob
    
    if not target_uids:
        return {}
    
    remaining = set(target_uids)
    disorder_map = {}
    
    # ================================================================
    # 策略1: 使用注册表定向扫描
    # ================================================================
    registry_path = os.path.join(SILVER_DATA_DIR, "_registry", "user_lookup.parquet")
    
    if os.path.exists(registry_path):
        logger.info(f"  [disorder_scan] Using registry for targeted scan: {registry_path}")
        try:
            # 只读取目标用户的分区映射
            reg = pd.read_parquet(registry_path, columns=["user_id_hash", "partition_id"])
            reg = reg[reg["user_id_hash"].isin(remaining)]
            
            if len(reg) == 0:
                logger.warning("  [disorder_scan] Registry has no matching users")
            else:
                # 构建分区→用户集合的映射
                partition_to_users = {}
                for _, row in reg.iterrows():
                    pid = int(row["partition_id"])
                    uid = row["user_id_hash"]
                    if pid not in partition_to_users:
                        partition_to_users[pid] = set()
                    partition_to_users[pid].add(uid)
                
                logger.info(f"  [disorder_scan] Target users spread across "
                           f"{len(partition_to_users)} partitions")
                
                # 逐分区定向扫描
                for pid in sorted(partition_to_users.keys()):
                    if not remaining:
                        break
                    
                    users_in_part = partition_to_users[pid] & remaining
                    if not users_in_part:
                        continue
                    
                    part_dir = os.path.join(
                        SILVER_DATA_DIR, "data", f"part_{pid:04d}"
                    )
                    
                    if not os.path.isdir(part_dir):
                        # 尝试不带 data/ 子目录的路径
                        part_dir = os.path.join(
                            SILVER_DATA_DIR, f"part_{pid:04d}"
                        )
                    
                    if not os.path.isdir(part_dir):
                        logger.warning(f"  [disorder_scan] Partition dir not found: part_{pid:04d}")
                        continue
                    
                    found_in_part = _scan_partition_for_disorder(
                        part_dir, users_in_part, max_files=max_files_per_partition
                    )
                    disorder_map.update(found_in_part)
                    remaining -= set(found_in_part.keys())
                
                logger.info(f"  [disorder_scan] Registry-based scan recovered: "
                           f"{len(disorder_map)}/{len(target_uids)}")
        
        except Exception as e:
            logger.warning(f"  [disorder_scan] Registry scan failed: {e}, falling back to full scan")
            remaining = set(target_uids)  # 重置
            disorder_map = {}
    
    # ================================================================
    # 策略2: 全分区扫描（注册表不可用或仍有剩余用户时）
    # ================================================================
    if remaining:
        logger.info(f"  [disorder_scan] Full partition scan for remaining {len(remaining)} users...")
        
        partitions = _discover_silver_partitions()
        
        if max_partitions:
            partitions = partitions[:max_partitions]
        
        for part_dir in partitions:
            if not remaining:
                break
            
            found_in_part = _scan_partition_for_disorder(
                part_dir, remaining, max_files=max_files_per_partition
            )
            disorder_map.update(found_in_part)
            remaining -= set(found_in_part.keys())
        
        if remaining:
            logger.warning(f"  [disorder_scan] Still missing {len(remaining)} users after full scan")
        logger.info(f"  [disorder_scan] Total recovered: {len(disorder_map)}/{len(target_uids)}")
    
    return disorder_map


def _discover_silver_partitions() -> List[str]:
    """
    鲁棒地发现 Silver 数据分区目录。
    处理两种可能的目录结构：
      - SILVER_DATA_DIR/data/part_XXXX/
      - SILVER_DATA_DIR/part_XXXX/
    """
    partitions = []
    
    # 尝试1: SILVER_DATA_DIR/data/ 子目录
    data_subdir = os.path.join(SILVER_DATA_DIR, "data")
    base_dir = data_subdir if os.path.isdir(data_subdir) else SILVER_DATA_DIR
    
    if os.path.isdir(base_dir):
        for entry in sorted(os.listdir(base_dir)):
            full_path = os.path.join(base_dir, entry)
            if os.path.isdir(full_path) and entry.startswith("part_"):
                partitions.append(full_path)
    
    if not partitions:
        logger.warning(f"  [discover] No partitions found in {SILVER_DATA_DIR}")
    else:
        logger.info(f"  [discover] Found {len(partitions)} partitions in {base_dir}")
    
    return partitions


def _scan_partition_for_disorder(
    part_dir: str,
    target_uids: set,
    max_files: int = 20,
) -> Dict[str, str]:
    """
    扫描单个分区目录，从中提取目标用户的 disorder_group。
    
    使用向量化操作替代逐行 iterrows，大幅提升速度。
    
    参数:
        part_dir: 分区目录路径 (如 .../part_0042/)
        target_uids: 目标用户ID集合
        max_files: 最多扫描的文件数
    
    返回:
        {user_id_hash: disorder_group} 映射
    """
    import pyarrow.parquet as pq
    import glob
    
    disorder_map = {}
    
    parquet_files = sorted(glob.glob(os.path.join(part_dir, "*.parquet")))
    
    for pf_path in parquet_files[:max_files]:
        if not target_uids:
            break
        
        try:
            pf = pq.ParquetFile(pf_path)
            available_cols = set(pf.schema.names)
            
            # 检查必要列是否存在
            required = {"user_id_hash", "disorder_group"}
            if not required.issubset(available_cols):
                continue
            
            read_cols = ["user_id_hash", "disorder_group"]
            
            for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=read_cols):
                df = batch.to_pandas()
                
                # ✅ 向量化过滤：替代逐行 iterrows
                mask = (
                    df["user_id_hash"].isin(target_uids) &
                    df["disorder_group"].notna()
                )
                matched = df[mask]
                
                if len(matched) > 0:
                    # 每个用户取第一条有效记录
                    for uid, dg in matched.drop_duplicates(subset=["user_id_hash"]).values:
                        dg_str = str(dg)
                        # 排除无效标签
                        if dg_str not in ("control", "control_candidate", "unknown", "nan", "None", ""):
                            disorder_map[uid] = dg_str
                
                del df, matched
                
        except Exception as e:
            logger.debug(f"  [disorder_scan] Error reading {pf_path}: {e}")
            continue
    
    return disorder_map

def build_user_groups(
    t0_df: pd.DataFrame,
    psm_matched_path: str = PSM_MATCHED_PATH,
    user_meta_path: str = USER_META_PATH,
) -> Dict[str, Dict]:
    """
    [FIX-1a] 从多个来源构建完整的 user_groups 字典。
    
    解决:
      - pair_id 未从 PSM 配对文件传递到轨迹数据
      - iptw_att_weight / eb_weight 未从 user_meta 传递到轨迹数据
      - group 列 "control_candidate" 未标准化为 "control"
    
    参数:
        t0_df: load_t0_mapping() 的返回值
        psm_matched_path: PSM 匹配结果文件
        user_meta_path: 用户元数据文件（含权重）
    
    返回:
        {user_id_hash: {
            "group": "patient" / "control",
            "disorder_group": str,
            "pair_id": str or None,
            "iptw_att_weight": float or NaN,
            "eb_weight": float or NaN,
        }}
    """
    user_groups = {}
    
    # ===== Step 1: 从 t0_df 初始化基础信息 =====
    for _, row in t0_df.iterrows():
        uid = row["user_id_hash"]
        # group 标准化：防御性确认，解决 C 项问题
        group = str(row["group"])
        if group not in ("patient", "control"):
            group = "control"
        
        disorder = str(row.get("disorder_group", "control"))
        if group == "control":
            disorder = "control"
            
        user_groups[uid] = {
            "group": group,
            "disorder_group": disorder,
            "pair_id": None,
            "iptw_att_weight": np.nan,
            "eb_weight": np.nan,
        }
    
    # ===== Step 2: 从 user_meta 加载权重 =====
    if os.path.exists(user_meta_path):
        try:
            import pyarrow.parquet as pq
            schema = pq.ParquetFile(user_meta_path).schema.names
            read_cols = ["user_id_hash"]
            if "iptw_att_weight" in schema: read_cols.append("iptw_att_weight")
            if "eb_weight" in schema: read_cols.append("eb_weight")
                
            um = pd.read_parquet(user_meta_path, columns=read_cols)
            um = um[um["user_id_hash"].isin(user_groups.keys())]
            
            n_iptw = 0
            n_eb = 0
            
            if "iptw_att_weight" in um.columns:
                iptw_map = um.dropna(subset=["iptw_att_weight"]).set_index("user_id_hash")["iptw_att_weight"].to_dict()
                for uid, w in iptw_map.items():
                    user_groups[uid]["iptw_att_weight"] = float(w)
                n_iptw = len(iptw_map)
                
            if "eb_weight" in um.columns:
                eb_map = um.dropna(subset=["eb_weight"]).set_index("user_id_hash")["eb_weight"].to_dict()
                for uid, w in eb_map.items():
                    user_groups[uid]["eb_weight"] = float(w)
                n_eb = len(eb_map)
            
            logger.info(f"  [build_user_groups] IPTW weights: {n_iptw}/{len(user_groups)} loaded")
            logger.info(f"  [build_user_groups] EB weights: {n_eb}/{len(user_groups)} loaded")
        except Exception as e:
            logger.warning(f"  [build_user_groups] user_meta load failed: {e}")
    else:
        logger.warning(f"  [build_user_groups] user_meta not found: {user_meta_path}")
    
    # ===== Step 3: 加载或重建 pair_id =====
    pair_info_path = os.path.join(FEATURES_DIR, "psm_pair_info.parquet")
    
    # 如果配对文件不存在，尝试触发重建
    if not os.path.exists(pair_info_path) and os.path.exists(psm_matched_path):
        logger.info("  [build_user_groups] Pair info not found, attempting reconstruction...")
        try:
            psm_df = pd.read_parquet(psm_matched_path)
            # 确定 UID 列
            uid_col = None
            for candidate in ["user_id_hash", "user_id", "uid"]:
                if candidate in psm_df.columns:
                    uid_col = candidate
                    break
            # 确定 group 列
            group_col = None
            for candidate in ["is_patient", "group"]:
                if candidate in psm_df.columns:
                    group_col = candidate
                    break
            if uid_col and group_col:
                _save_pair_info_from_psm(psm_df, uid_col, group_col)
        except Exception as e:
            logger.warning(f"  [build_user_groups] Pair reconstruction failed: {e}")
    
    # 加载配对文件
    if os.path.exists(pair_info_path):
        try:
            pairs = pd.read_parquet(pair_info_path)
            n_pair = 0
            
            for _, row in pairs.iterrows():
                pid = row["pair_id"]
                pat_uid = row.get("patient_id")
                ctrl_uid = row.get("control_id")
                
                if pat_uid and pat_uid in user_groups:
                    user_groups[pat_uid]["pair_id"] = pid
                    n_pair += 1
                if ctrl_uid and ctrl_uid in user_groups:
                    user_groups[ctrl_uid]["pair_id"] = pid
                    n_pair += 1
            
            n_users_with_pair = sum(1 for v in user_groups.values() if v["pair_id"] is not None)
            logger.info(f"  [build_user_groups] Pair IDs: {n_users_with_pair}/{len(user_groups)} users have pair_id")
        except Exception as e:
            logger.warning(f"  [build_user_groups] Pair info load failed: {e}")
    else:
        logger.warning("  [build_user_groups] No pair info available. "
                      "Pair FE DID will be unavailable in s10.")
    
    # ===== Step 4: 汇总统计 =====
    n_pat = sum(1 for v in user_groups.values() if v["group"] == "patient")
    n_ctrl = sum(1 for v in user_groups.values() if v["group"] == "control")
    n_has_pair = sum(1 for v in user_groups.values() if v["pair_id"] is not None)
    n_has_iptw = sum(1 for v in user_groups.values() 
                     if not (isinstance(v.get("iptw_att_weight"), float) and np.isnan(v["iptw_att_weight"])))
    
    logger.info(f"  [build_user_groups] Final: {len(user_groups)} users "
               f"(patient={n_pat}, control={n_ctrl})")
    logger.info(f"    With pair_id: {n_has_pair}")
    logger.info(f"    With iptw_att_weight: {n_has_iptw}")
    
    return user_groups

def _save_pair_info_from_psm(psm_df, uid_col, group_col):
    """
    [NEW] 尝试从 PSM 文件中提取/重建配对信息。
    
    预处理的 PSM 文件是扁平结构，但匹配是在 tenure_quartile 分层内做的。
    我们可以从 ps_logit 的距离重建近似的配对关系。
    """
    logger.info("  Attempting to reconstruct pair information from PSM...")
    
    pair_info_path = os.path.join(FEATURES_DIR, "psm_pair_info.parquet")
    
    # 如果已经有配对信息，跳过
    if os.path.exists(pair_info_path):
        logger.info(f"    Pair info already exists: {pair_info_path}")
        return
    
    # 需要有 ps_logit 或 ps_score 才能重建配对
    ps_col = None
    for candidate in ["ps_logit", "ps_score"]:
        if candidate in psm_df.columns:
            ps_col = candidate
            break
    
    if ps_col is None or "tenure_quartile" not in psm_df.columns:
        logger.warning("    Cannot reconstruct pairs: missing ps_logit or tenure_quartile")
        return
    
    from sklearn.neighbors import NearestNeighbors
    
    pair_records = []
    
    # 在每个分层内重建配对
    for q_val in sorted(psm_df["tenure_quartile"].dropna().unique()):
        stratum = psm_df[psm_df["tenure_quartile"] == q_val].copy()
        
        # 确定患者/对照
        if group_col == "is_patient":
            patients = stratum[stratum[group_col] == 1]
            controls = stratum[stratum[group_col] == 0]
        else:
            patients = stratum[stratum[group_col] == "patient"]
            controls = stratum[stratum[group_col] != "patient"]
        
        if len(patients) == 0 or len(controls) == 0:
            continue
        
        # 基于倾向得分距离重建配对
        n_neighbors = min(5, len(controls))  # 1:5 匹配
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        nbrs.fit(controls[[ps_col]].values)
        dists, idxs = nbrs.kneighbors(patients[[ps_col]].values)
        
        pair_id_counter = 0
        for i, (_, pat_row) in enumerate(patients.iterrows()):
            pat_uid = pat_row[uid_col]
            for j in range(min(5, len(idxs[i]))):
                ctrl_row = controls.iloc[idxs[i][j]]
                ctrl_uid = ctrl_row[uid_col]
                pair_records.append({
                    "patient_id": pat_uid,
                    "control_id": ctrl_uid,
                    "pair_id": f"pair_{q_val}_{pair_id_counter}",
                    "ps_distance": dists[i][j],
                    "stratum": q_val,
                })
            pair_id_counter += 1
    
    if pair_records:
        pair_df = pd.DataFrame(pair_records)
        pair_df.to_parquet(pair_info_path, index=False)
        logger.info(f"    ✓ Reconstructed {len(pair_df)} pair relationships → {pair_info_path}")
    else:
        logger.warning("    No pair relationships reconstructed")


def _construct_t0_from_psm_real_dates(psm_df, uid_col, group_col, disorder_col):
    """
    从 PSM 文件的 real_t0_date 构建 t0 映射。
    为对照分配配对患者的 t0（如果无配对信息，则从患者分布采样）。
    """
    t0_records = []
    
    # 解析患者的 real_t0_date
    patient_t0_map = {}
    
    # 确定哪些是患者
    if group_col == "is_patient":
        patient_mask = psm_df[group_col] == 1
    else:
        patient_mask = psm_df[group_col] == "patient"
    
    patients = psm_df[patient_mask]
    controls = psm_df[~patient_mask]
    
    # 解析患者 t0
    for _, row in patients.iterrows():
        t0_ts = _parse_single_t0(row.get("real_t0_date"))
        if t0_ts is not None:
            patient_t0_map[row[uid_col]] = t0_ts
            t0_records.append({
                "user_id_hash": row[uid_col],
                "t0_timestamp": t0_ts,
                "group": "patient",
                "disorder_group": str(row[disorder_col]) if disorder_col and pd.notna(row.get(disorder_col)) else "unknown",
                "t0_source": "psm_real_t0_date",
            })
    
    # 对照 t0：从患者分布中采样（带有效性校验）
    if len(controls) > 0 and len(patient_t0_map) > 0:
        valid_t0_values = list(patient_t0_map.values())
        
        # 如果有 first_post_ts / last_post_ts，做校验
        has_post_history = "first_post_ts" in psm_df.columns and "last_post_ts" in psm_df.columns
        min_pre = PSM_CONFIG_MIN_PRE_DAYS * 86400 if has_post_history else 0
        min_post = PSM_CONFIG_MIN_POST_DAYS * 86400 if has_post_history else 0
        
        rng = np.random.RandomState(RANDOM_SEED)
        
        for _, row in controls.iterrows():
            ctrl_uid = row[uid_col]
            
            # 尝试最多5次找到有效的 t0
            assigned_t0 = None
            for _ in range(5):
                sampled = rng.choice(valid_t0_values)
                
                if has_post_history:
                    first_ts = row.get("first_post_ts", 0)
                    last_ts = row.get("last_post_ts", float('inf'))
                    if pd.notna(first_ts) and pd.notna(last_ts):
                        if sampled > first_ts + min_pre and sampled < last_ts - min_post:
                            assigned_t0 = sampled
                            break
                else:
                    assigned_t0 = sampled
                    break
            
            if assigned_t0 is not None:
                t0_records.append({
                    "user_id_hash": ctrl_uid,
                    "t0_timestamp": assigned_t0,
                    "group": "control",
                    "disorder_group": "control",
                    "t0_source": "psm_sampled_from_patients",
                })
    
    if t0_records:
        return pd.DataFrame(t0_records)
    
    return pd.DataFrame()


def _load_t0_from_silver_data():
    """
    从 Silver 数据分区的 assigned_t0 列加载 t0。
    这是预处理在 process_final_build_chunk 中写入的。
    """
    logger.info("  Scanning Silver data partitions for assigned_t0...")
    
    if not os.path.exists(SILVER_DATA_DIR):
        return None
    
    # 发现分区
    partitions = []
    for entry in sorted(os.listdir(SILVER_DATA_DIR)):
        full_path = os.path.join(SILVER_DATA_DIR, entry)
        if os.path.isdir(full_path):
            partitions = _discover_silver_partitions()
    
    if not partitions:
        return None
    
    # 只需要读取第一批文件来获取 assigned_t0 + group 信息
    # 因为每个用户的 assigned_t0 在所有文件中是相同的
    user_t0_map = {}
    
    cols_needed = ["user_id_hash", "assigned_t0", "group", "disorder_group", "is_retrospective"]
    
    # 限制扫描的文件数量（t0 在所有文件中一致）
    import pyarrow.parquet as pq
    import glob
    
    parquet_files = []
    for part_dir in partitions[:3]:  # 只扫描前3个分区
        parquet_files.extend(glob.glob(os.path.join(part_dir, "*.parquet")))
    
    for pf_path in parquet_files[:10]:  # 每个分区只读前10个文件
        try:
            pf = pq.ParquetFile(pf_path)
            available_cols = set(pf.schema.names)
            read_cols = [c for c in cols_needed if c in available_cols]
            
            if "assigned_t0" not in available_cols:
                continue
            
            for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=read_cols):
                df = batch.to_pandas()
                for _, row in df.iterrows():
                    uid = row["user_id_hash"]
                    if uid in user_t0_map:
                        continue
                    
                    t0_ts = row.get("assigned_t0")
                    if pd.isna(t0_ts) or t0_ts == 0:
                        continue

                    t0_ts = float(t0_ts)
                    if t0_ts > 1e12:
                        t0_ts = t0_ts / 1000.0
                    if not (1e9 < t0_ts < 2e9):
                        continue

                    # [FIX-D1] 对照用户的 assigned_t0 额外校验：
                    # Phase 2 中对照的 assigned_t0 可能是占位值（如文件创建时间等），
                    # 需确保落在合理的 Reddit 活跃期内 (2005-2023)
                    REDDIT_MIN_TS = 1135987200   # 2006-01-01
                    REDDIT_MAX_TS = 1704038400   # 2024-01-01
                    if t0_ts < REDDIT_MIN_TS or t0_ts > REDDIT_MAX_TS:
                        continue

                    # [FIX-D1] 对照用户必须有非零的 assigned_t0
                    group_val = str(row.get("group", "unknown"))
                    if group_val != "patient" and t0_ts < 1e9:
                        logger.debug(f"  Skipping control user {uid[:8]} with invalid assigned_t0={t0_ts}")
                        continue
                    
                    group = str(row.get("group", "unknown"))
                    if group in ("control_candidate", "healthy_control", "Control"):
                        group = "control"
                    
                    disorder = str(row.get("disorder_group", "control"))
                    if group == "control":
                        disorder = "control"
                    
                    user_t0_map[uid] = {
                        "user_id_hash": uid,
                        "t0_timestamp": t0_ts,
                        "group": group,
                        "disorder_group": disorder,
                        "t0_source": "silver_assigned_t0",
                    }
                
                del df
        except Exception as e:
            logger.warning(f"    Error reading {pf_path}: {e}")
    
    if user_t0_map:
        t0_df = pd.DataFrame(list(user_t0_map.values()))
        logger.info(f"    ✓ Loaded from Silver assigned_t0: {len(t0_df)} users")
        return t0_df
    
    return None


def _load_t0_from_features_fallback():
    """
    最后手段：从 Module 1 的 features 文件解析 t0_date。
    """
    feat_path = os.path.join(FEATURES_DIR, RAW_FEATURES_FILE)
    if not os.path.exists(feat_path):
        raise FileNotFoundError(
            f"Cannot find t0 from any source! Last tried: {feat_path}"
        )
    
    logger.warning("  Using LAST RESORT: parsing t0_date from features file")
    
    feat_df = pd.read_parquet(feat_path, 
        columns=["user_id_hash", "group", "disorder_group", "t0_date"])
    
    # 患者：解析 t0_date
    patients = feat_df[feat_df["group"] == "patient"].copy()
    patients["t0_timestamp"] = _parse_t0_column(patients)
    valid_patients = patients[patients["t0_timestamp"].notna()].copy()
    
    if len(valid_patients) == 0:
        raise ValueError("No valid patient t0 timestamps found from any source!")
    
    # 对照：从患者 t0 分布采样（带校验）
    # [FIX-A2] 兼容 "control" 和 "control_candidate"
    controls = feat_df[feat_df["group"].isin(["control", "control_candidate"])].copy()
    
    if len(controls) > 0:
        patient_t0_values = valid_patients["t0_timestamp"].values
        rng = np.random.RandomState(RANDOM_SEED)
        
        # 尝试从 user_meta 获取发帖历史用于校验
        first_post_map = {}
        last_post_map = {}

        # PSM matched 文件有这两列
        if os.path.exists(PSM_MATCHED_PATH):
            try:
                psm = pd.read_parquet(PSM_MATCHED_PATH,
                    columns=["user_id_hash", "first_post_ts", "last_post_ts"])
                first_post_map = psm.set_index("user_id_hash")["first_post_ts"].to_dict()
                last_post_map = psm.set_index("user_id_hash")["last_post_ts"].to_dict()
            except Exception:
                pass
        
        min_pre = 30 * 86400  # 30天
        min_post = 30 * 86400
        
        sampled_t0 = rng.choice(patient_t0_values, size=len(controls), replace=True)
        controls["t0_timestamp"] = sampled_t0
        # [FIX-2a] 强制标准化 group 列
        controls["group"] = "control"
        
        # 校验有效性
        if first_post_map:
            first_ts = controls["user_id_hash"].map(first_post_map)
            last_ts = controls["user_id_hash"].map(last_post_map)
            valid_shadow = (
                (controls["t0_timestamp"] > first_ts + min_pre) &
                (controls["t0_timestamp"] < last_ts - min_post)
            )
            invalid_frac = 1 - valid_shadow.mean()
            if invalid_frac > 0.3:
                logger.warning(f"    {invalid_frac:.1%} controls have invalid shadow_t0")
            controls = controls[valid_shadow].copy()
    
    t0_df = pd.concat([valid_patients, controls], ignore_index=True)
    
    # 保存缓存
    save_path = os.path.join(FEATURES_DIR, "t0_mapping.parquet")
    t0_df.to_parquet(save_path, index=False)
    logger.info(f"  Saved t0 mapping to: {save_path}")
    
    return t0_df


# 兼容旧代码的常量（如果 PSM_CONFIG 不可用）
PSM_CONFIG_MIN_PRE_DAYS = 30
PSM_CONFIG_MIN_POST_DAYS = 30


def _parse_single_t0(value) -> Optional[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    placeholders = {"PENDING", "pending", "nan", "None", "NaT", "NULL", ""}
    if isinstance(value, str) and value.strip() in placeholders:
        return None
    try:
        numeric = float(value)
        if 1e9 < numeric < 2e9: return numeric
        elif 1e12 < numeric < 2e12: return numeric / 1000.0
    except (ValueError, TypeError): pass
    try:
        dt = pd.to_datetime(value)
        if pd.notna(dt): return dt.timestamp()
    except Exception: pass
    return None


def _parse_t0_column(df: pd.DataFrame) -> pd.Series:
    result = pd.Series(np.nan, index=df.index, dtype=float)
    raw = df["t0_date"].copy()
    if "t0_date" not in df.columns:
        return result
    
    placeholders = {"PENDING", "pending", "nan", "None", "NaT", "NULL", ""}
    mask_valid = raw.notna() & ~raw.isin(placeholders)
    raw = raw[mask_valid]
    if len(raw) == 0: return result
    
    numeric = pd.to_numeric(raw, errors='coerce')
    mask_s = (numeric > 1e9) & (numeric < 2e9)
    if mask_s.any(): result.loc[mask_s.index[mask_s]] = numeric[mask_s].values
    
    mask_ms = (numeric > 1e12) & (numeric < 2e12)
    if mask_ms.any(): result.loc[mask_ms.index[mask_ms]] = numeric[mask_ms].values / 1000.0
    
    still_missing = result.isna() & mask_valid
    if still_missing.any():
        try:
            parsed = pd.to_datetime(raw[still_missing], errors='coerce')
            valid_parsed = parsed.notna()
            if valid_parsed.any():
                result.loc[valid_parsed.index] = (parsed[valid_parsed].astype(np.int64) // 10**9).astype(float)
        except Exception: pass
    return result


# ======================== P-FACTOR WEIGHTS & DIRECTION ========================

def load_pfactor_weights() -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    [FIX v1.2] 加载权重和方向。
    权重 = Factor 1 载荷的绝对值 (归一化)
    方向 = Factor 1 载荷的原始符号 (数据驱动，替代先验硬编码)
    """
    logger.info("Loading p-factor weights and directions...")
    
    loadings_path = os.path.join(os.path.dirname(TRAJECTORY_DIR), "stats", "factor_loadings_features.csv")
    if not os.path.exists(loadings_path):
        loadings_path = FACTOR_LOADINGS_PATH
    
    if os.path.exists(loadings_path):
        loadings_df = pd.read_csv(loadings_path, index_col=0)
        if "Factor_1" in loadings_df.columns:
            raw_loadings = loadings_df["Factor_1"].to_dict()
            
            # 权重：绝对值归一化
            weights = {k: abs(v) for k, v in raw_loadings.items()}
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
            
            # [FIX v1.2] 方向：原始符号。正载荷=特征升高时p因子升高=severity加重
            directions = {k: np.sign(v) for k, v in raw_loadings.items()}
            
            n_matched = sum(1 for f in weights.keys() if translate_pfactor_name(f) in TRAJECTORY_FEATURE_SUBSET)
            logger.info(f"  Loaded Factor 1 weights: {len(weights)} features, {n_matched} matched (after translation)")
            return weights, directions
    
    logger.warning("  Factor 1 loadings not found, using equal weights and +1 directions")
    return (
        {f: 1.0 / len(TRAJECTORY_FEATURE_SUBSET) for f in TRAJECTORY_FEATURE_SUBSET},
        {f: 1.0 for f in TRAJECTORY_FEATURE_SUBSET}
    )


# ======================== FEATURE NAME MATCHING ========================

# 模块级缓存，避免重复警告
_match_warning_logged = set()

def _match_feature_name(feat_name: str, features: Dict) -> Optional[str]:
    """
    [FIX-C1] 增强版双向模糊匹配特征名。
    
    新增匹配策略：
      1. 精确匹配
      2. 加后缀 (_rate, _mean, _std, 空串)
      3. 去后缀再匹配
      4. [NEW] 核心词匹配：提取特征核心词（去掉 _score/_ratio/_index）再尝试
    """
    # 1. 精确匹配
    if feat_name in features:
        return feat_name
    
    # 2. 加后缀
    for suffix in ["_rate", "_mean", "_std", ""]:
        candidate = feat_name + suffix
        if candidate in features:
            return candidate
    
    # 3. 去后缀再匹配
    for suffix in ["_rate", "_mean", "_std", "_score", "_ratio", "_index"]:
        if feat_name.endswith(suffix):
            base = feat_name[:-len(suffix)]
            if base in features:
                return base
            for s2 in ["_rate", "_mean", "_std", ""]:
                if base + s2 in features:
                    return base + s2
    
    # 4. [NEW] 核心词匹配
    # 例: "negative_net" 匹配 "negative_net_score_mean"
    CORE_SUFFIXES = ["_score", "_ratio", "_index", "_count"]
    for cs in CORE_SUFFIXES:
        if feat_name.endswith(cs):
            core = feat_name[:-len(cs)]
            for suffix in ["_rate", "_mean", "_std", ""]:
                for cs2 in CORE_SUFFIXES + [""]:
                    candidate = core + cs2 + suffix
                    if candidate in features:
                        return candidate
    
    # 5. [NEW] 未匹配时记录警告（限次输出，避免刷屏）
    if feat_name not in _match_warning_logged:
        _match_warning_logged.add(feat_name)
        logger.warning(f"  [Feature Match] UNMATCHED: '{feat_name}' not found in "
                      f"trajectory features. Available keys sample: "
                      f"{list(features.keys())[:10]}")
    
    return None

# ======================== WINDOW FEATURE EXTRACTION ========================

def extract_window_features(
    window_posts: pd.DataFrame,
    pfactor_weights: Dict[str, float],
    pfactor_directions: Dict[str, float], # [FIX v1.2] 新增方向参数
) -> Optional[Dict]:
    # [FIX P0-1] 同步过滤文本和时间戳，保证索引严格对齐
    valid_rows = window_posts.dropna(subset=["clean_text"])
    texts_raw = valid_rows["clean_text"].values
    timestamps_raw = valid_rows["post_timestamp"].values.astype(float) if "post_timestamp" in valid_rows.columns else None
    
    post_features = []
    feat_timestamps = []  # 新增：仅追踪成功提取特征的时间戳
    total_words = 0
    
    for i, text in enumerate(texts_raw):
        if not isinstance(text, str) or len(text.strip()) < 10:
            continue
        pf = extract_linguistic_features_post(text)
        if pf is not None:
            post_features.append(pf)
            total_words += pf.get("n_words", 0)
            if timestamps_raw is not None:
                feat_timestamps.append(timestamps_raw[i])
    
    n_posts = len(post_features)
    # [REVIEW] 提取帖子时间戳，用于时间加权惯性 (现在与post_features严格对齐)
    post_timestamps = np.array(feat_timestamps) if timestamps_raw is not None else None
    
    if n_posts < MIN_POSTS_PER_WINDOW or total_words < MIN_WORDS_PER_WINDOW:
        return None
    
    def sum_and_rate(key: str) -> float:
        total_val = 0
        total_w = 0
        for pf in post_features:
            if key not in pf:
                continue
            val = pf[key]
            w = pf.get("n_words", 0)
            if val is not None and not (isinstance(val, float) and np.isnan(val)) and w > 0:
                total_val += val      # 只累加计数
                total_w += w           # 累加词数
        return safe_divide(total_val, total_w, 0) * 100.0
    
    def weighted_mean_of(key: str) -> float:
        weighted_sum = 0.0
        total_weight = 0.0
        for pf in post_features:
            if key not in pf:  # 核心修复：特征不存在时跳过，而非填0
                continue
            val = pf[key]
            weight = pf.get("n_words", 0)
            if val is not None and not (isinstance(val, float) and np.isnan(val)) and weight > 0:
                weighted_sum += val * weight
                total_weight += weight
        return safe_divide(weighted_sum, total_weight, np.nan)
    
    def std_of(key: str) -> float:
        vals = [pf.get(key, 0) for pf in post_features if pf.get(key) is not None and not (isinstance(pf.get(key), float) and np.isnan(pf.get(key)))]
        return np.std(vals) if len(vals) > 1 else 0.0
    
    features = {
        "n_posts": n_posts,
        "total_words": total_words,
        "avg_words_per_post": total_words / n_posts,
        # [FIX v1.2] 增加高质量窗口标记
        "is_high_quality": n_posts >= MIN_POSTS_HIGH_QUALITY and total_words >= MIN_WORDS_HIGH_QUALITY,
    }
    
    for feat_name, agg_type in FEATURE_AGGREGATION_MAP.items():
        output_name = f"{feat_name}_{agg_type}"
        if agg_type == "rate":
            features[output_name] = sum_and_rate(feat_name)
        elif agg_type == "mean":
            features[output_name] = weighted_mean_of(feat_name)
    
    features["vader_between_post_std"] = std_of("vader_compound") # [FIX P1-1] 重命名：跨帖标准差
    features["vader_compound_min_mean"] = weighted_mean_of("vader_compound_min")
    features["vader_within_post_volatility"] = weighted_mean_of("vader_compound_std") # 帖内波动均值
    
    # [FIX v1.2] 情感惯性计算：引入时间间隔权重
    vader_series = [pf.get("vader_compound", 0) for pf in post_features]
    if len(vader_series) >= MIN_POSTS_PER_WINDOW * 2:  # 至少6-10条帖子才算有效惯性
        vader_arr = np.array(vader_series, dtype=float)
        vader_var = np.var(vader_arr)
        if vader_var > 1e-10:
            # 时间加权自相关：相邻帖子间隔越近，惯性权重越大
            time_diffs_days = np.diff(post_timestamps[:len(vader_arr)]) / 86400.0
            # 间隔30天以上的帖子关联度衰减为近0
            weights_ts = np.exp(-time_diffs_days / 30.0) 
            autocorr = np.average(
                (vader_arr[:-1] - np.mean(vader_arr)) * (vader_arr[1:] - np.mean(vader_arr)),
                weights=weights_ts
            ) / vader_var
            features["emotion_inertia_window"] = float(np.clip(autocorr, -1, 1))
        else:
            features["emotion_inertia_window"] = 0.0
    else:
        features["emotion_inertia_window"] = np.nan
    
    features["posting_intensity"] = n_posts / WINDOW_SIZE_DAYS  
    
    return features


# ======================== P-FACTOR SEVERITY ========================

def _compute_pfactor_severity(
    features: Dict,
    pfactor_weights: Dict[str, float],
    pfactor_directions: Dict[str, float], # [FIX v1.2] 新增
    user_baseline_stats: Optional[Dict[str, Tuple[float, float]]] = None,
) -> float:
    """
    [FIX v1.2] 计算窗口级 p-factor severity score。
    方向由 Factor 1 原始载荷符号自动决定，消除先验硬编码的偏差。
    """
    score = 0.0
    total_weight = 0.0
    matched_count = 0
    used_trajectory_features = set()
    for feat_name, weight in pfactor_weights.items():
        # [FIX] 翻译特征名：将 Factor Loading 名映射为 Trajectory 特征名
        trajectory_feat_name = translate_pfactor_name(feat_name)
        if trajectory_feat_name in used_trajectory_features:
            continue  # 跳过重复映射
        used_trajectory_features.add(trajectory_feat_name)
        # 然后用翻译后的名字去匹配
        matched_name = _match_feature_name(trajectory_feat_name, features)
        if matched_name is None:
            continue
            
        value = features[matched_name]
        matched_count += 1
        
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        
        # Step 1: Z-score
        if user_baseline_stats and matched_name in user_baseline_stats:
            bl_mean, bl_std = user_baseline_stats[matched_name]
            if bl_std > 1e-10:
                z_value = (value - bl_mean) / bl_std
            else:
                z_value = 0.0
        else:
            continue # 无基线时跳过，避免量纲灾难
        
        # Step 2: 方向统一 (由数据驱动)
        direction = pfactor_directions.get(feat_name, 1.0)
        directed_z = z_value * direction
        
        # Step 3: 加权
        score += weight * directed_z
        total_weight += weight
    
    # [FIX P1-3] 监控匹配率并输出警告
    if matched_count < len(pfactor_weights) * 0.8:
        logger.warning(f"  [P-factor] Low feature match rate: {matched_count}/{len(pfactor_weights)} "
                       f"({matched_count/len(pfactor_weights):.1%}). P-factor score may be unreliable.")
    
    result = safe_divide(score, total_weight, np.nan)
    # [FIX P1-2] 显式记录无基线导致的NaN，便于排查
    if np.isnan(result) and matched_count > 0:
        logger.warning(f"  [P-factor] Computed as NaN despite {matched_count} matched features. "
                       f"Likely due to missing baseline_stats for z-score standardization.")
    return result


# ======================== USER TRAJECTORY BUILDER ========================

def build_user_trajectory(
    user_id: str,
    user_posts: pd.DataFrame,
    t0_timestamp: float,
    pfactor_weights: Dict[str, float],
    pfactor_directions: Dict[str, float], # [FIX v1.2]
) -> Optional[List[Dict]]:
    user_posts = user_posts.sort_values("post_timestamp").reset_index(drop=True)
    
    timestamps = user_posts["post_timestamp"].values.astype(float)
    valid_ts = (timestamps > 1e9) & (timestamps < 2e9)
    if valid_ts.sum() < MIN_POSTS_PER_WINDOW:
        return None
    
    user_posts = user_posts[valid_ts].copy()
    timestamps = user_posts["post_timestamp"].values.astype(float)
    
    t0_s = int(t0_timestamp)  # 强制整型，消除浮点精度引起的窗口错位
    window_size_s = int(WINDOW_SIZE_DAYS * 86400)
    window_step_s = int(WINDOW_STEP_DAYS * 86400)
    
    EPOCH_REF = 946684800
    offset_from_ref = t0_s - EPOCH_REF
    steps_from_ref = offset_from_ref / window_step_s
    t0_window_start_idx = int(np.floor(steps_from_ref))
    
    n_pre_windows = TRAJECTORY_PRE_DAYS // WINDOW_STEP_DAYS
    n_post_windows = TRAJECTORY_POST_DAYS // WINDOW_STEP_DAYS
    
    first_window_idx = t0_window_start_idx - n_pre_windows
    last_window_idx = t0_window_start_idx + n_post_windows
    
    windows = []
    
    for abs_idx in range(first_window_idx, last_window_idx + 1):
        window_start_ts = EPOCH_REF + abs_idx * window_step_s
        window_end_ts = window_start_ts + window_size_s
        
        window_index = abs_idx - t0_window_start_idx
        days_from_t0 = (window_start_ts - t0_s) / 86400.0
        days_from_t0_center = days_from_t0 + WINDOW_SIZE_DAYS / 2.0
        
        mask = (timestamps >= window_start_ts) & (timestamps < window_end_ts)
        window_posts = user_posts[mask]
        
        # [FIX v1.2] 传入方向字典
        window_feats = extract_window_features(window_posts, pfactor_weights, pfactor_directions)
        
        # [FIX v1.2] 明确划分 Phase: pre / transition / post
        window_start = days_from_t0
        window_end = days_from_t0 + WINDOW_SIZE_DAYS

        if window_end <= 0:  # 窗口完全在 t0 之前
            phase = "pre"
        elif window_start >= 0:  # 窗口完全在 t0 之后（包含从 t0 开始的窗口）
            phase = "post"
        else:  # 窗口跨越 t0
            phase = "transition"
            
        window_record = {
            "user_id_hash": user_id,
            "t0_timestamp": t0_s,
            "window_index": window_index,
            "days_from_t0": days_from_t0,
            "days_from_t0_center": days_from_t0_center,
            "window_start_ts": window_start_ts,
            "window_end_ts": window_end_ts,
            "n_posts_in_window": len(window_posts),
            "is_reliable": window_feats is not None,
            "phase": phase, 
            "calendar_year": int(datetime.fromtimestamp(window_start_ts, tz=timezone.utc).year) # [FIX P2-2] 替换弃用方法
        }
        
        if window_feats is not None:
            window_record.update(window_feats)
        else:
            # TRAJECTORY_FEATURE_SUBSET 中的元素已经是带后缀的完整列名(如 anxiety_score_rate)
            for feat_name in TRAJECTORY_FEATURE_SUBSET:
                window_record[feat_name] = np.nan
            # 补充 extract_window_features 中额外生成的非 SUBSET 特征
            window_record["vader_between_post_std"] = np.nan  # [FIX P1-1] 与重命名保持一致
            window_record["vader_compound_min_mean"] = np.nan
            window_record["vader_within_post_volatility"] = np.nan
            window_record["posting_intensity"] = 0
            window_record["emotion_inertia_window"] = np.nan
            window_record["is_high_quality"] = False
        
        windows.append(window_record)
    
    if not windows:
        return None
    
    n_reliable = sum(1 for w in windows if w["is_reliable"])
    if n_reliable < MIN_TOTAL_RELIABLE_WINDOWS:
        return None
    
    pre_windows = [w for w in windows if w["phase"] == "pre"]
    post_windows = [w for w in windows if w["phase"] == "post"]
    
    n_pre_reliable = sum(1 for w in pre_windows if w["is_reliable"])
    n_post_reliable = sum(1 for w in post_windows if w["is_reliable"])
    
    for w in windows:
        w["n_reliable_windows_total"] = n_reliable
        w["n_pre_reliable_windows"] = n_pre_reliable
        w["n_post_reliable_windows"] = n_post_reliable
        w["has_sufficient_pre"] = n_pre_reliable >= MIN_PRE_WINDOWS
        w["has_sufficient_post"] = n_post_reliable >= MIN_POST_WINDOWS
    
    return windows


def _compute_window_temporal_social_features(posts: List[Dict]) -> Dict:
    """复刻 s01 逻辑，计算窗口内的时间节律、社交互动和发帖行为特征"""
    if not posts:
        return {}
    
    timestamps = np.array([p["post_timestamp"] for p in posts], dtype=float)
    n_posts = len(timestamps)
    result = {}
    
    # === 1. 时间节律 ===
    hours = (timestamps % 86400) / 3600.0
    rads = hours * (2 * pi / 24)
    sin_mean = np.mean(np.sin(rads))
    cos_mean = np.mean(np.cos(rads))
    
    mean_angle = np.arctan2(sin_mean, cos_mean)
    circadian_mean_hour = mean_angle * (24 / (2 * pi))
    if circadian_mean_hour < 0: circadian_mean_hour += 24.0
    result['circadian_mean_hour'] = circadian_mean_hour
    result['circadian_concentration'] = np.sqrt(sin_mean**2 + cos_mean**2)
    result['night_activity_ratio'] = np.mean((hours >= 23) | (hours < 5))
    
    days_of_week = np.array([datetime.utcfromtimestamp(int(ts)).weekday() for ts in timestamps])
    result['weekend_activity_ratio'] = np.mean(days_of_week >= 5)
    
    weekday_mask = days_of_week < 5
    if np.sum(weekday_mask) > 0 and np.sum(~weekday_mask) > 0:
        wkd_rads = hours[weekday_mask] * (2 * pi / 24)
        wke_rads = hours[~weekday_mask] * (2 * pi / 24)
        wkd_sin, wkd_cos = np.mean(np.sin(wkd_rads)), np.mean(np.cos(wkd_rads))
        wke_sin, wke_cos = np.mean(np.sin(wke_rads)), np.mean(np.cos(wke_rads))  # 修复 wwe_cos
        wkd_mean_hour = np.arctan2(wkd_sin, wkd_cos) * (24 / (2 * pi))
        if wkd_mean_hour < 0: wkd_mean_hour += 24.0
        wke_mean_hour = np.arctan2(wke_sin, wke_cos) * (24 / (2 * pi))  # 修复 wwe_cos
        if wke_mean_hour < 0: wke_mean_hour += 24.0
        diff = wke_mean_hour - wkd_mean_hour
        if diff > 12: diff -= 24
        elif diff < -12: diff += 24
        result['social_jetlag_hours'] = diff
    else:
        result['social_jetlag_hours'] = np.nan
        
    result['sleep_disruption_index'] = np.mean((hours >= 0) & (hours < 6))
    
    sorted_ts = np.sort(timestamps)
    intervals = np.diff(sorted_ts).astype(np.float64)
    intervals = intervals[intervals > 0]
    if len(intervals) > 1:
        interval_mean = np.mean(intervals)
        interval_std = np.std(intervals)
        interval_median = np.median(intervals)
        result['posting_regularity_cv'] = interval_std / interval_mean if interval_mean > 0 else np.nan
        result['posting_burstiness'] = (interval_std - interval_mean) / (interval_std + interval_mean + 1e-10)
        result['interval_kurtosis'] = kurtosis(intervals, fisher=True)
        result['interval_median_hours'] = interval_median / 3600.0
        result['interval_mean_hours'] = interval_mean / 3600.0
    else:
        for k in ['posting_regularity_cv', 'posting_burstiness', 'interval_kurtosis', 'interval_median_hours', 'interval_mean_hours']:
            result[k] = np.nan
            
    hour_bins, _ = np.histogram(hours, bins=24, range=(0, 24))
    hour_probs = hour_bins / hour_bins.sum() if hour_bins.sum() > 0 else np.ones(24) / 24
    result['hourly_entropy'] = entropy(hour_probs + 1e-10)
    
    dow_bins, _ = np.histogram(days_of_week, bins=7, range=(0, 7))
    dow_probs = dow_bins / dow_bins.sum() if dow_bins.sum() > 0 else np.ones(7) / 7
    result['weekly_entropy'] = entropy(dow_probs + 1e-10)
    result['temporal_diversity_index'] = (result['hourly_entropy'] + result['weekly_entropy']) / 2
    
    # === 2. 发帖行为 ===
    min_ts, max_ts = timestamps.min(), timestamps.max()
    tenure_days = (max_ts - min_ts) / 86400.0
    result['tenure_days'] = tenure_days
    result['n_posts_total'] = n_posts
    
    active_dates = {int(ts) // 86400 for ts in timestamps}
    n_active_days = len(active_dates)
    result['n_active_days'] = n_active_days
    result['active_day_ratio'] = n_active_days / max(1, int(tenure_days) + 1) if tenure_days > 0 else 1.0
    result['posts_per_day'] = n_posts / max(1.0, tenure_days)
    result['posts_per_active_day'] = n_posts / max(1, n_active_days)
    
    sorted_days = sorted(active_dates)
    if len(sorted_days) > 1:
        day_gaps = np.diff(sorted_days)
        result['max_gap_days'] = int(np.max(day_gaps))
        result['mean_gap_days'] = float(np.mean(day_gaps))
        result['gap_regularity_cv'] = np.std(day_gaps) / np.mean(day_gaps) if np.mean(day_gaps) > 0 else 0.0
    else:
        result['max_gap_days'], result['mean_gap_days'], result['gap_regularity_cv'] = 0, 0.0, 0.0
        
    # === 3. 社交互动 ===
    scores = np.array([p.get("score", 0) for p in posts], dtype=float)
    data_types = np.array([p.get("data_type", "") for p in posts])
    
    submission_mask = (data_types == "submission")
    n_submissions = int(np.sum(submission_mask))
    n_comments = n_posts - n_submissions
    
    result['submission_ratio'] = n_submissions / n_posts if n_posts > 0 else 0
    result['n_submissions'] = n_submissions
    result['n_comments'] = n_comments
    
    if len(scores) > 0:
        result['score_mean'] = float(np.mean(scores))
        result['score_median'] = float(np.median(scores))
        result['score_std'] = float(np.std(scores)) if len(scores) > 1 else 0.0
        result['score_max'] = float(np.max(scores))
        result['score_min'] = float(np.min(scores))
        result['score_range'] = result['score_max'] - result['score_min']
        result['neg_score_ratio'] = float(np.mean(scores < 0))
        result['high_engagement_ratio'] = float(np.mean(scores > np.percentile(scores, 90))) if len(scores) > 10 else 0.0
    else:
        for k in ['score_mean', 'score_median', 'score_std', 'score_max', 'score_min', 'score_range', 'neg_score_ratio', 'high_engagement_ratio']:
            result[k] = np.nan
            
    num_comments_arr = np.array([p.get("num_comments", 0) for p in posts], dtype=float)
    if n_submissions > 0:
        comments_received = num_comments_arr[submission_mask]
        result['avg_comments_received'] = float(np.mean(comments_received))
        result['total_comments_received'] = int(np.sum(comments_received))
        result['discussion_initiated_ratio'] = float(np.mean(comments_received > 0))
    else:
        result['avg_comments_received'], result['total_comments_received'], result['discussion_initiated_ratio'] = np.nan, np.nan, np.nan
        
    is_top_level_arr = np.array([p.get("is_top_level", False) for p in posts])
    if n_comments > 0:
        comment_mask = ~submission_mask
        if np.any(comment_mask):
            top_level_ratio = float(np.mean(is_top_level_arr[comment_mask]))
            result['top_level_comment_ratio'] = top_level_ratio
            result['avg_comment_depth'] = 1.0 if top_level_ratio > 0.5 else 2.0
        else:
            result['top_level_comment_ratio'], result['avg_comment_depth'] = np.nan, np.nan
    else:
        result['top_level_comment_ratio'], result['avg_comment_depth'] = np.nan, np.nan
        
    controversiality_arr = np.array([p.get("controversiality", 0) for p in posts])
    result['controversy_rate'] = float(np.mean(controversiality_arr > 0))
    
    gilded_arr = np.array([p.get("gilded", 0) for p in posts])
    result['gilded_rate'] = gilded_arr.sum() / n_posts if n_posts > 0 else 0
    
    # === 4. 社区分布 ===
    try:
        from Module1.config_module1 import SUBREDDIT_CATEGORIES, MENTAL_HEALTH_SUBREDDITS
    except ImportError:
        SUBREDDIT_CATEGORIES, MENTAL_HEALTH_SUBREDDITS = {}, set()
        
    subreddits = np.array([str(p.get("subreddit", "unknown")).lower() for p in posts])
    sub_counter = Counter(subreddits)
    total_sub_posts = sum(sub_counter.values())
    if total_sub_posts > 0:
        sub_probs = np.array(list(sub_counter.values())) / total_sub_posts
        result['subreddit_entropy'] = float(entropy(sub_probs + 1e-10))
        result['subreddit_herfindahl'] = float(np.sum(sub_probs ** 2))
        result['n_unique_subreddits'] = len(sub_counter)
        
        mh_posts = sum(1 for s in subreddits if s in MENTAL_HEALTH_SUBREDDITS)
        result['mental_health_sub_ratio'] = mh_posts / n_posts
        
        category_counts = defaultdict(int)
        for s in subreddits:
            assigned = False
            for cat, subs_set in SUBREDDIT_CATEGORIES.items():
                if s in subs_set:
                    category_counts[cat] += 1
                    assigned = True
                    break
            if not assigned: category_counts['other'] += 1
            
        result['support_sub_ratio'] = category_counts.get('support', 0) / n_posts
        result['entertainment_sub_ratio'] = category_counts.get('entertainment', 0) / n_posts
        result['information_sub_ratio'] = category_counts.get('information', 0) / n_posts
        result['social_sub_ratio'] = category_counts.get('social', 0) / n_posts
        result['other_sub_ratio'] = category_counts.get('other', 0) / n_posts
    else:
        for k in ['subreddit_entropy', 'subreddit_herfindahl', 'n_unique_subreddits', 'mental_health_sub_ratio', 'support_sub_ratio', 'entertainment_sub_ratio', 'information_sub_ratio', 'social_sub_ratio', 'other_sub_ratio']:
            result[k] = np.nan

    # 对数转换
    log_cols_map = {'posts_per_day': 'posts_per_day_log', 'posts_per_active_day': 'posts_per_active_day_log', 'interval_median_hours': 'interval_median_hours_log', 'max_gap_days': 'max_gap_days_log', 'tenure_days': 'tenure_days_log', 'n_posts_total': 'n_posts_total_log'}
    for orig, log_c in log_cols_map.items():
        if orig in result and not np.isnan(result[orig]):
            result[log_c] = np.log1p(max(0, result[orig]))
        else:
            result[log_c] = np.nan
            
    if 'score_mean' in result and not np.isnan(result['score_mean']):
        result['score_mean_log_signed'] = np.sign(result['score_mean']) * np.log1p(np.abs(result['score_mean']))
    else:
        result['score_mean_log_signed'] = np.nan
        
    return result

def extract_window_features_from_cache(
    cached_posts: List[Dict],
    pfactor_weights: Dict[str, float],
    pfactor_directions: Dict[str, float],
) -> Optional[Dict]:
    """从预提取的帖子特征聚合窗口特征（零NLP调用）"""
    valid_posts = [p for p in cached_posts if p["features"] is not None]
    n_posts = len(valid_posts)
    
    if n_posts < MIN_POSTS_PER_WINDOW:
        return None
    
    total_words = sum(p["n_words"] for p in valid_posts)
    if total_words < MIN_WORDS_PER_WINDOW:
        return None
    
    post_features = [p["features"] for p in valid_posts]
    post_timestamps = np.array([p["post_timestamp"] for p in valid_posts], dtype=float)
    
    def sum_and_rate(key: str) -> float:
        vals_weights = [(pf.get(key), pf.get("n_words", 0)) for pf in post_features if key in pf]
        if not vals_weights: return np.nan
        total_val = sum(v * w for v, w in vals_weights if v is not None and not (isinstance(v, float) and np.isnan(v)))
        total_w = sum(w for v, w in vals_weights if v is not None and not (isinstance(v, float) and np.isnan(v)))
        return safe_divide(total_val, total_w, 0) * 100.0

    def weighted_mean_of(key: str) -> float:
        weighted_sum, total_weight = 0.0, 0.0
        for pf in post_features:
            if key not in pf: continue
            val, weight = pf[key], pf.get("n_words", 0)
            if val is not None and not (isinstance(val, float) and np.isnan(val)) and weight > 0:
                weighted_sum += val * weight
                total_weight += weight
        return safe_divide(weighted_sum, total_weight, np.nan)
    
    def std_of(key: str) -> float:
        vals = [pf.get(key, 0) for pf in post_features if pf.get(key) is not None and not (isinstance(pf.get(key), float) and np.isnan(pf.get(key)))]
        return np.std(vals) if len(vals) > 1 else 0.0
    
    features = {
        "n_posts": n_posts,
        "total_words": total_words,
        "avg_words_per_post": total_words / n_posts,
        "is_high_quality": n_posts >= MIN_POSTS_HIGH_QUALITY and total_words >= MIN_WORDS_HIGH_QUALITY,
    }
    
    for feat_name, agg_type in FEATURE_AGGREGATION_MAP.items():
        output_name = f"{feat_name}_{agg_type}"
        if agg_type == "rate":
            features[output_name] = sum_and_rate(feat_name)
        elif agg_type == "mean":
            features[output_name] = weighted_mean_of(feat_name)
    
    features["vader_between_post_std"] = std_of("vader_compound")  # [FIX P1-1] 与主提取函数重命名保持一致
    features["vader_compound_min_mean"] = weighted_mean_of("vader_compound_min")
    features["vader_within_post_volatility"] = weighted_mean_of("vader_compound_std")
    
    vader_series = [pf.get("vader_compound", 0) for pf in post_features]
    if len(vader_series) > 3 and len(post_timestamps) > 0:
        vader_arr = np.array(vader_series, dtype=float)
        vader_var = np.var(vader_arr)
        if vader_var > 1e-10:
            time_diffs_days = np.diff(post_timestamps[:len(vader_arr)]) / 86400.0
            weights_ts = np.exp(-time_diffs_days / 30.0)
            autocorr = np.average(
                (vader_arr[:-1] - np.mean(vader_arr)) * (vader_arr[1:] - np.mean(vader_arr)),
                weights=weights_ts
            ) / vader_var
            features["emotion_inertia_window"] = float(np.clip(autocorr, -1, 1))
        else:
            features["emotion_inertia_window"] = 0.0
    else:
        features["emotion_inertia_window"] = np.nan
    
    features["posting_intensity"] = n_posts / WINDOW_SIZE_DAYS  
    # --- 补充语言学维度的缺失特征 ---
    features['ari_std'] = std_of('ari_score')
    word_counts = [pf['n_words'] for pf in post_features]
    features['post_length_mean'] = float(np.mean(word_counts))
    features['post_length_std'] = float(np.std(word_counts)) if len(word_counts) > 1 else 0.0
    features['post_length_cv'] = safe_divide(features['post_length_std'], features['post_length_mean'], 0)
    features['self_focus_variability'] = std_of('self_focus_ratio')
    features['social_focus_variability'] = std_of('social_focus_ratio')
    features['vader_compound_last_mean'] = weighted_mean_of('vader_compound_last')
    features['url_usage_rate'] = sum_and_rate('n_urls')
    
    # --- 补充时间、行为、社交维度特征 ---
    # [FIX] 行为特征应包含所有帖子(包括短帖子)，而不是仅包含提取了语言学特征的帖子
    features.update(_compute_window_temporal_social_features(cached_posts))
    
    return features


def build_user_trajectory_cached(
    user_id: str,
    pre_extracted: List[Dict],
    t0_timestamp: float,
    pfactor_weights: Dict[str, float],
    pfactor_directions: Dict[str, float],
) -> Optional[List[Dict]]:
    """基于预提取特征的轨迹构建（避免重复NLP调用）"""
    timestamps = np.array([p["post_timestamp"] for p in pre_extracted], dtype=float)
    valid_ts = (timestamps > 1e9) & (timestamps < 2e9)
    
    if valid_ts.sum() < MIN_POSTS_PER_WINDOW:
        return None
    
    valid_posts = [p for p, v in zip(pre_extracted, valid_ts) if v]
    timestamps = timestamps[valid_ts]
    
    t0_s = t0_timestamp
    window_size_s = WINDOW_SIZE_DAYS * 86400
    window_step_s = WINDOW_STEP_DAYS * 86400
    
    EPOCH_REF = 946684800
    offset_from_ref = t0_s - EPOCH_REF
    steps_from_ref = offset_from_ref / window_step_s
    t0_window_start_idx = int(np.floor(steps_from_ref))
    
    n_pre_windows = TRAJECTORY_PRE_DAYS // WINDOW_STEP_DAYS
    n_post_windows = TRAJECTORY_POST_DAYS // WINDOW_STEP_DAYS
    
    first_window_idx = t0_window_start_idx - n_pre_windows
    last_window_idx = t0_window_start_idx + n_post_windows
    
    windows = []
    
    for abs_idx in range(first_window_idx, last_window_idx + 1):
        window_start_ts = EPOCH_REF + abs_idx * window_step_s
        window_end_ts = window_start_ts + window_size_s
        
        window_index = abs_idx - t0_window_start_idx
        days_from_t0 = (window_start_ts - t0_s) / 86400.0
        days_from_t0_center = days_from_t0 + WINDOW_SIZE_DAYS / 2.0
        
        # 基于时间戳切片预提取特征（核心加速点）
        mask = (timestamps >= window_start_ts) & (timestamps < window_end_ts)
        window_cached = [valid_posts[i] for i in range(len(mask)) if mask[i]]
        
        window_feats = extract_window_features_from_cache(window_cached, pfactor_weights, pfactor_directions)
        
        window_start = days_from_t0
        window_end = days_from_t0 + WINDOW_SIZE_DAYS

        if window_end <= 0:
            phase = "pre"
        elif window_start >= 0:
            phase = "post"
        else:
            phase = "transition"
            
        window_record = {
            "user_id_hash": user_id,
            "t0_timestamp": t0_s,
            "window_index": window_index,
            "days_from_t0": days_from_t0,
            "days_from_t0_center": days_from_t0_center,
            "window_start_ts": window_start_ts,
            "window_end_ts": window_end_ts,
            "n_posts_in_window": len(window_cached),
            "is_reliable": window_feats is not None,
            "phase": phase, 
            "calendar_year": int(datetime.fromtimestamp(window_start_ts, tz=timezone.utc).year) # [FIX] 替换弃用方法
        }
        
        if window_feats is not None:
            window_record.update(window_feats)
        else:
            for feat_name in TRAJECTORY_FEATURE_SUBSET:
                window_record[feat_name] = np.nan
            window_record["vader_between_post_std"] = np.nan  
            window_record["vader_compound_min_mean"] = np.nan
            window_record["vader_within_post_volatility"] = np.nan
            window_record["posting_intensity"] = 0
            window_record["emotion_inertia_window"] = np.nan
            window_record["is_high_quality"] = False
            
            # [NEW] 填补行为、社交等新增特征的 NaN
            for feat_name in BEHAVIORAL_FEATURES:
                window_record[feat_name] = np.nan
        
        windows.append(window_record)
    
    if not windows:
        return None
    
    n_reliable = sum(1 for w in windows if w["is_reliable"])
    if n_reliable < MIN_TOTAL_RELIABLE_WINDOWS:
        return None
    
    pre_windows = [w for w in windows if w["phase"] == "pre"]
    post_windows = [w for w in windows if w["phase"] == "post"]
    
    n_pre_reliable = sum(1 for w in pre_windows if w["is_reliable"])
    n_post_reliable = sum(1 for w in post_windows if w["is_reliable"])
    
    for w in windows:
        w["n_reliable_windows_total"] = n_reliable
        w["n_pre_reliable_windows"] = n_pre_reliable
        w["n_post_reliable_windows"] = n_post_reliable
        w["has_sufficient_pre"] = n_pre_reliable >= MIN_PRE_WINDOWS
        w["has_sufficient_post"] = n_post_reliable >= MIN_POST_WINDOWS
    
    return windows



def _z_score_value(value: float, baseline: Tuple[float, float]) -> float:
    if isinstance(value, float) and np.isnan(value): return np.nan
    bl_mean, bl_std = baseline
    if bl_std > 1e-10: return (value - bl_mean) / bl_std
    return 0.0


# ======================== PARTITION PROCESSOR ========================

def process_partition_trajectories(
    partition_path: str,
    t0_map: Dict[str, float],
    user_groups: Dict[str, Dict],
    pfactor_weights: Dict[str, float],
    pfactor_directions: Dict[str, float],
) -> List[Dict]:
    logger.info(f"Processing {os.path.basename(partition_path)}")
    init_worker()
    
    import pyarrow.parquet as pq
    parquet_files = [f for f in os.listdir(partition_path) if f.endswith('.parquet')]
    if not parquet_files: return []
    
    # [优化] 使用轻量级字典代替 groupby 存储原始数据，降低 80% 内存占用
    user_posts_dict = defaultdict(list)
    
    for pf_name in parquet_files:
        pf_path = os.path.join(partition_path, pf_name)
        try:
            pf = pq.ParquetFile(pf_path)
            for batch in pf.iter_batches(batch_size=BATCH_SIZE):
                df = batch.to_pandas()
                if "user_id_hash" not in df.columns: continue
                mask = df["user_id_hash"].isin(t0_map.keys())
                df = df[mask]
                if len(df) == 0: continue
                
                # [FIX F] 使用 to_dict('records') 替代 iterrows，大幅提升性能
                needed_cols = ["user_id_hash", "post_timestamp", "clean_text", "score", "data_type", "is_top_level", "num_comments", "subreddit", "controversiality", "gilded"]
                existing_cols = [c for c in needed_cols if c in df.columns]
                records = df[existing_cols].to_dict('records')
                
                for record in records:
                    ts = record.get("post_timestamp")
                    if pd.isna(ts): continue  # [加固] 跳过极少数时间戳异常的脏数据，防止 int() 转换崩溃
                    
                    user_posts_dict[record["user_id_hash"]].append(
                        (
                            int(ts), 
                            record.get("clean_text", ""),
                            record.get("score", 0),
                            record.get("data_type", ""),
                            record.get("is_top_level", False),
                            record.get("num_comments", 0),
                            record.get("subreddit", "unknown"),
                            record.get("controversiality", 0),
                            record.get("gilded", 0)
                        )
                    )
                del df, records
        except Exception as e:
            logger.error(f"Error reading {pf_name}: {e}")
            continue
    
    all_windows = []
    
    # 将 for uid, posts_list in user_posts_dict.items():
    # 改为先提取 keys，再 pop
    user_ids = list(user_posts_dict.keys())
    for uid in tqdm(user_ids, desc=f"Trajectories {os.path.basename(partition_path)}"):
        posts_list = user_posts_dict.pop(uid, None)  # [FIX] 彻底从字典中弹出，释放内存
        if posts_list is None or uid not in t0_map: 
            continue
            
        try:
            t0_ts = t0_map[uid]
            posts_list.sort(key=lambda x: x[0])  # 按时间排序
            
            # [优化] 即用即提取，提取完立即释放文本内存
            pre_extracted = []
            for p_tuple in posts_list:
                ts, text, score, dtype, is_top, num_com, sub, contr, gilded = p_tuple
                if isinstance(text, str) and len(text.strip()) >= 10:
                    pf = extract_linguistic_features_post(text)
                else:
                    pf = None
                pre_extracted.append({
                    "post_timestamp": ts,
                    "features": pf,
                    "n_words": pf.get("n_words", 0) if pf else 0,
                    "score": score,
                    "data_type": dtype,
                    "is_top_level": is_top,
                    "num_comments": num_com,
                    "subreddit": sub,
                    "controversiality": contr,
                    "gilded": gilded
                })
            
            windows = build_user_trajectory_cached(uid, pre_extracted, t0_ts, pfactor_weights, pfactor_directions)
            
            if windows:
                meta = user_groups.get(uid, {})
                for w in windows:
                    w["group"] = meta.get("group", "unknown")
                    w["disorder_group"] = meta.get("disorder_group", "unknown")
                    w["pair_id"] = meta.get("pair_id", None)
                    w["iptw_att_weight"] = meta.get("iptw_att_weight", np.nan)
                    w["eb_weight"] = meta.get("eb_weight", np.nan)
                all_windows.extend(windows)
                
            del pre_extracted, posts_list, windows  # 显式释放
        except Exception as e:
            logger.error(f"User {uid[:8]} trajectory failed: {e}")
            if 'posts_list' in locals(): del posts_list
    
    del user_posts_dict
    gc.collect()
    return all_windows


# ======================== P-FACTOR NORMALIZATION ========================

def compute_derived_indices(df: pd.DataFrame, pfactor_weights: Dict, pfactor_directions: Dict) -> pd.DataFrame:
    """
    [FIX v3.0] 完全向量化计算衍生指标，消除 iterrows 和逐行字典操作。
    执行顺序：
      1. 向量化计算个人/全局基线及收缩估计 (仅基于可靠窗口)
      2. [NEW] 提前执行向量化 MNAR 插补，确保 z-score 和 p-factor 能正确利用插补数据
      3. 向量化 Z-score
      4. 向量化 P-factor 加权求和及波动性/社交退缩指数
    """
    logger.info("Computing derived indices (fully vectorized)...")
    df = df.copy()
    
    # --- 0. 建立 P-factor 特征映射 (base_name -> df_col_name) ---
    df_cols_set = set(df.columns)
    feat_map = {}
    for base_feat in pfactor_weights.keys():
        trajectory_feat_name = translate_pfactor_name(base_feat)
        matched_name = _match_feature_name(trajectory_feat_name, {c: 0 for c in df_cols_set})
        if matched_name is not None:
            feat_map[base_feat] = matched_name

    target_cols = list(set(feat_map.values()))
    
    # 强制将社交退缩计算所需的特征加入 target_cols
    withdraw_required_cols = ['second_person_score_rate', 'posting_intensity']
    for c in withdraw_required_cols:
        if c in df_cols_set:
            target_cols.append(c)
    target_cols = list(set(target_cols)) # 去重
    
    if not target_cols:
        logger.warning("No P-factor target columns found in dataframe!")
        return df
        
    logger.info(f"  [Derived Indices] Matched {len(feat_map)}/{len(pfactor_weights)} p-factor features. Target cols: {len(target_cols)}")

    # --- 1. 向量化基线计算 (必须仅使用真实可靠的窗口) ---
    baseline_mask_strict = df['is_reliable'] & (df['days_from_t0_center'] > -360) & (df['days_from_t0_center'] < -270)
    baseline_mask_fallback = df['is_reliable'] & (df['days_from_t0_center'] < -360)
    
    strict_agg = df[baseline_mask_strict].groupby('user_id_hash')[target_cols].agg(['mean', 'std', 'count'])
    fallback_agg = df[baseline_mask_fallback].groupby('user_id_hash')[target_cols].agg(['mean', 'std', 'count'])
    
    all_baseline_users = strict_agg.index.union(fallback_agg.index)
    
    def _extract_aligned(agg, level, idx):
        if agg.empty:
            return pd.DataFrame(np.nan, index=idx, columns=target_cols)
        return agg.xs(level, level=1, axis=1).reindex(idx)
    
    strict_means   = _extract_aligned(strict_agg,   'mean',  all_baseline_users)
    strict_stds    = _extract_aligned(strict_agg,   'std',   all_baseline_users)
    strict_counts  = _extract_aligned(strict_agg,   'count', all_baseline_users).fillna(0)
    
    fallback_means  = _extract_aligned(fallback_agg, 'mean',  all_baseline_users)
    fallback_stds   = _extract_aligned(fallback_agg, 'std',   all_baseline_users)
    fallback_counts = _extract_aligned(fallback_agg, 'count', all_baseline_users).fillna(0)
    
    is_personal_enough = (strict_counts >= 2)
    
    chosen_means  = strict_means.where(is_personal_enough, fallback_means)
    chosen_stds   = strict_stds.where(is_personal_enough, fallback_stds)
    chosen_counts = strict_counts.where(is_personal_enough, fallback_counts)
    
    personal_means  = chosen_means
    personal_stds   = chosen_stds
    personal_counts = chosen_counts.fillna(0)
    
    global_means = df[baseline_mask_strict | baseline_mask_fallback].groupby('group')[target_cols].mean()
    global_stds = df[baseline_mask_strict | baseline_mask_fallback].groupby('group')[target_cols].std()
    
    user_group_map = df.drop_duplicates('user_id_hash').set_index('user_id_hash')['group'] \
                      .map(lambda x: 'patient' if x == 'patient' else 'control')
                      
    user_groups_for_baseline = user_group_map.loc[all_baseline_users]
    global_means_mapped = global_means.loc[user_groups_for_baseline].copy()
    global_means_mapped.index = all_baseline_users
    global_stds_mapped = global_stds.loc[user_groups_for_baseline].copy()
    global_stds_mapped.index = all_baseline_users
    
    w_p = personal_counts / (personal_counts + 2)
    bl_means = w_p * personal_means + (1 - w_p) * global_means_mapped
    bl_stds = np.maximum(personal_stds.fillna(0), global_stds_mapped.fillna(0))
    bl_stds = bl_stds.replace(0, np.nan) # 避免除0
    
    # --- 2. 向量化 MNAR 插补 (提前到 Z-score 之前，解决链路断裂问题) ---
    logger.info(f"  Applying vectorized MNAR imputation (quantile={MNAR_IMPUTATION_QUANTILE})...")
    silent_mask = ~df["is_reliable"]
    n_silent = silent_mask.sum()
    
    if n_silent > 0:
        # 沉默窗口的真实行为值应为 0，直接赋值，后续不再插补
        df.loc[silent_mask, "posting_intensity"] = 0
        df.loc[silent_mask, "n_posts_in_window"] = 0
        
        behavioral_log_cols = [c for c in target_cols if c.endswith('_log') or c.endswith('_log_signed') or 'n_posts' in c or 'n_active' in c or 'n_submissions' in c or 'n_comments' in c or 'n_unique' in c]
        behavioral_keywords = ['score', 'comment', 'submission', 'post', 'active', 'circadian', 'night', 'weekend', 'sleep', 'interval', 'hour', 'entropy', 'gap', 'tenure', 'subreddit', 'gilded', 'controversy', 'depth', 'intensity']
        
        # 精确剥离各类特征维度
        linguistic_negative_cols = [c for c in target_cols if c not in behavioral_log_cols and not any(k in c for k in behavioral_keywords) and 'positive' not in c and 'vader_pos' not in c and 'vader_compound' not in c]
        linguistic_positive_cols = [c for c in target_cols if c not in behavioral_log_cols and ('positive' in c or 'vader_pos' in c)]
        compound_cols = [c for c in df.columns if 'vader_compound' in c and c.endswith(('_mean', '_std'))]
        
        # [FIX] 把 withdrawal_index 公式实际使用的特征单独拆出来，避免与通用语义分类冲突。
        # second_person_score_rate 越低 = 越退缩；withdrawal_index = -z(second_person) - z(posting_intensity)
        # 所以沉默窗口应插补为低分位数 (P10)，而不是按"社交聚焦特征沉默=升高"的通用规则插补为 P90。
        withdrawal_directional_cols = [c for c in target_cols if c in withdraw_required_cols and c != 'posting_intensity']
        social_cols = [c for c in target_cols if ('second_person' in c or 'social_focus' in c) and c not in withdrawal_directional_cols]
        other_cols = [c for c in target_cols if c not in behavioral_log_cols and c not in linguistic_negative_cols and c not in linguistic_positive_cols and c not in compound_cols and c not in social_cols and c not in withdrawal_directional_cols and c not in ['posting_intensity', 'n_posts_in_window']]
        
        reliable_mask = df["is_reliable"]
        
        for group in ["patient", "control"]:
            g_mask = (df["group"] == group) & reliable_mask
            g_silent = (df["group"] == group) & silent_mask
            if g_silent.sum() == 0: continue
            
            # 1. 语言学消极特征：插补为 P90
            for col in linguistic_negative_cols:
                df.loc[g_silent, col] = df.loc[g_mask, col].quantile(MNAR_IMPUTATION_QUANTILE)
            # 2. 语言学积极特征：插补为 P10
            for col in linguistic_positive_cols:
                df.loc[g_silent, col] = df.loc[g_mask, col].quantile(1 - MNAR_IMPUTATION_QUANTILE)
            # 3. VADER 复合特征
            for col in compound_cols:
                q_val = MNAR_IMPUTATION_QUANTILE if ('std' in col or 'min' in col) else (1 - MNAR_IMPUTATION_QUANTILE)
                df.loc[g_silent, col] = df.loc[g_mask, col].quantile(q_val)
            # 4. 社交聚焦特征 (不含 second_person_score_rate)：插补为 P90 (高分位数)
            for col in social_cols:
                df.loc[g_silent, col] = df.loc[g_mask, col].quantile(MNAR_IMPUTATION_QUANTILE)
            # 4b. [FIX] withdrawal 専用方向特征 (second_person_score_rate)：沉默=退缩=该值更低 -> 插补为 P10
            for col in withdrawal_directional_cols:
                df.loc[g_silent, col] = df.loc[g_mask, col].quantile(1 - MNAR_IMPUTATION_QUANTILE)
            # 5. 行为/活跃度特征：插补为 P10 (score类用中位数)
            for col in behavioral_log_cols + other_cols:
                if 'score' in col:
                    df.loc[g_silent, col] = df.loc[g_mask, col].median()
                else:
                    df.loc[g_silent, col] = df.loc[g_mask, col].quantile(1 - MNAR_IMPUTATION_QUANTILE)
                
        df["is_mnar_imputed"] = silent_mask
        logger.info(f"    MNAR imputed {n_silent} silent windows")
    else:
        df["is_mnar_imputed"] = False

    # --- 3. 向量化 Z-score 计算 (此时 silent 窗口已被正确插补) ---
    z_scores = pd.DataFrame(index=df.index, columns=target_cols, dtype=float)
    for col in target_cols:
        mean_map = bl_means[col]
        std_map = bl_stds[col]
        z_scores[col] = (df[col] - df['user_id_hash'].map(mean_map)) / df['user_id_hash'].map(std_map)
    
    # --- 4. 向量化 P-factor 计算 ---
    p_factor_num = np.zeros(len(df))
    p_factor_den = np.zeros(len(df))
    
    for base_feat, weight in pfactor_weights.items():
        col = feat_map.get(base_feat)
        if col and col in z_scores.columns:
            direction = pfactor_directions.get(base_feat, 1.0)
            z_vals = z_scores[col].fillna(0).values
            p_factor_num += weight * direction * z_vals
            p_factor_den += weight * np.where(~z_scores[col].isna(), 1, 0)
            
    df['p_factor_severity_z'] = np.where(p_factor_den > 0, p_factor_num / p_factor_den, np.nan)
    
    # --- 5. 波动性与社交退缩 (向量化) ---
    # [FIX] 不要用字面字符串硬编码列名，应通过 feat_map 解析 "vader_compound_std" 这一
    # pfactor loading 名对应的真实窗口级列名 (即 vader_between_post_std)。
    volatility_col = feat_map.get('vader_compound_std')
    if volatility_col and volatility_col in z_scores.columns:
        df['volatility_index'] = z_scores[volatility_col]
    else:
        df['volatility_index'] = np.nan
        logger.warning(
            "  [Derived Indices] 'vader_compound_std' 未能匹配到任何窗口级列 "
            "(预期匹配到 vader_between_post_std)，volatility_index 将全部为 NaN。"
        )
    
    withdraw_parts = []
    for col in ['second_person_score_rate', 'posting_intensity']:
        if col in z_scores.columns:
            withdraw_parts.append(-z_scores[col])
    if withdraw_parts:
        df['social_withdrawal_index'] = sum(withdraw_parts) / len(withdraw_parts)
        
    return df


# ======================== TRAJECTORY SMOOTHING ========================

def apply_mnar_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """
    [FIX v1.4] MNAR 感知插值：沉默即信号。
    """
    logger.info("Applying MNAR-aware imputation: high-withdrawal proxy for silent periods...")
    
    df["is_missing_window"] = ~df["is_reliable"]
    
    mnar_updated = 0
    for uid, user_df in tqdm(df.groupby("user_id_hash"), desc="MNAR Imputation"):
        # [FIX v1.5] 使用向量化操作替代行级循环
        if "social_withdrawal_index" in user_df.columns:
            valid_withdrawal = user_df.loc[user_df["is_reliable"], "social_withdrawal_index"].dropna()
            if len(valid_withdrawal) >= 2:
                mnar_value = valid_withdrawal.quantile(MNAR_IMPUTATION_QUANTILE)
                
                # 构建掩码：该用户下，且满足缺失且不活跃的条件
                fill_mask = (user_df["social_withdrawal_index"].isna()) & (~user_df["is_reliable"])
                target_idx = user_df.index[fill_mask]
                
                if len(target_idx) > 0:
                    df.loc[target_idx, "social_withdrawal_index"] = mnar_value
                    mnar_updated += len(target_idx)
    
    logger.info(f"  MNAR update: Applied {MNAR_IMPUTATION_QUANTILE} quantile withdrawal value to {mnar_updated} missing windows.")
    return df


# ======================== T0 VALIDATION (NEW) ========================

def validate_t0_linguistic_shift(traj_df: pd.DataFrame) -> Dict:
    """
    [FIX v1.4] 代理 t0 效度验证。使用 ttest_1samp 替代 ttest_rel，并增加正态性检验。
    """
    from scipy.stats import ttest_1samp, wilcoxon, shapiro
    
    logger.info("Validating t0 linguistic shift proxy...")
    results = {"patient": {}, "control": {}}
    
    for group in ["patient", "control"]:
        group_df = traj_df[traj_df["group"] == group]
        pre_post_diffs = []
        
        for uid, udf in group_df.groupby("user_id_hash"):
            pre_val = udf.loc[(udf["phase"] == "pre") & udf["is_reliable"], "p_factor_severity_z"].mean()
            post_val = udf.loc[(udf["phase"] == "post") & udf["is_reliable"], "p_factor_severity_z"].mean()
            
            if not np.isnan(pre_val) and not np.isnan(post_val):
                pre_post_diffs.append(post_val - pre_val)
        
        if len(pre_post_diffs) >= 20:
            diffs = np.array(pre_post_diffs)
            cohen_d = np.mean(diffs) / np.std(diffs, ddof=1) if np.std(diffs, ddof=1) > 0 else 0
            
            # [FIX v1.4] 正态性检验与统计方法选择
            if len(diffs) < 5000:
                _, shapiro_p = shapiro(diffs)
            else:
                shapiro_p = 1.0  # 大样本假设正态
                
            if shapiro_p > 0.05:
                t_stat, p_val = ttest_1samp(diffs, 0)
                test_method = "ttest_1samp"
            else:
                # Wilcoxon 不返回 t_stat，返回 Z 近似值
                res = wilcoxon(diffs)
                p_val = res.pvalue
                t_stat = norm.ppf(p_val / 2) # 近似 Z 值用于报告
                test_method = "wilcoxon"
            
            results[group] = {
                "n_users": len(diffs),
                "mean_shift": float(np.mean(diffs)),
                "test_method": test_method,
                "statistic": float(t_stat),
                "p_val": float(p_val),
                "cohen_d": float(cohen_d),
                "is_significant": p_val < 0.05 and abs(cohen_d) > 0.2
            }
            logger.info(f"  {group} t0 shift: mean={np.mean(diffs):.3f}, d={cohen_d:.3f}, p={p_val:.4f} ({test_method})")
        else:
            results[group] = {"n_users": len(pre_post_diffs), "is_significant": None}
            
    return results

# 在 run_trajectory_construction 函数之前添加：
def process_partition_trajectories_wrapper(args):
    """多进程包装函数"""
    return process_partition_trajectories(*args)

# ======================== MAIN PIPELINE ========================

def run_trajectory_construction():
    logger.info("=" * 80)
    logger.info("MODULE 2 - Step 07: Longitudinal Trajectory Construction (v1.2)")
    logger.info(f"Window: {WINDOW_SIZE_DAYS}d / Step: {WINDOW_STEP_DAYS}d")
    logger.info(f"Range: t0-{TRAJECTORY_PRE_DAYS}d to t0+{TRAJECTORY_POST_DAYS}d")
    logger.info("=" * 80)
    
    # [FIX v1.6] 必须先加载 p-factor 权重，因为无论是否断点续跑都需要它
    pfactor_weights, pfactor_directions = load_pfactor_weights()
    
    # [FIX v1.6] 断点恢复机制：如果存在 checkpoint 则直接加载，跳过耗时4天的窗口提取
    checkpoint_path = os.path.join(TRAJECTORY_DIR, "_checkpoint_raw_traj.parquet")
    
    if os.path.exists(checkpoint_path):
        logger.warning(f"⚠️ Resuming from checkpoint: {checkpoint_path}")
        traj_df = pd.read_parquet(checkpoint_path)
        # [FIX v1.7] checkpoint 恢复路径此前直接读取 parquet，跳过了
        # build_user_groups() 的 disorder_group 增强，导致患者 disorder_group
        # 在 checkpoint 中若缺失/为 unknown，则整条下游（s08 per-disorder、
        # s09 disorder 分析、s10 disorder DID）全部静默失效。
        # 这里补回 disorder_group：从 t0_df（load_t0_mapping，含 Silver 增强）
        # 重建 user_groups 并覆盖患者的 disorder_group。
        t0_df = load_t0_mapping()
        user_groups = build_user_groups(t0_df, PSM_MATCHED_PATH, USER_META_PATH)
        uid_to_disorder = {
            uid: meta.get("disorder_group", "unknown")
            for uid, meta in user_groups.items()
        }
        traj_df["disorder_group"] = traj_df["user_id_hash"].map(uid_to_disorder).fillna("unknown")
        n_pat_known = (
            (traj_df.groupby("user_id_hash")["disorder_group"].first() != "unknown")
            & (traj_df.groupby("user_id_hash")["group"].first() == "patient")
        ).sum()
        n_pat = (traj_df.groupby("user_id_hash")["group"].first() == "patient").sum()
        logger.info(f"  [checkpoint-fix] disorder_group restored: {n_pat_known}/{n_pat} patients known")
    else:
        # 1. t0 映射
        t0_df = load_t0_mapping()
        t0_map = dict(zip(t0_df["user_id_hash"], t0_df["t0_timestamp"]))
        logger.info(f"Users with valid t0: {len(t0_map)}")
        
        # 2. [FIX B+C] 统一使用 build_user_groups()，确保：
        #    - group 列归一化 (control_candidate 等异常值 -> control)
        #    - disorder_group 使用 t0_df 中经 Silver 增强后的版本，而非重新从
        #      user_features_standardized.parquet 读取一份脱钩的版本
        #    - pair_id / iptw_att_weight / eb_weight 一并加载
        logger.info("Building user groups via build_user_groups()...")
        user_groups = build_user_groups(t0_df, PSM_MATCHED_PATH, USER_META_PATH)
        
        # [FIX D] 诊断 t0_map 与 user_groups 的用户集合差异
        missing_meta = set(t0_map.keys()) - set(user_groups.keys())
        if missing_meta:
            frac = len(missing_meta) / len(t0_map)
            logger.warning(
                f"  [FIX D] {len(missing_meta)}/{len(t0_map)} ({frac:.1%}) users in t0_map "
                f"have NO group metadata — their windows will be tagged group='unknown'. "
                f"Sample missing uids: {list(missing_meta)[:5]}"
            )
            if frac > 0.05:
                logger.error(
                    "  [FIX D] Missing-metadata fraction exceeds 5% — investigate before "
                    "proceeding to full-scale run, this will distort group-level baselines."
                )
                
        logger.info(f"User metadata loaded: {len(user_groups)} users")
        
        # 4. 发现分区
        partitions = []
        if os.path.exists(SILVER_DATA_DIR):
            for entry in sorted(os.listdir(SILVER_DATA_DIR)):
                full_path = os.path.join(SILVER_DATA_DIR, entry)
                if os.path.isdir(full_path) and entry.startswith('part_'):
                    partitions.append(full_path)
        
        if not partitions:
            logger.error(f"No partitions found in {SILVER_DATA_DIR}")
            return None
        
        logger.info(f"Found {len(partitions)} Silver Data partitions")
                    
        # 5. 构建轨迹 (分区块落盘)
        temp_dir = os.path.join(TRAJECTORY_DIR, "_temp_partitions")
        os.makedirs(temp_dir, exist_ok=True)
        
        partition_idx = 0
        if NUM_CORES > 1 and len(partitions) > 1:
            # [FIX] 使用更稳健的进程池管理，确保所有分区都被处理
            logger.info(f"Parallel processing {len(partitions)} partitions with {NUM_CORES} cores...")
            with Pool(processes=NUM_CORES, initializer=init_worker) as pool:
                # 使用 imap_unordered 按需分配任务，避免某个慢的分区阻塞全局
                results_iter = pool.imap_unordered(
                    process_partition_trajectories_wrapper,
                    [(p, t0_map, user_groups, pfactor_weights, pfactor_directions) for p in partitions]
                )
                for res in tqdm(results_iter, total=len(partitions), desc="Partitions (Parallel)"):
                    if res:
                        pd.DataFrame(res).to_parquet(
                            os.path.join(temp_dir, f"part_{partition_idx}.parquet"), index=False
                        )
                        partition_idx += 1
                    gc.collect()
        else:
            for partition in tqdm(partitions, desc="Partitions"):
                results = process_partition_trajectories(partition, t0_map, user_groups, pfactor_weights, pfactor_directions)
                if results:
                    pd.DataFrame(results).to_parquet(
                        os.path.join(temp_dir, f"part_{partition_idx}.parquet"), index=False
                    )
                    partition_idx += 1
                gc.collect()

        # 6. 构建 DataFrame (从磁盘读取)
        logger.info("Building trajectory DataFrame from temp files...")
        import pyarrow.parquet as pq
        traj_df = pd.read_parquet(temp_dir)
        
        # [FIX v1.6] 保存 Checkpoint，防止后续步骤崩溃导致4天白跑
        logger.info(f"Saving checkpoint to {checkpoint_path} ...")
        traj_df.to_parquet(checkpoint_path, compression='zstd', index=False)
        logger.info("Checkpoint saved successfully.")
        
        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir)
    
    logger.info(f"Raw trajectory shape: {traj_df.shape}")
    logger.info(f"Unique users: {traj_df['user_id_hash'].nunique()}")
    logger.info(f"Reliable windows: {traj_df['is_reliable'].sum()}/{len(traj_df)} ({traj_df['is_reliable'].mean()*100:.1f}%)")
    
    # 7. [FIX v1.4] 后置计算所有衍生指标 (替代原先错误的 normalize_pfactor_severity)
    # 这里会直接从 traj_df 中正确计算窗口级的全局基线和个人基线
    traj_df = compute_derived_indices(traj_df, pfactor_weights, pfactor_directions)
    
    # 8. [FIX v1.4] MNAR 插值平滑 (替代原先的 smooth_trajectories)
    # [FIX v1.5] 移除二次 MNAR 插补：compute_derived_indices (步骤 7) 内已在
    # z-score/p_factor 计算之前完成向量化 MNAR 插补（含 withdrawal 方向特征填 P10）。
    # 此处若再次调用 apply_mnar_imputation，会：(a) 对已插补的 social_withdrawal_index
    # 再次覆盖为 P90，与首次插补的 P10 语义冲突，形成双重插补；(b) 掩码 is_reliable
    # 为 False 的窗口仍会被当作"真实值"参与 p_factor，污染下游 PELT 拐点与形状特征
    # （该污染由 s08/s09/s10 统一 is_reliable 过滤兜底）。
    # traj_df = apply_mnar_imputation(traj_df)
    
    # [FIX v1.2] 9. 社交退缩已在 build_user_trajectory 中三维计算，无需再在此处降级内联
    # 仅做日志验证
    if "social_withdrawal_index" in traj_df.columns:
        logger.info(f"Social Withdrawal Index coverage: {traj_df['social_withdrawal_index'].notna().mean()*100:.1f}%")
    
    # [FIX v1.2] 10. 代理 t0 效度验证
    t0_validation = validate_t0_linguistic_shift(traj_df)
    
    # 11. 保存
    output_path = os.path.join(TRAJECTORY_DIR, TRAJECTORY_OUTPUT_FILE)
    traj_df.to_parquet(output_path, compression='zstd', index=False)
    logger.info(f"Saved trajectories to: {output_path}")
    logger.info(f"Final shape: {traj_df.shape}")
    
    # 12. 元数据
    metadata = {
        "pipeline_version": "v1.5",
        "window_size_days": WINDOW_SIZE_DAYS,
        "window_step_days": WINDOW_STEP_DAYS,
        "trajectory_pre_days": TRAJECTORY_PRE_DAYS,
        "trajectory_post_days": TRAJECTORY_POST_DAYS,
        "min_posts_per_window": MIN_POSTS_PER_WINDOW,
        "min_reliable_windows": MIN_TOTAL_RELIABLE_WINDOWS,
        "pfactor_weight_source": PFACTOR_WEIGHT_SOURCE,
        "pfactor_direction_method": "data_driven_factor1_sign", # [FIX v1.2]
        "baseline_definition_days": "(-360, -270)", # [FIX v1.2]
        "n_users": int(traj_df["user_id_hash"].nunique()),
        "n_patients": int((traj_df.groupby("user_id_hash")["group"].first() == "patient").sum()),
        "n_controls": int((traj_df.groupby("user_id_hash")["group"].first() == "control").sum()),
        "n_windows_total": len(traj_df),
        "n_windows_reliable": int(traj_df["is_reliable"].sum()),
        "n_windows_high_quality": int(traj_df["is_high_quality"].sum()), # [FIX v1.2]
        "reliable_rate": float(traj_df["is_reliable"].mean()),
        "mean_windows_per_user": float(traj_df.groupby("user_id_hash").size().mean()),
        "t0_validation": t0_validation, # [FIX v1.2]
        "construction_timestamp": datetime.now().isoformat(),
    }
    
    meta_path = os.path.join(TRAJECTORY_DIR, TRAJECTORY_METADATA_FILE)
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # [NEW] 保存聚合轨迹数据，供可视化直接调用，避免绘图时重复聚合耗时
    logger.info("Aggregating trajectories for visualization...")
    agg_cols = ["window_index", "days_from_t0_center", "group"]
    metric_cols = ["p_factor_severity_z", "volatility_index", "social_withdrawal_index", "posting_intensity"]
    metric_cols = [c for c in metric_cols if c in traj_df.columns]
    
    # 仅对可靠窗口聚合
    # 包含插补后的窗口进行聚合，解决 A 项问题
    valid_traj = traj_df[traj_df["is_reliable"] | traj_df["is_mnar_imputed"]].copy()
    
    if len(valid_traj) > 0 and len(metric_cols) > 0:
        agg_df = valid_traj.groupby(agg_cols)[metric_cols].agg(["mean", "std", "count"]).reset_index()
        # 展平多级列名: (p_factor_severity_z, mean) -> p_factor_severity_z_mean
        agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns]
        
        # 计算标准误
        for mc in metric_cols:
            std_col = f"{mc}_std"
            count_col = f"{mc}_count"
            se_col = f"{mc}_se"
            if std_col in agg_df.columns and count_col in agg_df.columns:
                agg_df[se_col] = agg_df[std_col] / np.sqrt(agg_df[count_col])
        
        agg_path = os.path.join(TRAJECTORY_DIR, "trajectory_aggregated_by_window.csv")
        agg_df.to_csv(agg_path, index=False)
        logger.info(f"Saved aggregated trajectories for visualization: {agg_path}")    
    
    _generate_trajectory_quality_report(traj_df)
    
    logger.info("=" * 80)
    logger.info("TRAJECTORY CONSTRUCTION COMPLETE")
    logger.info("=" * 80)
    
    return traj_df

def _build_psm_long_format(psm_df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """[FIX v1.3] 明确处理 PSM 表到长格式的映射，避免列丢失"""
    # 确保必须包含的主键和配对键
    required_pat_cols = ["patient_id", "pair_id"]
    required_ctrl_cols = ["control_id", "pair_id"]
    
    # 提取额外需要的权重列 (只保留 PSM 表中实际存在的)
    weight_cols = [c for c in cols if c in psm_df.columns and c not in required_pat_cols]
    
    # 患者端
    pat_cols_to_select = required_pat_cols + weight_cols
    patient_data = psm_df[pat_cols_to_select].rename(columns={"patient_id": "user_id_hash"})
    
    # 对照端
    ctrl_cols_to_select = required_ctrl_cols + weight_cols
    control_data = psm_df[ctrl_cols_to_select].rename(columns={"control_id": "user_id_hash"})
    
    long_df = pd.concat([patient_data, control_data], ignore_index=True)
    return long_df.drop_duplicates(subset=["user_id_hash"])

def _generate_trajectory_quality_report(traj_df: pd.DataFrame):
    report = {}
    user_stats = traj_df.groupby("user_id_hash").agg(
        n_windows=("window_index", "count"),
        n_reliable=("is_reliable", "sum"),
        n_high_quality=("is_high_quality", "sum"), # [FIX v1.2]
        has_sufficient_pre=("has_sufficient_pre", "first"),
        has_sufficient_post=("has_sufficient_post", "first"),
        group=("group", "first"),
    )
    
    report["n_users_total"] = len(user_stats)
    report["n_users_with_sufficient_pre"] = int(user_stats["has_sufficient_pre"].sum())
    report["n_users_with_sufficient_post"] = int(user_stats["has_sufficient_post"].sum())
    report["n_users_with_both"] = int((user_stats["has_sufficient_pre"] & user_stats["has_sufficient_post"]).sum())
    report["mean_windows_per_user"] = float(user_stats["n_windows"].mean())
    report["mean_reliable_windows"] = float(user_stats["n_reliable"].mean())
    report["mean_high_quality_windows"] = float(user_stats["n_high_quality"].mean()) # [FIX v1.2]
    
    for group in ["patient", "control"]:
        group_mask = user_stats["group"] == group
        n_group = group_mask.sum()
        n_with_pre = user_stats.loc[group_mask, "has_sufficient_pre"].sum()
        n_with_post = user_stats.loc[group_mask, "has_sufficient_post"].sum()
        report[f"{group}_n_users"] = int(n_group)
        report[f"{group}_with_sufficient_pre"] = int(n_with_pre)
        report[f"{group}_with_sufficient_post"] = int(n_with_post)
    
    reliable_rates = user_stats["n_reliable"] / user_stats["n_windows"]
    report["reliable_rate_median"] = float(reliable_rates.median())
    report["reliable_rate_q25"] = float(reliable_rates.quantile(0.25))
    
    report_path = os.path.join(TRAJECTORY_DIR, "trajectory_quality_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info("Trajectory Quality Report:")
    for k, v in report.items():
        logger.info(f"  {k}: {v}")

if __name__ == "__main__":
    run_trajectory_construction()