"""
Step 02: Feature Aggregation & Matching Integration (v1.3)

Merges extracted features with PSM matching results.
Produces the final analysis-ready dataset for Module 1.

v1.2 → v1.3 Changelog:
  [FIX] PSM merge column conflict → group KeyError (root cause)
  [FIX] PSM weight merge: only carry weight/pair columns, not all PSM columns
  [FIX] Vectorized PSM weight merge (replaced iterrows)
  [FIX] pair_strat logic: identical branches → properly use patient's strat
  [FIX] Vectorized pair stratification (replaced iterrows)
  [FIX] temporal_features intersected with feature_cols
  [FIX] STATS_DIR auto-creation
  [FIX] Group value validation with catch-all
  [FIX] Column existence validation after load and after merge
  [FIX] Empty DataFrame handling
  [FIX] Stratified split fallback for rare categories
  [FIX] Quality report defensive column access
  [FIX] Imputation group_medians fallback for all-NaN groups
  [NEW] Comprehensive column conflict detection before merge
  [NEW] Data validation between phases
"""

import os
import logging
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Tuple, List, Dict, Optional, Set

from config_module1 import (
    FEATURES_DIR, PSM_MATCHED_PATH, USER_META_PATH,
    REGISTRY_PATH, STATS_DIR, RANDOM_SEED,
    DISORDER_GROUPS, MIN_POSTS_FOR_FEATURES,
    RAW_FEATURES_FILE, FEATURE_METADATA_FILE
)

# [FIX] Ensure all output directories exist
for _d in [FEATURES_DIR, STATS_DIR]:
    os.makedirs(_d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(FEATURES_DIR, "aggregation.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)

np.random.seed(RANDOM_SEED)


# ======================== PSM CLUSTER RESOLUTION ========================

class UnionFind:
    """
    轻量级并查集，用于解析 PSM 1:k 或复用匹配的连通图。
    保证在 train/val/test 切分时，同一个匹配网络中的所有用户同进同退。
    """
    def __init__(self, elements):
        self.parent = {e: e for e in elements}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # 路径压缩
            x = self.parent[x]
        return x

    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x

def resolve_psm_clusters(psm_df: pd.DataFrame) -> pd.DataFrame:
    """
    将 PSM 的配对关系解析为连通区块。
    无论 1:1, 1:k, 还是存在 Control 复用，都能将相互关联的用户归入同一个 cluster_id。
    """
    all_users = set(psm_df["patient_id"].dropna().unique()) | \
                set(psm_df["control_id"].dropna().unique())
    
    uf = UnionFind(all_users)
    
    # 根据配对关系连通节点
    for _, row in psm_df.iterrows():
        uf.union(row["patient_id"], row["control_id"])
    
    # 生成 cluster_id 映射
    cluster_map = {}
    for uid in all_users:
        root = uf.find(uid)
        if root not in cluster_map:
            cluster_map[root] = f"cluster_{len(cluster_map)}"
        cluster_map[uid] = cluster_map[root]
        
    return cluster_map
# ======================== VALIDATION HELPERS ========================

def validate_essential_columns(df: pd.DataFrame, required_cols: List[str],
                                context: str = "") -> bool:
    """
    [NEW] Validate that essential columns exist in a DataFrame.
    Returns True if all present, False otherwise.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"  [{context}] Missing essential columns: {missing}")
        logger.error(f"  Available columns: {list(df.columns)[:20]}...")
        return False
    return True


def detect_merge_conflicts(left_df: pd.DataFrame, right_df: pd.DataFrame,
                           on_cols: List[str]) -> List[str]:
    """
    [NEW] Detect column name conflicts that would cause _x/_y suffixing
    during a merge operation.
    """
    left_only = set(left_df.columns) - set(on_cols)
    right_only = set(right_df.columns) - set(on_cols)
    conflicts = left_only & right_only
    return sorted(conflicts)


# ======================== PHASE 1: LOAD & MERGE ========================

def load_and_merge():
    """
    Load features and merge with PSM matching + user metadata.
    
    [FIX v1.3] Root cause of group KeyError: PSM file may contain columns
    that conflict with feature columns. Now:
      1. Only merge PSM-specific weight/pair columns
      2. Detect and resolve column conflicts BEFORE merge
      3. Validate essential columns after every operation
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: Loading and merging data sources")
    logger.info("=" * 60)

    # --- Load raw features ---
    raw_path = os.path.join(FEATURES_DIR, RAW_FEATURES_FILE)
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Raw features not found: {raw_path}. "
            f"Run s01_feature_extraction.py first."
        )

    logger.info("Loading raw features...")
    features_df = pd.read_parquet(raw_path)
    logger.info(f"  Features shape: {features_df.shape}")

    # [FIX v1.3] Validate essential columns immediately after loading
    essential_cols = ["user_id_hash", "group", "disorder_group"]
    if not validate_essential_columns(features_df, essential_cols, "Raw features"):
        # Attempt to infer missing columns
        if "group" not in features_df.columns:
            logger.error("  'group' column missing — cannot proceed. "
                        "Check s01 output or re-run feature extraction.")
            raise ValueError("'group' column missing from raw features")
        if "disorder_group" not in features_df.columns:
            logger.warning("  'disorder_group' missing — inferring from group")
            features_df["disorder_group"] = features_df["group"].apply(
                lambda g: "control" if g in ("control", "control_candidate", "healthy_control")
                else "unknown"
            )

    # [FIX v1.3] Defensive column access with .get()
    if "group" in features_df.columns:
        logger.info(f"  Groups (raw): {features_df['group'].value_counts().to_dict()}")
    else:
        logger.warning("  'group' column not found in raw features!")

    # [FIX v1.3] Standardize group names with validation
    group_mapping = {
        "control_candidate": "control",
        "healthy_control": "control",
        "Control": "control",
        "Patient": "patient",
    }
    if "group" in features_df.columns:
        features_df["group"] = features_df["group"].replace(group_mapping)

        # Validate: after mapping, only "patient" and "control" should remain
        unexpected_groups = set(features_df["group"].unique()) - {"patient", "control"}
        if unexpected_groups:
            logger.warning(f"  Unexpected group values after standardization: "
                          f"{unexpected_groups}")
            # Map any remaining non-patient values to control as fallback
            features_df["group"] = features_df["group"].apply(
                lambda g: g if g in ("patient", "control") else "control"
            )
            logger.warning(f"  Mapped unexpected values to 'control'")

    if "disorder_group" in features_df.columns:
        # Ensure controls always have disorder_group = "control"
        features_df.loc[features_df["group"] == "control", "disorder_group"] = "control"

    if "group" in features_df.columns:
        logger.info(f"  Groups (standardized): "
                    f"{features_df['group'].value_counts().to_dict()}")
    if "disorder_group" in features_df.columns:
        logger.info(f"  Disorder groups: "
                    f"{features_df['disorder_group'].value_counts().to_dict()}")

    # [FIX v1.3] Remove any PSM weight columns leaked from Step 01
    # More comprehensive detection: any column matching known PSM patterns
    psm_leak_patterns = [
        "iptw_att_weight", "ps_score", "eb_weight",
        "pair_id", "match_strata",
    ]
    # Also catch _x/_y suffixed versions from previous failed merges
    psm_leak_cols = []
    for pattern in psm_leak_patterns:
        for col in features_df.columns:
            if col == pattern or col.startswith(pattern + "_"):
                psm_leak_cols.append(col)

    if psm_leak_cols:
        logger.warning(f"  Dropping {len(psm_leak_cols)} PSM columns leaked "
                      f"from Step 01: {psm_leak_cols}")
        features_df = features_df.drop(columns=psm_leak_cols)

    # --- Load PSM matching results ---
    logger.info("Loading PSM matching results...")
    if not os.path.exists(PSM_MATCHED_PATH):
        raise FileNotFoundError(
            f"PSM matching results not found: {PSM_MATCHED_PATH}. "
            f"Run PSM matching first."
        )

    psm_df = pd.read_parquet(PSM_MATCHED_PATH)
    logger.info(f"  PSM shape: {len(psm_df)}")
    logger.info(f"  PSM columns: {list(psm_df.columns)}")

    # Determine PSM structure
    psm_has_pairs = ("patient_id" in psm_df.columns and
                     "control_id" in psm_df.columns)

    if psm_has_pairs:
        matched_patients = set(psm_df["patient_id"].unique())
        matched_controls = set(psm_df["control_id"].unique())
        matched_users = matched_patients | matched_controls
        logger.info(f"  PSM pairs: {len(psm_df)}")
        logger.info(f"  Matched patients: {len(matched_patients)}, "
                    f"controls: {len(matched_controls)}")

        # Validate pair completeness
        feature_users = set(features_df["user_id_hash"])
        missing_patients = matched_patients - feature_users
        missing_controls = matched_controls - feature_users

        if missing_patients:
            logger.warning(f"  {len(missing_patients)} matched patients not in "
                          f"features (filtered by quality in Step 01)")
        if missing_controls:
            logger.warning(f"  {len(missing_controls)} matched controls not "
                          f"in features")

        # Drop broken pairs
        valid_pairs = psm_df[
            psm_df["patient_id"].isin(feature_users) &
            psm_df["control_id"].isin(feature_users)
        ].copy()

        n_broken = len(psm_df) - len(valid_pairs)
        if n_broken > 0:
            logger.warning(f"  Dropping {n_broken} broken pairs (missing member)")
            psm_df = valid_pairs
            matched_patients = set(psm_df["patient_id"].unique())
            matched_controls = set(psm_df["control_id"].unique())
            matched_users = matched_patients | matched_controls

        logger.info(f"  Valid pairs after integrity check: {len(psm_df)}")

    elif "user_id_hash" in psm_df.columns:
        matched_users = set(psm_df["user_id_hash"].unique())
        logger.info(f"  Total matched users: {len(matched_users)}")
        psm_has_pairs = False
    else:
        raise ValueError(
            "PSM file must contain either (patient_id, control_id) "
            "or user_id_hash column"
        )

    # Filter to matched users
    features_matched = features_df[
        features_df["user_id_hash"].isin(matched_users)
    ].copy()

    # [FIX v1.3] Validate after filtering
    if len(features_matched) == 0:
        logger.error("  No matched users found in features! "
                    "Check user_id_hash consistency between features and PSM.")
        raise ValueError("Empty features after PSM filter")

    logger.info(f"  Features after matching filter: {features_matched.shape[0]} users "
                f"({features_matched.shape[0] / len(features_df) * 100:.1f}% retained)")

    # [FIX v1.3] Authoritative PSM weight merge with conflict detection
    if psm_has_pairs:
        # Generate pair_id if not present
        if "pair_id" not in psm_df.columns:
            psm_df = psm_df.copy()
            psm_df["pair_id"] = [f"pair_{i}" for i in range(len(psm_df))]

        # [FIX v1.3] CRITICAL: Only merge PSM-specific columns (weights + pair_id)
        # Do NOT merge columns that already exist in features_matched
        # This is the root cause of the group KeyError
        psm_weight_only_cols = ["pair_id"]
        weight_col_candidates = [
            "ps_score", "iptw_att_weight", "eb_weight",
            "match_strata", "distance", "caliper",
        ]
        for col in weight_col_candidates:
            if col in psm_df.columns:
                psm_weight_only_cols.append(col)

        # [FIX v1.3] Detect conflicts before merge
        conflicts = detect_merge_conflicts(
            features_matched,
            pd.DataFrame(columns=psm_weight_only_cols + ["user_id_hash"]),
            on_cols=["user_id_hash"]
        )
        if conflicts:
            logger.warning(f"  PSM merge column conflicts detected: {conflicts}")
            logger.warning(f"  These columns will be renamed with _psm suffix")
            # Rename conflicting columns in PSM data
            rename_map = {c: f"{c}_psm" for c in conflicts}
            # But actually, we should just drop them from PSM side
            # since features_matched already has them
            psm_weight_only_cols = [c for c in psm_weight_only_cols
                                     if c not in conflicts]
            logger.warning(f"  Kept PSM columns after conflict resolution: "
                          f"{psm_weight_only_cols}")

        # [FIX v1.3] Vectorized PSM weight merge (replaced iterrows)
        # Build long-format weights DataFrame from pair structure
        patient_weights = psm_df[["patient_id"] + psm_weight_only_cols].copy()
        patient_weights = patient_weights.rename(
            columns={"patient_id": "user_id_hash"}
        )

        control_weights = psm_df[["control_id"] + psm_weight_only_cols].copy()
        control_weights = control_weights.rename(
            columns={"control_id": "user_id_hash"}
        )

        weights_long_raw = pd.concat(
            [patient_weights, control_weights], ignore_index=True
        )
        
        # [FIX] 检测 control 复用 (1:k 匹配)
        # [FIX v1.3.1] 使用并查集解决 1:k 匹配及复用导致的 Pair 结构撕裂问题
        dup_users = weights_long_raw[weights_long_raw.duplicated(subset=["user_id_hash"], keep=False)]
        if len(dup_users) > 0:
            n_dup = dup_users["user_id_hash"].nunique()
            logger.warning(f"  ⚠️ 检测到 {n_dup} 个用户在 PSM 中存在 1:k 匹配或复用。")
            logger.warning(f"  启动连通图解析：将配对网络升级为 Cluster，确保 Split 时的严格封闭性。")
            
            # 1. 解析连通分量，获取真正的区块 ID
            cluster_map = resolve_psm_clusters(psm_df)
            
            # 2. 将 cluster_id 映射回长表 (覆盖原有的 pair_id)
            weights_long_raw["pair_id"] = weights_long_raw["user_id_hash"].map(cluster_map)
            
            # 3. 按用户聚合：cluster_id 必然一致(取first)，权重取均值(消除多对重复带来的方差膨胀)
            agg_dict = {"pair_id": "first"}
            for w_col in psm_weight_only_cols:
                if w_col != "pair_id":
                    agg_dict[w_col] = "mean"
            
            weights_long = weights_long_raw.groupby("user_id_hash", as_index=False).agg(agg_dict)
            
            n_clusters = weights_long["pair_id"].nunique()
            logger.info(f"  解析完成：{len(matched_users)} 个用户被归入 {n_clusters} 个独立连通区块。")
        else:
            weights_long = weights_long_raw.drop_duplicates(subset=["user_id_hash"])

        logger.info(f"  PSM weights long format: {len(weights_long)} records, "
                    f"columns: {list(weights_long.columns)}")

        # [FIX v1.3] Final conflict check before merge
        final_conflicts = detect_merge_conflicts(
            features_matched, weights_long, on_cols=["user_id_hash"]
        )
        if final_conflicts:
            logger.error(f"  UNEXPECTED: Still have merge conflicts: "
                        f"{final_conflicts}. Dropping from PSM side.")
            weights_long = weights_long.drop(
                columns=[c for c in final_conflicts if c != "user_id_hash"]
            )

        features_matched = features_matched.merge(
            weights_long, on="user_id_hash", how="left"
        )

        # [FIX v1.3] Validate essential columns survived the merge
        if "group" not in features_matched.columns:
            logger.error("  'group' column lost during PSM merge! "
                        "This indicates a column conflict bug.")
            # Attempt recovery from original features_df
            if "group" in features_df.columns:
                logger.warning("  Attempting recovery from original features_df")
                group_map = features_df.set_index("user_id_hash")["group"]
                features_matched["group"] = features_matched["user_id_hash"].map(
                    group_map
                )

        if "disorder_group" not in features_matched.columns:
            logger.error("  'disorder_group' column lost during PSM merge!")
            if "disorder_group" in features_df.columns:
                dg_map = features_df.set_index("user_id_hash")["disorder_group"]
                features_matched["disorder_group"] = features_matched[
                    "user_id_hash"
                ].map(dg_map)

        logger.info(f"  Merged PSM pair IDs and weights for "
                    f"{len(weights_long)} users")
    else:
        # Direct merge by user_id_hash (non-paired PSM)
        psm_merge_cols = [c for c in psm_df.columns if c != "user_id_hash"]
        # [FIX v1.3] Filter to only weight-type columns
        psm_merge_cols = [c for c in psm_merge_cols
                          if c in {"ps_score", "iptw_att_weight", "eb_weight",
                                   "match_strata", "pair_id"}]
        if psm_merge_cols:
            # Check conflicts
            conflicts = detect_merge_conflicts(
                features_matched,
                psm_df[["user_id_hash"] + psm_merge_cols],
                on_cols=["user_id_hash"]
            )
            if conflicts:
                logger.warning(f"  PSM merge conflicts: {conflicts}. Dropping.")
                psm_merge_cols = [c for c in psm_merge_cols if c not in conflicts]

            if psm_merge_cols:
                features_matched = features_matched.merge(
                    psm_df[["user_id_hash"] + psm_merge_cols].drop_duplicates(
                        subset=["user_id_hash"]
                    ),
                    on="user_id_hash",
                    how="left"
                )
                logger.info(f"  Merged PSM weights: {psm_merge_cols}")

    # Merge additional metadata
    if os.path.exists(USER_META_PATH):
        logger.info("Loading user metadata...")
        try:
            user_meta = pd.read_parquet(USER_META_PATH)
            logger.info(f"  User meta shape: {user_meta.shape}")

            meta_cols_to_merge = ["user_id_hash"]
            for col in ["account_age_days", "total_posts", "total_subreddits",
                         "is_comorbid", "comorbid_disorders"]:
                if (col in user_meta.columns and
                        col not in features_matched.columns):
                    meta_cols_to_merge.append(col)

            if len(meta_cols_to_merge) > 1:
                # [FIX v1.3] Check conflicts before merge
                conflicts = detect_merge_conflicts(
                    features_matched,
                    user_meta[meta_cols_to_merge],
                    on_cols=["user_id_hash"]
                )
                if conflicts:
                    logger.warning(f"  User meta merge conflicts: {conflicts}. "
                                  f"Dropping from meta side.")
                    meta_cols_to_merge = [c for c in meta_cols_to_merge
                                          if c not in conflicts]

                if len(meta_cols_to_merge) > 1:
                    features_matched = features_matched.merge(
                        user_meta[meta_cols_to_merge].drop_duplicates(
                            subset=["user_id_hash"]
                        ),
                        on="user_id_hash",
                        how="left"
                    )
                    logger.info(f"  Merged {len(meta_cols_to_merge) - 1} "
                                f"metadata columns")
        except Exception as e:
            logger.warning(f"  Failed to load user metadata: {e}. Skipping.")

    # [FIX v1.3] Final validation
    if not validate_essential_columns(
        features_matched, ["user_id_hash", "group", "disorder_group"],
        "After all merges"
    ):
        raise ValueError("Essential columns missing after merge operations")

    logger.info(f"  Final merged shape: {features_matched.shape}")

    # [FIX v1.3] Defensive group distribution logging
    if "group" in features_matched.columns:
        logger.info(f"  Final group distribution: "
                    f"{features_matched['group'].value_counts().to_dict()}")
    if "disorder_group" in features_matched.columns:
        logger.info(f"  Final disorder distribution: "
                    f"{features_matched['disorder_group'].value_counts().to_dict()}")

    return features_matched, psm_df, psm_has_pairs


# ======================== PHASE 2: FEATURE IDENTIFICATION ========================

def identify_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], Dict]:
    """
    Properly identify numeric feature columns, excluding metadata,
    quality flags, non-numeric columns, and all-NaN columns.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: Identifying feature columns")
    logger.info("=" * 60)

    # Load feature metadata
    meta_path = os.path.join(FEATURES_DIR, FEATURE_METADATA_FILE)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Feature metadata not found: {meta_path}. "
            f"Run s01_feature_extraction.py first."
        )

    with open(meta_path, 'r') as f:
        feature_meta = json.load(f)

    # Collect all candidate feature columns from metadata
    candidate_cols = []
    for dim, cols in feature_meta.items():
        if isinstance(cols, list):
            candidate_cols.extend(cols)

    # Add log-transformed features that exist in the DataFrame
    log_cols = [c for c in df.columns if c.endswith("_log") or c.endswith("_log_signed")]
    candidate_cols.extend(log_cols)

    # Deduplicate while preserving order
    candidate_cols = list(dict.fromkeys(candidate_cols))

    # Metadata / non-feature columns to always exclude
    always_exclude = {
        'user_id_hash', 'group', 'disorder_group', 'is_retrospective',
        't0_date', 'analysis_timestamp', 'split', 'strat_key',
        'pair_id', 'data_version', 'patch_applied',
        'primary_subreddit',  # string column in community features
        'is_temporal_structurally_imputed',  # flag, not feature
    }

    feature_cols = []
    excluded_cols = []

    # [FIX v1.3] Also detect columns from PSM merge that aren't features
    psm_artifact_cols = {"ps_score", "iptw_att_weight", "eb_weight",
                         "match_strata", "distance", "caliper",
                         "pair_id", "ps_score_psm"}

    for col in candidate_cols:
        if col not in df.columns:
            excluded_cols.append((col, "not_in_dataframe"))
            continue

        if col in always_exclude or col in psm_artifact_cols:
            excluded_cols.append((col, "metadata_or_psm_column"))
            continue

        if df[col].dtype == object:
            excluded_cols.append((col, "non_numeric"))
            continue

        # Check if column has any valid numeric values
        if df[col].notna().sum() == 0:
            excluded_cols.append((col, "all_nan"))
            continue

        feature_cols.append(col)

    # [FIX v1.3] Also scan DataFrame for numeric columns that should be excluded
    # (quality flags, source percentages, PSM artifacts, etc.)
    metadata_prefixes = ('quality_', 'pct_', 'source_')
    feature_cols = [c for c in feature_cols if not c.startswith(metadata_prefixes)]

    # [FIX v1.3] Also exclude any _x/_y suffixed columns from failed merges
    feature_cols = [c for c in feature_cols
                    if not c.endswith('_x') and not c.endswith('_y')]

    logger.info(f"  Feature columns identified: {len(feature_cols)}")
    logger.info(f"  Excluded columns: {len(excluded_cols)}")
    for col, reason in excluded_cols[:15]:
        logger.info(f"    Excluded: {col} ({reason})")
    if len(excluded_cols) > 15:
        logger.info(f"    ... and {len(excluded_cols) - 15} more")

    # Check for infinite values
    inf_counts = {}
    for col in feature_cols:
        n_inf = np.isinf(df[col].dropna()).sum()
        if n_inf > 0:
            inf_counts[col] = int(n_inf)

    if inf_counts:
        logger.warning(f"  Found infinite values in {len(inf_counts)} columns:")
        for col, count in inf_counts.items():
            logger.warning(f"    {col}: {count} inf values")

    # [FIX v1.3] Log feature dimensions from metadata
    for dim, cols in feature_meta.items():
        if isinstance(cols, list):
            existing = [c for c in cols if c in feature_cols]
            logger.info(f"    {dim}: {len(existing)}/{len(cols)} columns "
                        f"retained as features")

    # [FIX] 检测并移除 log 对中的原始特征，避免完全共线性
    # 策略：保留 _log 或 _log_signed 版本（分布更正态），移除原始版本
    log_suffixes = ('_log', '_log_signed')
    original_cols_to_drop = set()
    
    for col in feature_cols:
        if col.endswith(log_suffixes):
            for suffix in log_suffixes:
                if col.endswith(suffix):
                    original_col = col[:-len(suffix)]
                    if original_col in feature_cols and original_col not in original_cols_to_drop:
                        original_cols_to_drop.add(original_col)
                    break
    
    if original_cols_to_drop:
        logger.info(f"  检测到 {len(original_cols_to_drop)} 对 log 共线特征，"
                   f"移除原始版本保留 log 版本:")
        for orig in sorted(original_cols_to_drop):
            logger.info(f"    移除: {orig}")
        
        feature_cols = [c for c in feature_cols if c not in original_cols_to_drop]

    return feature_cols, excluded_cols, feature_meta


# ======================== PHASE 3: DATA QUALITY & SPLITTING ========================

def create_analysis_splits(df, feature_cols, psm_has_pairs, psm_df):
    """
    Create train/validation/test splits.
    
    [FIX v1.3] 
      - Fixed pair_strat logic (was identical branches)
      - Vectorized pair stratification
      - Added fallback for rare stratification categories
    """
    logger.info("=" * 60)
    logger.info("PHASE 3: Creating analysis splits (70/15/15)")
    logger.info("=" * 60)

    # Stratification key: disorder_group × group
    df["strat_key"] = (df["disorder_group"].astype(str) + "_" +
                       df["group"].astype(str))
    strat_col = df["strat_key"].values

    # [FIX v1.3] Check stratification category sizes
    strat_counts = df["strat_key"].value_counts()
    min_category_size = strat_counts.min()
    logger.info(f"  Stratification categories: {len(strat_counts)}")
    logger.info(f"  Min category size: {min_category_size}")

    # [NEW 6.5] 罕见分层键合并，防止小样本导致 StratifiedShuffleSplit 崩溃
    MIN_STRAT_CELL_SIZE = 20
    rare_cats = strat_counts[strat_counts < MIN_STRAT_CELL_SIZE].index.tolist()
    
    if rare_cats:
        logger.warning(f"  发现 {len(rare_cats)} 个稀有分层键 (n<{MIN_STRAT_CELL_SIZE}): {rare_cats}")
        logger.warning(f"  将其合并为 'rare_disorder_patient' / 'rare_disorder_control'")
        
        # 将罕见键合并
        df["strat_key"] = df["strat_key"].apply(
            lambda x: ("rare_disorder_" + x.split("_")[-1]) if x in rare_cats else x
        )
        strat_col = df["strat_key"].values
        strat_counts = df["strat_key"].value_counts()
        logger.info(f"  合并后分层键数量: {len(strat_counts)}")

    if min_category_size < 5:
        rare_cats = strat_counts[strat_counts < 5].index.tolist()
        logger.warning(f"  Rare stratification categories (n<5): {rare_cats}")
        logger.warning(f"  These may cause splitting failures. "
                      f"Consider merging rare categories.")

    if psm_has_pairs and "pair_id" in df.columns:
        # [FIX v1.3] Pair-aware splitting with FIXED stratification
        logger.info("  Using PAIR-AWARE splitting (PSM matched pairs)")

        # [FIX v1.3] Vectorized pair stratification
        # Priority: use patient's strat_key for each pair
        patient_strat = (
            df[df["group"] == "patient"]
            .groupby("pair_id")["strat_key"]
            .first()
            .to_dict()
        )
        # Fallback: for pairs without patient (shouldn't happen), use control
        control_strat = (
            df[df["group"] == "control"]
            .groupby("pair_id")["strat_key"]
            .first()
            .to_dict()
        )
        pair_strat = {}
        for pid in set(df["pair_id"].dropna().unique()):
            if pid in patient_strat:
                pair_strat[pid] = patient_strat[pid]
            elif pid in control_strat:
                pair_strat[pid] = control_strat[pid]
            else:
                pair_strat[pid] = "unknown_unknown"

        unique_pairs = list(pair_strat.keys())
        pair_strat_labels = [pair_strat[p] for p in unique_pairs]

        logger.info(f"  Total pairs to split: {len(unique_pairs)}")

        # [FIX v1.3] Handle rare categories in pair stratification
        pair_strat_array = np.array(pair_strat_labels)
        pair_strat_counts = pd.Series(pair_strat_array).value_counts()

        # Split at the pair level
        pair_array = np.arange(len(unique_pairs))

        # [FIX v1.3] If some categories are too small for stratification,
        # fall back to non-stratified splitting for those
        try:
            # First split: 70% train vs 30% holdout
            sss1 = StratifiedShuffleSplit(
                n_splits=1, test_size=0.30, random_state=RANDOM_SEED
            )
            train_pair_idx, holdout_pair_idx = next(
                sss1.split(pair_array, pair_strat_array)
            )

            # Second split: 50/50 of holdout → 15% val + 15% test
            holdout_strat = pair_strat_array[holdout_pair_idx]
            sss2 = StratifiedShuffleSplit(
                n_splits=1, test_size=0.50, random_state=RANDOM_SEED
            )
            val_pair_idx_rel, test_pair_idx_rel = next(
                sss2.split(np.zeros(len(holdout_pair_idx)), holdout_strat)
            )
        except ValueError as e:
            logger.warning(f"  Stratified splitting failed: {e}")
            logger.warning(f"  Falling back to non-stratified pair splitting")
            # Non-stratified fallback
            rng = np.random.RandomState(RANDOM_SEED)
            shuffled = rng.permutation(len(unique_pairs))
            n_train = int(0.70 * len(unique_pairs))
            n_val = int(0.15 * len(unique_pairs))
            train_pair_idx = shuffled[:n_train]
            holdout_pair_idx = shuffled[n_train:]
            val_pair_idx_rel = np.arange(n_val)
            test_pair_idx_rel = np.arange(n_val, len(holdout_pair_idx))

        train_pairs = set(unique_pairs[i] for i in train_pair_idx)
        val_pairs = set(unique_pairs[holdout_pair_idx[i]]
                        for i in val_pair_idx_rel)
        test_pairs = set(unique_pairs[holdout_pair_idx[i]]
                         for i in test_pair_idx_rel)

        df["split"] = "train"
        df.loc[df["pair_id"].isin(val_pairs), "split"] = "val"
        df.loc[df["pair_id"].isin(test_pairs), "split"] = "test"

    else:
        # Standard stratified splitting (no pair structure)
        logger.info("  Using STANDARD stratified splitting (no pair structure)")

        try:
            sss1 = StratifiedShuffleSplit(
                n_splits=1, test_size=0.30, random_state=RANDOM_SEED
            )
            train_idx, holdout_idx = next(sss1.split(df, strat_col))

            holdout_strat = strat_col[holdout_idx]
            sss2 = StratifiedShuffleSplit(
                n_splits=1, test_size=0.50, random_state=RANDOM_SEED
            )
            val_idx_rel, test_idx_rel = next(
                sss2.split(np.zeros(len(holdout_idx)), holdout_strat)
            )
            val_idx = holdout_idx[val_idx_rel]
            test_idx = holdout_idx[test_idx_rel]

            df["split"] = "train"
            df.iloc[val_idx, df.columns.get_loc("split")] = "val"
            df.iloc[test_idx, df.columns.get_loc("split")] = "test"

        except ValueError as e:
            logger.warning(f"  Stratified splitting failed: {e}")
            logger.warning(f"  Falling back to random splitting")
            rng = np.random.RandomState(RANDOM_SEED)
            indices = rng.permutation(len(df))
            n_train = int(0.70 * len(df))
            n_val = int(0.15 * len(df))
            df["split"] = "train"
            df.iloc[indices[n_train:n_train + n_val],
                    df.columns.get_loc("split")] = "val"
            df.iloc[indices[n_train + n_val:],
                    df.columns.get_loc("split")] = "test"

    # Report split statistics
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        logger.info(f"  {split_name}: {len(split_df)} users "
                    f"({len(split_df)/len(df)*100:.1f}%)")
        if 'disorder_group' in split_df.columns:
            dist = split_df['disorder_group'].value_counts().to_dict()
            logger.info(f"    Distribution: {dist}")

    # Verify no pair is split across train/val/test
    if psm_has_pairs and "pair_id" in df.columns:
        pair_split_counts = df.groupby("pair_id")["split"].nunique()
        broken_pairs = (pair_split_counts > 1).sum()
        if broken_pairs > 0:
            logger.error(f"  🔴 {broken_pairs} pairs are split across "
                        f"different splits!")
        else:
            logger.info(f"  ✅ All {len(pair_split_counts)} pairs kept "
                        f"intact within splits")

    return df


# ======================== PHASE 4: MISSING VALUES ========================

def handle_missing_values(df, feature_cols, feature_meta):
    """
    Handle missing values in features.
    """
    logger.info("=" * 60)
    logger.info("PHASE 4: Handling missing values (train-statistics only)")
    logger.info("=" * 60)

    train_mask = df["split"] == "train"

    # --- Per-user missingness: exclude users with >50% missing features ---
    missing_per_user = df[feature_cols].isnull().mean(axis=1)
    high_missing_mask = missing_per_user > 0.5
    n_excluded = high_missing_mask.sum()

    if n_excluded > 0:
        logger.info(f"  Excluding {n_excluded} users with >50% missing features "
                    f"({n_excluded/len(df)*100:.1f}%)")

        if "pair_id" in df.columns:
            excluded_pairs = set(df.loc[high_missing_mask, "pair_id"].dropna().unique())
            pair_excluded_mask = df["pair_id"].isin(excluded_pairs)
            pair_excluded_mask = pair_excluded_mask & ~df["pair_id"].isna()
            pair_excluded_mask = pair_excluded_mask | high_missing_mask
            n_pair_excluded = pair_excluded_mask.sum()
            logger.info(f"  Pair-aware exclusion: {n_pair_excluded} total users "
                        f"({len(excluded_pairs)} pairs)")
            df = df[~pair_excluded_mask].copy()
        else:
            df = df[~high_missing_mask].copy()

        train_mask = df["split"] == "train"

    # --- Structural missingness analysis ---
    temporal_meta_keys = feature_meta.get('temporal', []) + \
                         feature_meta.get('temporal_linguistic', [])
    temporal_features = [c for c in temporal_meta_keys if c in feature_cols]
    temporal_features += [f"{f}_log" for f in temporal_meta_keys
                         if f"{f}_log" in feature_cols and f"{f}_log" not in temporal_features]
    non_temporal_features = [c for c in feature_cols if c not in temporal_features]

    logger.info(f"  Temporal features: {len(temporal_features)}, "
                f"Non-temporal: {len(non_temporal_features)}")

    if "quality_is_temporal_reliable" in df.columns:
        temporal_unreliable_mask = df["quality_is_temporal_reliable"] == False
        n_structurally_missing = temporal_unreliable_mask.sum()

        logger.info(f"  Structural missingness analysis:")
        logger.info(f"    Users with unreliable temporal data: "
                    f"{n_structurally_missing} "
                    f"({n_structurally_missing/len(df)*100:.1f}%)")

        if n_structurally_missing > 0 and len(temporal_features) > 0:
            struct_missing_pct = df.loc[
                temporal_unreliable_mask, temporal_features
            ].isnull().mean().mean()
            nonstruct_missing_pct = df.loc[
                temporal_unreliable_mask, non_temporal_features
            ].isnull().mean().mean()
            logger.info(f"    Temporal feature missingness (these users): "
                        f"{struct_missing_pct*100:.1f}%")
            logger.info(f"    Non-temporal feature missingness (these users): "
                        f"{nonstruct_missing_pct*100:.1f}%")

            df["is_temporal_structurally_imputed"] = False
            df.loc[temporal_unreliable_mask,
                   "is_temporal_structurally_imputed"] = True

            logger.info(f"    Added is_temporal_structurally_imputed flag "
                        f"({n_structurally_missing} users marked)")

    # --- Per-feature missingness report ---
    missing_per_feature = df[feature_cols].isnull().mean()
    logger.info(f"  Feature missingness range: "
                f"{missing_per_feature.min():.3f} - {missing_per_feature.max():.3f}")

    high_missing_features = missing_per_feature[missing_per_feature > 0.3]
    if len(high_missing_features) > 0:
        logger.warning(f"  {len(high_missing_features)} features with "
                      f">30% missingness:")
        for feat, pct in high_missing_features.items():
            feat_type = "TEMPORAL" if feat in temporal_features else "OTHER"
            logger.warning(f"    [{feat_type}] {feat}: {pct:.1%}")

    # --- Compute imputation statistics from TRAINING SET ONLY ---
    logger.info("  Computing imputation statistics from training set...")

    train_df = df[train_mask]

    # [FIX] 移除按 disorder_group 分组插补，改用训练集全局中位数，防止标签泄露
    # 缺失模式本身可能包含诊断信息，imputed_flag 已经捕获了这一信号
    imputation_map = {}
    for col in feature_cols:
        global_median = train_df[col].median()
        imputation_map[col] = {
            "global_median": float(global_median) if pd.notna(global_median) else None
        }

    # ================================================================
    # [FIX v1.3] Build imputed_flags using pd.concat (NOT loop insert)
    # Eliminates DataFrame fragmentation warning
    # ================================================================
    logger.info("  Building imputation flags...")

    # 构建 imputed_flags 时加入 user_id_hash
    # [FIX] 必须包含 user_id_hash 以保证后续对齐安全
    imputed_flag_dict = {"user_id_hash": df["user_id_hash"].values}
    
    for col in feature_cols:
        imputed_flag_dict[f"{col}_was_imputed"] = df[col].isnull().astype(np.int8)

    if "is_temporal_structurally_imputed" in df.columns:
        imputed_flag_dict["is_temporal_structurally_imputed"] = \
            df["is_temporal_structurally_imputed"].astype(np.int8)

    # Single concat → no fragmentation
    imputed_flags = pd.DataFrame(imputed_flag_dict)
    del imputed_flag_dict

    logger.info(f"  Imputation flags shape: {imputed_flags.shape}")

    # [FIX] 应用全局中位数插补（无标签泄露）
    n_imputed_total = 0
    for col in feature_cols:
        global_median = imputation_map[col]["global_median"]
        
        missing_mask = df[col].isnull()
        n_missing = missing_mask.sum()
        
        if n_missing > 0:
            if global_median is not None:
                df.loc[missing_mask, col] = global_median
            else:
                logger.warning(f"    {col}: The global median is NaN, filled with 0")
                df.loc[missing_mask, col] = 0
            n_imputed_total += n_missing

    logger.info(f"  Total values imputed: {n_imputed_total}")

    # Handle infinite values
    # [FIX 6.6] 使用训练集的分位数替代无穷值，防止被极端离群点带偏
    n_inf_total = 0
    for col in feature_cols:
        inf_mask = np.isinf(df[col])
        n_inf = inf_mask.sum()
        if n_inf > 0:
            # 仅从训练集计算鲁棒边界 (与 Winsorization 逻辑对齐)
            train_col_vals = df.loc[train_mask, col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(train_col_vals) > 0:
                upper_bound = train_col_vals.quantile(0.995)
                lower_bound = train_col_vals.quantile(0.005)
                df.loc[df[col] == np.inf, col] = upper_bound
                df.loc[df[col] == -np.inf, col] = lower_bound
            else:
                df.loc[inf_mask, col] = 0
            n_inf_total += n_inf

    if n_inf_total > 0:
        logger.info(f"  Replaced {n_inf_total} infinite values with "
                    f"finite bounds")

    remaining_missing = df[feature_cols].isnull().sum().sum()
    remaining_inf = sum(np.isinf(df[col]).sum() for col in feature_cols)
    logger.info(f"  Remaining missing values: {remaining_missing}")
    logger.info(f"  Remaining infinite values: {remaining_inf}")

    # Save imputation map
    imputation_path = os.path.join(FEATURES_DIR, "imputation_params.json")
    with open(imputation_path, 'w') as f:
        json.dump(imputation_map, f, indent=2)
    logger.info(f"  Saved imputation parameters to: {imputation_path}")

    return df, imputed_flags


def winsorize_features(df, feature_cols, lower=0.5, upper=99.5):
    """
    Winsorize extreme outliers at specified percentiles.
    Percentiles computed from TRAINING SET ONLY.
    """
    logger.info("=" * 60)
    logger.info(f"PHASE 5: Winsorizing features at [{lower}, {upper}] percentiles")
    logger.info("  (percentiles from training set only)")
    logger.info("=" * 60)

    train_mask = df["split"] == "train"
    train_df = df[train_mask]

    winsorize_params = {}
    total_clipped = 0

    for col in feature_cols:
        if col not in df.columns:
            continue

        train_vals = train_df[col].dropna()
        if len(train_vals) < 10:
            continue

        lo_val = float(np.percentile(train_vals, lower))
        hi_val = float(np.percentile(train_vals, upper))

        # Skip if lo == hi (constant feature)
        if lo_val == hi_val:
            continue

        winsorize_params[col] = {"lower": lo_val, "upper": hi_val}

        n_clipped_lo = (df[col] < lo_val).sum()
        n_clipped_hi = (df[col] > hi_val).sum()
        n_clipped = int(n_clipped_lo + n_clipped_hi)

        if n_clipped > 0:
            df[col] = df[col].clip(lo_val, hi_val)
            total_clipped += n_clipped

    logger.info(f"  Total values clipped: {total_clipped}")

    # Save winsorization parameters
    winsorize_path = os.path.join(FEATURES_DIR, "winsorize_params.json")
    with open(winsorize_path, 'w') as f:
        json.dump(winsorize_params, f, indent=2)
    logger.info(f"  Saved winsorization parameters to: {winsorize_path}")

    return df


def standardize_features_train_only(df, feature_cols):
    """
    Standardize features using RobustScaler (median/IQR).
    Fit on training set only. Handle zero-IQR features.
    
    [FIX v1.3] Zero-IQR features are DROPPED from the feature set
    rather than set to 0, because:
      1. They provide zero discriminative power
      2. Setting to 0 creates misleading "no signal" for downstream models
      3. Duplicate features (inhibition_rate == inhibition_marker_rate)
         are detected and one is dropped
    """
    logger.info("=" * 60)
    logger.info("PHASE 6: Standardizing features (RobustScaler, train-fit only)")
    logger.info("=" * 60)

    train_mask = df["split"] == "train"

    # Detect zero-IQR features before scaling
    train_df = df.loc[train_mask, feature_cols]
    zero_iqr_cols = []
    valid_feature_cols = []

    for col in feature_cols:
        if col not in df.columns:
            continue
        q75 = train_df[col].quantile(0.75)
        q25 = train_df[col].quantile(0.25)
        iqr = q75 - q25
        if iqr < 1e-10:
            zero_iqr_cols.append(col)
        else:
            valid_feature_cols.append(col)

    if zero_iqr_cols:
        logger.warning(f"  {len(zero_iqr_cols)} features have zero IQR "
                      f"(constant or near-constant in training set):")
        for col in zero_iqr_cols:
            n_unique = train_df[col].nunique()
            n_nonzero = (train_df[col] != 0).sum()
            pct_nonzero = n_nonzero / len(train_df) * 100
            logger.warning(
                f"    {col}: median={train_df[col].median():.6f}, "
                f"unique_values={n_unique}, "
                f"nonzero={pct_nonzero:.2f}%"
            )

        # [FIX v1.3] Check for duplicate features among zero-IQR
        # If two features are computed identically (like inhibition_rate
        # and inhibition_marker_rate), detect and flag them
        if len(zero_iqr_cols) >= 2:
            logger.info("  Checking for duplicate features among zero-IQR set...")
            for i, col_a in enumerate(zero_iqr_cols):
                for col_b in zero_iqr_cols[i+1:]:
                    if train_df[col_a].equals(train_df[col_b]):
                        logger.warning(
                            f"    DUPLICATE: {col_a} == {col_b} "
                            f"(identical values in training set)"
                        )

        # [FIX v1.3] Decision: drop zero-IQR features from modeling set
        # They are saved in the output file for transparency but
        # excluded from valid_feature_cols (used by downstream models)
        logger.info(f"  These {len(zero_iqr_cols)} features will be DROPPED "
                    f"from the modeling feature set (zero discriminative power)")
        logger.info(f"  They remain in the output parquet for audit purposes")

    logger.info(f"  Valid features for scaling: {len(valid_feature_cols)}")

    # Fit scaler on valid features only
    if len(valid_feature_cols) == 0:
        logger.error("  No valid features for scaling!")
        return df.copy(), None, valid_feature_cols, zero_iqr_cols

    # [FIX] 检测零膨胀且 IQR 极小的特征，先进行 log1p 变换压缩长尾
    # 否则 RobustScaler (x-median)/IQR 会产生极端异常值
    df_standardized = df.copy()
    zero_inflated_transformed = []
    for col in valid_feature_cols:
        zero_rate = (train_df[col] == 0).mean()
        if zero_rate > 0.80:
            iqr = train_df[col].quantile(0.75) - train_df[col].quantile(0.25)
            if 1e-10 < iqr < 1.0:
                df_standardized[col] = np.log1p(df_standardized[col])  # 修改副本
                zero_inflated_transformed.append((col, zero_rate, iqr))
    
    if zero_inflated_transformed:
        logger.warning(f"  检测到 {len(zero_inflated_transformed)} 个零膨胀+微IQR特征，"
                      f"已在标准化前应用 log1p 变换:")
        for col, zr, iqr in zero_inflated_transformed:
            logger.warning(f"    {col}: zero_rate={zr:.2%}, IQR={iqr:.4f}")
        
    scaler = RobustScaler()
    scaler.fit(df_standardized.loc[train_mask, valid_feature_cols].values)
    df_standardized[valid_feature_cols] = scaler.transform(df_standardized[valid_feature_cols].values)

    # [FIX v1.3] Zero-IQR features: keep original values (not set to 0)
    # This preserves the rare non-zero values for potential manual inspection
    # Downstream models should use valid_feature_cols only
    # for col in zero_iqr_cols:
    #     df_standardized[col] = 0.0  

    # Save scaler parameters
    scaler_params = {
        "valid_features": valid_feature_cols,
        "zero_iqr_features": zero_iqr_cols,
        "dropped_features_reason": {
            col: "zero_IQR_constant" for col in zero_iqr_cols
        },
        "center": dict(zip(valid_feature_cols, scaler.center_.tolist())),
        "scale": dict(zip(valid_feature_cols, scaler.scale_.tolist())),
        "zero_inflated_log1p_columns": [col for col, _, _ in zero_inflated_transformed],
    }
    scaler_path = os.path.join(FEATURES_DIR, "robust_scaler_params.json")
    with open(scaler_path, 'w') as f:
        json.dump(scaler_params, f, indent=2)
    logger.info(f"  Saved scaler parameters to: {scaler_path}")

    # [FIX v1.3] Save feature selection manifest for downstream clarity
    feature_selection = {
        "total_candidate_features": len(feature_cols),
        "valid_features": valid_feature_cols,
        "n_valid_features": len(valid_feature_cols),
        "dropped_features": zero_iqr_cols,
        "n_dropped_features": len(zero_iqr_cols),
        "drop_reason": "zero_IQR_in_training_set",
        # [NEW 6.3] 显式标记权重列，严禁下游作为特征输入
        "psm_weight_columns": [c for c in df.columns if c in 
                               {"iptw_att_weight", "ps_score", "eb_weight", "match_strata"}],
        "sample_weight_recommendation": "Use psm_weight_columns as sample_weight in model.fit(), NOT as features.",
        "selection_timestamp": pd.Timestamp.now().isoformat(),
    }
    selection_path = os.path.join(FEATURES_DIR, "feature_selection.json")
    with open(selection_path, 'w') as f:
        json.dump(feature_selection, f, indent=2)
    logger.info(f"  Saved feature selection manifest to: {selection_path}")

    return df_standardized, scaler, valid_feature_cols, zero_iqr_cols


# ======================== PHASE 7: QUALITY REPORT ========================

def generate_quality_report(df, df_standardized, feature_cols, valid_feature_cols,
                           zero_iqr_cols, feature_meta, psm_has_pairs):
    """Generate comprehensive data quality report."""
    logger.info("=" * 60)
    logger.info("PHASE 7: Generating quality report")
    logger.info("=" * 60)

    # [FIX v1.3] Defensive column access throughout
    train_mask = df["split"] == "train" if "split" in df.columns else np.ones(len(df), dtype=bool)
    val_mask = df["split"] == "val" if "split" in df.columns else np.zeros(len(df), dtype=bool)
    test_mask = df["split"] == "test" if "split" in df.columns else np.zeros(len(df), dtype=bool)

    # Group counts with defensive access
    n_patients = int((df["group"] == "patient").sum()) if "group" in df.columns else 0
    n_controls = int(df["group"].isin(["control", "control_candidate"]).sum()) if "group" in df.columns else 0

    disorder_dist = {}
    if "disorder_group" in df.columns and "group" in df.columns:
        disorder_dist = df[df["group"] == "patient"]["disorder_group"].value_counts().to_dict()

    split_dist = df["split"].value_counts().to_dict() if "split" in df.columns else {}

    # Feature dimension counts
    feature_dim_counts = {}
    for dim, cols in feature_meta.items():
        if isinstance(cols, list):
            feature_dim_counts[dim] = len(cols)

    # Data quality metrics with defensive computation
    remaining_missing = 0
    remaining_inf = 0
    for col in feature_cols:
        if col in df.columns:
            remaining_missing += int(df[col].isnull().sum())
            remaining_inf += int(np.isinf(df[col]).sum())

    report = {
        "pipeline_version": "v1.3",
        "timestamp": pd.Timestamp.now().isoformat(),

        # Sample sizes
        "total_users": len(df),
        "patients": n_patients,
        "controls": n_controls,
        "disorders": disorder_dist,

        # Splits
        "splits": split_dist,
        "split_ratios": {
            split: f"{len(df[df['split']==split])/len(df)*100:.1f}%"
            for split in ["train", "val", "test"]
            if "split" in df.columns
        },

        # Features
        "n_features_total": len(feature_cols),
        "n_features_valid": len(valid_feature_cols),
        "n_features_zero_iqr": len(zero_iqr_cols),
        "feature_dimensions": feature_dim_counts,

        # PSM
        "psm_pair_structure": psm_has_pairs,

        # Data quality
        "remaining_missing": remaining_missing,
        "remaining_infinite": remaining_inf,

        # Leakage prevention checklist
        "leakage_prevention": {
            "imputation_source": "train_only",
            "winsorization_source": "train_only",
            "scaler_fit_source": "train_only",
            "split_before_preprocessing": True,
        },
    }

    # Save report
    os.makedirs(STATS_DIR, exist_ok=True)
    report_path = os.path.join(STATS_DIR, "aggregation_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"  Saved quality report to: {report_path}")

    # Print summary
    logger.info(f"\n  SUMMARY:")
    logger.info(f"    Users: {report['total_users']} "
                f"(patients={report['patients']}, controls={report['controls']})")
    logger.info(f"    Features: {report['n_features_valid']} valid, "
                f"{report['n_features_zero_iqr']} constant (zero IQR)")
    logger.info(f"    Splits: train={int(train_mask.sum())}, "
                f"val={int(val_mask.sum())}, test={int(test_mask.sum())}")
    logger.info(f"    Leakage prevention: ✅ All statistics from training set only")

    return report


# ======================== MAIN PIPELINE ========================

def run_aggregation():
    """
    Main pipeline for feature aggregation.
    
    Execution order (CRITICAL for leakage prevention):
      1. Load & merge data sources
      2. Identify feature columns (exclude non-numeric)
      3. Split into train/val/test FIRST
      4. Handle missing values (using training set statistics)
      5. Winsorize (using training set percentiles)
      6. Standardize (fit on training set only)
      7. Save outputs
      8. Generate quality report
    """
    logger.info("=" * 80)
    logger.info("MODULE 1 - Step 02: Feature Aggregation & Matching Integration (v1.3)")
    logger.info("CRITICAL: All preprocessing statistics derived from training set only")
    logger.info("=" * 80)

    # ============================================================
    # PHASE 1: Load & Merge
    # ============================================================
    df, psm_df, psm_has_pairs = load_and_merge()

    # [FIX v1.3] Post-merge validation
    if len(df) == 0:
        logger.error("Empty DataFrame after merge. Cannot proceed.")
        return None, None, None

    # ============================================================
    # PHASE 2: Identify Feature Columns
    # ============================================================
    feature_cols, excluded_cols, feature_meta = identify_feature_columns(df)

    if len(feature_cols) == 0:
        logger.error("No valid feature columns identified!")
        return None, None, None

    # ============================================================
    # PHASE 3: Split FIRST (before any preprocessing)
    # ============================================================
    df = create_analysis_splits(df, feature_cols, psm_has_pairs, psm_df)

    # ============================================================
    # PHASE 4: Handle Missing Values (train statistics only)
    # ============================================================
    df, imputed_flags = handle_missing_values(df, feature_cols, feature_meta)

    # ============================================================
    # PHASE 5: Winsorize (train percentiles only)
    # ============================================================
    df = winsorize_features(df, feature_cols, lower=0.5, upper=99.5)

    # ============================================================
    # PHASE 6: Standardize (fit on train only)
    # ============================================================
    df_standardized, scaler, valid_feature_cols, zero_iqr_cols = \
        standardize_features_train_only(df, feature_cols)

    # ============================================================
    # PHASE 7: Save Outputs
    # ============================================================
    logger.info("=" * 60)
    logger.info("Saving outputs...")
    logger.info("=" * 60)

    # Save imputation flags separately (not in main feature file)
    imputed_path = os.path.join(FEATURES_DIR, "imputation_flags.parquet")
    imputed_flags.to_parquet(imputed_path, index=False)
    logger.info(f"  Saved imputation flags: {imputed_path}")

    # Raw (unscaled) version — for descriptive statistics
    raw_output = os.path.join(FEATURES_DIR, "user_features_matched.parquet")
    df.to_parquet(raw_output, index=False)
    logger.info(f"  Saved matched raw features: {raw_output} ({df.shape})")

    # Standardized version — for modeling
    std_output = os.path.join(FEATURES_DIR, "user_features_standardized.parquet")
    df_standardized.to_parquet(std_output, index=False)
    logger.info(f"  Saved standardized features: {std_output} "
                f"({df_standardized.shape})")

    # ============================================================
    # PHASE 8: Quality Report
    # ============================================================
    report = generate_quality_report(
        df, df_standardized, feature_cols, valid_feature_cols,
        zero_iqr_cols, feature_meta, psm_has_pairs
    )

    logger.info("=" * 80)
    logger.info("AGGREGATION COMPLETE")
    logger.info("=" * 80)

    return df, df_standardized, valid_feature_cols


if __name__ == "__main__":
    run_aggregation()