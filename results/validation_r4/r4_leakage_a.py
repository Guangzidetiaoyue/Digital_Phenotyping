"""
R4-1: Leakage A — classification excluding diagnosis-triggering posts
=====================================================================
Re-estimates single-disorder vs control AUROC using user-level features
computed ONLY from posts that are NOT the self-report diagnosis posts
(is_diagnosis_post == False). Reuses Module1 s01's exact per-user aggregation
and the original RobustScaler/winsorize parameters, so the ONLY difference vs
the headline AUROC is the exclusion of diagnosis-adjacent language.

Pipeline (server):
  1. matched users (PSM) + group/disorder
  2. scan silver partitions, keep is_diagnosis_post != True for matched users
  3. aggregate_user_features per user (s01 function) -> nodiag feature matrix
  4. winsorize + RobustScaler using saved params (Module1 FEATURES_DIR)
  5. per-disorder LASSO-logistic CV AUROC vs control
  6. compare to original AUROC (module1_results/statistics/discrimination_disorder_vs_control.csv)

Output (NAS): r4_leakage_a_features.parquet, r4_leakage_a_results.csv/.json
"""
import os
import sys
import gc
import json
import logging
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

PROJ = "/path/to/code"
sys.path.insert(0, os.path.join(PROJ, "Module1"))
from s01_feature_extraction import (aggregate_user_features, get_partition_dirs,  # noqa: E402
                                    init_worker)
from config_module1 import (SILVER_DATA_DIR, FEATURES_DIR, PSM_MATCHED_PATH,  # noqa: E402
                            NUM_CORES)

R4 = "/path/to/project/module4_results/validation_r4"
os.makedirs(R4, exist_ok=True)
M1_STAT = "/path/to/project/module1_results/statistics"

MATCHED_UIDS = set()


def load_matched():
    df = pd.read_parquet(PSM_MATCHED_PATH,
                         columns=["user_id_hash", "group", "disorder_group"])
    return df


def process_partition_nodiag(part):
    """Scan one silver partition, keep matched users' non-diagnosis posts,
    aggregate per-user features (mirrors s01.process_partition but filtered)."""
    out = []
    n_total = 0
    n_drop = 0
    files = [f for f in os.listdir(part) if f.endswith(".parquet")]
    if not files:
        return out, n_total, n_drop
    user_data = defaultdict(list)
    for pf_name in files:
        pf = pq.ParquetFile(os.path.join(part, pf_name))
        for batch in pf.iter_batches(batch_size=50_000):
            df = batch.to_pandas()
            if "user_id_hash" not in df.columns:
                continue
            df = df[df["user_id_hash"].isin(MATCHED_UIDS)]
            if df.empty:
                continue
            n_total += len(df)
            if "is_diagnosis_post" in df.columns:
                n_drop += int(df["is_diagnosis_post"].fillna(False).astype(bool).sum())
                df = df[df["is_diagnosis_post"].fillna(False) != True]
            for uid, udf in df.groupby("user_id_hash"):
                user_data[uid].append(udf)
            del df
            gc.collect()
    for uid in list(user_data.keys()):
        dfs = user_data[uid]
        try:
            feats = aggregate_user_features(uid, pd.concat(dfs, ignore_index=True))
            if feats:
                out.append(feats)
        except Exception as e:
            logger.error(f"user {uid[:8]} failed: {e}")
        del user_data[uid]
    del user_data
    gc.collect()
    return out, n_total, n_drop


def extract_nodiag_features(matched_uids, n_cores=None):
    """Parallel (per-partition) nodiag feature extraction, mirroring s01."""
    global MATCHED_UIDS
    MATCHED_UIDS = matched_uids
    n_cores = n_cores or (NUM_CORES if NUM_CORES > 1 else cpu_count())
    parts = get_partition_dirs(SILVER_DATA_DIR)
    logger.info(f"partitions: {len(parts)}, cores: {n_cores}")
    rows = []
    n_total_posts, n_dropped_posts = 0, 0
    if n_cores > 1 and len(parts) > 1:
        with Pool(processes=n_cores, initializer=init_worker) as pool:
            for out, nt, nd in tqdm(pool.imap_unordered(process_partition_nodiag, parts),
                                    total=len(parts), desc="partitions"):
                rows.extend(out)
                n_total_posts += nt
                n_dropped_posts += nd
    else:
        for part in tqdm(parts, desc="partitions"):
            out, nt, nd = process_partition_nodiag(part)
            rows.extend(out)
            n_total_posts += nt
            n_dropped_posts += nd
    return pd.DataFrame(rows), {"total_posts_scanned": n_total_posts,
                                "diagnosis_posts_dropped": n_dropped_posts}


def load_scaler():
    with open(os.path.join(FEATURES_DIR, "robust_scaler_params.json")) as f:
        sp = json.load(f)
    with open(os.path.join(FEATURES_DIR, "winsorize_params.json")) as f:
        wp = json.load(f)
    return sp, wp


def transform_features(feat_df, sp, wp):
    cols = list(sp["center"].keys())
    cols = [c for c in cols if c in feat_df.columns]
    X = feat_df[cols].copy()
    # winsorize
    for c in cols:
        lo, hi = wp.get(c, {}).get("lower"), wp.get(c, {}).get("upper")
        if lo is not None and hi is not None:
            X[c] = X[c].clip(lo, hi)
    # robust scale
    center = np.array([sp["center"][c] for c in cols])
    scale = np.array([sp["scale"][c] for c in cols])
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    Xv = (X.values - center) / scale
    return Xv, cols


def cv_auroc(X, y, seed=2025):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(C=1.0, penalty="l1", solver="liblinear",
                                 max_iter=2000, random_state=seed)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    logger.info("loading matched cohort")
    psm = load_matched()
    # PSM file is user-level matched rows; keep patients (case) + controls
    matched_uids = set(psm["user_id_hash"].unique())
    logger.info(f"matched users: {len(matched_uids)}")

    logger.info("extracting nodiag features (long job)")
    feat_df, counts = extract_nodiag_features(matched_uids)
    feat_df.to_parquet(os.path.join(R4, "r4_leakage_a_features.parquet"), index=False)
    logger.info(f"nodiag users: {len(feat_df)}, counts={counts}")

    # merge group labels (keep per-user one row)
    lbl = psm.drop_duplicates("user_id_hash")[["user_id_hash", "group", "disorder_group"]]
    m = feat_df.merge(lbl, on="user_id_hash", how="inner")
    logger.info(f"labeled nodiag users: {len(m)}")

    sp, wp = load_scaler()
    X, cols = transform_features(m, sp, wp)
    logger.info(f"scaled features: {X.shape}, cols={len(cols)}")

    # original AUROC reference
    orig = pd.read_csv(os.path.join(M1_STAT, "discrimination_disorder_vs_control.csv"))
    orig_map = dict(zip(orig["disorder"], orig["auroc"])) if "auroc" in orig.columns else {}

    ctrl = m["group"] == "control"
    results = []
    for d in sorted(m.loc[~ctrl, "disorder_group"].dropna().unique()):
        mask = (~ctrl) & (m["disorder_group"] == d)
        if mask.sum() < 100:
            continue
        y = (mask | ctrl).astype(int)  # 1=case, 0=control
        auc, sd = cv_auroc(X, y)
        results.append({
            "disorder": d, "n_case": int(mask.sum()), "n_control": int(ctrl.sum()),
            "auroc_nodiag": auc, "auroc_nodiag_sd": sd,
            "auroc_original": float(orig_map.get(d, np.nan)),
            "delta": float(auc - orig_map.get(d, np.nan)) if d in orig_map else np.nan,
        })
        logger.info(f"{d}: nodiag AUROC={auc:.3f} vs original {orig_map.get(d, np.nan):.3f}")

    res = pd.DataFrame(results)
    res.to_csv(os.path.join(R4, "r4_leakage_a_results.csv"), index=False)
    report = {"counts": counts, "feature_cols": len(cols), "n_users": len(m), "results": results}
    with open(os.path.join(R4, "r4_leakage_a_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("DONE -> %s", R4)


if __name__ == "__main__":
    main()
