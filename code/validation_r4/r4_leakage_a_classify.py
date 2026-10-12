"""
R4-1 (classifier stage, s04-matching): Leakage A AUROC on cached nodiag features
===============================================================================
Loads the cached nodiag feature matrix (r4_leakage_a_features.parquet, produced
by the long extraction run), then runs EXACTLY the Module1 s04 logistic_l2
disorder-vs-control protocol:
  - controls subsampled to max(3*n_case, 5000), CONTROL_SUBSAMPLE_TRIALS=5 trials
  - LogisticRegressionCV(Cs=10, cv=3, l1_ratios=(0,), solver="lbfgs",
        max_iter=2000, scoring="roc_auc", n_jobs=-1, class_weight="balanced")
  - 5-fold stratified CV (CV_FOLDS=5), RANDOM_SEED=2025; trial seeds 2025+i
  - features standardized with the ORIGINAL saved RobustScaler/winsorize params
Compares per-disorder AUROC to the atlas values (auroc_vs_control.csv).

Outputs (NAS): r4_leakage_a_results.csv, r4_leakage_a_report.json
"""
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

PROJ = "/path/to/code"
sys.path.insert(0, os.path.join(PROJ, "Module1"))
from config_module1 import FEATURES_DIR, PSM_MATCHED_PATH  # noqa: E402

R4 = "/path/to/project/module4_results/validation_r4"
M1_STAT = "/path/to/project/module1_results/statistics"
FEATURES_PQ = os.path.join(R4, "r4_leakage_a_features.parquet")

RANDOM_SEED = 2025
CV_FOLDS = 5
SUBSAMPLE_RATIO = 3
SUBSAMPLE_MIN = 5000
SUBSAMPLE_TRIALS = 5
DISORDERS = ["depression", "bipolar", "anxiety", "ocd", "ptsd", "bpd",
             "eating_disorders", "schizophrenia", "adhd", "autism"]


def transform_features(feat_df):
    with open(os.path.join(FEATURES_DIR, "robust_scaler_params.json")) as f:
        sp = json.load(f)
    with open(os.path.join(FEATURES_DIR, "winsorize_params.json")) as f:
        wp = json.load(f)
    cols = [c for c in sp["center"].keys() if c in feat_df.columns]
    X = feat_df[cols].copy()
    for c in cols:
        lo, hi = wp.get(c, {}).get("lower"), wp.get(c, {}).get("upper")
        if lo is not None and hi is not None:
            X[c] = X[c].clip(lo, hi)
    center = np.array([sp["center"][c] for c in cols])
    scale = np.array([sp["scale"][c] for c in cols])
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    return (X.values - center) / scale, cols


def subsample_controls(n_patient, control_idx, rng):
    target = min(max(SUBSAMPLE_RATIO * n_patient, SUBSAMPLE_MIN), len(control_idx))
    return rng.choice(control_idx, size=target, replace=False)


def fresh_clf():
    return LogisticRegressionCV(
        Cs=10, cv=3, l1_ratios=(0,), solver="lbfgs", max_iter=2000,
        scoring="roc_auc", random_state=RANDOM_SEED, n_jobs=-1,
        class_weight="balanced", use_legacy_attributes=False)


def trial_auroc(X, y):
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = fresh_clf()
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    logger.info("loading cached nodiag features")
    feat = pd.read_parquet(FEATURES_PQ)
    feat = feat.drop(columns=["group", "disorder_group"], errors="ignore")
    logger.info(f"nodiag users: {len(feat)}")

    logger.info("loading PSM labels")
    psm = pd.read_parquet(PSM_MATCHED_PATH, columns=["user_id_hash", "group", "disorder_group"])
    lbl = psm.drop_duplicates("user_id_hash")[["user_id_hash", "group", "disorder_group"]]
    m = feat.merge(lbl, on="user_id_hash", how="inner")
    logger.info(f"labeled nodiag users: {len(m)}")

    X, cols = transform_features(m)
    logger.info(f"scaled features: {X.shape} cols={len(cols)}")

    # drop rows with undefined features (mirrors s04: valid = ~any(isnan, axis=1))
    nan_mask = np.any(np.isnan(X), axis=1)
    if nan_mask.sum():
        logger.info(f"dropping {int(nan_mask.sum())} rows with NaN features")
        X = X[~nan_mask]
        m = m.iloc[~nan_mask].reset_index(drop=True)
    logger.info(f"clean features: {X.shape}")

    orig = pd.read_csv(os.path.join(M1_STAT, "auroc_vs_control.csv"))
    orig_map = dict(zip(orig["disorder"], orig["auroc_vs_control"]))

    X_df = m[["user_id_hash", "group", "disorder_group"]].copy()
    # PSM labels use group='control_candidate' (patients vs candidate controls)
    is_ctrl = X_df["group"].isin(["control", "control_candidate"]).values
    ctrl_idx = np.where(is_ctrl)[0]

    results = []
    for d in DISORDERS:
        case_mask = ((X_df["group"].values == "patient") &
                     (X_df["disorder_group"].values == d) & (~is_ctrl))
        case_idx = np.where(case_mask)[0]
        if len(case_idx) < 20:
            logger.warning(f"{d}: only {len(case_idx)} cases, skip")
            continue
        trial_aucs, trial_stds = [], []
        for t in range(SUBSAMPLE_TRIALS):
            rng = np.random.RandomState(RANDOM_SEED + t)
            ctrl = subsample_controls(len(case_idx), ctrl_idx, rng)
            idx = np.concatenate([case_idx, ctrl])
            y = np.concatenate([np.ones(len(case_idx)), np.zeros(len(ctrl))])
            a, s = trial_auroc(X[idx], y)
            trial_aucs.append(a); trial_stds.append(s)
            logger.info(f"  {d} trial {t}: AUROC={a:.4f} (n_case={len(case_idx)}, "
                        f"n_ctrl={len(ctrl)})")
        mean_auc = float(np.mean(trial_aucs))
        results.append({
            "disorder": d, "n_case": int(len(case_idx)),
            "auroc_nodiag": mean_auc,
            "auroc_nodiag_min": float(np.min(trial_aucs)),
            "auroc_nodiag_max": float(np.max(trial_aucs)),
            "auroc_original": float(orig_map.get(d, np.nan)),
            "delta": float(mean_auc - orig_map.get(d, np.nan)) if d in orig_map else np.nan,
            "trial_aurocs": trial_aucs,
        })
        logger.info(f"{d}: nodiag AUROC={mean_auc:.4f} vs original {orig_map.get(d, np.nan):.4f} "
                    f"(delta {mean_auc - orig_map.get(d, np.nan):+.4f})")

    res = pd.DataFrame([{k: v for k, v in r.items() if k != "trial_aurocs"} for r in results])
    res.to_csv(os.path.join(R4, "r4_leakage_a_results.csv"), index=False)
    report = {"feature_cols": len(cols), "n_users": int(len(m)),
              "protocol": "s04 logistic_l2: 3x control subsample, 5 trials, LogisticRegressionCV lbfgs l2, 5-fold",
              "results": results}
    with open(os.path.join(R4, "r4_leakage_a_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("DONE -> %s", R4)


if __name__ == "__main__":
    main()
