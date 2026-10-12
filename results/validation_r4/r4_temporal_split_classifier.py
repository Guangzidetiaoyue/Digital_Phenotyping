"""
R4-3b (v2, P0): Classifier-level temporal generalization — 93-feature atlas + train-only preprocessing
========================================================================================================
Revision addressing the round-5 audit:
  (1) use the IDENTICAL 93-feature primary-atlas set (98 scaler-mapped minus the
      5 circular subreddit features listed in effect_sizes_report.json);
  (2) STRICT train-only preprocessing: winsorize bounds (0.5/99.5 quantiles) and
      RobustScaler (median / IQR) are fitted on the TRAIN-era users only and
      applied to the TEST era (no test-era information in preprocessing);
  (3) forward transfer matrix: E1->E2, E1->E3, E2->E3.

Design otherwise unchanged: era anchored to each user's t0_date (real for
patients, shadow for controls); per disorder fit LogisticRegression(L2,
balanced, C=1.0) on train-era patients + 3:1 subsampled train-era controls;
evaluate AUROC on the FULL later-era patients+controls vs the atlas AUROC.

Input : module1_results/features/user_features_matched.parquet (raw features + t0)
Outputs (NAS): r4_temporal_split_93train_results.csv / _report.json
"""
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

PROJ = "/path/to/code"
FEAT = ("/path/to/project/module1_results/features/"
        "user_features_matched.parquet")
FEATURES_DIR = "/path/to/project/module1_results/features"
M1_STAT = "/path/to/project/module1_results/statistics"
R4 = "/path/to/project/module4_results/validation_r4"
OUT_BASE = "r4_temporal_split_93train"
RANDOM_SEED = 2025
DISORDERS = ["depression", "bipolar", "anxiety", "ocd", "ptsd", "bpd",
             "eating_disorders", "schizophrenia", "adhd", "autism"]
PAIRS = [("E1", "E2"), ("E1", "E3"), ("E2", "E3")]


def era_of(year):
    if year <= 2015:
        return "E1"
    if year <= 2019:
        return "E2"
    if year <= 2022:
        return "E3"
    return None


def robust_fit_train_only(train_vals, q_lo=0.5, q_hi=99.5):
    """Winsorize bounds + RobustScaler center/scale fitted on train only."""
    f = train_vals[~np.isnan(train_vals)]
    if len(f) == 0:
        return np.nan, np.nan, np.nan, np.nan
    lo = np.percentile(f, q_lo)
    hi = np.percentile(f, q_hi)
    center = np.median(f)
    scale = np.percentile(f, 75) - np.percentile(f, 25)
    if scale < 1e-12 or not np.isfinite(scale):
        scale = np.nan  # will be replaced by 1.0 at transform
    return lo, hi, center, scale


def transform_rows(df, idx, cols, params):
    """Apply train-fitted winsorize+scale to rows idx; returns (X, valid_mask)."""
    if len(idx) == 0:
        return np.empty((0, len(cols))), np.zeros(0, dtype=bool)
    Xc = []
    for c in cols:
        lo, hi, center, scale = params[c]
        s = scale if np.isfinite(scale) else 1.0
        v = np.clip(df.loc[idx, c].values, lo, hi)
        Xc.append((v - center) / s)
    X = np.column_stack(Xc)
    valid = ~np.any(np.isnan(X), axis=1)
    return X[valid], valid


def main():
    logger.info("loading matched features (raw)")
    df = pd.read_parquet(FEAT)

    # --- feature space: 98 scaler-mapped columns minus the 5 circular subreddit features -> 93
    sp_keys = set(json.load(open(os.path.join(FEATURES_DIR, "robust_scaler_params.json")))["center"].keys())
    with open(os.path.join(M1_STAT, "effect_sizes_report.json")) as f:
        art = set(json.load(f).get("selection_artifact_features", []))
    all98 = [c for c in df.columns if c in sp_keys]
    cols = [c for c in all98 if c not in art]
    logger.info(f"all98={len(all98)} artifact={len(all98) - len(cols)} cols93={len(cols)} "
                f"(artifact names: {sorted(art & set(all98))})")
    if len(cols) != 93:
        logger.warning(f"Expected 93 primary-atlas features, got {len(cols)}")

    t0 = pd.to_datetime(df["t0_date"], errors="coerce")
    df["era"] = t0.dt.year.map(era_of)
    df = df.dropna(subset=["era"]).reset_index(drop=True)

    era_arr = df["era"].values
    is_pat = (df["group"] == "patient").values
    dis = df["disorder_group"].values

    orig = pd.read_csv(os.path.join(M1_STAT, "auroc_vs_control.csv"))
    orig_map = dict(zip(orig["disorder"], orig["auroc_vs_control"]))

    results = []
    for d in DISORDERS:
        case_all = np.where(is_pat & (dis == d))[0]
        for tr_era, te_era in PAIRS:
            tr_case = case_all[era_arr[case_all] == tr_era]
            te_case = case_all[era_arr[case_all] == te_era]
            tr_ctrl = np.where((~is_pat) & (era_arr == tr_era))[0]
            te_ctrl = np.where((~is_pat) & (era_arr == te_era))[0]
            if len(tr_case) < 50 or len(te_case) < 100:
                logger.warning(f"{d} {tr_era}->{te_era}: skip "
                               f"(n_tr={len(tr_case)}, n_te={len(te_case)})")
                continue

            # ---- strict train-only preprocessing: fit on ALL train-era users ----
            train_era_idx = np.concatenate([tr_case, tr_ctrl])
            params = {c: robust_fit_train_only(df.loc[train_era_idx, c].values) for c in cols}

            # train design (cases + 3:1 subsampled controls)
            Xc_tr, _ = transform_rows(df, tr_case, cols, params)
            rng = np.random.RandomState(RANDOM_SEED)
            n_ctrl = min(max(3 * len(tr_case), 5000), len(tr_ctrl))
            tr_ctrl_s = rng.choice(tr_ctrl, size=n_ctrl, replace=False)
            Xctrl_tr, vm = transform_rows(df, tr_ctrl_s, cols, params)
            X_tr = np.vstack([Xc_tr, Xctrl_tr])
            y_tr = np.concatenate([np.ones(len(Xc_tr)), np.zeros(int(vm.sum()))])

            clf = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=2000,
                                     class_weight="balanced", random_state=RANDOM_SEED)
            clf.fit(X_tr, y_tr)
            in_auc = roc_auc_score(y_tr, clf.predict_proba(X_tr)[:, 1])

            # ---- full test era (cases + all controls), scaled with train params ----
            X_te_c, vm_c = transform_rows(df, te_case, cols, params)
            X_te_ctl, vm_ctl = transform_rows(df, te_ctrl, cols, params)
            X_te = np.vstack([X_te_c, X_te_ctl])
            y_te = np.concatenate([np.ones(int(vm_c.sum())), np.zeros(int(vm_ctl.sum()))])
            te_auc = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])

            results.append({
                "disorder": d, "train_era": tr_era, "test_era": te_era,
                "n_train_case": int(len(tr_case)), "n_test_case": int(vm_c.sum()),
                "n_test_control": int(vm_ctl.sum()),
                "n_features": len(cols),
                "auroc_train_insample": float(in_auc),
                "auroc_test_era": float(te_auc),
                "auroc_atlas": float(orig_map.get(d, np.nan)),
                "delta_vs_atlas": float(te_auc - orig_map.get(d, np.nan)) if d in orig_map else np.nan,
            })
            logger.info(f"{d} {tr_era}->{te_era}: test-era AUROC={te_auc:.4f} "
                        f"(atlas {orig_map.get(d, np.nan):.4f}, delta "
                        f"{te_auc - orig_map.get(d, np.nan):+.4f})")

    res = pd.DataFrame(results)
    res.to_csv(os.path.join(R4, f"{OUT_BASE}_results.csv"), index=False)
    with open(os.path.join(R4, f"{OUT_BASE}_report.json"), "w") as f:
        json.dump({"n_features": len(cols),
                   "artifact_excluded": sorted(art & set(all98)),
                   "pairs": PAIRS, "results": results}, f, indent=2, default=str)
    logger.info("DONE -> %s", R4)


if __name__ == "__main__":
    main()
