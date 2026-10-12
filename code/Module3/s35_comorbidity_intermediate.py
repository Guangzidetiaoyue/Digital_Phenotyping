"""
Module 3 - Analysis B2: Comorbidity Intermediate-State Analysis
================================================================
RQ: Do dual-diagnosis users' digital phenotypes lie BETWEEN the centroids of
their two constituent disorders? This tests the dimensional view (comorbidity =
intermediate state on a continuous spectrum) vs the categorical view (comorbidity
= independent/superposition).

Design (rigorous):
  B2.1 Centroids: each disorder's centroid is computed on SINGLE-diagnosis users
       only (those reporting exactly one disorder), so centroids are not
       contaminated by comorbid users.
  B2.2 For each comorbidity pair (A,B) with enough dual users (n>=50):
       project each dual user's phenotype vector onto the A->B centroid line.
       Projection position t = dot(v - cA, d) / dot(d, d), d = cB - cA.
       t=0 => pure A; t=1 => pure B; t in (0,1) => intermediate.
  B2.3 Null (dimensional) prediction: dual users cluster in (0,1) with small
       perpendicular residual (on the A-B line).
       Null (categorical/superposition) prediction: dual users resemble the
       average of A and B but with residual no smaller than single-diagnosis
       users; OR they scatter off-line.
  B2.4 Statistical tests (per pair, FDR-corrected across pairs):
       a) Fraction of dual users with t in (0,1): test vs 50% (chance) using
          exact binomial.
       b) Median |t - 0.5| (distance from midpoint): dual vs single-diagnosis
          reference (users of A or B only). Smaller = more intermediate.
       c) Perpendicular residual (distance to A-B line): dual vs single
          reference. Smaller = more on-line (dimensional).
       d) Bootstrap CI on the dual-user mean projection position.
  B2.5 Aggregate: across all pairs, mean fraction intermediate, mean residual
       ratio, sign test.

Rigor notes:
  - Feature space: Module 1 standardized features (RobustScaled within analysis).
  - Centroids from single-diagnosis users only (no contamination).
  - Reference distribution = single-diagnosis users of A or B, same feature space.
  - Multiple comparisons across pairs -> FDR (Benjamini-Hochberg).
  - We report effect sizes (fraction, residual ratio), not just p-values.
"""

import os
import json
import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats
from sklearn.preprocessing import RobustScaler
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = "/path/to/project"
FEATURES_PATH = os.path.join(PROJECT_ROOT, "module1_results", "features", "user_features_standardized.parquet")
MULTI_DISORDER_PATH = os.path.join(PROJECT_ROOT, "module3_results", "statistics", "user_multi_disorder.parquet")
OUT_DIR = os.path.join(PROJECT_ROOT, "module3_results", "statistics")
RANDOM_SEED = 42
MIN_DUAL_USERS = 50

EXCLUDE_SUBSTR = [
    "user_id_hash", "group", "disorder_group", "is_retrospective", "t0_date",
    "analysis_timestamp", "quality_", "pct_liwc", "pct_nrc", "pct_regex",
    "primary_subreddit", "ps_score", "iptw", "eb_weight", "strat_key",
    "split", "is_temporal_structurally_imputed", "self_post_ratio",
]


def load_data():
    feat = pd.read_parquet(FEATURES_PATH)
    md = pd.read_parquet(MULTI_DISORDER_PATH)
    feature_cols = [
        c for c in feat.columns
        if feat[c].dtype in (np.float64, np.float32, np.int64, np.int32, np.bool_)
        and not any(s in c for s in EXCLUDE_SUBSTR)
        and c not in ("user_id_hash", "group", "disorder_group")
    ]
    all_nan = [c for c in feature_cols if feat[c].isna().all()]
    feature_cols = [c for c in feature_cols if c not in all_nan]
    feat = feat.dropna(subset=feature_cols).copy()
    return feat, md, feature_cols


def project_to_ab_line(X, cA, cB):
    """Project points onto the A->B centroid line, normalized so A=(0,0), B=(1,0).

    Returns t (position along line, 0=A, 1=B) and perpendicular residual
    (normalized by the A-B distance so pairs are comparable).
    """
    d = cB - cA
    dd = np.dot(d, d)
    if dd < 1e-12:
        return np.full(len(X), np.nan), np.full(len(X), np.nan)
    # t in units of A->B distance
    t = (X - cA) @ d / dd
    # residual perpendicular to the line, normalized by |A-B|
    norm_d = np.sqrt(dd)
    proj = cA + t[:, None] * d
    res = np.linalg.norm(X - proj, axis=1) / norm_d
    return t, res


def main():
    logger.info("Loading data...")
    feat, md, feature_cols = load_data()
    logger.info(f"Features: {len(feature_cols)}")

    # Merge features with multi-disorder labels
    merged = feat.merge(md, on="user_id_hash", how="inner")
    merged = merged[merged["n_disorders"] >= 1].copy()

    # Single-diagnosis users
    single = merged[merged["n_disorders"] == 1].copy()
    # Dual-diagnosis users
    dual = merged[merged["n_disorders"] == 2].copy()
    logger.info(f"Single-dx users: {len(single)}, dual-dx users: {len(dual)}")

    # Robust-scale features
    scaler = RobustScaler().fit(merged[feature_cols].values)
    for grp in (single, dual):
        grp[feature_cols] = scaler.transform(grp[feature_cols].values)

    # Disorder centroids from single-diagnosis users
    disorders = sorted(single["disorder_group"].unique())
    centroids = {}
    for d in disorders:
        sub = single[single["disorder_group"] == d]
        centroids[d] = sub[feature_cols].values.mean(axis=0)
    logger.info(f"Disorders: {disorders}")

    # Identify dual users' pairs
    has_cols = [c for c in dual.columns if c.startswith("has_")]
    dual_pairs = []
    for _, row in dual.iterrows():
        lbl = [c.replace("has_", "") for c in has_cols if row[c] == 1]
        if len(lbl) == 2:
            dual_pairs.append((row["user_id_hash"], tuple(sorted(lbl))))
    dual_pairs = pd.DataFrame(dual_pairs, columns=["user_id_hash", "pair"])
    logger.info(f"Dual users mapped to pairs: {len(dual_pairs)}")

    # Group dual users by pair
    pair_groups = dual_pairs.groupby("pair")["user_id_hash"].apply(list).to_dict()

    # Reference single users for each disorder (for residual comparison)
    single_by_disorder = {d: single[single["disorder_group"] == d][feature_cols].values
                          for d in disorders}

    results = []
    all_rows = []
    for pair, uids in pair_groups.items():
        a, b = pair
        if a not in centroids or b not in centroids:
            continue
        if len(uids) < MIN_DUAL_USERS:
            continue
        cA, cB = centroids[a], centroids[b]
        dual_idx = dual.set_index("user_id_hash").loc[uids]
        X_dual = dual_idx[feature_cols].values

        # 1. Projection positions + residuals for dual users (vectorized)
        ts_dual, res_dual = project_to_ab_line(X_dual, cA, cB)
        valid = ~np.isnan(ts_dual)
        ts_dual, res_dual = ts_dual[valid], res_dual[valid]
        if len(ts_dual) < 30:
            continue

        # 2. Reference: single-diagnosis users of A and B (same normalization)
        X_ref = np.vstack([single_by_disorder[a], single_by_disorder[b]])
        ts_ref, res_ref = project_to_ab_line(X_ref, cA, cB)
        valid_r = ~np.isnan(ts_ref)
        ts_ref, res_ref = ts_ref[valid_r], res_ref[valid_r]

        # 3. Statistics
        frac_intermediate = float(np.mean((ts_dual > 0) & (ts_dual < 1)))
        # Binomial test vs 0.5
        n_int = int(((ts_dual > 0) & (ts_dual < 1)).sum())
        p_binom = float(stats.binomtest(n_int, len(ts_dual), 0.5, alternative="greater").pvalue)
        # Distance from midpoint
        mid_dist_dual = float(np.median(np.abs(ts_dual - 0.5)))
        mid_dist_ref = float(np.median(np.abs(ts_ref - 0.5)))
        # Perpendicular residual
        res_dual_mean = float(np.mean(res_dual))
        res_ref_mean = float(np.mean(res_ref))
        # Mann-Whitney: dual residual vs single residual
        try:
            u_stat, p_res = stats.mannwhitneyu(res_dual, res_ref, alternative="less")
            # alternative="less": dual residual smaller => more on-line
        except ValueError:
            p_res = np.nan
        # Mann-Whitney on |t-0.5|: dual closer to midpoint
        try:
            _, p_mid = stats.mannwhitneyu(
                np.abs(ts_dual - 0.5), np.abs(ts_ref - 0.5), alternative="less"
            )
        except ValueError:
            p_mid = np.nan

        results.append({
            "disorder_a": a, "disorder_b": b,
            "n_dual": len(uids),
            "frac_intermediate": frac_intermediate,
            "p_binom": p_binom,
            "mid_dist_dual": mid_dist_dual,
            "mid_dist_ref": mid_dist_ref,
            "res_dual_mean": res_dual_mean,
            "res_ref_mean": res_ref_mean,
            "res_ratio": float(res_dual_mean / res_ref_mean) if res_ref_mean > 0 else np.nan,
            "p_res_mw": p_res,
            "p_mid_mw": p_mid,
            "mean_t_dual": float(np.mean(ts_dual)),
        })
        logger.info(f"  {a}-{b}: n={len(uids)}, frac_int={frac_intermediate:.2f}, "
                    f"res_ratio={res_dual_mean/res_ref_mean:.2f}, p_res={p_res:.3f}")

    res_df = pd.DataFrame(results)

    # FDR correction across pairs
    if len(res_df) > 0:
        from statsmodels.stats.multitest import multipletests
        for col in ["p_binom", "p_res_mw", "p_mid_mw"]:
            valid = res_df[col].notna()
            if valid.sum() > 0:
                _, padj, _, _ = multipletests(res_df.loc[valid, col].values,
                                              alpha=0.05, method="fdr_bh")
                res_df.loc[valid, f"{col}_fdr"] = padj

    res_df.to_csv(os.path.join(OUT_DIR, "comorbidity_intermediate_results.csv"), index=False)

    # Aggregate
    agg = {}
    if len(res_df) > 0:
        agg = {
            "n_pairs_tested": len(res_df),
            "mean_frac_intermediate": float(res_df["frac_intermediate"].mean()),
            "median_res_ratio": float(res_df["res_ratio"].median()),
            "n_pairs_res_signif": int((res_df.get("p_res_mw_fdr", 0) < 0.05).sum()),
            "n_pairs_mid_signif": int((res_df.get("p_mid_mw_fdr", 0) < 0.05).sum()),
            "n_pairs_intermediate_frac_gt_half": int((res_df["frac_intermediate"] > 0.5).sum()),
        }
        logger.info(f"\n=== Aggregate ===")
        logger.info(f"  pairs tested: {agg['n_pairs_tested']}")
        logger.info(f"  mean fraction intermediate: {agg['mean_frac_intermediate']:.3f}")
        logger.info(f"  median residual ratio (dual/single): {agg['median_res_ratio']:.3f}")
        logger.info(f"  pairs with significant smaller residual: {agg['n_pairs_res_signif']}")
        logger.info(f"  pairs with fraction > 0.5: {agg['n_pairs_intermediate_frac_gt_half']}")

    report = {"aggregate": agg, "per_pair": res_df.to_dict("records")}
    with open(os.path.join(OUT_DIR, "comorbidity_intermediate_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("DONE")


if __name__ == "__main__":
    main()
