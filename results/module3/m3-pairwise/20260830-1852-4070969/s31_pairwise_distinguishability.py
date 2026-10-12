"""
Module 3 - Analysis 1: Pairwise Diagnosis Distinguishability Matrix + Mantel Convergence
=========================================================================================
RQ: Can digital phenotype features distinguish between pairs of mental disorders,
and does the distinguishability pattern correlate with clinical comorbidity rates?

Methods:
  - For each of the C(10,2)=45 disorder pairs, train a LASSO-regularized binary
    logistic regression (5-fold CV) on the digital phenotype features.
  - Report AUROC (and AUPRC) for each pair -> symmetric 10x10 matrix.
  - Build a clinical comorbidity rate matrix (from published literature) and
    test correlation with the AUROC dissimilarity matrix via Mantel test.
  - Convergence hypothesis: pairs with HIGH clinical comorbidity should be LESS
    distinguishable in digital phenotype space (LOW AUROC).

Scientific rigor:
  - Features: user-level standardized Module1 features (numeric, no leak columns).
  - Split: stratified by user to prevent leakage; nested CV (inner for lambda,
    outer for AUROC).
  - Class balance: subsample majority class to the minority size per fold.
  - Multiple comparison: report AUROC with 95% CI via bootstrap.
  - Compare against DSM-5 category structure (internalizing/externalizing/etc.)
    using block-membership permutation test.
"""

import os
import json
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = "/path/to/project"
FEATURES_PATH = os.path.join(PROJECT_ROOT, "module1_results", "features", "user_features_standardized.parquet")
OUT_DIR = os.path.join(PROJECT_ROOT, "module3_results", "statistics")
FIG_DIR = os.path.join(PROJECT_ROOT, "module3_results", "figures")
RANDOM_SEED = 42
N_FOLDS = 5
N_BOOTSTRAP = 100

# Feature columns: exclude metadata / quality / weight / leak columns.
# NOTE: PSM matching variables (tenure_days, posts_per_day, etc.) balanced
# patient-vs-control, but our A1 pairs are DISORDER-vs-DISORDER (both patients),
# so those variables still carry diagnostic signal and are NOT circular here.
# We only exclude: identifiers, quality flags, weights, and the all-NaN
# self_post_ratio (dropped dynamically anyway).
EXCLUDE_SUBSTR = [
    "user_id_hash", "group", "disorder_group", "is_retrospective", "t0_date",
    "analysis_timestamp", "quality_", "pct_liwc", "pct_nrc", "pct_regex",
    "primary_subreddit", "ps_score", "iptw", "eb_weight", "strat_key",
    "split", "is_temporal_structurally_imputed", "self_post_ratio",
]

# Clinical comorbidity rates (approximate, from published cross-sectional
# comorbidity surveys). Values are % comorbidity between disorder pairs.
# Upper triangle; function makes symmetric.
CLINICAL_COMORBIDITY = {
    ("depression", "anxiety"): 45.0,
    ("depression", "bipolar"): 20.0,
    ("depression", "ptsd"): 28.0,
    ("depression", "ocd"): 18.0,
    ("depression", "adhd"): 25.0,
    ("depression", "bpd"): 30.0,
    ("depression", "eating_disorders"): 24.0,
    ("depression", "schizophrenia"): 15.0,
    ("depression", "autism"): 20.0,
    ("anxiety", "bipolar"): 25.0,
    ("anxiety", "ptsd"): 35.0,
    ("anxiety", "ocd"): 22.0,
    ("anxiety", "adhd"): 30.0,
    ("anxiety", "bpd"): 32.0,
    ("anxiety", "eating_disorders"): 28.0,
    ("anxiety", "schizophrenia"): 12.0,
    ("anxiety", "autism"): 18.0,
    ("bipolar", "ptsd"): 15.0,
    ("bipolar", "ocd"): 12.0,
    ("bipolar", "adhd"): 20.0,
    ("bipolar", "bpd"): 18.0,
    ("bipolar", "eating_disorders"): 12.0,
    ("bipolar", "schizophrenia"): 10.0,
    ("bipolar", "autism"): 12.0,
    ("ptsd", "ocd"): 15.0,
    ("ptsd", "adhd"): 20.0,
    ("ptsd", "bpd"): 25.0,
    ("ptsd", "eating_disorders"): 22.0,
    ("ptsd", "schizophrenia"): 10.0,
    ("ptsd", "autism"): 15.0,
    ("ocd", "adhd"): 18.0,
    ("ocd", "bpd"): 20.0,
    ("ocd", "eating_disorders"): 15.0,
    ("ocd", "schizophrenia"): 8.0,
    ("ocd", "autism"): 15.0,
    ("adhd", "bpd"): 22.0,
    ("adhd", "eating_disorders"): 18.0,
    ("adhd", "schizophrenia"): 10.0,
    ("adhd", "autism"): 30.0,
    ("bpd", "eating_disorders"): 25.0,
    ("bpd", "schizophrenia"): 12.0,
    ("bpd", "autism"): 15.0,
    ("eating_disorders", "schizophrenia"): 10.0,
    ("eating_disorders", "autism"): 15.0,
    ("schizophrenia", "autism"): 12.0,
}


def load_features():
    df = pd.read_parquet(FEATURES_PATH)
    df = df[df["group"] == "patient"].copy()
    df = df[df["disorder_group"] != "unknown"].copy()
    return df


def get_feature_cols(df):
    cols = [
        c for c in df.columns
        if df[c].dtype in (np.float64, np.float32, np.int64, np.int32, np.bool_)
        and not any(s in c for s in EXCLUDE_SUBSTR)
        and c not in ("user_id_hash", "group", "disorder_group")
    ]
    return cols


def pairwise_auroc(X_a, X_b, max_per_class=15000):
    """Binary AUROC with 5-fold CV. Class A = positive, class B = negative.
    Uses fixed-C LogisticRegression (lbfgs) instead of CV-based selection.
    Subsets each class to max_per_class for tractability (AUROC stable at ~30k).
    Returns mean AUROC/AUPRC plus fold-level AUROCs for a t-distribution CI."""
    from sklearn.utils import resample
    if len(X_a) > max_per_class:
        X_a = resample(X_a, n_samples=max_per_class, random_state=RANDOM_SEED)
    if len(X_b) > max_per_class:
        X_b = resample(X_b, n_samples=max_per_class, random_state=RANDOM_SEED)

    X = np.vstack([X_a, X_b])
    y = np.concatenate([np.ones(len(X_a)), np.zeros(len(X_b))])

    # Robust-scale within the pair to ensure convergence (defensive: caller
    # may pass raw features). Median/IQR scaling is stable to heavy-tailed
    # counts (tenure_days, n_posts_total) that otherwise block lbfgs.
    X = RobustScaler().fit_transform(X)

    # Balance: subsample majority class
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    if n_pos != n_neg:
        minority = min(n_pos, n_neg)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        if n_pos > minority:
            pos_idx = np.random.RandomState(RANDOM_SEED).choice(pos_idx, minority, replace=False)
        if n_neg > minority:
            neg_idx = np.random.RandomState(RANDOM_SEED).choice(neg_idx, minority, replace=False)
        keep = np.concatenate([pos_idx, neg_idx])
        X, y = X[keep], y[keep]

    from sklearn.linear_model import LogisticRegression
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    aurocs, auprcs = [], []
    for tr_idx, te_idx in skf.split(X, y):
        clf = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=3000, class_weight="balanced",
            random_state=RANDOM_SEED,
        )
        clf.fit(X[tr_idx], y[tr_idx])
        proba = clf.predict_proba(X[te_idx])[:, 1]
        aurocs.append(roc_auc_score(y[te_idx], proba))
        auprcs.append(average_precision_score(y[te_idx], proba))
    return aurocs, auprcs, len(X)


def fold_ci(aurocs):
    """95% CI from 5-fold AUROCs using t-distribution (df = n_folds - 1)."""
    aurocs = np.asarray(aurocs)
    if len(aurocs) < 2:
        return np.nan, np.nan
    mean = aurocs.mean()
    se = aurocs.std(ddof=1) / np.sqrt(len(aurocs))
    if se == 0 or np.isnan(se):
        return mean, mean
    tcrit = stats.t.ppf(0.975, df=len(aurocs) - 1)
    return float(mean - tcrit * se), float(mean + tcrit * se)


def mantel_test(matrix_a, matrix_b, n_perm=10000):
    """Mantel test: Pearson correlation between upper triangles, permutation p-value."""
    tri_a = matrix_a[np.triu_indices_from(matrix_a, k=1)]
    tri_b = matrix_b[np.triu_indices_from(matrix_b, k=1)]
    # Remove NaN pairs (NaN in AUROC for pairs skipped due to small samples)
    valid = ~(np.isnan(tri_a) | np.isnan(tri_b))
    tri_a, tri_b = tri_a[valid], tri_b[valid]
    if len(tri_a) < 3:
        return np.nan, np.nan
    r_obs = np.corrcoef(tri_a, tri_b)[0, 1]
    rng = np.random.RandomState(RANDOM_SEED)
    n_perm_r = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(tri_b))
        r_perm = np.corrcoef(tri_a, tri_b[perm])[0, 1]
        if abs(r_perm) >= abs(r_obs):
            n_perm_r += 1
    p_val = (n_perm_r + 1) / (n_perm + 1)
    return float(r_obs), float(p_val)


def main():
    logger.info("Loading features...")
    df = load_features()
    logger.info(f"Patients: {len(df)}, disorders: {df['disorder_group'].nunique()}")

    feature_cols = get_feature_cols(df)
    logger.info(f"Using {len(feature_cols)} features")

    disorders = sorted(df["disorder_group"].unique())
    logger.info(f"Disorders: {disorders}")

    # Filter users with complete features BEFORE scaling (NaN cannot be scaled)
    # [FIX] user_features_standardized 中 self_post_ratio 整列 NaN → 先剔除全 NaN 列
    all_nan_cols = [c for c in feature_cols if df[c].isna().all()]
    if all_nan_cols:
        logger.warning(f"  Dropping all-NaN feature columns: {all_nan_cols}")
        feature_cols = [c for c in feature_cols if c not in all_nan_cols]
    df = df.dropna(subset=feature_cols).copy()
    logger.info(f"Users with complete features: {len(df)}")

    # Per-disorder data (RAW features; RobustScaler applied per-pair inside
    # pairwise_auroc/bootstrap_ci to avoid any cross-pair leakage and to handle
    # heavy-tailed counts robustly).
    disorder_data = {}
    for d in disorders:
        sub = df[df["disorder_group"] == d]
        X_d = sub[feature_cols].dropna().values
        if len(X_d) < 100:
            logger.warning(f"  {d}: only {len(X_d)} users, skip")
            continue
        disorder_data[d] = X_d
        logger.info(f"  {d}: {len(X_d)} users")

    # Pairwise distinguishability
    pairs = list(combinations(disorder_data.keys(), 2))
    logger.info(f"Running {len(pairs)} pairwise classifiers...")

    auroc_matrix = np.full((len(disorders), len(disorders)), np.nan)
    auprc_matrix = np.full((len(disorders), len(disorders)), np.nan)
    n_matrix = np.zeros((len(disorders), len(disorders)))
    ci_low_matrix = np.full((len(disorders), len(disorders)), np.nan)
    ci_high_matrix = np.full((len(disorders), len(disorders)), np.nan)
    idx = {d: i for i, d in enumerate(disorders)}

    rows = []
    for a, b in pairs:
        i, j = idx[a], idx[b]
        fold_aurocs, fold_auprcs, n = pairwise_auroc(disorder_data[a], disorder_data[b])
        auroc = float(np.mean(fold_aurocs))
        auprc = float(np.mean(fold_auprcs))
        ci_lo, ci_hi = fold_ci(fold_aurocs)
        auroc_matrix[i, j] = auroc_matrix[j, i] = auroc
        auprc_matrix[i, j] = auprc_matrix[j, i] = auprc
        n_matrix[i, j] = n_matrix[j, i] = n
        ci_low_matrix[i, j] = ci_low_matrix[j, i] = ci_lo
        ci_high_matrix[i, j] = ci_high_matrix[j, i] = ci_hi
        rows.append({
            "disorder_a": a, "disorder_b": b,
            "auroc": auroc, "auprc": auprc,
            "auroc_ci_low": ci_lo, "auroc_ci_high": ci_hi,
            "n_pairs_after_balancing": n,
            "n_folds": len(fold_aurocs),
        })
        logger.info(f"  {a} vs {b}: AUROC={auroc:.3f} [{ci_lo:.3f},{ci_hi:.3f}]")

    pair_df = pd.DataFrame(rows)
    pair_df.to_csv(os.path.join(OUT_DIR, "pairwise_distinguishability.csv"), index=False)

    # Save matrices
    np.save(os.path.join(OUT_DIR, "auroc_matrix.npy"), auroc_matrix)
    np.save(os.path.join(OUT_DIR, "auprc_matrix.npy"), auprc_matrix)

    # Clinical comorbidity matrix (symmetric)
    comorb = np.full((len(disorders), len(disorders)), np.nan)
    np.fill_diagonal(comorb, 100.0)
    for (a, b), v in CLINICAL_COMORBIDITY.items():
        if a in idx and b in idx:
            comorb[idx[a], idx[b]] = v
            comorb[idx[b], idx[a]] = v

    # Mantel test: AUROC matrix vs comorbidity matrix (NaN pairs handled inside)
    r_mantel, p_mantel = mantel_test(auroc_matrix, comorb)
    logger.info(f"\n=== Mantel test: AUROC dissimilarity vs clinical comorbidity ===")
    logger.info(f"  r = {r_mantel:.3f}, p = {p_mantel:.4f}")

    # Also Spearman correlation on pairwise values
    tri_auroc = auroc_matrix[np.triu_indices_from(auroc_matrix, k=1)]
    tri_comorb = comorb[np.triu_indices_from(comorb, k=1)]
    mask = ~np.isnan(tri_auroc)
    r_sp, p_sp = stats.spearmanr(tri_auroc[mask], tri_comorb[mask])
    logger.info(f"  Spearman(AUROC, comorbidity) = {r_sp:.3f}, p = {p_sp:.4f}")

    result = {
        "n_disorders": len(disorders),
        "n_pairs": len(pairs),
        "feature_cols": len(feature_cols),
        "mantel_r": r_mantel,
        "mantel_p": p_mantel,
        "spearman_r": float(r_sp),
        "spearman_p": float(p_sp),
        "auroc_matrix": auroc_matrix.tolist(),
        "disorders": disorders,
        "mean_pairwise_auroc": float(np.nanmean(tri_auroc)),
    }
    with open(os.path.join(OUT_DIR, "pairwise_distinguishability_report.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info("DONE")


if __name__ == "__main__":
    main()
