"""
Step 03: Effect Size Matrix Computation (v1.5 → v1.6)

Core Analysis 1 of Module 1: Cohen's d for each feature × disorder pair.
Produces the "Digital Phenotyping Atlas" heat map data.

v1.5 → v1.6 Changelog:
  [FIX] DiffSpec noise gate: magnitude floor eliminates spurious high
        differential specificity from near-zero |g| direction flips
  [FIX] IPTW weight trimming: clip extreme weights to [1st, 99th]
        percentile before computing weighted statistics, preventing
        over-optimistic variance from influential observations
  [FIX] Deterministic seed: replaced hash(disorder) with SHA-256
        based seed for cross-platform reproducibility
  [FIX] Strict direction inconsistency: added effect-size-gated
        criterion alongside the original exploratory flag, plus
        null-expectation baseline estimation
  [FIX] Clinical utility weight sensitivity: extracted weights to
        configuration dict, added 3-scheme sensitivity analysis with
        Jaccard stability index on top-K features
"""

import os
import json
import hashlib
import logging
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from typing import List, Dict, Tuple, Optional

from config_module1 import (
    FEATURES_DIR, STATS_DIR, DISORDER_GROUPS,
    DISORDER_DISPLAY_NAMES, FDR_METHOD, ALPHA,
    BOOTSTRAP_N, NUM_CORES, RANDOM_SEED
)

os.makedirs(STATS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(STATS_DIR, "effect_sizes.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)

np.random.seed(RANDOM_SEED)

# ======================== CONFIGURATION ========================

MIN_PATIENTS_PER_DISORDER = 30
MIN_CONTROLS_FOR_COMPARISON = 50
MIN_SAMPLES_FOR_BOOTSTRAP = 20

COMPUTE_ON_TRAIN_ONLY = True

BOOTSTRAP_N_ANALYSIS1 = 500
BOOTSTRAP_N_ANALYSIS2 = 0

EFFECT_THRESHOLDS = {
    "minimal": 0.10,
    "small": 0.20,
    "medium": 0.50,
    "large": 0.80,
}

SELECTION_ARTIFACT_FEATURES = {
    "mental_health_sub_ratio",
    "support_sub_ratio",
    "subreddit_entropy",
    "subreddit_herfindahl",
    "primary_subreddit_ratio",
}

# [FIX v1.6] DiffSpec noise gate threshold
# When max|g| across disorders < this value, differential specificity
# is attenuated proportionally because direction flips at this scale
# are indistinguishable from sampling noise.
# Set slightly below "small" effect (0.20) so that genuine small
# effects survive while noise is suppressed.
MIN_ABS_G_FOR_DIFFSPEC = 0.15

# [FIX v1.6] IPTW weight trimming percentiles
# Clip weights to [1st, 99th] percentile to prevent a single
# extreme weight from dominating the weighted estimate.
# (Cole & Hernán, 2008; Li et al., 2018)
WEIGHT_TRIM_LOWER_PCT = 1
WEIGHT_TRIM_UPPER_PCT = 99

# [FIX v1.6] Clinical utility weight configuration
# Extracted from hardcoded values to enable sensitivity analysis.
# Weights must satisfy: base + direction_spread + significance = 1.0
CU_WEIGHT_SCHEMES = {
    "default": {
        "base": 0.30,
        "direction_spread": 0.40,
        "significance": 0.30,
    },
    "spread_heavy": {
        "base": 0.20,
        "direction_spread": 0.50,
        "significance": 0.30,
    },
    "significance_heavy": {
        "base": 0.30,
        "direction_spread": 0.30,
        "significance": 0.40,
    },
}
CU_PRIMARY_SCHEME = "default"

# [FIX v1.6] Strict direction inconsistency thresholds
# A feature is flagged as STRICTLY direction-inconsistent only if:
#   1. It has ≥1 FDR-significant positive effect AND ≥1 FDR-significant negative effect
#   2. The significant positive effect has |g| ≥ this threshold
#   3. The significant negative effect has |g| ≥ this threshold
#   4. The range (max_sig_g - min_sig_g) ≥ this threshold
STRICT_DIRECTION_MIN_ABS_G = 0.10  # each direction must reach at least "minimal"
STRICT_DIRECTION_MIN_RANGE = 0.20  # total spread must reach at least "small"


# ======================== UTILITY FUNCTIONS ========================

# [FIX v1.6] Deterministic seed generation
def deterministic_seed(disorder: str, feature_idx: int,
                       base_seed: int = RANDOM_SEED) -> int:
    """
    Generate a cross-platform deterministic random seed from
    disorder name + feature index + base seed.
    
    Uses SHA-256 which is guaranteed to produce identical output
    across Python versions, platforms, and PYTHONHASHSEED values.
    This replaces the previous hash(disorder) % 10000 approach
    which was non-deterministic under Python's randomized hash seed.
    """
    key = f"{disorder}_{feature_idx}_{base_seed}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    # Take first 8 hex chars → 32-bit integer, fit in np.random seed range
    return int(digest[:8], 16) % (2**31)


def _trim_weights(weights: np.ndarray, lower_pct: float = WEIGHT_TRIM_LOWER_PCT,
                  upper_pct: float = WEIGHT_TRIM_UPPER_PCT) -> Tuple[np.ndarray, Dict]:
    """
    [FIX v1.6] Trim extreme IPTW weights to specified percentiles.
    
    Returns:
        trimmed_weights: weights clipped to [lower_pct, upper_pct] percentile
        trim_info: dict with trim statistics for logging/diagnostics
    """
    if len(weights) == 0:
        return weights, {"n_trimmed": 0, "pct_trimmed": 0.0}

    lower_bound = np.percentile(weights, lower_pct)
    upper_bound = np.percentile(weights, upper_pct)

    trimmed = np.clip(weights, lower_bound, upper_bound)

    n_trimmed = int(np.sum((weights < lower_bound) | (weights > upper_bound)))
    pct_trimmed = n_trimmed / len(weights) * 100

    trim_info = {
        "n_trimmed": n_trimmed,
        "pct_trimmed": float(pct_trimmed),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "original_max": float(np.max(weights)),
        "original_min": float(np.min(weights)),
    }

    return trimmed, trim_info


# ======================== CORE STATISTICS ========================

def cohens_d(group1, group2):
    """Compute Cohen's d with Hedges' correction."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan, np.nan

    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std < 1e-10:
        return 0.0, 0.0, 1.0

    d = (mean1 - mean2) / pooled_std
    correction = 1 - 3 / (4 * (n1 + n2) - 9)
    g = d * correction
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)

    return g, se_d, p_value


def weighted_cohens_d(values1, weights1, values2, weights2,
                      trim_weights: bool = True):
    """
    Compute IPTW-weighted Cohen's d (Hedges' g).
    
    [FIX v1.6] Added weight trimming to prevent extreme weights
    from inflating effective sample size and producing over-optimistic
    confidence intervals.
    
    Args:
        trim_weights: If True, clip weights to [1st, 99th] percentile
                      before computing weighted statistics.
    """
    mask1 = ~(np.isnan(values1) | np.isnan(weights1))
    mask2 = ~(np.isnan(values2) | np.isnan(weights2))
    v1, w1 = values1[mask1], weights1[mask1]
    v2, w2 = values2[mask2], weights2[mask2]

    n1, n2 = len(v1), len(v2)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan, np.nan

    # [FIX v1.6] Trim extreme weights before computing statistics
    if trim_weights:
        w1, trim_info1 = _trim_weights(w1)
        w2, trim_info2 = _trim_weights(w2)

        total_trimmed = trim_info1["n_trimmed"] + trim_info2["n_trimmed"]
        if total_trimmed > 0:
            logger.debug(
                f"  Weight trimming: {total_trimmed} weights clipped "
                f"(group1: {trim_info1['pct_trimmed']:.1f}%, "
                f"group2: {trim_info2['pct_trimmed']:.1f}%)"
            )

    w1_sum, w2_sum = w1.sum(), w2.sum()
    mean1_w = np.sum(w1 * v1) / w1_sum
    mean2_w = np.sum(w2 * v2) / w2_sum

    n1_eff = w1_sum**2 / np.sum(w1**2)
    n2_eff = w2_sum**2 / np.sum(w2**2)

    var1_w = np.sum(w1 * (v1 - mean1_w)**2) / w1_sum
    var2_w = np.sum(w2 * (v2 - mean2_w)**2) / w2_sum

    pooled_std_w = np.sqrt(
        ((n1_eff - 1) * var1_w + (n2_eff - 1) * var2_w) /
        (n1_eff + n2_eff - 2)
    )

    if pooled_std_w < 1e-10:
        return 0.0, 0.0, 1.0

    d_w = (mean1_w - mean2_w) / pooled_std_w
    correction = 1 - 3 / (4 * (n1_eff + n2_eff) - 9)
    g_w = d_w * correction

    se_w = np.sqrt(
        (n1_eff + n2_eff) / (n1_eff * n2_eff) +
        d_w**2 / (2 * (n1_eff + n2_eff))
    )

    se_diff = np.sqrt(var1_w / n1_eff + var2_w / n2_eff)
    if se_diff < 1e-10:
        p_value = 1.0
    else:
        t_stat = (mean1_w - mean2_w) / se_diff
        df_num = (var1_w / n1_eff + var2_w / n2_eff)**2
        df_denom = (var1_w / n1_eff)**2 / max(1, n1_eff - 1) + \
                   (var2_w / n2_eff)**2 / max(1, n2_eff - 1)
        df = df_num / max(df_denom, 1e-10)
        p_value = 2 * stats.t.sf(np.abs(t_stat), df=max(1, df))

    return g_w, se_w, p_value


def bootstrap_cohens_d(group1, group2, n_boot=500, ci=95, seed=None):
    """Bootstrap confidence interval for Cohen's d."""
    if seed is None:
        seed = RANDOM_SEED

    rng = np.random.RandomState(seed)
    boot_ds = []

    for _ in range(n_boot):
        b1 = rng.choice(group1, size=len(group1), replace=True)
        b2 = rng.choice(group2, size=len(group2), replace=True)
        g, _, _ = cohens_d(b1, b2)
        if not np.isnan(g):
            boot_ds.append(g)

    if len(boot_ds) < 50:
        return np.nan, np.nan

    alpha_half = (100 - ci) / 2
    lo = np.percentile(boot_ds, alpha_half)
    hi = np.percentile(boot_ds, 100 - alpha_half)
    return lo, hi


# ======================== SINGLE COMPARISON ========================

def compute_single_comparison(feature_col, patient_vals, control_vals,
                              do_bootstrap=True, bootstrap_seed=None,
                              n_boot=500,
                              patient_weights=None, control_weights=None,
                              use_weights=False):
    """Compute effect size for a single feature × disorder comparison."""
    if use_weights and patient_weights is not None and control_weights is not None:
        mask_p = ~(np.isnan(patient_vals) | np.isnan(patient_weights))
        mask_c = ~(np.isnan(control_vals) | np.isnan(control_weights))
    else:
        mask_p = ~np.isnan(patient_vals)
        mask_c = ~np.isnan(control_vals)

    p_clean = patient_vals[mask_p]
    c_clean = control_vals[mask_c]

    if len(p_clean) < 10 or len(c_clean) < 10:
        return {
            "feature": feature_col,
            "hedges_g": np.nan, "se": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_patient": len(p_clean), "n_control": len(c_clean),
            "patient_mean": np.nan, "patient_std": np.nan,
            "control_mean": np.nan, "control_std": np.nan,
            "weighted": use_weights,
        }

    if use_weights and patient_weights is not None and control_weights is not None:
        pw = patient_weights[mask_p]
        cw = control_weights[mask_c]
        g, se, p = weighted_cohens_d(p_clean, pw, c_clean, cw, trim_weights=True)
    else:
        g, se, p = cohens_d(p_clean, c_clean)

    if do_bootstrap and len(p_clean) >= MIN_SAMPLES_FOR_BOOTSTRAP and n_boot > 0:
        ci_lo, ci_hi = bootstrap_cohens_d(
            p_clean, c_clean, n_boot=n_boot, ci=95, seed=bootstrap_seed
        )
    else:
        ci_lo = g - 1.96 * se
        ci_hi = g + 1.96 * se

    return {
        "feature": feature_col,
        "hedges_g": g, "se": se, "p_value": p,
        "ci_lower": ci_lo, "ci_upper": ci_hi,
        "n_patient": len(p_clean), "n_control": len(c_clean),
        "patient_mean": float(np.mean(p_clean)),
        "patient_std": float(np.std(p_clean, ddof=1)),
        "control_mean": float(np.mean(c_clean)),
        "control_std": float(np.std(c_clean, ddof=1)),
        "weighted": use_weights,
    }


# ======================== DISORDER-LEVEL COMPUTATION ========================

def compute_disorder_effect_sizes(df, feature_cols, disorder,
                                   do_bootstrap=True, n_boot=500,
                                   use_iptw=True):
    """Compute effect sizes for all features for a single disorder vs controls."""
    logger.info(f"  Computing effect sizes for: {disorder}")

    patient_mask = (df["group"] == "patient") & (df["disorder_group"] == disorder)
    control_mask = (df["group"] == "control")

    patient_df = df[patient_mask]
    control_df = df[control_mask]

    n_patients = len(patient_df)
    n_controls = len(control_df)

    if n_patients < MIN_PATIENTS_PER_DISORDER:
        logger.warning(f"    {disorder}: Only {n_patients} patients. Skipping.")
        return []

    if n_controls < MIN_CONTROLS_FOR_COMPARISON:
        logger.warning(f"    {disorder}: Only {n_controls} controls. Skipping.")
        return []

    logger.info(f"    Patients: {n_patients}, Controls: {n_controls}")

    has_iptw = use_iptw and "iptw_att_weight" in df.columns
    if has_iptw:
        iptw_valid_p = patient_df["iptw_att_weight"].notna().sum()
        iptw_valid_c = control_df["iptw_att_weight"].notna().sum()
        if iptw_valid_p < MIN_PATIENTS_PER_DISORDER or \
           iptw_valid_c < MIN_CONTROLS_FOR_COMPARISON:
            logger.warning(f"    Insufficient valid IPTW weights, "
                          f"falling back to unweighted")
            has_iptw = False

    results = []
    for i, feat in enumerate(feature_cols):
        p_vals = patient_df[feat].values.astype(np.float64)
        c_vals = control_df[feat].values.astype(np.float64)

        # [FIX v1.6] Deterministic seed via SHA-256
        bootstrap_seed = deterministic_seed(disorder, i, RANDOM_SEED)

        if has_iptw:
            pw = patient_df["iptw_att_weight"].values.astype(np.float64)
            cw = control_df["iptw_att_weight"].values.astype(np.float64)
            res = compute_single_comparison(
                feat, p_vals, c_vals,
                do_bootstrap=do_bootstrap,
                bootstrap_seed=bootstrap_seed,
                n_boot=n_boot,
                patient_weights=pw, control_weights=cw,
                use_weights=True
            )
        else:
            res = compute_single_comparison(
                feat, p_vals, c_vals,
                do_bootstrap=do_bootstrap,
                bootstrap_seed=bootstrap_seed,
                n_boot=n_boot,
                use_weights=False
            )

        res["disorder"] = disorder
        res["is_selection_artifact"] = feat in SELECTION_ARTIFACT_FEATURES
        results.append(res)

    g_vals = [r["hedges_g"] for r in results
              if not np.isnan(r.get("hedges_g", np.nan))]
    if g_vals:
        n_sig = sum(1 for r in results if r.get('p_value', 1) < 0.05)
        logger.info(f"    Effect sizes: mean |g|={np.mean(np.abs(g_vals)):.3f}, "
                    f"max |g|={np.max(np.abs(g_vals)):.3f}, "
                    f"n_sig(p<0.05)={n_sig}")

    return results


def compute_pairwise_disorder_effects(df, feature_cols, disorder_a, disorder_b):
    """
    Compute effect sizes between two patient groups.
    No bootstrap for pairwise — analytical CI is sufficient.
    """
    mask_a = (df["group"] == "patient") & (df["disorder_group"] == disorder_a)
    mask_b = (df["group"] == "patient") & (df["disorder_group"] == disorder_b)

    df_a = df[mask_a]
    df_b = df[mask_b]

    n_a, n_b = len(df_a), len(df_b)

    if n_a < MIN_PATIENTS_PER_DISORDER or n_b < MIN_PATIENTS_PER_DISORDER:
        logger.warning(
            f"    Insufficient samples for pairwise {disorder_a} vs "
            f"{disorder_b}: n_a={n_a}, n_b={n_b}. Skipping."
        )
        return []

    results = []
    for i, feat in enumerate(feature_cols):
        a_vals = df_a[feat].values.astype(np.float64)
        b_vals = df_b[feat].values.astype(np.float64)

        # [FIX v1.6] Deterministic seed (for consistency, even though
        # do_bootstrap=False here; future use if bootstrap re-enabled)
        bootstrap_seed = deterministic_seed(
            f"{disorder_a}_vs_{disorder_b}", i, RANDOM_SEED
        )

        res = compute_single_comparison(
            feat, a_vals, b_vals,
            do_bootstrap=False,
            bootstrap_seed=bootstrap_seed,
            use_weights=False
        )
        res["disorder_a"] = disorder_a
        res["disorder_b"] = disorder_b
        res["is_selection_artifact"] = feat in SELECTION_ARTIFACT_FEATURES
        results.append(res)

    return results


# ======================== FDR CORRECTION ========================

def apply_fdr_correction(results_df):
    """Apply FDR correction across all tests."""
    logger.info("Applying FDR correction...")

    results_df["p_fdr_within"] = np.nan
    results_df["sig_fdr_within"] = False

    for disorder in results_df["disorder"].unique():
        disorder_mask = results_df["disorder"] == disorder
        disorder_idx = results_df.index[disorder_mask]

        p_vals = results_df.loc[disorder_idx, "p_value"].values
        valid_mask = ~np.isnan(p_vals)

        if valid_mask.sum() > 0:
            valid_idx = disorder_idx[valid_mask]
            reject, p_adj, _, _ = multipletests(
                p_vals[valid_mask], alpha=ALPHA, method=FDR_METHOD
            )
            results_df.loc[valid_idx, "p_fdr_within"] = p_adj
            results_df.loc[valid_idx, "sig_fdr_within"] = reject

    all_p = results_df["p_value"].values
    valid_global = ~np.isnan(all_p)

    results_df["p_fdr_global"] = np.nan
    results_df["sig_fdr_global"] = False
    if valid_global.sum() > 0:
        valid_global_idx = results_df.index[valid_global]
        reject_g, p_adj_g, _, _ = multipletests(
            all_p[valid_global], alpha=ALPHA, method=FDR_METHOD
        )
        results_df.loc[valid_global_idx, "p_fdr_global"] = p_adj_g
        results_df.loc[valid_global_idx, "sig_fdr_global"] = reject_g
    else:
        results_df["p_fdr_global"] = np.nan
        results_df["sig_fdr_global"] = False

    results_df["effect_category"] = "negligible"
    results_df.loc[
        results_df["hedges_g"].abs() >= EFFECT_THRESHOLDS["minimal"],
        "effect_category"
    ] = "minimal"
    results_df.loc[
        results_df["hedges_g"].abs() >= EFFECT_THRESHOLDS["small"],
        "effect_category"
    ] = "small"
    results_df.loc[
        results_df["hedges_g"].abs() >= EFFECT_THRESHOLDS["medium"],
        "effect_category"
    ] = "medium"
    results_df.loc[
        results_df["hedges_g"].abs() >= EFFECT_THRESHOLDS["large"],
        "effect_category"
    ] = "large"

    n_total = len(results_df)
    n_sig_global = results_df["sig_fdr_global"].sum()

    n_artifact = results_df["is_selection_artifact"].sum()
    n_nonartifact = n_total - n_artifact
    n_sig_nonartifact = results_df[
        ~results_df["is_selection_artifact"]
    ]["sig_fdr_global"].sum()

    cat_dist = results_df["effect_category"].value_counts().to_dict()
    cat_nonartifact = results_df[
        ~results_df["is_selection_artifact"]
    ]["effect_category"].value_counts().to_dict()

    logger.info(f"  Total tests: {n_total} "
                f"({n_nonartifact} non-artifact, {n_artifact} artifact)")
    logger.info(f"  Significant (global FDR): {n_sig_global} "
                f"({n_sig_global/n_total*100:.1f}%)")
    logger.info(f"  Significant non-artifact: {n_sig_nonartifact} "
                f"({n_sig_nonartifact/max(1,n_nonartifact)*100:.1f}%)")
    logger.info(f"  Effect categories (all): {cat_dist}")
    logger.info(f"  Effect categories (non-artifact): {cat_nonartifact}")

    return results_df


# ======================== MATRIX & SPECIFICITY ========================

def build_effect_size_matrix(results_df):
    """Pivot results into feature × disorder matrices."""
    matrix = results_df.pivot_table(
        index="feature", columns="disorder",
        values="hedges_g", aggfunc="first"
    )
    sig_matrix = results_df.pivot_table(
        index="feature", columns="disorder",
        values="sig_fdr_global", aggfunc="first"
    )
    p_matrix = results_df.pivot_table(
        index="feature", columns="disorder",
        values="p_fdr_global", aggfunc="first"
    )
    return matrix, sig_matrix, p_matrix


def compute_specificity_scores(effect_matrix):
    """
    Compute feature specificity (magnitude-based).
    
    High specificity → |g| concentrated in 1-2 disorders
    Low specificity → |g| spread across many disorders (transdiagnostic)
    """
    logger.info("Computing feature specificity scores (magnitude-based)...")

    n_disorders = effect_matrix.shape[1]
    if n_disorders < 2:
        return pd.Series(dtype=float, name="specificity")

    max_entropy = np.log(n_disorders)
    specificities = {}

    for feature in effect_matrix.index:
        abs_d = effect_matrix.loc[feature].abs().values
        abs_d = np.nan_to_num(abs_d, nan=0.0)
        total = abs_d.sum()
        if total < 0.01:
            specificities[feature] = 0.0
            continue
        probs = abs_d / total
        h = -np.sum(probs * np.log(probs + 1e-10))
        specificity = 1 - h / max_entropy
        specificities[feature] = specificity

    return pd.Series(specificities, name="specificity")


def compute_differential_specificity(effect_matrix, sig_matrix):
    """
    [FIX v1.6] TRUE direction-aware specificity with noise gate.
    
    Previous (v1.5) issue: when max|g| is small (< 0.15), direction
    flips are indistinguishable from sampling noise, yet differential
    specificity could be artificially high.
    
    Fix: add magnitude floor factor that attenuates diff_spec
    proportionally when max|g| < MIN_ABS_G_FOR_DIFFSPEC.
    
      magnitude_floor = min(1.0, max_abs_g / MIN_ABS_G_FOR_DIFFSPEC)
      diff_spec = direction_diversity × magnitude_concentration × magnitude_floor
    
    This ensures:
      - g = [+0.5, -0.5, 0]: floor=1.0, full diff_spec (genuine signal)
      - g = [+0.1, -0.1, 0]: floor=0.67, attenuated (borderline)
      - g = [+0.01, -0.01, 0]: floor=0.07, near-zero (noise)
    """
    logger.info("Computing differential specificity (polarity-based, v1.6)...")

    n_disorders = effect_matrix.shape[1]
    if n_disorders < 2:
        return pd.Series(dtype=float, name="differential_specificity")

    max_entropy = np.log(n_disorders)
    diff_specificities = {}

    for feature in effect_matrix.index:
        g_vals = effect_matrix.loc[feature].values
        g_clean = np.nan_to_num(g_vals, nan=0.0)
        abs_g = np.abs(g_clean)
        abs_sum = abs_g.sum()

        if abs_sum < 0.01:
            diff_specificities[feature] = 0.0
            continue

        # Component 1: Polarity Score (direction consistency)
        signed_sum = np.sum(abs_g * np.sign(g_clean))
        polarity = signed_sum / abs_sum  # Range: [-1, 1]
        direction_diversity = 1 - abs(polarity)  # Range: [0, 1]

        # Component 2: Magnitude Concentration (standard specificity)
        probs = abs_g / abs_sum
        h = -np.sum(probs * np.log(probs + 1e-10))
        magnitude_concentration = 1 - h / max_entropy  # Range: [0, 1]

        # [FIX v1.6] Component 3: Magnitude floor (noise gate)
        # When max|g| is below threshold, direction information is unreliable.
        # Attenuate proportionally rather than hard-cutoff, to preserve
        # partial signal from features on the boundary.
        max_abs_g = float(np.max(abs_g))
        magnitude_floor = min(1.0, max_abs_g / MIN_ABS_G_FOR_DIFFSPEC)

        # Combine: must be concentrated, direction-diverse, AND have
        # sufficient magnitude for the direction signal to be trustworthy
        diff_specificities[feature] = (
            direction_diversity * magnitude_concentration * magnitude_floor
        )

    diff_series = pd.Series(diff_specificities, name="differential_specificity")

    if len(diff_series) > 0:
        logger.info(f"  Differential specificity range: "
                    f"{diff_series.min():.3f} - {diff_series.max():.3f}")

        # Log noise-gate impact
        n_gated = 0
        for feature in effect_matrix.index:
            g_vals = effect_matrix.loc[feature].values
            g_clean = np.nan_to_num(g_vals, nan=0.0)
            max_abs_g = float(np.max(np.abs(g_clean)))
            if max_abs_g < MIN_ABS_G_FOR_DIFFSPEC:
                n_gated += 1
        logger.info(f"  Noise gate impact: {n_gated}/{len(effect_matrix)} "
                    f"features attenuated (max|g| < {MIN_ABS_G_FOR_DIFFSPEC})")

        # Log features with high differential specificity
        high_diff = diff_series[diff_series > 0.3].sort_values(ascending=False)
        if len(high_diff) > 0:
            logger.info(f"  High differential specificity (>0.3): "
                        f"{len(high_diff)} features")
            for feat, ds in high_diff.head(5).items():
                g_vals = effect_matrix.loc[feat]
                n_pos = (g_vals > 0).sum()
                n_neg = (g_vals < 0).sum()
                max_abs_g = g_vals.abs().max()
                best_pos = g_vals.idxmax()
                best_neg = g_vals.idxmin()
                logger.info(
                    f"    {feat}: DiffSpec={ds:.3f} "
                    f"({n_pos}+/{n_neg}-), max|g|={max_abs_g:.3f}, "
                    f"most+={best_pos}({g_vals[best_pos]:.2f}), "
                    f"most-={best_neg}({g_vals[best_neg]:.2f})"
                )
        else:
            logger.info(f"  No features with high differential specificity (>0.3)")

    return diff_series


def _compute_clinical_utility_single(effect_matrix, sig_matrix,
                                      weights: Dict[str, float]) -> pd.Series:
    """
    [FIX v1.6] Internal: compute clinical utility with given weight scheme.
    
    Extracted from compute_clinical_utility to enable sensitivity analysis
    across multiple weight configurations.
    
    Args:
        weights: dict with keys 'base', 'direction_spread', 'significance'
                 values must sum to 1.0
    
    Returns:
        Series of clinical utility scores indexed by feature
    """
    n_disorders = effect_matrix.shape[1]
    if n_disorders < 2:
        return pd.Series(dtype=float, name="clinical_utility")

    w_base = weights["base"]
    w_spread = weights["direction_spread"]
    w_sig = weights["significance"]

    utility_scores = {}

    for feature in effect_matrix.index:
        g_vals = effect_matrix.loc[feature]
        sig_vals = sig_matrix.loc[feature] if feature in sig_matrix.index \
            else pd.Series(dtype=bool)

        valid_g = g_vals.dropna()
        if len(valid_g) < 2:
            utility_scores[feature] = 0.0
            continue

        abs_g = valid_g.abs()
        max_abs_g = abs_g.max()

        # Component 1: Effect magnitude (necessary condition)
        max_abs_g_score = min(max_abs_g / 0.8, 1.0)

        # Component 2: Direction spread (differential diagnosis value)
        g_range = valid_g.max() - valid_g.min()
        direction_spread = g_range / (2 * max_abs_g + 0.01)
        direction_spread = min(direction_spread, 1.0)

        # Component 3: Significance coverage (reliability)
        n_sig = sig_vals.sum() if len(sig_vals) > 0 else 0
        n_sig_score = n_sig / n_disorders

        # Composite with configurable weights
        clinical_utility = max_abs_g_score * (
            w_base + w_spread * direction_spread + w_sig * n_sig_score
        )

        utility_scores[feature] = clinical_utility

    return pd.Series(utility_scores, name="clinical_utility")


def compute_clinical_utility(effect_matrix, sig_matrix) -> Tuple[pd.Series, Dict]:
    """
    [FIX v1.6] Clinical Utility Score with weight sensitivity analysis.
    
    Returns:
        primary_utility: Series using the primary weight scheme
        sensitivity_report: dict with Jaccard stability across schemes
    """
    logger.info("Computing clinical utility scores (v1.6)...")

    primary_weights = CU_WEIGHT_SCHEMES[CU_PRIMARY_SCHEME]
    primary_utility = _compute_clinical_utility_single(
        effect_matrix, sig_matrix, primary_weights
    )

    if len(primary_utility) > 0:
        logger.info(f"  Clinical utility range: "
                    f"{primary_utility.min():.3f} - {primary_utility.max():.3f}")
        n_high = (primary_utility > 0.5).sum()
        n_moderate = ((primary_utility > 0.3) & (primary_utility <= 0.5)).sum()
        logger.info(f"  High utility (>0.5): {n_high} features")
        logger.info(f"  Moderate utility (0.3-0.5): {n_moderate} features")

        top5 = primary_utility.nlargest(min(5, len(primary_utility)))
        logger.info(f"  Top 5 clinical utility features:")
        for feat, cu in top5.items():
            g_vals = effect_matrix.loc[feat]
            best_disorder = g_vals.abs().idxmax()
            best_g = g_vals[best_disorder]
            logger.info(f"    {feat}: CU={cu:.3f} "
                        f"(best: {best_disorder} g={best_g:.3f})")

    # [FIX v1.6] Weight sensitivity analysis
    sensitivity_report = _compute_weight_sensitivity(
        effect_matrix, sig_matrix, primary_utility
    )

    return primary_utility, sensitivity_report


def _compute_weight_sensitivity(effect_matrix, sig_matrix,
                                 primary_utility: pd.Series,
                                 top_k: int = 20) -> Dict:
    """
    [FIX v1.6] Assess stability of clinical utility rankings across
    different weight configurations.
    
    For each alternative scheme, compute:
      1. Rank correlation (Spearman) with primary scheme
      2. Jaccard index of top-K feature sets
    
    This provides empirical evidence for whether the subjective weight
    choice materially affects conclusions.
    """
    logger.info("  Computing weight sensitivity analysis...")

    primary_top_k = set(primary_utility.nlargest(min(top_k, len(primary_utility)))
                        .index.tolist())

    sensitivity = {
        "primary_scheme": CU_PRIMARY_SCHEME,
        "primary_weights": CU_WEIGHT_SCHEMES[CU_PRIMARY_SCHEME],
        "top_k": top_k,
        "comparisons": {},
    }

    for scheme_name, weights in CU_WEIGHT_SCHEMES.items():
        if scheme_name == CU_PRIMARY_SCHEME:
            continue

        alt_utility = _compute_clinical_utility_single(
            effect_matrix, sig_matrix, weights
        )

        # Spearman rank correlation
        common_features = primary_utility.index.intersection(alt_utility.index)
        if len(common_features) < 5:
            continue

        rho, p_val = stats.spearmanr(
            primary_utility.loc[common_features],
            alt_utility.loc[common_features]
        )

        # Jaccard index of top-K
        alt_top_k = set(alt_utility.nlargest(min(top_k, len(alt_utility)))
                        .index.tolist())
        jaccard = len(primary_top_k & alt_top_k) / max(1, len(primary_top_k | alt_top_k))

        sensitivity["comparisons"][scheme_name] = {
            "weights": weights,
            "spearman_rho": float(rho),
            "spearman_p": float(p_val),
            "jaccard_top_k": float(jaccard),
            "n_shared_top_k": len(primary_top_k & alt_top_k),
        }

        logger.info(f"    vs {scheme_name}: ρ={rho:.3f} (p={p_val:.2e}), "
                    f"Jaccard(top-{top_k})={jaccard:.3f} "
                    f"({len(primary_top_k & alt_top_k)}/{top_k} shared)")

    # Overall stability assessment
    all_jaccards = [v["jaccard_top_k"] for v in sensitivity["comparisons"].values()]
    if all_jaccards:
        min_jaccard = min(all_jaccards)
        if min_jaccard >= 0.7:
            stability = "high"
        elif min_jaccard >= 0.5:
            stability = "moderate"
        else:
            stability = "low"
        sensitivity["overall_stability"] = stability
        sensitivity["min_jaccard"] = float(min_jaccard)
        logger.info(f"  Weight sensitivity: {stability} stability "
                    f"(min Jaccard = {min_jaccard:.3f})")
    else:
        sensitivity["overall_stability"] = "single_scheme"

    return sensitivity


def detect_direction_inconsistency(effect_matrix, sig_matrix):
    """
    [FIX v1.6] Detect features with inconsistent effect directions
    across disorders, with both exploratory and strict criteria.
    
    Two-tier system:
    
    1. EXPLORATORY (has_direction_inconsistency):
       ≥1 significant positive AND ≥1 significant negative
       → useful for hypothesis generation but may include false positives
       from multiple testing
    
    2. STRICT (has_strict_direction_inconsistency):
       Same as above, PLUS:
       - Each direction's max |g| ≥ STRICT_DIRECTION_MIN_ABS_G
       - Total g range ≥ STRICT_DIRECTION_MIN_RANGE
       → reduces false positives from tiny/significant-by-N effects
    
    Also estimates the null-expectation baseline: if all effects were
    independently significant with probability p_sig and equally likely
    positive/negative, the expected number of direction-inconsistent
    features follows a Binomial distribution.
    """
    logger.info("Detecting direction inconsistency across disorders (v1.6)...")

    direction_records = []

    # Pre-compute overall significance rate for null expectation
    n_total_tests = 0
    n_total_sig = 0

    for feature in effect_matrix.index:
        g_vals = effect_matrix.loc[feature]
        sig_vals = sig_matrix.loc[feature] if feature in sig_matrix.index \
            else pd.Series()

        valid_g = g_vals.dropna()
        if len(valid_g) < 2:
            continue

        n_positive = (valid_g > 0).sum()
        n_negative = (valid_g < 0).sum()

        # Exploratory criterion: any significant in both directions
        sig_pos_disorders = []
        sig_neg_disorders = []
        sig_pos_g_values = []
        sig_neg_g_values = []

        for disorder in valid_g.index:
            g = valid_g[disorder]
            is_sig = sig_vals.get(disorder, False)

            n_total_tests += 1
            if is_sig:
                n_total_sig += 1

            if is_sig and g > 0:
                sig_pos_disorders.append(disorder)
                sig_pos_g_values.append(g)
            elif is_sig and g < 0:
                sig_neg_disorders.append(disorder)
                sig_neg_g_values.append(g)

        has_inconsistency = (len(sig_pos_disorders) > 0) and \
                            (len(sig_neg_disorders) > 0)

        # [FIX v1.6] Strict criterion: effect-size-gated
        max_sig_pos_g = max(sig_pos_g_values) if sig_pos_g_values else 0.0
        min_sig_neg_g = min(sig_neg_g_values) if sig_neg_g_values else 0.0
        sig_g_range = max_sig_pos_g - min_sig_neg_g

        has_strict_inconsistency = (
            has_inconsistency and
            max_sig_pos_g >= STRICT_DIRECTION_MIN_ABS_G and
            abs(min_sig_neg_g) >= STRICT_DIRECTION_MIN_ABS_G and
            sig_g_range >= STRICT_DIRECTION_MIN_RANGE
        )

        direction_records.append({
            "feature": feature,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "n_sig_positive": len(sig_pos_disorders),
            "n_sig_negative": len(sig_neg_disorders),
            "has_direction_inconsistency": has_inconsistency,
            "has_strict_direction_inconsistency": has_strict_inconsistency,
            "max_sig_positive_g": float(max_sig_pos_g),
            "min_sig_negative_g": float(min_sig_neg_g),
            "sig_g_range": float(sig_g_range),
            "sig_positive_disorders": ";".join(sig_pos_disorders),
            "sig_negative_disorders": ";".join(sig_neg_disorders),
            "max_positive_g": float(valid_g.max()),
            "max_negative_g": float(valid_g.min()),
            "direction_spread": float(valid_g.max() - valid_g.min()),
            "is_selection_artifact": feature in SELECTION_ARTIFACT_FEATURES,
        })

    direction_df = pd.DataFrame(direction_records)

    # Report both tiers
    n_exploratory = direction_df["has_direction_inconsistency"].sum()
    n_strict = direction_df["has_strict_direction_inconsistency"].sum()
    n_exploratory_nonartifact = direction_df[
        direction_df["has_direction_inconsistency"] &
        ~direction_df["is_selection_artifact"]
    ].shape[0]
    n_strict_nonartifact = direction_df[
        direction_df["has_strict_direction_inconsistency"] &
        ~direction_df["is_selection_artifact"]
    ].shape[0]

    logger.info(f"  Direction-inconsistent features (exploratory): "
                f"{n_exploratory}/{len(direction_df)} "
                f"({n_exploratory_nonartifact} non-artifact)")
    logger.info(f"  Direction-inconsistent features (strict): "
                f"{n_strict}/{len(direction_df)} "
                f"({n_strict_nonartifact} non-artifact)")

    # [FIX v1.6] Null-expectation baseline estimation
    # If each test is independently significant with probability p_sig,
    # and among significant tests, positive/negative are equally likely,
    # then for a feature with K significant disorders:
    #   P(direction inconsistent | K sig) = 1 - 2 * (0.5)^K
    # Expected count = Σ_features P(inconsistent | their K)
    null_report = _estimate_null_direction_inconsistency(
        direction_df, n_total_tests, n_total_sig
    )
    logger.info(f"  Null expectation: {null_report['expected_exploratory']:.1f} "
                f"exploratory direction-inconsistent features by chance "
                f"(observed: {n_exploratory})")
    logger.info(f"  Observed/Expected ratio: "
                f"{n_exploratory / max(1, null_report['expected_exploratory']):.1f}x")

    if n_strict_nonartifact > 0:
        nonartifact_strict = direction_df[
            direction_df["has_strict_direction_inconsistency"] &
            ~direction_df["is_selection_artifact"]
        ].sort_values("sig_g_range", ascending=False)

        logger.info(f"  Strict direction-inconsistent features (non-artifact, "
                    f"top 10 by sig_g_range):")
        for _, row in nonartifact_strict.head(10).iterrows():
            logger.info(
                f"    {row['feature']}: "
                f"range={row['sig_g_range']:.3f}, "
                f"+max={row['max_sig_positive_g']:.3f}, "
                f"-min={row['min_sig_negative_g']:.3f}, "
                f"+{row['n_sig_positive']} ({row['sig_positive_disorders']}), "
                f"-{row['n_sig_negative']} ({row['sig_negative_disorders']})"
            )

    return direction_df, null_report


def _estimate_null_direction_inconsistency(direction_df: pd.DataFrame,
                                            n_total_tests: int,
                                            n_total_sig: int) -> Dict:
    """
    [FIX v1.6] Estimate expected number of direction-inconsistent features
    under the null hypothesis that significant effects are equally likely
    to be positive or negative.
    
    For a feature with K significant disorders, the probability of
    seeing both positive and negative directions is:
        P(inconsistent | K) = 1 - P(all same sign | K) = 1 - 2*(0.5^K)
    
    We sum this across all features using their observed K values.
    """
    n_features = len(direction_df)
    if n_features == 0:
        return {"expected_exploratory": 0.0, "method": "none"}

    p_sig = n_total_sig / max(1, n_total_tests)

    expected_count = 0.0
    for _, row in direction_df.iterrows():
        k = row["n_sig_positive"] + row["n_sig_negative"]
        if k >= 2:
            # P(at least one + AND at least one - | K significant)
            p_inconsistent = 1.0 - 2.0 * (0.5 ** k)
            expected_count += p_inconsistent
        # k=0 or k=1: impossible to have inconsistency, contributes 0

    return {
        "expected_exploratory": float(expected_count),
        "p_sig_overall": float(p_sig),
        "method": "binomial_null (independent, p=0.5 sign assignment)",
        "caveat": (
            "This is a lower bound because it assumes independence "
            "and equal sign probability. Real correlations among "
            "disorders will increase the expected count."
        ),
    }


# ======================== FEATURE COLUMN LOADING ========================

def load_feature_columns():
    """Load valid feature columns from Step 02's outputs."""
    logger.info("Loading feature columns from Step 02 outputs...")

    selection_path = os.path.join(FEATURES_DIR, "feature_selection.json")
    if os.path.exists(selection_path):
        with open(selection_path, 'r') as f:
            selection = json.load(f)
        valid_features = selection.get("valid_features", [])

        if isinstance(valid_features, int):
            logger.error("  BUG: feature_selection.json has int valid_features. "
                        "Falling back.")
            valid_features = []

        if isinstance(valid_features, list) and len(valid_features) > 0:
            str_features = [f for f in valid_features if isinstance(f, str)]
            dropped = selection.get("dropped_features",
                                    selection.get("dropped_list", []))
            if isinstance(dropped, int):
                dropped = []
            logger.info(f"  Loaded from feature_selection.json: "
                        f"{len(str_features)} valid features "
                        f"({len(dropped)} dropped: {dropped})")
            return str_features

    scaler_path = os.path.join(FEATURES_DIR, "robust_scaler_params.json")
    if os.path.exists(scaler_path):
        with open(scaler_path, 'r') as f:
            scaler_params = json.load(f)
        valid_features = scaler_params.get("valid_features", [])
        if isinstance(valid_features, list) and len(valid_features) > 0:
            str_features = [f for f in valid_features if isinstance(f, str)]
            logger.info(f"  Loaded from robust_scaler_params.json: "
                        f"{len(str_features)} valid features")
            return str_features

    logger.warning("  Falling back to metadata-derived features.")
    meta_path = os.path.join(FEATURES_DIR, "feature_metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No feature metadata found in {FEATURES_DIR}")

    with open(meta_path, 'r') as f:
        feature_meta = json.load(f)

    df_path = os.path.join(FEATURES_DIR, "user_features_standardized.parquet")
    df = pd.read_parquet(df_path)

    candidate_cols = []
    for dim, cols in feature_meta.items():
        if isinstance(cols, list):
            candidate_cols.extend(cols)
    log_cols = [c for c in df.columns
                if (c.endswith("_log") or c.endswith("_log_signed"))]
    candidate_cols.extend(log_cols)
    candidate_cols = list(dict.fromkeys(candidate_cols))

    always_exclude = {
        'user_id_hash', 'group', 'disorder_group', 'is_retrospective',
        't0_date', 'analysis_timestamp', 'split', 'strat_key',
        'pair_id', 'data_version', 'patch_applied',
        'primary_subreddit', 'is_temporal_structurally_imputed',
    }
    psm_artifacts = {"ps_score", "iptw_att_weight", "eb_weight",
                     "match_strata", "distance", "caliper"}
    metadata_prefixes = ('quality_', 'pct_', 'source_')

    valid_features = []
    for col in candidate_cols:
        if col not in df.columns:
            continue
        if col in always_exclude or col in psm_artifacts:
            continue
        if col.startswith(metadata_prefixes):
            continue
        if col.endswith('_x') or col.endswith('_y'):
            continue
        if df[col].dtype == object:
            continue
        if df[col].notna().sum() == 0:
            continue
        valid_features.append(col)

    logger.info(f"  Derived {len(valid_features)} features from metadata")
    return valid_features


# ======================== MAIN PIPELINE ========================

def run_effect_sizes():
    """Main pipeline for effect size computation."""
    logger.info("=" * 80)
    logger.info("MODULE 1 - Step 03: Effect Size Matrix Computation (v1.6)")
    logger.info("Fixes: DiffSpec noise gate, weight trimming, deterministic seed,")
    logger.info("       strict direction inconsistency, CU weight sensitivity")
    logger.info("=" * 80)

    # Load standardized features
    df_path = os.path.join(FEATURES_DIR, "user_features_standardized.parquet")
    if not os.path.exists(df_path):
        raise FileNotFoundError(
            f"Standardized features not found: {df_path}. "
            f"Run s02_feature_aggregation.py first."
        )

    df = pd.read_parquet(df_path)
    logger.info(f"Loaded features: {df.shape}")

    # Load feature columns from Step 02
    all_feature_cols = load_feature_columns()

    missing_cols = [c for c in all_feature_cols if c not in df.columns]
    if missing_cols:
        logger.warning(f"  {len(missing_cols)} features not in DataFrame")
        all_feature_cols = [c for c in all_feature_cols if c in df.columns]

    # Split features into analysis vs artifact
    artifact_cols = [c for c in all_feature_cols
                     if c in SELECTION_ARTIFACT_FEATURES]
    analysis_cols = [c for c in all_feature_cols
                     if c not in SELECTION_ARTIFACT_FEATURES]

    logger.info(f"Total features from Step 02: {len(all_feature_cols)}")
    logger.info(f"  Selection artifact features (excluded from primary "
                f"analysis): {len(artifact_cols)}: {artifact_cols}")
    logger.info(f"  Analysis features: {len(analysis_cols)}")

    # Filter to training set
    if COMPUTE_ON_TRAIN_ONLY and "split" in df.columns:
        n_before = len(df)
        df = df[df["split"] == "train"].copy()
        logger.info(f"Restricted to training set: "
                    f"{len(df)} users ({len(df)/n_before*100:.1f}%)")

    # Validate disorders
    available_disorders = []
    for disorder in DISORDER_GROUPS:
        n_patients = ((df["group"] == "patient") &
                      (df["disorder_group"] == disorder)).sum()
        if n_patients >= MIN_PATIENTS_PER_DISORDER:
            available_disorders.append(disorder)
        else:
            logger.warning(f"  Skipping {disorder}: only {n_patients} patients")

    logger.info(f"Disorders to analyze: "
                f"{len(available_disorders)}/{len(DISORDER_GROUPS)}")

    if not available_disorders:
        logger.error("No disorders with sufficient patients!")
        return None, None, None

    # ====== ANALYSIS 1: Disorder vs Controls ======
    logger.info("\n--- Analysis 1: Disorder vs Controls Effect Sizes ---")
    logger.info(f"    Computing all {len(all_feature_cols)} features (artifact features flagged for exclusion)")
    logger.info(f"    Bootstrap iterations: {BOOTSTRAP_N_ANALYSIS1}")

    all_results = []
    for disorder in available_disorders:
        results = compute_disorder_effect_sizes(
            df, all_feature_cols, disorder,
            do_bootstrap=True,
            n_boot=BOOTSTRAP_N_ANALYSIS1,
            use_iptw=True
        )
        all_results.extend(results)

    if not all_results:
        logger.error("No effect sizes computed!")
        return None, None, None

    results_df = pd.DataFrame(all_results)
    results_df = apply_fdr_correction(results_df)

    # Save full results
    results_df.to_csv(os.path.join(STATS_DIR, "effect_sizes_full.csv"),
                      index=False)
    results_df.to_parquet(
        os.path.join(STATS_DIR, "effect_sizes_full.parquet"), index=False
    )

    # Build matrices
    effect_matrix, sig_matrix, p_matrix = build_effect_size_matrix(results_df)

    effect_matrix.to_csv(os.path.join(STATS_DIR, "effect_size_matrix.csv"))
    sig_matrix.to_csv(os.path.join(STATS_DIR, "significance_matrix.csv"))
    p_matrix.to_csv(os.path.join(STATS_DIR, "p_value_matrix.csv"))

    logger.info(f"Effect size matrix shape: {effect_matrix.shape}")

    # Separate artifact-free matrices for primary analysis
    nonartifact_features = [f for f in effect_matrix.index
                            if f not in SELECTION_ARTIFACT_FEATURES]
    effect_matrix_clean = effect_matrix.loc[nonartifact_features]
    sig_matrix_clean = sig_matrix.loc[nonartifact_features]
    p_matrix_clean = p_matrix.loc[nonartifact_features]

    effect_matrix_clean.to_csv(
        os.path.join(STATS_DIR, "effect_size_matrix_clean.csv"))
    sig_matrix_clean.to_csv(
        os.path.join(STATS_DIR, "significance_matrix_clean.csv"))
    p_matrix_clean.to_csv(
        os.path.join(STATS_DIR, "p_value_matrix_clean.csv"))

    logger.info(f"Clean effect size matrix (no artifacts): "
                f"{effect_matrix_clean.shape}")

    # ====== SPECIFICITY & CLINICAL UTILITY (on clean matrix) ======
    specificity = compute_specificity_scores(effect_matrix_clean)
    specificity.to_csv(os.path.join(STATS_DIR, "feature_specificity.csv"))

    diff_specificity = compute_differential_specificity(
        effect_matrix_clean, sig_matrix_clean
    )
    diff_specificity.to_csv(
        os.path.join(STATS_DIR, "feature_differential_specificity.csv")
    )

    # [FIX v1.6] clinical_utility now returns (series, sensitivity_report)
    clinical_utility, cu_sensitivity = compute_clinical_utility(
        effect_matrix_clean, sig_matrix_clean
    )
    clinical_utility.to_csv(
        os.path.join(STATS_DIR, "feature_clinical_utility.csv")
    )

    # [FIX v1.6] Save CU sensitivity report
    cu_sensitivity_path = os.path.join(STATS_DIR, "cu_weight_sensitivity.json")
    with open(cu_sensitivity_path, 'w') as f:
        json.dump(cu_sensitivity, f, indent=2, default=str)
    logger.info(f"  Saved CU weight sensitivity to: {cu_sensitivity_path}")

    # [FIX v1.6] Direction inconsistency now returns (df, null_report)
    direction_df, direction_null = detect_direction_inconsistency(
        effect_matrix_clean, sig_matrix_clean
    )
    direction_df.to_csv(
        os.path.join(STATS_DIR, "feature_direction_analysis.csv"),
        index=False
    )

    # ====== FEATURE RANKING (on clean features) ======
    feature_ranking = pd.DataFrame({
        "specificity": specificity,
        "differential_specificity": diff_specificity,
        "clinical_utility": clinical_utility,
    })

    for feat in feature_ranking.index:
        if feat in effect_matrix_clean.index:
            row = effect_matrix_clean.loc[feat]
            best_disorder = row.abs().idxmax()
            feature_ranking.loc[feat, "max_abs_g"] = abs(row[best_disorder])
            feature_ranking.loc[feat, "best_disorder"] = best_disorder
            feature_ranking.loc[feat, "best_g"] = row[best_disorder]

    if len(direction_df) > 0:
        dir_map = direction_df.set_index("feature")[
            "has_direction_inconsistency"
        ]
        feature_ranking["direction_inconsistent"] = \
            feature_ranking.index.map(dir_map).fillna(False)

        # [FIX v1.6] Also map strict direction inconsistency
        strict_dir_map = direction_df.set_index("feature")[
            "has_strict_direction_inconsistency"
        ]
        feature_ranking["strict_direction_inconsistent"] = \
            feature_ranking.index.map(strict_dir_map).fillna(False)

    feature_ranking = feature_ranking.sort_values(
        "clinical_utility", ascending=False
    )
    feature_ranking.to_csv(os.path.join(STATS_DIR, "feature_ranking.csv"))

    logger.info(f"  Feature ranking (top 10 by clinical utility):")
    for feat, row in feature_ranking.head(10).iterrows():
        dir_flag = " ↕" if row.get("direction_inconsistent", False) else ""
        strict_flag = " ↕S" if row.get("strict_direction_inconsistent", False) else ""
        logger.info(f"    {feat}: CU={row.get('clinical_utility', 0):.3f}, "
                    f"diff_spec={row.get('differential_specificity', 0):.3f}, "
                    f"max|g|={row.get('max_abs_g', 0):.3f}, "
                    f"best={row.get('best_disorder', '?')}{dir_flag}{strict_flag}")

    # ====== ANALYSIS 2: Pairwise Disorder Comparisons ======
    logger.info("\n--- Analysis 2: Pairwise Disorder Effect Sizes ---")
    logger.info(f"    Using analytical CI only (no bootstrap)")
    logger.info(f"    Computing all {len(all_feature_cols)} features (artifact features flagged for exclusion)")

    pairwise_results = []
    disorder_pairs = list(combinations(available_disorders, 2))
    logger.info(f"Total disorder pairs: {len(disorder_pairs)}")

    for disorder_a, disorder_b in disorder_pairs:
        results = compute_pairwise_disorder_effects(
            df, analysis_cols, disorder_a, disorder_b
        )
        pairwise_results.extend(results)

    if pairwise_results:
        pairwise_df = pd.DataFrame(pairwise_results)

        valid_pw_mask = pairwise_df["p_value"].notna()
        if valid_pw_mask.sum() > 0:
            valid_pw_idx = pairwise_df.index[valid_pw_mask]
            reject_pw, p_adj_pw, _, _ = multipletests(
                pairwise_df.loc[valid_pw_idx, "p_value"].values,
                alpha=ALPHA, method=FDR_METHOD
            )
            pairwise_df.loc[valid_pw_idx, "p_fdr"] = p_adj_pw
            pairwise_df.loc[valid_pw_idx, "sig_fdr"] = reject_pw
        else:
            pairwise_df["p_fdr"] = np.nan
            pairwise_df["sig_fdr"] = False

        pairwise_df.to_csv(
            os.path.join(STATS_DIR, "pairwise_effect_sizes.csv"), index=False
        )
        pairwise_df.to_parquet(
            os.path.join(STATS_DIR, "pairwise_effect_sizes.parquet"),
            index=False
        )

        pairwise_summary = pairwise_df.groupby(
            ["disorder_a", "disorder_b"]
        ).agg(
            mean_abs_g=("hedges_g", lambda x: np.nanmean(np.abs(x))),
            median_abs_g=("hedges_g", lambda x: np.nanmedian(np.abs(x))),
            max_abs_g=("hedges_g", lambda x: np.nanmax(np.abs(x))),
            n_total_features=("feature", "count"),
        ).reset_index()

        if "sig_fdr" in pairwise_df.columns:
            n_sig = pairwise_df.groupby(
                ["disorder_a", "disorder_b"]
            )["sig_fdr"].sum().reset_index(name="n_sig_features")
            pairwise_summary = pairwise_summary.merge(
                n_sig, on=["disorder_a", "disorder_b"], how="left"
            )

        pairwise_summary.to_csv(
            os.path.join(STATS_DIR, "pairwise_summary.csv"), index=False
        )
    else:
        pairwise_df = pd.DataFrame()
        pairwise_summary = pd.DataFrame()

    # ====== SUMMARY REPORT ======
    results_clean = results_df[~results_df["is_selection_artifact"]].copy()  # 加上 .copy()
    results_clean["abs_hedges_g"] = results_clean["hedges_g"].abs()

    top10_largest = results_clean.nlargest(10, "abs_hedges_g")[
        ["disorder", "feature", "hedges_g", "p_fdr_global", "effect_category"]
    ].to_dict("records")

    top10_cu = clinical_utility.nlargest(
        min(10, len(clinical_utility))
    ).to_dict()

    most_similar = None
    most_distinct = None
    if len(pairwise_summary) > 0 and "mean_abs_g" in pairwise_summary.columns:
        most_similar = pairwise_summary.nsmallest(1, "mean_abs_g")[
            ["disorder_a", "disorder_b", "mean_abs_g"]
        ].to_dict("records")
        most_distinct = pairwise_summary.nlargest(1, "mean_abs_g")[
            ["disorder_a", "disorder_b", "mean_abs_g"]
        ].to_dict("records")

    n_inconsistent_exploratory = 0
    n_inconsistent_strict = 0
    inconsistent_features_exploratory = []
    inconsistent_features_strict = []
    if len(direction_df) > 0:
        nonartifact_dir = direction_df[~direction_df["is_selection_artifact"]]
        n_inconsistent_exploratory = nonartifact_dir[
            "has_direction_inconsistency"
        ].sum()
        n_inconsistent_strict = nonartifact_dir[
            "has_strict_direction_inconsistency"
        ].sum()
        inconsistent_features_exploratory = nonartifact_dir[
            nonartifact_dir["has_direction_inconsistency"]
        ]["feature"].tolist()[:10]
        inconsistent_features_strict = nonartifact_dir[
            nonartifact_dir["has_strict_direction_inconsistency"]
        ]["feature"].tolist()[:10]

    report = {
        "pipeline_version": "v1.6",
        "analysis_scope": "train_only" if COMPUTE_ON_TRAIN_ONLY else "full",
        "n_features_total": len(all_feature_cols),
        "n_features_analysis": len(analysis_cols),
        "selection_artifact_features": artifact_cols,
        "selection_artifact_rationale": (
            "These features are circular with patient identification: "
            "patients were identified by posting in mental health subreddits, "
            "so mental_health_sub_ratio and support_sub_ratio are "
            "confounded with the selection criterion."
        ),

        # [FIX v1.6] v1.6-specific parameters
        "v16_parameters": {
            "min_abs_g_for_diffspec": MIN_ABS_G_FOR_DIFFSPEC,
            "weight_trim_percentiles": [
                WEIGHT_TRIM_LOWER_PCT, WEIGHT_TRIM_UPPER_PCT
            ],
            "cu_primary_scheme": CU_PRIMARY_SCHEME,
            "cu_primary_weights": CU_WEIGHT_SCHEMES[CU_PRIMARY_SCHEME],
            "strict_direction_min_abs_g": STRICT_DIRECTION_MIN_ABS_G,
            "strict_direction_min_range": STRICT_DIRECTION_MIN_RANGE,
        },

        "n_disorders": len(available_disorders),
        "disorders_analyzed": available_disorders,
        "effect_thresholds": EFFECT_THRESHOLDS,
        "total_tests": len(results_df),
        "total_tests_nonartifact": len(results_clean),
        "significant_global_fdr": int(results_df["sig_fdr_global"].sum()),
        "significant_nonartifact": int(results_clean["sig_fdr_global"].sum()),
        "pct_significant_nonartifact": float(
            results_clean["sig_fdr_global"].mean() * 100
        ),
        "effect_categories_nonartifact": results_clean[
            "effect_category"
        ].value_counts().to_dict(),
        "specificity_range": [
            float(specificity.min()), float(specificity.max())
        ] if len(specificity) > 0 else None,
        "differential_specificity_range": [
            float(diff_specificity.min()),
            float(diff_specificity.max())
        ] if len(diff_specificity) > 0 else None,
        "clinical_utility_range": [
            float(clinical_utility.min()),
            float(clinical_utility.max())
        ] if len(clinical_utility) > 0 else None,

        # [FIX v1.6] Direction inconsistency — both tiers
        "n_direction_inconsistent_exploratory": int(n_inconsistent_exploratory),
        "n_direction_inconsistent_strict": int(n_inconsistent_strict),
        "direction_inconsistent_features_exploratory": inconsistent_features_exploratory,
        "direction_inconsistent_features_strict": inconsistent_features_strict,

        # [FIX v1.6] Null expectation
        "direction_inconsistency_null_expectation": direction_null,

        # [FIX v1.6] Weight sensitivity
        "cu_weight_sensitivity": cu_sensitivity,

        "top10_largest_effects_nonartifact": top10_largest,
        "top10_clinical_utility": top10_cu,
        "most_similar_pair": most_similar,
        "most_distinct_pair": most_distinct,
        "iptw_weighted": "iptw_att_weight" in df.columns,
        "bootstrap_iterations_analysis1": BOOTSTRAP_N_ANALYSIS1,
        "bootstrap_iterations_analysis2": BOOTSTRAP_N_ANALYSIS2,
        "computation_timestamp": pd.Timestamp.now().isoformat(),
    }

    report_path = os.path.join(STATS_DIR, "effect_sizes_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY REPORT (v1.6)")
    logger.info(f"{'='*60}")
    logger.info(f"  Features: {report['n_features_analysis']} analysis + "
                f"{len(artifact_cols)} artifacts (excluded)")
    logger.info(f"  Disorders: {report['n_disorders']}")
    logger.info(f"  Tests (non-artifact): {report['total_tests_nonartifact']}")
    logger.info(f"  Significant (FDR, non-artifact): "
                f"{report['significant_nonartifact']} "
                f"({report['pct_significant_nonartifact']:.1f}%)")
    logger.info(f"  Effect categories: "
                f"{report['effect_categories_nonartifact']}")

    if top10_largest:
        logger.info(f"\n  Top 5 largest effects (non-artifact):")
        for item in top10_largest[:5]:
            logger.info(f"    {item['disorder']} / {item['feature']}: "
                        f"g={item['hedges_g']:.3f} "
                        f"({item['effect_category']})")

    if len(clinical_utility) > 0:
        top5_cu = clinical_utility.nlargest(min(5, len(clinical_utility)))
        logger.info(f"\n  Top 5 clinical utility:")
        for feat, cu in top5_cu.items():
            logger.info(f"    {feat}: CU={cu:.3f}")

    logger.info(f"\n  Direction inconsistency:")
    logger.info(f"    Exploratory: {n_inconsistent_exploratory} features "
                f"(null expectation: "
                f"{direction_null['expected_exploratory']:.1f})")
    logger.info(f"    Strict: {n_inconsistent_strict} features "
                f"(effect-size-gated)")

    logger.info(f"\n  CU weight sensitivity: "
                f"{cu_sensitivity.get('overall_stability', 'unknown')} "
                f"(min Jaccard = "
                f"{cu_sensitivity.get('min_jaccard', 'N/A')})")

    logger.info(f"\n  Artifact features (excluded from primary analysis):")
    for feat in artifact_cols:
        if feat in effect_matrix.index:
            max_g = effect_matrix.loc[feat].abs().max()
            logger.info(f"    {feat}: max|g|={max_g:.3f} "
                        f"(circular with patient identification)")

    logger.info(f"\n  Report saved to: {report_path}")
    logger.info("=" * 80)
    logger.info("EFFECT SIZE COMPUTATION COMPLETE")
    logger.info("=" * 80)

    return results_df, effect_matrix, pairwise_df


if __name__ == "__main__":
    run_effect_sizes()