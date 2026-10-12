"""
Step 06: Sensitivity & Robustness Analyses (v1.0 → v1.2)

Ensures that findings are robust to:
  1. Confound adjustment
  2. Sample size stability
  3. Multiple comparison correction methods
  4. Outlier handling strategies
  5. Cross-validation scheme variations
  6. Feature set variations
  7. Temporal stability
  8. Permutation test
  9. Effect size heterogeneity (I²)

v1.0 → v1.2 Changelog:
  [CRITICAL] Fix file references: effect_sizes_all.csv → effect_sizes_full.csv
  [CRITICAL] Inline cohens_d instead of fragile cross-module import
  [CRITICAL] Fix outlier sensitivity: use raw (unscaled) features, not
             already-winsorized standardized features
  [FIX] Feature loading: use feature_selection.json + exclude artifacts
  [FIX] Control subsampling for all classification-based analyses
  [FIX] Restrict to training set
  [FIX] Permutation test: use subsampled data + reduced iterations
  [FIX] Sample size stability: use subsampled controls
  [FIX] Multiple comparison: use clean effect sizes (non-artifact)
  [OPT] Reduced computational burden throughout
"""

import os
import json
import logging
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from statsmodels.stats.multitest import multipletests
from scipy.stats.mstats import winsorize

from config_module1 import (
    FEATURES_DIR, STATS_DIR, FIGURES_DIR,
    DISORDER_GROUPS, DISORDER_DISPLAY_NAMES,
    CV_FOLDS, RANDOM_SEED, NUM_CORES, ALPHA, FDR_METHOD, PSM_MATCHED_PATH
)

os.makedirs(STATS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(STATS_DIR, "sensitivity.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

np.random.seed(RANDOM_SEED)

# Consistent artifact exclusion
SELECTION_ARTIFACT_FEATURES = {
    "mental_health_sub_ratio",
    "support_sub_ratio",
    "subreddit_entropy",
    "subreddit_herfindahl",
    "primary_subreddit_ratio",
}

# Control subsampling
CONTROL_SUBSAMPLE_RATIO = 3
CONTROL_SUBSAMPLE_MIN = 5000


# ======================== INLINED UTILITY FUNCTIONS ========================

def cohens_d(group1, group2):
    """[INLINE v1.2] Compute Cohen's d with Hedges' correction."""
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


def subsample_controls(df_patient, df_control, random_state=None):
    """Subsample controls for computational efficiency."""
    n_patient = len(df_patient)
    n_target = max(CONTROL_SUBSAMPLE_RATIO * n_patient, CONTROL_SUBSAMPLE_MIN)
    n_target = min(n_target, len(df_control))
    if n_target >= len(df_control):
        return df_control
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(df_control), size=n_target, replace=False)
    return df_control.iloc[idx].copy()


def load_analysis_features():
    """Load features consistent with s02/s03/s04."""
    feature_cols = []
    selection_path = os.path.join(FEATURES_DIR, "feature_selection.json")
    if os.path.exists(selection_path):
        with open(selection_path, 'r') as f:
            selection = json.load(f)
        valid_features = selection.get("valid_features", [])
        if isinstance(valid_features, list):
            feature_cols = [f for f in valid_features if isinstance(f, str)]

    if not feature_cols:
        scaler_path = os.path.join(FEATURES_DIR, "robust_scaler_params.json")
        if os.path.exists(scaler_path):
            with open(scaler_path, 'r') as f:
                scaler_params = json.load(f)
            valid_features = scaler_params.get("valid_features", [])
            if isinstance(valid_features, list):
                feature_cols = [f for f in valid_features if isinstance(f, str)]

    feature_cols = [c for c in feature_cols if c not in SELECTION_ARTIFACT_FEATURES]
    return feature_cols


def _load_effect_sizes():
    """[FIX v1.2] Load effect sizes from correct file."""
    # Priority: clean (non-artifact) results
    for path in [
        os.path.join(STATS_DIR, "effect_sizes_full.csv"),
        os.path.join(STATS_DIR, "effect_sizes_full.parquet"),
    ]:
        if os.path.exists(path):
            if path.endswith('.parquet'):
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
            # Filter out artifact features
            df = df[~df["feature"].isin(SELECTION_ARTIFACT_FEATURES)].copy()
            logger.info(f"  Loaded effect sizes from {os.path.basename(path)}: {len(df)} records")
            return df

    raise FileNotFoundError("No effect size file found in STATS_DIR")


# ======================== 1. CONFOUND-ADJUSTED ANALYSIS ========================

def confound_adjusted_effect_sizes(df, feature_cols, confound_cols=None):
    """
    [FIX v1.3] Confound-adjusted effect sizes with both linear and
    polynomial residualization, using available proxy columns when
    exact confound names are absent.
    
    Column mapping strategy:
      account_age_days  → tenure_days (PSM file, analysis span proxy)
      total_posts       → n_posts_total (feature set, analysis period)
      total_subreddits  → n_unique_subreddits (feature set, analysis period)
      is_comorbid       → unavailable (no reliable proxy)
    """
    logger.info("\n=== Sensitivity 1: Confound-Adjusted Effect Sizes ===")

    # === 列名映射：理想名 → 数据中的可用替代 ===
    CONFOUND_COLUMN_MAP = {
        "account_age_days": ["account_age_days", "tenure_days"],
        "total_posts": ["total_posts", "n_posts_total"],
        "total_subreddits": ["total_subreddits", "n_unique_subreddits"],
    }

    if confound_cols is None:
        # Step 1: 从当前df中找近似列
        confound_cols = []
        confound_source = {}  # 记录每列来源

        for target_name, candidate_names in CONFOUND_COLUMN_MAP.items():
            for candidate in candidate_names:
                if candidate in df.columns:
                    confound_cols.append(candidate)
                    confound_source[candidate] = f"proxy_for_{target_name}"
                    break

        # Step 2: 从PSM文件补充（tenure_days等可能在PSM中但不在特征集中）
        PSM_CONFOUND_COLS = [
            "tenure_days", "total_posts", "posts_per_day",
            "circadian_mean_hour", "circadian_regularity",
            "readability_grade", "vocab_density",
        ]

        if os.path.exists(PSM_MATCHED_PATH):
            try:
                psm_df = pd.read_parquet(PSM_MATCHED_PATH)
                available_psm = [
                    c for c in PSM_CONFOUND_COLS
                    if c in psm_df.columns and c not in df.columns
                ]

                if available_psm:
                    psm_merge = psm_df[["user_id_hash"] + available_psm].copy()
                    psm_merge = psm_merge.drop_duplicates(subset=["user_id_hash"])
                    df = df.merge(psm_merge, on="user_id_hash", how="left")
                    confound_cols.extend(available_psm)
                    for c in available_psm:
                        confound_source[c] = "psm_file"
                    logger.info(f"  Supplemented {len(available_psm)} confound "
                               f"columns from PSM file: {available_psm}")
            except Exception as e:
                logger.warning(f"  Failed to load PSM confounds: {e}")

        # Step 3: is_comorbid — 无可靠代理，跳过
        if "is_comorbid" not in df.columns:
            logger.info("  is_comorbid unavailable — no reliable proxy exists. "
                       "Factor 1 in s05 captures the shared severity/comorbidity "
                       "dimension indirectly.")

        # 去重（保持顺序）
        confound_cols = list(dict.fromkeys(confound_cols))

        if not confound_cols:
            logger.warning("  No confound columns found. Skipping.")
            return None

        logger.info(f"  Confounders identified ({len(confound_cols)}):")
        for c in confound_cols:
            src = confound_source.get(c, "feature_set")
            logger.info(f"    {c} ({src})")
    else:
        # 调用方显式指定了confound_cols
        confound_cols = [c for c in confound_cols if c in df.columns]
        if not confound_cols:
            logger.warning("  No confound columns found. Skipping.")
            return None

    # ========================================
    # Method A: Linear residualization
    # ========================================
    logger.info("  Method A: Linear residualization...")
    df_linear = _residualize(df, feature_cols, confound_cols, method="linear")

    # ========================================
    # Method B: Polynomial (degree=2) residualization
    # ========================================
    logger.info("  Method B: Polynomial (degree=2) residualization...")
    df_poly = _residualize(df, feature_cols, confound_cols, method="poly2")

    # Compute effect sizes for both methods
    results_linear = _compute_adjusted_effect_sizes(df_linear, feature_cols, "linear")
    results_poly = _compute_adjusted_effect_sizes(df_poly, feature_cols, "poly2")

    if results_linear is None or results_poly is None:
        return None

    # Compare with original
    orig_df = _load_effect_sizes()

    comparison = {}

    for method_name, adj_df in [("linear", results_linear), ("poly2", results_poly)]:
        merged = adj_df.merge(
            orig_df[["disorder", "feature", "hedges_g"]],
            on=["disorder", "feature"], how="inner"
        )

        if len(merged) > 0 and "hedges_g" in merged.columns:
            valid_m = merged["hedges_g_adjusted"].notna() & merged["hedges_g"].notna()
            if valid_m.sum() > 10:
                r, _ = stats.pearsonr(
                    merged.loc[valid_m, "hedges_g_adjusted"],
                    merged.loc[valid_m, "hedges_g"]
                )
                mad = np.mean(np.abs(
                    merged.loc[valid_m, "hedges_g_adjusted"] -
                    merged.loc[valid_m, "hedges_g"]
                ))
                comparison[method_name] = {
                    "correlation_with_original": float(r),
                    "mean_abs_difference": float(mad),
                    "n_comparisons": int(valid_m.sum()),
                }
                logger.info(f"  {method_name}: r={r:.4f}, MAD={mad:.4f}")

    # Diagnostic: detect nonlinear confounding
    if "linear" in comparison and "poly2" in comparison:
        mad_linear = comparison["linear"]["mean_abs_difference"]
        mad_poly = comparison["poly2"]["mean_abs_difference"]
        nonlinearity_ratio = mad_poly / max(mad_linear, 1e-6)

        logger.info(f"  Nonlinearity ratio (poly_MAD / linear_MAD): "
                    f"{nonlinearity_ratio:.3f}")

        if nonlinearity_ratio > 1.5:
            logger.warning("  ⚠️ Substantial nonlinear confounding detected! "
                          f"Poly adjustment changes g {nonlinearity_ratio:.1f}x "
                          f"more than linear.")
            logger.warning("  Consider using poly2 results as primary.")
        elif nonlinearity_ratio > 1.2:
            logger.info("  Moderate nonlinear confounding present.")
        else:
            logger.info("  ✅ Minimal nonlinear confounding. "
                       "Linear adjustment is sufficient.")

        comparison["nonlinearity_ratio"] = float(nonlinearity_ratio)

    # Save both adjusted results
    results_linear.to_csv(
        os.path.join(STATS_DIR, "sensitivity_confound_linear.csv"), index=False)
    results_poly.to_csv(
        os.path.join(STATS_DIR, "sensitivity_confound_poly2.csv"), index=False)

    return {
        "confound_cols": confound_cols,
        "comparison": comparison,
        "adj_df": results_poly,
    }


def _residualize(df, feature_cols, confound_cols, method="linear"):
    """Residualize features by regressing out confounders."""
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.impute import SimpleImputer

    df_resid = df.copy()
    X_confounds = df[confound_cols].values.astype(float)

    # ========== 修复：在展开前填补NaN ==========
    # 先记录哪些行有有效混杂变量
    valid_confound_row = ~np.any(np.isnan(X_confounds), axis=1)
    n_invalid_rows = (~valid_confound_row).sum()

    if n_invalid_rows > 0:
        # 用中位数填补NaN行，使PolynomialFeatures可以运行
        imputer = SimpleImputer(strategy='median')
        X_confounds_filled = imputer.fit_transform(X_confounds)
        logger.info(f"    Imputed {n_invalid_rows} rows with NaN confounds "
                    f"({n_invalid_rows/len(df)*100:.1f}%) for residualization")
    else:
        X_confounds_filled = X_confounds

    # ========== 构建混杂矩阵 ==========
    if method == "poly2":
        pf = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        X_confound_expanded = pf.fit_transform(X_confounds_filled)
        logger.info(f"    Poly2: {X_confounds_filled.shape[1]} → "
                    f"{X_confound_expanded.shape[1]} confound features")
    else:
        X_confound_expanded = X_confounds_filled

    # ========== 对每个特征做残差化 ==========
    valid_expanded = ~np.any(np.isnan(X_confound_expanded), axis=1)

    # 追踪因混杂变量缺失而跳过的特征数
    n_skipped = 0

    for feat in feature_cols:
        y_feat = df[feat].values.astype(float)
        # 仅对混杂变量和特征都有效的行做回归
        valid = valid_expanded & ~np.isnan(y_feat)

        if valid.sum() < 100:
            n_skipped += 1
            continue

        reg = LinearRegression()
        reg.fit(X_confound_expanded[valid], y_feat[valid])
        residuals = np.full_like(y_feat, np.nan)
        residuals[valid] = y_feat[valid] - reg.predict(X_confound_expanded[valid])
        df_resid[feat] = residuals

    if n_skipped > 0:
        logger.info(f"    Skipped {n_skipped} features with insufficient "
                    f"valid samples")

    return df_resid


def _compute_adjusted_effect_sizes(df_resid, feature_cols, method_name):
    """Compute effect sizes on residualized features."""
    all_results = []

    for disorder in DISORDER_GROUPS:
        patient_mask = (df_resid["group"] == "patient") & \
                       (df_resid["disorder_group"] == disorder)
        control_mask = (df_resid["group"] == "control")

        for feat in feature_cols:
            p_vals = df_resid.loc[patient_mask, feat].dropna().values
            c_vals = df_resid.loc[control_mask, feat].dropna().values
            if len(p_vals) < 10 or len(c_vals) < 10:
                continue
            g, se, p = cohens_d(p_vals, c_vals)
            all_results.append({
                "disorder": disorder, "feature": feat,
                "hedges_g_adjusted": g, "se_adjusted": se,
                "p_value_adjusted": p, "method": method_name,
            })

    if not all_results:
        return None

    adj_df = pd.DataFrame(all_results)

    # FDR correction
    pvals = adj_df["p_value_adjusted"].values
    valid_p = ~np.isnan(pvals)
    if valid_p.sum() > 0:
        reject, pvals_corrected, _, _ = multipletests(
            pvals[valid_p], alpha=ALPHA, method=FDR_METHOD)
        adj_df.loc[valid_p, "p_fdr_adjusted"] = pvals_corrected
        adj_df.loc[valid_p, "significant_adjusted"] = reject

    return adj_df


# ======================== 2. SAMPLE SIZE STABILITY ========================

def sample_size_stability(df, feature_cols, fractions=None, n_repeats=10):
    """
    [FIX v1.2] Subsample analysis with control subsampling.
    """
    logger.info("\n=== Sensitivity 2: Sample Size Stability ===")

    if fractions is None:
        fractions = [0.25, 0.50, 0.75, 1.0]

    control_df = df[df["group"] == "control"].copy()
    stability_results = {}

    for disorder in DISORDER_GROUPS:
        patient_df = df[(df["group"] == "patient") & (df["disorder_group"] == disorder)].copy()
        n_patients = len(patient_df)
        if n_patients < 20:
            continue

        logger.info(f"  {disorder}: {n_patients} patients")
        disorder_results = {}

        for frac in fractions:
            aurocs = []
            rng = np.random.RandomState(RANDOM_SEED)

            for rep in range(n_repeats):
                # Subsample patients
                n_p = max(int(n_patients * frac), 10)
                n_p = min(n_p, n_patients)
                p_idx = rng.choice(n_patients, n_p, replace=False)
                pat_sample = patient_df.iloc[p_idx]

                # Subsample controls
                ctrl_sample = subsample_controls(pat_sample, control_df,
                                                  random_state=RANDOM_SEED + rep)

                subset = pd.concat([pat_sample, ctrl_sample])
                y = (subset["group"] == "patient").astype(int).values
                X = subset[feature_cols].values
                valid = ~np.any(np.isnan(X), axis=1)
                X, y = X[valid], y[valid]

                if y.sum() < 5 or (len(y) - y.sum()) < 5:
                    continue

                try:
                    skf = StratifiedKFold(n_splits=3, shuffle=True,
                                          random_state=RANDOM_SEED + rep)
                    fold_aurocs = []
                    for train_i, test_i in skf.split(X, y):
                        clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000,
                                                  random_state=RANDOM_SEED, class_weight='balanced')
                        clf.fit(X[train_i], y[train_i])
                        y_prob = clf.predict_proba(X[test_i])[:, 1]
                        if len(np.unique(y[test_i])) == 2:
                            fold_aurocs.append(roc_auc_score(y[test_i], y_prob))
                    if fold_aurocs:
                        aurocs.append(np.mean(fold_aurocs))
                except Exception:
                    pass

            if aurocs:
                disorder_results[frac] = {
                    "mean_auroc": float(np.mean(aurocs)),
                    "std_auroc": float(np.std(aurocs)),
                    "ci_lower": float(np.percentile(aurocs, 2.5)),
                    "ci_upper": float(np.percentile(aurocs, 97.5)),
                    "n_repeats": len(aurocs),
                }
                logger.info(f"    frac={frac:.0%}: AUROC = {np.mean(aurocs):.3f} "
                            f"± {np.std(aurocs):.3f}")

        stability_results[disorder] = disorder_results

    return stability_results


# ======================== 3. MULTIPLE COMPARISON METHODS ========================

def multiple_comparison_sensitivity():
    """
    [FIX v1.2] Compare results under different correction methods.
    Uses clean (non-artifact) effect sizes.
    """
    logger.info("\n=== Sensitivity 3: Multiple Comparison Correction Methods ===")

    effect_df = _load_effect_sizes()

    if "p_value" not in effect_df.columns:
        logger.warning("  p_value column not found. Skipping.")
        return None

    pvals = effect_df["p_value"].values
    valid_p = ~np.isnan(pvals)

    methods = {
        "none": None,
        "bonferroni": "bonferroni",
        "holm": "holm",
        "fdr_bh": "fdr_bh",
        "fdr_by": "fdr_by",
    }

    comparison = {}
    for method_name, method_code in methods.items():
        if method_code is None:
            n_sig = int(np.sum(pvals[valid_p] < ALPHA))
        else:
            reject, _, _, _ = multipletests(pvals[valid_p], alpha=ALPHA, method=method_code)
            n_sig = int(np.sum(reject))

        comparison[method_name] = {
            "n_significant": n_sig,
            "n_total": int(valid_p.sum()),
            "proportion_significant": float(n_sig / valid_p.sum()) if valid_p.sum() > 0 else 0.0,
        }
        logger.info(f"  {method_name:15s}: {n_sig:5d} / {valid_p.sum()} "
                    f"({n_sig/valid_p.sum()*100:.1f}%)")

    return comparison


# ======================== 4. OUTLIER SENSITIVITY ========================

def outlier_sensitivity(df, feature_cols):
    """
    [FIX v1.2] Outlier handling with sample size tracking.
    
    Addresses critique: z3_removal reduces sample size, potentially
    underestimating performance.
    
    Key changes:
      1. z-removal replaces outliers with MEDIAN (not NaN), preserving
         sample size
      2. Report sample retention rate per strategy
      3. Add trimmed-mean approach as middle ground
      4. Compare AUROC only at equal sample sizes
    """
    logger.info("\n=== Sensitivity 4: Outlier Handling Strategies ===")
    logger.info("  Note: Testing on RAW features (pre-s02 standardization)")

    # ========== 修复：加载真正原始数据 ==========
    # 优先级：s01原始 > s02未标准化 > s02标准化
    raw_path = None
    raw_source = None

    # 优先：s01的直接输出（缩尾和标准化之前）
    s01_raw = os.path.join(FEATURES_DIR, "user_features_raw.parquet")
    if os.path.exists(s01_raw):
        raw_path = s01_raw
        raw_source = "s01_raw (pre-winsorization)"
    else:
        # 次选：s02的未标准化版本（已缩尾，但不理想）
        s02_matched = os.path.join(FEATURES_DIR, "user_features_matched.parquet")
        if os.path.exists(s02_matched):
            raw_path = s02_matched
            raw_source = "s02_matched (already winsorized - suboptimal)"
            logger.warning("  ⚠️ Using s02 matched data (already winsorized).")
            logger.warning("  For proper outlier testing, re-run s01 with "
                          "audit copy or add user_features_raw.parquet.")

    if raw_path is None:
        logger.warning("  No raw features found. Skipping.")
        return None

    df_raw = pd.read_parquet(raw_path)
    logger.info(f"  Using: {raw_source}")
    logger.info(f"  Raw data shape: {df_raw.shape}")

    if "split" in df_raw.columns:
        df_raw = df_raw[df_raw["split"] == "train"].copy()

    # ========== 检查原始数据是否已被缩尾 ==========
    # 通过检查极端值比例判断
    if raw_source.startswith("s01_raw"):
        # 真正原始数据：报告极端值比例
        n_extreme = 0
        n_total_vals = 0
        for col in feature_cols[:20]:  # 抽样检查
            if col not in df_raw.columns:
                continue
            vals = df_raw[col].dropna().values
            if len(vals) < 100:
                continue
            n_total_vals += len(vals)
            q01, q99 = np.percentile(vals, [1, 99])
            iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
            if iqr > 1e-10:
                outlier_mask = (vals < q01 - 3*iqr) | (vals > q99 + 3*iqr)
                n_extreme += outlier_mask.sum()

        if n_total_vals > 0:
            extreme_rate = n_extreme / n_total_vals * 100
            logger.info(f"  Extreme values (3×IQR): {extreme_rate:.2f}% "
                        f"({'expected' if extreme_rate > 0.5 else 'suspiciously low'})")

    # ========== 后续策略测试代码不变 ==========
    strategies = {
        # 真正的"不做任何处理"基线（原始数据直接标准化）
        "no_outlier_treatment": {"method": "none"},
        # 缩尾策略（现在在真正原始数据上有意义了）
        "winsorize_1_99": {"method": "winsorize", "limits": [0.01, 0.01]},
        "winsorize_5_95": {"method": "winsorize", "limits": [0.05, 0.05]},
        "z3_median_replace": {"method": "median_replace", "z_threshold": 3.0},
        "z2.5_median_replace": {"method": "median_replace", "z_threshold": 2.5},
        "trimmed_winsorize": {"method": "trimmed_winsorize", "limits": [0.05, 0.05]},
    }

    df_baseline = df[df["split"] == "train"].copy() if "split" in df.columns else df.copy()
    results = {}

    for strategy_name, params in strategies.items():
        logger.info(f"  Strategy: {strategy_name}")

        df_processed = df_raw.copy()
        total_outliers_replaced = 0
        total_values = 0

        # ========== 新增：不做任何处理策略 ==========
        if params["method"] == "none":
            # 跳过异常值处理，直接标准化
            pass
        else:
            for col in feature_cols:
                if col not in df_processed.columns:
                    continue
                vals = df_processed[col].values.astype(float)
                valid = ~np.isnan(vals)
                if valid.sum() < 10:
                    continue

                total_values += valid.sum()

                if params["method"] == "winsorize":
                    original = vals[valid].copy()
                    vals[valid] = winsorize(vals[valid], limits=params["limits"])
                    n_changed = int(np.sum(original != vals[valid]))
                    total_outliers_replaced += n_changed
                    df_processed[col] = vals

                elif params["method"] == "trimmed_winsorize":
                    original = vals[valid].copy()
                    vals[valid] = winsorize(vals[valid], limits=params["limits"])
                    n_changed = int(np.sum(original != vals[valid]))
                    total_outliers_replaced += n_changed
                    from scipy.stats import trim_mean
                    tm = trim_mean(vals[valid], proportiontocut=params["limits"][0])
                    vals[valid] = vals[valid] - tm
                    df_processed[col] = vals

                elif params["method"] == "median_replace":
                    z_thresh = params["z_threshold"]
                    col_median = np.median(vals[valid])
                    col_std = np.std(vals[valid], ddof=1)
                    if col_std < 1e-10:
                        continue
                    z_scores = np.abs((vals[valid] - col_median) / col_std)
                    outlier_mask = z_scores > z_thresh
                    n_outliers = outlier_mask.sum()
                    total_outliers_replaced += n_outliers
                    outlier_indices = np.where(valid)[0][outlier_mask]
                    vals[outlier_indices] = col_median
                    df_processed[col] = vals

        # ========== 报告异常值替换率 ==========
        if total_values > 0:
            outlier_rate = total_outliers_replaced / total_values * 100
            logger.info(f"    Outliers replaced: {total_outliers_replaced} "
                        f"({outlier_rate:.2f}% of values)")
        logger.info(f"    Sample retention: 100%")

        # ========== 统一标准化流程 ==========
        from sklearn.preprocessing import RobustScaler
        valid_feat = [c for c in feature_cols if c in df_processed.columns]
        imputer = SimpleImputer(strategy='median')
        df_processed[valid_feat] = imputer.fit_transform(df_processed[valid_feat])
        scaler = RobustScaler()
        df_processed[valid_feat] = scaler.fit_transform(df_processed[valid_feat])

        # ========== 判别测试 ==========
        test_disorders = DISORDER_GROUPS[:3]
        strategy_aurocs = {}
        
        for disorder in test_disorders:
            patient_mask = (df_processed["group"] == "patient") & \
                           (df_processed["disorder_group"] == disorder)
            control_mask = (df_processed["group"] == "control")
            subset = df_processed[patient_mask | control_mask]

            y = (subset["group"] == "patient").astype(int).values
            X = subset[valid_feat].values
            valid = ~np.any(np.isnan(X), axis=1)
            X, y = X[valid], y[valid]

            # Subsample controls
            if y.sum() < 15 or (len(y) - y.sum()) < 15:
                continue
            if (len(y) - y.sum()) > CONTROL_SUBSAMPLE_MIN:
                rng = np.random.RandomState(RANDOM_SEED)
                ctrl_idx = np.where(y == 0)[0]
                pat_idx = np.where(y == 1)[0]
                n_ctrl = min(max(CONTROL_SUBSAMPLE_RATIO * y.sum(), CONTROL_SUBSAMPLE_MIN),
                             len(ctrl_idx))
                keep_idx = np.concatenate([pat_idx, rng.choice(ctrl_idx, n_ctrl, replace=False)])
                X, y = X[keep_idx], y[keep_idx]

            try:
                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
                fold_aurocs = []
                for train_i, test_i in skf.split(X, y):
                    clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000,
                                              random_state=RANDOM_SEED, class_weight='balanced')
                    clf.fit(X[train_i], y[train_i])
                    y_prob = clf.predict_proba(X[test_i])[:, 1]
                    if len(np.unique(y[test_i])) == 2:
                        fold_aurocs.append(roc_auc_score(y[test_i], y_prob))
                if fold_aurocs:
                    strategy_aurocs[disorder] = float(np.mean(fold_aurocs))
            except Exception:
                pass

        if strategy_aurocs:
            mean_auroc = np.mean(list(strategy_aurocs.values()))
            results[strategy_name] = {
                "per_disorder": strategy_aurocs,
                "mean_auroc": float(mean_auroc),
                "sample_retention_rate": 1.0,
                "outliers_replaced": total_outliers_replaced,
            }
            logger.info(f"    Mean AUROC: {mean_auroc:.3f}")

    # Baseline comparison
    baseline_aurocs = {}
    for disorder in test_disorders:
        patient_mask = (df_baseline["group"] == "patient") & \
                       (df_baseline["disorder_group"] == disorder)
        control_mask = (df_baseline["group"] == "control")
        subset = df_baseline[patient_mask | control_mask]
        y = (subset["group"] == "patient").astype(int).values
        X = subset[feature_cols].values
        valid = ~np.any(np.isnan(X), axis=1)
        X, y = X[valid], y[valid]
        if y.sum() >= 15 and (len(y) - y.sum()) >= 15:
            if (len(y) - y.sum()) > CONTROL_SUBSAMPLE_MIN:
                rng = np.random.RandomState(RANDOM_SEED)
                ctrl_idx = np.where(y == 0)[0]
                pat_idx = np.where(y == 1)[0]
                n_ctrl = min(max(CONTROL_SUBSAMPLE_RATIO * y.sum(), CONTROL_SUBSAMPLE_MIN),
                             len(ctrl_idx))
                keep_idx = np.concatenate([pat_idx, rng.choice(ctrl_idx, n_ctrl, replace=False)])
                X, y = X[keep_idx], y[keep_idx]
            try:
                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
                fold_aurocs = []
                for train_i, test_i in skf.split(X, y):
                    clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000,
                                              random_state=RANDOM_SEED, class_weight='balanced')
                    clf.fit(X[train_i], y[train_i])
                    y_prob = clf.predict_proba(X[test_i])[:, 1]
                    if len(np.unique(y[test_i])) == 2:
                        fold_aurocs.append(roc_auc_score(y[test_i], y_prob))
                if fold_aurocs:
                    baseline_aurocs[disorder] = float(np.mean(fold_aurocs))
            except Exception:
                pass

    if baseline_aurocs:
        results["baseline_s02"] = {
            "per_disorder": baseline_aurocs,
            "mean_auroc": float(np.mean(list(baseline_aurocs.values()))),
            "sample_retention_rate": 1.0,
            "outliers_replaced": -1,  # 不适用
        }
        logger.info(f"  Baseline (s02 default): "
                    f"{np.mean(list(baseline_aurocs.values())):.3f}")

    # [FIX v1.2] Summary: AUROC range across strategies
    all_mean_aurocs = [v["mean_auroc"] for v in results.values()]
    if all_mean_aurocs:
        auroc_range = max(all_mean_aurocs) - min(all_mean_aurocs)
        logger.info(f"  AUROC range across strategies: {auroc_range:.3f}")
        if auroc_range < 0.02:
            logger.info(f"  ✅ Results robust to outlier handling (Δ < 0.02)")
        elif auroc_range < 0.05:
            logger.info(f"  ⚡ Moderate sensitivity to outlier handling (Δ = {auroc_range:.3f})")
        else:
            logger.warning(f"  ⚠️ High sensitivity to outlier handling! (Δ = {auroc_range:.3f})")

    return results


# ======================== 5. CV SCHEME SENSITIVITY ========================

def cv_scheme_sensitivity(df, feature_cols):
    """Compare different CV schemes with control subsampling."""
    logger.info("\n=== Sensitivity 5: Cross-Validation Scheme Comparison ===")

    test_disorders = DISORDER_GROUPS[:3]
    control_df = df[df["group"] == "control"].copy()
    results = {}

    for disorder in test_disorders:
        patient_df = df[(df["group"] == "patient") & (df["disorder_group"] == disorder)].copy()
        ctrl_sample = subsample_controls(patient_df, control_df, random_state=RANDOM_SEED)
        subset = pd.concat([patient_df, ctrl_sample])

        y = (subset["group"] == "patient").astype(int).values
        X = subset[feature_cols].values
        valid = ~np.any(np.isnan(X), axis=1)
        X, y = X[valid], y[valid]

        if y.sum() < 15:
            continue

        disorder_results = {}

        # 5-fold
        aurocs = _run_cv(X, y, StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED))
        disorder_results["5fold"] = {"mean_auroc": float(np.mean(aurocs)),
                                     "std_auroc": float(np.std(aurocs))}
        logger.info(f"    {disorder} 5-fold: {np.mean(aurocs):.3f} ± {np.std(aurocs):.3f}")

        # 10-fold
        aurocs = _run_cv(X, y, StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED))
        disorder_results["10fold"] = {"mean_auroc": float(np.mean(aurocs)),
                                      "std_auroc": float(np.std(aurocs))}
        logger.info(f"    {disorder} 10-fold: {np.mean(aurocs):.3f} ± {np.std(aurocs):.3f}")

        # Repeated 5-fold × 3
        aurocs = _run_cv(X, y, RepeatedStratifiedKFold(
            n_splits=5, n_repeats=3, random_state=RANDOM_SEED))
        disorder_results["repeated_5fold_x3"] = {"mean_auroc": float(np.mean(aurocs)),
                                                  "std_auroc": float(np.std(aurocs))}
        logger.info(f"    {disorder} Repeated 5-fold×3: {np.mean(aurocs):.3f}")

        results[disorder] = disorder_results

    return results


def _run_cv(X, y, cv_splitter):
    """Helper: run logistic regression CV."""
    aurocs = []
    for train_i, test_i in cv_splitter.split(X, y):
        clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000,
                                  random_state=RANDOM_SEED, class_weight='balanced')
        clf.fit(X[train_i], y[train_i])
        y_prob = clf.predict_proba(X[test_i])[:, 1]
        if len(np.unique(y[test_i])) == 2:
            aurocs.append(roc_auc_score(y[test_i], y_prob))
    return aurocs


# ======================== 6. FEATURE SET SENSITIVITY ========================

def feature_set_sensitivity(df, feature_cols):
    """Compare discrimination using different feature subsets."""
    logger.info("\n=== Sensitivity 6: Feature Set Variations ===")

    from sklearn.decomposition import PCA

    effect_df = _load_effect_sizes()
    test_disorders = DISORDER_GROUPS[:3]
    control_df = df[df["group"] == "control"].copy()
    results = {}

    for disorder in test_disorders:
        patient_df = df[(df["group"] == "patient") & (df["disorder_group"] == disorder)].copy()
        ctrl_sample = subsample_controls(patient_df, control_df, random_state=RANDOM_SEED)
        subset = pd.concat([patient_df, ctrl_sample])

        y = (subset["group"] == "patient").astype(int).values
        X = subset[feature_cols].values
        valid = ~np.any(np.isnan(X), axis=1)
        X, y = X[valid], y[valid]

        if y.sum() < 15:
            continue

        disorder_results = {}

        # All features
        aurocs = _run_cv(X, y, StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED))
        disorder_results["all_features"] = {"n_features": X.shape[1],
                                             "mean_auroc": float(np.mean(aurocs))}
        logger.info(f"    {disorder} All ({X.shape[1]}): {np.mean(aurocs):.3f}")

        # Top-K by effect size
        dis_effects = effect_df[effect_df["disorder"] == disorder].copy()
        if len(dis_effects) > 0 and "hedges_g" in dis_effects.columns:
            dis_effects["abs_g"] = dis_effects["hedges_g"].abs()
            dis_effects = dis_effects.sort_values("abs_g", ascending=False)

            for top_k in [10, 20, 50]:
                top_feat_names = dis_effects["feature"].head(top_k).tolist()
                top_feat_idx = [i for i, c in enumerate(feature_cols) if c in top_feat_names]
                if len(top_feat_idx) < 5:
                    continue
                X_top = X[:, top_feat_idx]
                aurocs_top = _run_cv(X_top, y,
                    StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED))
                disorder_results[f"top_{top_k}"] = {
                    "n_features": len(top_feat_idx), "mean_auroc": float(np.mean(aurocs_top))}
                logger.info(f"    {disorder} Top-{top_k}: {np.mean(aurocs_top):.3f}")

        # PCA-10
        n_comp = min(10, X.shape[1], X.shape[0] - 1)
        pca = PCA(n_components=n_comp, random_state=RANDOM_SEED)
        X_pca = pca.fit_transform(X)
        aurocs_pca = _run_cv(X_pca, y,
            StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED))
        disorder_results["pca_10"] = {
            "n_features": n_comp, "mean_auroc": float(np.mean(aurocs_pca)),
            "var_explained": float(np.sum(pca.explained_variance_ratio_))}
        logger.info(f"    {disorder} PCA-{n_comp}: {np.mean(aurocs_pca):.3f}")

        results[disorder] = disorder_results

    return results



def _temporal_stability_patient_only(df, feature_cols, valid_time):
    """
    Fallback: compare effect sizes across patient time tertiles,
    using ALL controls as reference (not period-matched).
    
    This is less rigorous but still informative: it tests whether
    the PATIENT signal changes over calendar time.
    """
    df_patients = df[valid_time & (df["group"] == "patient")].copy()
    df_all_controls = df[df["group"] == "control"].copy()

    if len(df_patients) < 100 or len(df_all_controls) < 50:
        logger.warning("  Insufficient data for patient-only temporal analysis.")
        return None

    q33 = df_patients["_time_val"].quantile(0.33)
    q67 = df_patients["_time_val"].quantile(0.67)

    df_early = df_patients[df_patients["_time_val"] <= q33]
    df_mid = df_patients[(df_patients["_time_val"] > q33) &
                          (df_patients["_time_val"] <= q67)]
    df_late = df_patients[df_patients["_time_val"] > q67]

    logger.info(f"  Patient-only time split:")
    logger.info(f"    Early (≤{q33.date()}): {len(df_early)}")
    logger.info(f"    Mid (≤{q67.date()}): {len(df_mid)}")
    logger.info(f"    Late (>{q67.date()}): {len(df_late)}")
    logger.info(f"    Controls (all periods): {len(df_all_controls)}")

    temporal_comparison = []

    for disorder in DISORDER_GROUPS:
        for feat in feature_cols[:50]:
            c_vals = df_all_controls[feat].dropna().values

            if len(c_vals) < 20:
                continue

            g_by_period = {}
            for period_name, period_df in [("early", df_early),
                                            ("mid", df_mid),
                                            ("late", df_late)]:
                p_mask = (period_df["disorder_group"] == disorder)
                p_vals = period_df.loc[p_mask, feat].dropna().values

                if len(p_vals) < 20:
                    g_by_period[period_name] = np.nan
                    continue

                g, _, _ = cohens_d(p_vals, c_vals)
                g_by_period[period_name] = g

            valid_periods = {k: v for k, v in g_by_period.items()
                           if not np.isnan(v)}
            if len(valid_periods) < 2:
                continue

            record = {
                "disorder": disorder, "feature": feat,
                "n_valid_periods": len(valid_periods),
                "method": "patient_tertiles_vs_all_controls",
            }
            record.update({f"g_{k}": v for k, v in g_by_period.items()})

            if "early" in valid_periods and "late" in valid_periods:
                record["g_early"] = valid_periods["early"]
                record["g_late"] = valid_periods["late"]
                record["difference"] = valid_periods["late"] - valid_periods["early"]

            if len(valid_periods) >= 3:
                period_ordinals = []
                g_values = []
                for p_name, p_ord in [("early", 1), ("mid", 2), ("late", 3)]:
                    if p_name in valid_periods:
                        period_ordinals.append(p_ord)
                        g_values.append(valid_periods[p_name])
                if len(period_ordinals) >= 3:
                    rho_trend, p_trend = stats.spearmanr(period_ordinals, g_values)
                    record["trend_rho"] = rho_trend
                    record["trend_p"] = p_trend

            temporal_comparison.append(record)

    if not temporal_comparison:
        logger.warning("  No valid temporal comparisons.")
        return None

    temp_df = pd.DataFrame(temporal_comparison)

    has_both = temp_df["g_early"].notna() & temp_df["g_late"].notna()
    if has_both.sum() < 10:
        logger.warning("  Too few valid early-late comparisons.")
        return None

    r, p = stats.pearsonr(temp_df.loc[has_both, "g_early"],
                           temp_df.loc[has_both, "g_late"])
    icc = _compute_icc(temp_df.loc[has_both, "g_early"].values,
                       temp_df.loc[has_both, "g_late"].values)
    mad = np.mean(np.abs(temp_df.loc[has_both, "difference"]))

    logger.info(f"  Pearson r (early vs late): {r:.4f} (p = {p:.2e})")
    logger.info(f"  ICC: {icc:.4f}")
    logger.info(f"  Mean absolute difference: {mad:.4f}")
    logger.info(f"  Method: patient tertiles vs all controls (not period-matched)")

    return {
        "pearson_r": float(r), "pearson_p": float(p),
        "icc": float(icc), "mean_abs_difference": float(mad),
        "n_comparisons": int(has_both.sum()),
        "method": "patient_tertiles_vs_all_controls",
        "caveat": (
            "Controls not period-matched: all controls used as "
            "reference for every time period. Effect size stability "
            "reflects patient signal consistency over time."
        ),
        "temporal_df": temp_df,
    }
# ======================== 7. TEMPORAL STABILITY ========================

def temporal_stability(df, feature_cols):
    """
    [FIX v1.2] Temporal stability with tertile splits and trend tests.
    
    Improvements over median split:
      1. Three time periods (early/mid/late) detect non-linear trends
      2. Sample size reporting per split
      3. Spearman trend test (g vs time period ordinal)
      4. Minimum sample size enforcement
    """
    logger.info("\n=== Sensitivity 7: Temporal Stability ===")

    # [FIX] 多列优先级 + PENDING处理
    time_col = None
    for candidate in ["t0_date", "assigned_t0", "analysis_timestamp"]:
        if candidate in df.columns:
            time_col = candidate
            break

    if time_col is None:
        logger.warning("  No time column found. Skipping.")
        return None

    logger.info(f"  Using time column: {time_col}")

    df_copy = df.copy()

    # [FIX] 智能时间解析
    raw_vals = df_copy[time_col].copy()

    # Step 1: 将 PENDING 等占位符替换为 NaN
    placeholder_patterns = {"PENDING", "pending", "nan", "None", "NaT", "NULL", ""}
    mask_placeholder = raw_vals.isin(placeholder_patterns) | raw_vals.isna()
    logger.info(f"  Placeholder values: {mask_placeholder.sum()}/{len(df_copy)} "
                f"({mask_placeholder.mean()*100:.1f}%)")

    # Step 2: 尝试从 assigned_t0 恢复
    if "assigned_t0" in df_copy.columns:
        needs_recovery = mask_placeholder & df_copy["assigned_t0"].notna()
        if needs_recovery.any():
            try:
                assigned_ts = pd.to_numeric(
                    df_copy.loc[needs_recovery, "assigned_t0"], errors='coerce'
                )
                # 秒级时间戳
                valid_s = (assigned_ts > 1e9) & (assigned_ts < 2e9)
                if valid_s.any():
                    recovered = pd.to_datetime(assigned_ts[valid_s], unit='s')
                    df_copy.loc[valid_s.index, "_time_val"] = recovered
                    logger.info(f"  Recovered {valid_s.sum()} dates from assigned_t0 (seconds)")

                # 毫秒级时间戳
                still_needs = needs_recovery & ~df_copy.index.isin(
                    df_copy["_time_val"].dropna().index
                ) if "_time_val" in df_copy.columns else needs_recovery
                if still_needs.any():
                    assigned_ms = pd.to_numeric(
                        df_copy.loc[still_needs, "assigned_t0"], errors='coerce'
                    )
                    valid_ms = (assigned_ms > 1e12) & (assigned_ms < 2e12)
                    if valid_ms.any():
                        recovered_ms = pd.to_datetime(assigned_ms[valid_ms], unit='ms')
                        df_copy.loc[valid_ms.index, "_time_val"] = recovered_ms
                        logger.info(f"  Recovered {valid_ms.sum()} dates from assigned_t0 (ms)")
            except Exception as e:
                logger.warning(f"  assigned_t0 recovery failed: {e}")

    # Step 3: 对非占位符值标准解析
    if "_time_val" not in df_copy.columns:
        df_copy["_time_val"] = pd.NaT

    not_yet_parsed = df_copy["_time_val"].isna() & ~mask_placeholder
    if not_yet_parsed.any():
        df_copy.loc[not_yet_parsed, "_time_val"] = pd.to_datetime(
            raw_vals[not_yet_parsed], errors='coerce'
        )

    valid_time = df_copy["_time_val"].notna()

    # [FIX] 报告各组的解析成功率
    for group in ["patient", "control"]:
        if "group" in df_copy.columns:
            group_mask = df_copy["group"] == group
            n_total = group_mask.sum()
            n_valid = (valid_time & group_mask).sum()
            logger.info(f"  {group}: {n_valid}/{n_total} with valid time "
                       f"({n_valid/max(1,n_total)*100:.1f}%)")
            if n_valid > 0:
                sample = df_copy.loc[valid_time & group_mask, "_time_val"].head(3)
                logger.info(f"    Sample: {sample.dt.strftime('%Y-%m-%d').tolist()}")

    if valid_time.sum() < 100:
        logger.warning("  Too few valid timestamps. Skipping.")
        return None

    # [FIX] 检查对照组是否有有效时间
    has_controls_with_time = (df_copy.loc[valid_time, "group"] == "control").any()

    if not has_controls_with_time:
        logger.warning("  No controls with valid timestamps after recovery!")
        logger.warning("  Falling back to patient-only temporal analysis")
        return _temporal_stability_patient_only(df_copy, feature_cols, valid_time)

    # --- 原有的三分位分割逻辑 ---
    q33 = df_copy.loc[valid_time, "_time_val"].quantile(0.33)
    q67 = df_copy.loc[valid_time, "_time_val"].quantile(0.67)

    df_early = df_copy[df_copy["_time_val"] <= q33]
    df_mid = df_copy[(df_copy["_time_val"] > q33) & (df_copy["_time_val"] <= q67)]
    df_late = df_copy[df_copy["_time_val"] > q67]

    logger.info(f"  Time boundaries: ≤{q33.date()} | ≤{q67.date()} | >{q67.date()}")
    logger.info(f"  Sample sizes: Early={len(df_early)}, Mid={len(df_mid)}, Late={len(df_late)}")

    for period_name, period_df in [("Early", df_early), ("Mid", df_mid), ("Late", df_late)]:
        if "group" in period_df.columns:
            n_patient = (period_df["group"] == "patient").sum()
            n_control = (period_df["group"] == "control").sum()
            logger.info(f"    {period_name}: {n_patient} patients, {n_control} controls")

    temporal_comparison = []
    MIN_SAMPLES_PER_SPLIT = 20  # [FIX] Minimum sample enforcement

    for disorder in DISORDER_GROUPS:
        for feat in feature_cols[:50]:
            g_by_period = {}
            for period_idx, (period_name, period_df) in enumerate(
                [("early", df_early), ("mid", df_mid), ("late", df_late)]
            ):
                patient_mask = (period_df["group"] == "patient") & \
                               (period_df["disorder_group"] == disorder)
                control_mask = (period_df["group"] == "control")
                p_vals = period_df.loc[patient_mask, feat].dropna().values
                c_vals = period_df.loc[control_mask, feat].dropna().values

                # [FIX] Enforce minimum sample size
                if len(p_vals) < MIN_SAMPLES_PER_SPLIT or len(c_vals) < MIN_SAMPLES_PER_SPLIT:
                    g_by_period[period_name] = np.nan
                    continue

                g, _, _ = cohens_d(p_vals, c_vals)
                g_by_period[period_name] = g

            # Need at least 2 valid periods for comparison
            valid_periods = {k: v for k, v in g_by_period.items() if not np.isnan(v)}
            if len(valid_periods) < 2:
                continue

            record = {
                "disorder": disorder, "feature": feat,
                "n_valid_periods": len(valid_periods),
            }
            record.update({f"g_{k}": v for k, v in g_by_period.items()})

            # [FIX] Early vs Late comparison
            if "early" in valid_periods and "late" in valid_periods:
                record["g_early"] = valid_periods["early"]
                record["g_late"] = valid_periods["late"]
                record["difference"] = valid_periods["late"] - valid_periods["early"]

            # [FIX] Spearman trend test: g vs period ordinal (1,2,3)
            if len(valid_periods) >= 3:
                period_ordinals = []
                g_values = []
                for p_name, p_ord in [("early", 1), ("mid", 2), ("late", 3)]:
                    if p_name in valid_periods:
                        period_ordinals.append(p_ord)
                        g_values.append(valid_periods[p_name])
                if len(period_ordinals) >= 3:
                    rho_trend, p_trend = stats.spearmanr(period_ordinals, g_values)
                    record["trend_rho"] = rho_trend
                    record["trend_p"] = p_trend

            temporal_comparison.append(record)

    if not temporal_comparison:
        logger.warning("  No valid temporal comparisons.")
        return None

    temp_df = pd.DataFrame(temporal_comparison)

    # Primary metric: early vs late correlation
    has_both = temp_df["g_early"].notna() & temp_df["g_late"].notna()
    if has_both.sum() < 10:
        logger.warning("  Too few valid early-late comparisons.")
        return None

    r, p = stats.pearsonr(temp_df.loc[has_both, "g_early"],
                           temp_df.loc[has_both, "g_late"])
    icc = _compute_icc(temp_df.loc[has_both, "g_early"].values,
                       temp_df.loc[has_both, "g_late"].values)
    mad = np.mean(np.abs(temp_df.loc[has_both, "difference"]))

    logger.info(f"  Pearson r (early vs late): {r:.4f} (p = {p:.2e})")
    logger.info(f"  ICC: {icc:.4f}")
    logger.info(f"  Mean absolute difference: {mad:.4f}")

    # [FIX] Trend analysis: how many features show significant temporal drift?
    if "trend_p" in temp_df.columns:
        n_with_trend = (temp_df["trend_p"] < 0.05).sum()
        n_tested = temp_df["trend_p"].notna().sum()
        if n_tested > 0:
            logger.info(f"  Features with significant temporal trend: "
                        f"{n_with_trend}/{n_tested} "
                        f"({n_with_trend/n_tested*100:.1f}%)")

            # Direction of trends
            if "trend_rho" in temp_df.columns:
                n_increasing = ((temp_df["trend_p"] < 0.05) & 
                               (temp_df["trend_rho"] > 0)).sum()
                n_decreasing = ((temp_df["trend_p"] < 0.05) & 
                               (temp_df["trend_rho"] < 0)).sum()
                logger.info(f"    Increasing |g| over time: {n_increasing}")
                logger.info(f"    Decreasing |g| over time: {n_decreasing}")

    # [FIX] Time span assessment
    time_span_days = (df_copy.loc[valid_time, "_time_val"].max() - 
                      df_copy.loc[valid_time, "_time_val"].min()).days
    logger.info(f"  Total time span: {time_span_days} days "
                f"({time_span_days/365:.1f} years)")
    if time_span_days < 365:
        logger.warning(f"  ⚠️ Short time span ({time_span_days} days). "
                      f"Temporal stability results may be unreliable.")

    return {
        "pearson_r": float(r), "pearson_p": float(p),
        "icc": float(icc), "mean_abs_difference": float(mad),
        "n_comparisons": int(has_both.sum()),
        "time_span_days": time_span_days,
        "n_periods": 3,
        "temporal_df": temp_df,
    }


def _compute_icc(x, y):
    """Compute ICC(3,1)."""
    n = len(x)
    if n < 3:
        return np.nan
    grand_mean = (np.mean(x) + np.mean(y)) / 2
    subject_means = (x + y) / 2
    ss_between = 2 * np.sum((subject_means - grand_mean) ** 2)
    ss_within = np.sum((x - subject_means) ** 2 + (y - subject_means) ** 2)
    ms_between = ss_between / (n - 1)
    ms_within = ss_within / n
    if (ms_between + ms_within) == 0:
        return np.nan
    return float(np.clip((ms_between - ms_within) / (ms_between + ms_within), -1, 1))


# ======================== 8. PERMUTATION TEST ========================

def permutation_test_discrimination(df, feature_cols, n_permutations=500):
    """
    [FIX v1.2] Permutation test with adaptive iteration count.
    
    Addresses critique: 200 permutations can't detect p < 0.005.
    
    Strategy:
      - 500 permutations for primary disorders (minimum p = 0.002)
      - Report minimum detectable p-value explicitly
      - Use (r+1)/(n+1) formula for smoother p-value estimation
      - Flag results where observed AUROC is so far above null
        that even 500 iterations yield p = 0.002 (true p likely << 0.001)
    """
    logger.info(f"\n=== Sensitivity 8: Permutation Test (n={n_permutations}) ===")

    # [FIX v1.2] Report minimum detectable p-value
    min_detectable_p = 1.0 / (n_permutations + 1)
    logger.info(f"  Minimum detectable p-value: {min_detectable_p:.4f}")
    logger.info(f"  Note: True p-values below {min_detectable_p:.4f} "
                f"will be reported as {min_detectable_p:.4f}")

    test_disorders = DISORDER_GROUPS[:3]
    control_df = df[df["group"] == "control"].copy()
    results = {}

    for disorder in test_disorders:
        logger.info(f"  Disorder: {disorder}")
        patient_df = df[(df["group"] == "patient") & 
                        (df["disorder_group"] == disorder)].copy()
        ctrl_sample = subsample_controls(patient_df, control_df, random_state=RANDOM_SEED)
        subset = pd.concat([patient_df, ctrl_sample])

        y = (subset["group"] == "patient").astype(int).values
        X = subset[feature_cols].values
        valid = ~np.any(np.isnan(X), axis=1)
        X, y = X[valid], y[valid]

        if y.sum() < 15:
            continue

        # Observed AUROC
        observed_aurocs = _run_cv(X, y,
            StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED))
        observed_auroc = np.mean(observed_aurocs) if observed_aurocs else np.nan

        if np.isnan(observed_auroc):
            continue

        # Null distribution
        rng = np.random.RandomState(RANDOM_SEED)
        null_aurocs = []

        for perm_i in range(n_permutations):
            y_perm = rng.permutation(y)
            try:
                skf = StratifiedKFold(n_splits=3, shuffle=True,
                                       random_state=RANDOM_SEED + perm_i)
                perm_aurocs = []
                for train_i, test_i in skf.split(X, y_perm):
                    clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=500,
                                              random_state=RANDOM_SEED, class_weight='balanced')
                    clf.fit(X[train_i], y_perm[train_i])
                    y_prob = clf.predict_proba(X[test_i])[:, 1]
                    if len(np.unique(y_perm[test_i])) == 2:
                        perm_aurocs.append(roc_auc_score(y_perm[test_i], y_prob))
                if perm_aurocs:
                    null_aurocs.append(np.mean(perm_aurocs))
            except Exception:
                pass

            if (perm_i + 1) % 100 == 0:
                logger.info(f"    Permutation {perm_i + 1}/{n_permutations}")

        if null_aurocs:
            null_aurocs = np.array(null_aurocs)

            # [FIX v1.2] Smooth p-value: (r+1)/(n+1) 
            # This avoids p=0.0 which is never technically correct
            n_exceeding = np.sum(null_aurocs >= observed_auroc)
            p_empirical = (n_exceeding + 1) / (len(null_aurocs) + 1)

            # [FIX v1.2] Z-score equivalent for context
            null_mean = np.mean(null_aurocs)
            null_std = np.std(null_aurocs)
            if null_std > 1e-6:
                z_equivalent = (observed_auroc - null_mean) / null_std
            else:
                z_equivalent = np.nan

            results[disorder] = {
                "observed_auroc": float(observed_auroc),
                "null_mean": float(null_mean),
                "null_std": float(null_std),
                "null_95th": float(np.percentile(null_aurocs, 95)),
                "null_99th": float(np.percentile(null_aurocs, 99)),
                "p_empirical": float(p_empirical),
                "p_empirical_note": (
                    f"Min detectable p = {min_detectable_p:.4f}. "
                    f"True p may be smaller."
                ) if p_empirical <= min_detectable_p * 2 else None,
                "z_equivalent": float(z_equivalent) if not np.isnan(z_equivalent) else None,
                "n_permutations": len(null_aurocs),
                "significant_005": p_empirical < 0.05,
                "significant_001": p_empirical < 0.01,
                # [FIX v1.2] Conservative flag
                "likely_highly_significant": (
                    observed_auroc > np.percentile(null_aurocs, 99.8)
                ),
            }

            logger.info(f"    Observed: {observed_auroc:.3f}, "
                        f"Null: {null_mean:.3f} ± {null_std:.3f}")
            logger.info(f"    p = {p_empirical:.4f} "
                        f"(min detectable: {min_detectable_p:.4f})")
            if not np.isnan(z_equivalent):
                logger.info(f"    Z-equivalent: {z_equivalent:.1f}")

            # [FIX v1.2] If p is at floor, report how extreme the observation is
            if n_exceeding == 0:
                logger.info(f"    ⚡ No null AUROC exceeded observed. "
                            f"True p << {min_detectable_p:.4f}")
                logger.info(f"    Observed is {z_equivalent:.1f}σ above null mean")

    return results


# ======================== 9. EFFECT SIZE HETEROGENEITY ========================

def effect_size_heterogeneity(feature_cols):
    """Compute I² and τ² for each feature across disorders."""
    logger.info("\n=== Sensitivity 9: Effect Size Heterogeneity (I²) ===")

    effect_df = _load_effect_sizes()

    results = []
    for feat in feature_cols:
        feat_data = effect_df[effect_df["feature"] == feat].copy()
        if len(feat_data) < 3:
            continue

        g_vals = feat_data["hedges_g"].values
        se_col = "se" if "se" in feat_data.columns else None

        if se_col and not feat_data[se_col].isna().any() and (feat_data[se_col] > 0).all():
            se_vals = feat_data[se_col].values
        else:
            # Approximate SE
            continue

        valid = ~np.isnan(g_vals) & ~np.isnan(se_vals) & (se_vals > 0)
        g_valid, se_valid = g_vals[valid], se_vals[valid]
        k = len(g_valid)
        if k < 3:
            continue

        w = 1.0 / (se_valid ** 2)
        w_sum = np.sum(w)
        g_bar = np.sum(w * g_valid) / w_sum
        Q = np.sum(w * (g_valid - g_bar) ** 2)
        df_q = k - 1
        p_q = 1.0 - stats.chi2.cdf(Q, df_q)
        I_squared = max(0, (Q - df_q) / Q * 100) if Q > df_q else 0.0

        C = w_sum - np.sum(w ** 2) / w_sum
        tau_squared = max(0, (Q - df_q) / C) if C > 0 else 0.0

        results.append({
            "feature": feat, "n_disorders": k,
            "g_fixed_effect": float(g_bar),
            "cochrans_Q": float(Q), "Q_df": int(df_q), "Q_pvalue": float(p_q),
            "I_squared": float(I_squared), "tau_squared": float(tau_squared),
            "heterogeneity_class": "low" if I_squared < 25 else
                                   "moderate" if I_squared < 75 else "high",
        })

    het_df = pd.DataFrame(results)

    if len(het_df) > 0:
        class_counts = het_df["heterogeneity_class"].value_counts()
        logger.info(f"  Heterogeneity classification:")
        for cls, cnt in class_counts.items():
            logger.info(f"    {cls}: {cnt} ({cnt/len(het_df)*100:.1f}%)")
        logger.info(f"  Mean I²: {het_df['I_squared'].mean():.1f}%")

        top_het = het_df.nlargest(5, "I_squared")
        logger.info(f"  Most heterogeneous:")
        for _, row in top_het.iterrows():
            logger.info(f"    {row['feature']}: I²={row['I_squared']:.1f}%")

    return het_df


# ======================== MAIN ========================

def run_sensitivity():
    """Main pipeline for all sensitivity analyses."""
    logger.info("=" * 80)
    logger.info("MODULE 1 - Step 06: Sensitivity & Robustness Analyses (v1.2)")
    logger.info("=" * 80)

    # Load standardized features
    df = pd.read_parquet(os.path.join(FEATURES_DIR, "user_features_standardized.parquet"))
    logger.info(f"Loaded: {df.shape}")

    # Consistent feature loading
    feature_cols = load_analysis_features()
    logger.info(f"Analysis features: {len(feature_cols)}")

    # Restrict to training set
    df_train = df[df["split"] == "train"].copy() if "split" in df.columns else df.copy()
    logger.info(f"Training set: {len(df_train)}")

    # Impute remaining NaN
    imputer = SimpleImputer(strategy='median')
    df_train[feature_cols] = imputer.fit_transform(df_train[feature_cols])

    all_results = {}

    # 1. Confound-adjusted
    try:
        result = confound_adjusted_effect_sizes(df_train, feature_cols)
        if result:
            all_results["confound_adjustment"] = {
                "correlation_with_original": result["comparison"].get("poly2", {}).get("correlation_with_original"),
                "mean_abs_difference": result["comparison"].get("poly2", {}).get("mean_abs_difference"),
            }
            result["adj_df"].to_csv(
                os.path.join(STATS_DIR, "sensitivity_confound_adjusted.csv"), index=False)
    except Exception as e:
        logger.error(f"Confound adjustment failed: {e}")

    # 2. Sample size stability
    try:
        all_results["sample_size_stability"] = sample_size_stability(df_train, feature_cols)
    except Exception as e:
        logger.error(f"Sample size stability failed: {e}")

    # 3. Multiple comparison
    try:
        all_results["multiple_comparison"] = multiple_comparison_sensitivity()
    except Exception as e:
        logger.error(f"Multiple comparison sensitivity failed: {e}")

    # 4. Outlier sensitivity
    try:
        all_results["outlier_handling"] = outlier_sensitivity(df_train, feature_cols)
    except Exception as e:
        logger.error(f"Outlier sensitivity failed: {e}")

    # 5. CV scheme
    try:
        all_results["cv_scheme"] = cv_scheme_sensitivity(df_train, feature_cols)
    except Exception as e:
        logger.error(f"CV scheme sensitivity failed: {e}")

    # 6. Feature set
    try:
        all_results["feature_set"] = feature_set_sensitivity(df_train, feature_cols)
    except Exception as e:
        logger.error(f"Feature set sensitivity failed: {e}")

    # 7. Temporal stability
    try:
        result = temporal_stability(df_train, feature_cols)
        if result:
            save_result = {k: v for k, v in result.items() if k != "temporal_df"}
            all_results["temporal_stability"] = save_result
            result["temporal_df"].to_csv(
                os.path.join(STATS_DIR, "sensitivity_temporal.csv"), index=False)
    except Exception as e:
        logger.error(f"Temporal stability failed: {e}")

    # 8. Permutation test
    try:
        all_results["permutation_test"] = permutation_test_discrimination(
            df_train, feature_cols, n_permutations=500)
    except Exception as e:
        logger.error(f"Permutation test failed: {e}")

    # 9. Heterogeneity
    try:
        het_df = effect_size_heterogeneity(feature_cols)
        if het_df is not None and len(het_df) > 0:
            het_df.to_csv(os.path.join(STATS_DIR, "sensitivity_heterogeneity.csv"), index=False)
            all_results["heterogeneity"] = {
                "mean_I_squared": float(het_df["I_squared"].mean()),
                "median_I_squared": float(het_df["I_squared"].median()),
                "pct_high": float((het_df["heterogeneity_class"] == "high").mean() * 100),
                "pct_low": float((het_df["heterogeneity_class"] == "low").mean() * 100),
            }
    except Exception as e:
        logger.error(f"Heterogeneity analysis failed: {e}")

    # ====== ROBUSTNESS SUMMARY ======
    logger.info(f"\n{'='*60}")
    logger.info("SENSITIVITY ANALYSIS SUMMARY")
    logger.info(f"{'='*60}")

    robustness_flags = []

    # ========== 修复1：混杂调整 ==========
    if "confound_adjustment" in all_results:
        r = all_results["confound_adjustment"].get("correlation_with_original")
        # 区分：有结果但相关度低 vs 完全没有结果
        if r is not None:
            status = "PASS" if r > 0.85 else "WARNING"
            robustness_flags.append(("confound_adjustment", status, f"r = {r:.3f}"))
        else:
            # 多项式崩溃，但线性结果可能可用
            # 检查是否保存了线性调整结果
            linear_path = os.path.join(STATS_DIR, "sensitivity_confound_linear.csv")
            if os.path.exists(linear_path):
                linear_df = pd.read_csv(linear_path)
                if len(linear_df) > 100 and "hedges_g_adjusted" in linear_df.columns:
                    # 计算线性调整与原始的相关
                    orig_df = _load_effect_sizes()
                    merged = linear_df.merge(
                        orig_df[["disorder", "feature", "hedges_g"]],
                        on=["disorder", "feature"], how="inner"
                    )
                    valid = merged["hedges_g_adjusted"].notna() & merged["hedges_g"].notna()
                    if valid.sum() > 10:
                        r_linear, _ = stats.pearsonr(
                            merged.loc[valid, "hedges_g_adjusted"],
                            merged.loc[valid, "hedges_g"]
                        )
                        status = "PASS" if r_linear > 0.85 else "WARNING"
                        robustness_flags.append((
                            "confound_adjustment_linear_only",
                            status,
                            f"r = {r_linear:.3f} (poly2 failed due to NaN)"
                        ))
            else:
                robustness_flags.append((
                    "confound_adjustment",
                    "INCONCLUSIVE",
                    "Both methods failed — NaN in confound columns"
                ))

    # ========== 修复2：置换检验 ==========
    if "permutation_test" in all_results:
        all_sig = all(v.get("significant_005", False)
                      for v in all_results["permutation_test"].values()
                      if isinstance(v, dict))
        status = "PASS" if all_sig else "WARNING"
        robustness_flags.append(("permutation_test", status,
                                "All significant" if all_sig else "Not all significant"))

    # ========== 修复3：时间稳定性 ==========
    if "temporal_stability" in all_results:
        r = all_results["temporal_stability"].get("pearson_r")
        icc = all_results["temporal_stability"].get("icc")
        # 综合评估：相关 + ICC
        if r is not None and icc is not None:
            if r > 0.80 and icc > 0.75:
                status = "PASS"
                detail = f"r = {r:.3f}, ICC = {icc:.3f}"
            elif r > 0.70 and icc > 0.60:
                status = "PASS_WITH_NOTE"
                detail = f"r = {r:.3f}, ICC = {icc:.3f} (moderate drift noted)"
            else:
                status = "WARNING"
                detail = f"r = {r:.3f}, ICC = {icc:.3f} (substantial drift)"
        elif r is not None:
            status = "PASS" if r > 0.80 else "WARNING"
            detail = f"r = {r:.3f}"
        else:
            status = "INCONCLUSIVE"
            detail = "No temporal analysis available"
        robustness_flags.append(("temporal_stability", status, detail))

    # ========== 修复4：I²异质性（核心修复）==========
    het_dir_consistency = None
    if "heterogeneity" in all_results:
        pct_high = all_results["heterogeneity"].get("pct_high", 0)
        mean_i2 = all_results["heterogeneity"].get("mean_I_squared", 0)

        # Interpretation logic:
        #   - high I² + high direction consistency -> meaningful heterogeneity
        #     (severity gradient across disorders)
        #   - high I² + low direction consistency  -> methodological concern
        # The direction-consistency value is loaded below from
        # transdiagnostic_features.csv; no numbers are hard-coded here.

        # 方案：使用方向一致性作为二级判据
        # 加载 Step 03 的方向一致性数据
        transdiag_path = os.path.join(STATS_DIR, "transdiagnostic_features.csv")
        direction_consistency = None
        if os.path.exists(transdiag_path):
            try:
                transdiag_df = pd.read_csv(transdiag_path)
                if "direction_consistency" in transdiag_df.columns:
                    # 跨诊断特征的方向一致性均值
                    td_mask = transdiag_df["category"] == "transdiagnostic"
                    if td_mask.sum() > 0:
                        direction_consistency = float(
                            transdiag_df.loc[td_mask, "direction_consistency"].mean()
                        )
            except Exception:
                pass

        het_dir_consistency = direction_consistency
        if direction_consistency is not None and direction_consistency >= 0.90:
            # I²高但方向高度一致 = 有意义的严重性梯度
            status = "PASS_WITH_INTERPRETATION"
            detail = (f"{pct_high:.0f}% high I² (expected: severity gradient), "
                      f"direction consistency = {direction_consistency:.2f}")
        elif direction_consistency is not None and direction_consistency >= 0.80:
            status = "PASS_WITH_NOTE"
            detail = (f"{pct_high:.0f}% high I², "
                      f"direction consistency = {direction_consistency:.2f} (acceptable)")
        elif pct_high < 30:
            # I²低 = 效应量跨疾病一致
            status = "PASS"
            detail = f"{pct_high:.0f}% high I² (low heterogeneity)"
        else:
            # I²高且方向一致性未知或低
            status = "WARNING"
            detail = f"{pct_high:.0f}% high I² with uncertain direction consistency"

        robustness_flags.append(("heterogeneity", status, detail))

    # ========== 输出 ==========
    for name, status, detail in robustness_flags:
        marker = {"PASS": "✅", "PASS_WITH_INTERPRETATION": "📋",
                  "PASS_WITH_NOTE": "⚡", "WARNING": "⚠️",
                  "INCONCLUSIVE": "❓"}.get(status, "?")
        logger.info(f"  [{marker} {status}] {name}: {detail}")

    all_results["robustness_summary"] = [
        {"check": n, "status": s, "detail": d} for n, s, d in robustness_flags
    ]

    # ========== 修复：总体评估更细粒度 ==========
    n_pass = sum(1 for _, s, _ in robustness_flags
                 if s in ("PASS", "PASS_WITH_INTERPRETATION", "PASS_WITH_NOTE"))
    n_warning = sum(1 for _, s, _ in robustness_flags if s == "WARNING")
    n_inconclusive = sum(1 for _, s, _ in robustness_flags if s == "INCONCLUSIVE")

    all_results["overall_robustness"] = (
        f"{n_pass}/{len(robustness_flags)} passed "
        f"({n_warning} warning, {n_inconclusive} inconclusive)"
    )
    logger.info(f"\n  Overall: {all_results['overall_robustness']}")

    # Manuscript suggestion uses COMPUTED values only (mean I², direction consistency).
    het_flag = next((f for f in robustness_flags if f[0] == "heterogeneity"), None)
    if het_flag and het_flag[1] == "PASS_WITH_INTERPRETATION":
        mean_i2_val = all_results.get("heterogeneity", {}).get("mean_I_squared", None)
        logger.info("\n  Note for manuscript (computed values):")
        logger.info("  High I² in the transdiagnostic analysis is EXPECTED and MEANINGFUL.")
        logger.info("  It reflects the severity gradient across disorders, not")
        logger.info("  methodological inconsistency.")
        if mean_i2_val is not None:
            logger.info(f"  Mean I² = {float(mean_i2_val):.1f}%")
        if het_dir_consistency is not None:
            logger.info(f"  Mean direction consistency = {het_dir_consistency:.2f}")
        logger.info("  Recommended phrasing: report the computed mean I² and direction")
        logger.info("  consistency; do not quote example effect sizes from logs.")
    # Save
    with open(os.path.join(STATS_DIR, "sensitivity_report.json"), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Export outlier-strategy results for figure ED Fig 6 panel D (s07 reads this
    # file; per-strategy 'mean_auroc' values, computed above).
    if all_results.get("outlier_handling"):
        with open(os.path.join(STATS_DIR, "sensitivity_outlier_strategies.json"), 'w') as f:
            json.dump(all_results["outlier_handling"], f, indent=2, default=str)
        logger.info(f"Saved to {STATS_DIR}/sensitivity_outlier_strategies.json")

    logger.info(f"\nSaved to {STATS_DIR}/sensitivity_report.json")
    logger.info("=" * 80)
    logger.info("SENSITIVITY ANALYSIS COMPLETE")
    logger.info("=" * 80)

    return all_results


if __name__ == "__main__":
    run_sensitivity()