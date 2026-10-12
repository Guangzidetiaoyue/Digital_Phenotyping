"""
Step 05: Transdiagnostic Factor Analysis (v1.0 → v1.2)

Core Analysis 3 of Module 1.
Identifies shared (transdiagnostic) vs. specific latent dimensions
using Exploratory Factor Analysis on the effect-size profiles.

v1.0 → v1.2 Changelog:
  [CRITICAL] Use effect_size_matrix_clean.csv (exclude artifacts)
  [CRITICAL] Fix FA matrix orientation: features as observations (98×10),
             not disorders as observations (10×98)
  [FIX] Cap max_factors at min(n_vars//2, 4) for 10-variable case
  [FIX] Feature loading consistent with s03/s04 (from feature_selection.json)
  [FIX] User-level PCA: train set only + subsample for t-SNE
  [FIX] Transdiagnostic classification: exclude artifact features
  [NEW] Varimax rotation for factor interpretability
  [NEW] Factor structure interpretation: map factors to HiTOP spectra
  [NEW] Disorder PCA (10×10 correlation) as complementary analysis
  [OPT] t-SNE subsampling: 50K max for feasible computation
"""

import os
import json
import logging
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform

from config_module1 import (
    FEATURES_DIR, STATS_DIR, FIGURES_DIR,
    DISORDER_GROUPS, DISORDER_DISPLAY_NAMES,
    HITOP_SPECTRA, RANDOM_SEED
)

os.makedirs(STATS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(STATS_DIR, "factor_analysis.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

np.random.seed(RANDOM_SEED)

# [CRITICAL v1.2] Same artifact set as s03/s04
SELECTION_ARTIFACT_FEATURES = {
    "mental_health_sub_ratio",
    "support_sub_ratio",
    "subreddit_entropy",
    "subreddit_herfindahl",
    "primary_subreddit_ratio",
}

# t-SNE subsampling limit
TSNE_MAX_SAMPLES = 50000


# ======================== FEATURE LOADING ========================

def load_analysis_features():
    """Load feature columns consistent with s02/s03/s04."""
    logger.info("Loading analysis features (consistent with s03/s04)...")

    feature_cols = []

    selection_path = os.path.join(FEATURES_DIR, "feature_selection.json")
    if os.path.exists(selection_path):
        with open(selection_path, 'r') as f:
            selection = json.load(f)
        valid_features = selection.get("valid_features", [])
        if isinstance(valid_features, list) and len(valid_features) > 0:
            feature_cols = [f for f in valid_features if isinstance(f, str)]

    if not feature_cols:
        scaler_path = os.path.join(FEATURES_DIR, "robust_scaler_params.json")
        if os.path.exists(scaler_path):
            with open(scaler_path, 'r') as f:
                scaler_params = json.load(f)
            valid_features = scaler_params.get("valid_features", [])
            if isinstance(valid_features, list) and len(valid_features) > 0:
                feature_cols = [f for f in valid_features if isinstance(f, str)]

    # Exclude artifacts
    feature_cols = [c for c in feature_cols if c not in SELECTION_ARTIFACT_FEATURES]
    logger.info(f"  Analysis features: {len(feature_cols)}")

    return feature_cols


# ======================== FACTOR ANALYSIS ========================

def determine_n_factors(data_matrix, max_factors=4):
    """
    [FIX v1.2] Determine optimal number of factors.
    
    For features-as-observations (98×10) matrix:
      - Maximum meaningful factors ≤ n_variables // 2
      - Cap at 4 for 10-variable case
      - Parallel analysis uses correct dimensions
    
    Criteria:
      1. Parallel Analysis (primary)
      2. Kaiser criterion (eigenvalue > 1)
      3. Scree inspection
    """
    logger.info("Determining optimal number of factors...")

    n_samples, n_vars = data_matrix.shape
    max_factors = min(max_factors, n_vars // 2, n_samples - 1)
    max_factors = max(2, max_factors)  # At least 2 for interpretation

    logger.info(f"  Matrix: {n_samples} observations × {n_vars} variables")
    logger.info(f"  Max factors to test: {max_factors}")

    # Correlation matrix of variables (disorders)
    corr_matrix = np.corrcoef(data_matrix.T)  # n_vars × n_vars
    eigenvalues, _ = np.linalg.eigh(corr_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]

    # 1. Kaiser criterion
    n_kaiser = int(np.sum(eigenvalues > 1.0))
    logger.info(f"  Kaiser criterion: {n_kaiser} factors")

    # 2. Parallel analysis
    n_iterations = 1000
    random_eigenvalues = np.zeros((n_iterations, n_vars))
    rng = np.random.RandomState(RANDOM_SEED)

    for i in range(n_iterations):
        # [FIX v1.2] Random data with same dimensions as actual data
        random_data = rng.standard_normal((n_samples, n_vars))
        random_corr = np.corrcoef(random_data.T)
        random_eig, _ = np.linalg.eigh(random_corr)
        random_eigenvalues[i] = np.sort(random_eig)[::-1]

    random_95th = np.percentile(random_eigenvalues, 95, axis=0)
    n_parallel = int(np.sum(eigenvalues > random_95th))
    n_parallel = max(1, n_parallel)  # At least 1
    logger.info(f"  Parallel analysis: {n_parallel} factors")

    # 3. Scree: find elbow via ratio of consecutive eigenvalue differences
    if len(eigenvalues) >= 3:
        diffs = np.diff(eigenvalues)
        ratios = np.abs(diffs[:-1]) / (np.abs(diffs[1:]) + 1e-10)
        n_scree = int(np.argmax(ratios < 1.0)) + 1  # First point where ratio < 1
        n_scree = max(1, min(n_scree, max_factors))
    else:
        n_scree = 2
    logger.info(f"  Scree heuristic: {n_scree} factors")

    # Consensus: parallel analysis, bounded by [2, max_factors]
    n_optimal = max(2, min(n_parallel, max_factors))
    logger.info(f"  Selected n_factors = {n_optimal}")

    factor_selection = {
        "eigenvalues": eigenvalues.tolist(),
        "random_95th_eigenvalues": random_95th.tolist(),
        "kaiser_n": n_kaiser,
        "parallel_n": n_parallel,
        "scree_n": n_scree,
        "selected_n": n_optimal,
        "n_samples": n_samples,
        "n_variables": n_vars,
    }

    return n_optimal, factor_selection


def varimax_rotation(loadings, gamma=1.0, q=20, tol=1e-6):
    """
    Varimax rotation for factor interpretability (Kaiser normalization).
    
    Args:
        loadings: (p, k) matrix — p variables × k factors
        gamma: 1.0 for varimax, 0.0 for quartimax
        q: max iterations
        tol: convergence tolerance
    
    Returns:
        rotated_loadings: (p, k)
        rotation_matrix: (k, k)
    """
    p, k = loadings.shape

    # Kaiser normalization: normalize rows before rotation
    row_norms = np.sqrt((loadings ** 2).sum(axis=1, keepdims=True))
    row_norms = np.where(row_norms < 1e-10, 1.0, row_norms)
    normalized = loadings / row_norms

    rotation_matrix = np.eye(k)
    d = 0

    for _ in range(q):
        Lambda = normalized @ rotation_matrix          # (p, k)

        # --- Varimax 梯度计算 ---
        # 目标函数: 最大化 Σ_j [ p Σ_i λ⁴_ij − (Σ_i λ²_ij)² ]
        # 对旋转矩阵 R 的梯度通过链式法则得到:
        #
        #   A = Λ³ − (γ/p) · Λ · diag(1ᵀΛ²)        形状: (p, k)
        #   G = Λᵀ · A                                形状: (k, k)  ← 关键修复点
        #
        # 原始代码写成了 A · Λᵀ 得到 (p, p), 导致 rotation_matrix 膨胀为 (p, p)

        Lambda2 = Lambda ** 2
        col_sums_Lambda2 = Lambda2.sum(axis=0)         # (k,)

        A = Lambda ** 3 - (gamma / p) * Lambda * col_sums_Lambda2[np.newaxis, :]  # (p, k)

        G = Lambda.T @ A                                # (k, p) @ (p, k) → (k, k)

        U_g, s_g, Vt_g = np.linalg.svd(G)
        rotation_matrix = U_g @ Vt_g                    # (k, k)

        d_new = np.sum(s_g)
        if d_new < d + tol:
            break
        d = d_new

    # 旋转 + 反归一化
    rotated = loadings @ rotation_matrix                # (p, k) @ (k, k) → (p, k)
    return rotated, rotation_matrix


def run_factor_analysis_on_effects(effect_matrix):
    """
    [FIX v1.2] Factor analysis on the effect-size profile matrix.
    
    Orientation: features as observations, disorders as variables.
      X = effect_matrix.values  →  98 × 10  (features × disorders)
    
    Each factor represents a latent dimension in "disorder space":
      - Factor 1: general psychopathology severity (all disorders same direction)
      - Factor 2: neurodevelopmental vs emotional (autism/ADHD vs rest)
    
    This is the CORRECT orientation because:
      1. n_observation (98) > n_variables (10) — FA is valid
      2. Factor loadings on features tell us which features define each dimension
      3. Factor scores on disorders tell us each disorder's position in latent space
    """
    logger.info("\n--- Factor Analysis on Effect-Size Profiles ---")

    # [FIX v1.2] Features as observations, disorders as variables
    X = effect_matrix.values  # 98 × 10
    feature_names = effect_matrix.index.tolist()
    disorder_names = effect_matrix.columns.tolist()

    logger.info(f"  Matrix shape: {X.shape} (features × disorders)")
    logger.info(f"  Features: {len(feature_names)}, Disorders: {len(disorder_names)}")

    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)

    # Determine number of factors.
    # The data-driven selection (parallel analysis) is COMPUTED for transparency,
    # but the retained number of factors is an explicit ANALYST/THEORY choice
    # (HiTOP 3-layer hierarchy), not a data-driven result.
    try:
        n_data_driven, _sel = determine_n_factors(X, max_factors=4)
    except Exception as e:
        n_data_driven = None
        logger.warning(f"  Parallel analysis failed ({e}); data-driven n unavailable")
    n_factors = 3
    logger.info(f"  Analyst-selected n_factors = {n_factors} (HiTOP theory); "
                f"data-driven parallel analysis suggested n = {n_data_driven}")
    factor_selection = {
        "selected_n": n_factors,
        "data_driven_n": n_data_driven,
        "selection_method": "analyst/theory (HiTOP hierarchy), not data-driven",
        "rationale": (
            "n_factors=3 retained as an explicit analyst/theory choice for the HiTOP "
            "3-layer hierarchy (p-factor + Internalizing + Neurodevelopmental). "
            "The data-driven parallel-analysis value is reported in 'data_driven_n'. "
            "Variance explained by the retained components is computed from the "
            "effect-size matrix (see 'variance_explained_selected')."
        ),
    }
    
    # 运行因子分析
    fa = FactorAnalysis(n_components=n_factors, random_state=RANDOM_SEED)
    fa.fit(X)

    # ========== 修改2：正确应用Varimax旋转 ==========
    # 获取未旋转的载荷和得分
    feature_scores = fa.transform(X)           # (90, 3) — 特征的因子得分
    disorder_loadings = fa.components_.T       # (10, 3) — 疾病的因子载荷

    # 关键：对疾病载荷矩阵做Varimax旋转
    # 目标：让每个因子只被少数疾病高载荷定义 → 因子可解释
    rotated_disorder_loadings, rotation_matrix = varimax_rotation(disorder_loadings)

    # 特征得分跟着旋转
    rotated_feature_scores = feature_scores @ rotation_matrix

    # ========== 创建DataFrame ==========
    loading_df = pd.DataFrame(
        rotated_feature_scores,               # 特征在旋转后因子空间的得分
        index=feature_names,
        columns=[f"Factor_{i+1}" for i in range(n_factors)]
    )

    score_df = pd.DataFrame(
        rotated_disorder_loadings,            # 疾病在旋转后因子上的载荷
        index=disorder_names,
        columns=[f"Factor_{i+1}" for i in range(n_factors)]
    )

    # ========== 以下代码不变 ==========
    # Variance explained (approximate via PCA)
    pca_for_variance = PCA(n_components=n_factors, random_state=RANDOM_SEED)
    pca_for_variance.fit(X)
    variance_explained = pca_for_variance.explained_variance_ratio_
    # Record the ACTUAL variance explained by the retained components (computed).
    factor_selection["variance_explained_selected"] = float(np.sum(variance_explained))
    factor_selection["cumulative_variance_selected"] = float(np.cumsum(variance_explained)[-1])

    # Factor interpretation
    factor_interpretation = {}
    for i in range(n_factors):
        col = f"Factor_{i+1}"
        
        # 特征得分排名 (定义该因子的核心特征)
        sorted_scores = loading_df[col].abs().sort_values(ascending=False)
        top_features = sorted_scores.head(10).index.tolist()
        top_values = loading_df.loc[top_features, col].tolist()
        n_positive = sum(1 for v in top_values if v > 0)
        n_negative = sum(1 for v in top_values if v < 0)

        # 疾病载荷排名 (定义该因子的疾病分布)
        sorted_disorder = score_df[col].abs().sort_values(ascending=False)
        top_disorders = sorted_disorder.head(5).index.tolist()
        top_disorder_values = score_df.loc[top_disorders, col].tolist()

        factor_interpretation[col] = {
            "variance_explained": float(variance_explained[i]),
            "top_features": [
                {"feature": f, "loading": float(v)}
                for f, v in zip(top_features, top_values)
            ],
            "top_disorders": [
                {"disorder": d, "loading": float(v)}
                for d, v in zip(top_disorders, top_disorder_values)
            ],
            "direction_split": f"{n_positive}+/{n_negative}-",
            "disorder_scores": {
                d: float(score_df.loc[d, col]) for d in disorder_names
            },
        }

    # Log
    logger.info(f"\n  Factor Analysis Results (n_factors={n_factors}):")
    logger.info(f"  Variance explained: {variance_explained.round(3)}")
    logger.info(f"  Cumulative: {np.cumsum(variance_explained).round(3)}")

    for col, interp in factor_interpretation.items():
        logger.info(f"\n  {col} (variance={interp['variance_explained']:.3f}):")
        logger.info(f"    Top features:")
        for item in interp["top_features"][:5]:
            logger.info(f"      {item['feature']}: {item['loading']:+.3f}")
        logger.info(f"    Top disorders:")
        for item in interp["top_disorders"]:
            logger.info(f"      {item['disorder']}: {item['loading']:+.3f}")

    return {
        "loading_df": loading_df,
        "score_df": score_df,
        "n_factors": n_factors,
        "factor_selection": factor_selection,
        "variance_explained": variance_explained.tolist(),
        "cumulative_variance": np.cumsum(variance_explained).tolist(),
        "factor_interpretation": factor_interpretation,
        "rotation_matrix": rotation_matrix,
    }


# ======================== DISORDER CORRELATION & CLUSTERING ========================

def disorder_clustering(effect_matrix):
    """
    Hierarchical clustering of disorders based on effect-size profiles.
    """
    logger.info("\n--- Disorder Clustering Analysis ---")

    # Features as observations, disorders as variables
    X = effect_matrix.values  # 98 × 10
    X = np.nan_to_num(X, nan=0.0)
    disorder_names = effect_matrix.columns.tolist()

    # [FIX v1.2] Compute disorder correlation from feature profiles
    # Each disorder has a 98-dimensional profile
    # Correlation between disorder profiles = similarity
    disorder_profiles = X.T  # 10 × 98
    corr_matrix = np.corrcoef(disorder_profiles)  # 10 × 10

    dist_matrix = 1 - corr_matrix
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = np.clip(dist_matrix, 0, 2)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2  # Ensure symmetry

    # Hierarchical clustering
    condensed_dist = squareform(dist_matrix)
    linkage_matrix = linkage(condensed_dist, method='average')

    # Optimal clusters via silhouette
    best_k = 2
    best_sil = -1
    for k in range(2, min(len(disorder_names), 6)):
        labels = fcluster(linkage_matrix, t=k, criterion='maxclust')
        if len(set(labels)) > 1:
            sil = silhouette_score(dist_matrix, labels, metric='precomputed')
            if sil > best_sil:
                best_sil = sil
                best_k = k

    cluster_labels = fcluster(linkage_matrix, t=best_k, criterion='maxclust')
    cluster_assignment = dict(zip(disorder_names, cluster_labels.tolist()))

    logger.info(f"  Optimal clusters: {best_k} (silhouette = {best_sil:.3f})")
    logger.info(f"  Cluster assignments: {cluster_assignment}")

    # Compare to HiTOP
    hitop_labels = {}
    for d in disorder_names:
        for spectrum, disorders in HITOP_SPECTRA.items():
            if d in disorders:
                hitop_labels[d] = spectrum
                break

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.preprocessing import LabelEncoder

    ari = np.nan
    nmi = np.nan
    if len(hitop_labels) == len(disorder_names):
        hitop_encoded = [hitop_labels.get(d, "unknown") for d in disorder_names]
        le = LabelEncoder()
        hitop_numeric = le.fit_transform(hitop_encoded)
        ari = adjusted_rand_score(hitop_numeric, cluster_labels)
        nmi = normalized_mutual_info_score(hitop_numeric, cluster_labels)
        logger.info(f"  ARI (data-driven vs HiTOP): {ari:.3f}")
        logger.info(f"  NMI (data-driven vs HiTOP): {nmi:.3f}")

    corr_df = pd.DataFrame(corr_matrix, index=disorder_names, columns=disorder_names)

    # MDS embedding
    mds = MDS(n_components=2, dissimilarity='precomputed',
              random_state=RANDOM_SEED, normalized_stress='auto')
    embedding_2d = mds.fit_transform(dist_matrix)
    embedding_df = pd.DataFrame(embedding_2d, index=disorder_names,
                                columns=["MDS_1", "MDS_2"])

    # [NEW v1.2] Disorder PCA (on correlation matrix)
    disorder_pca = PCA(n_components=min(5, len(disorder_names) - 1))
    disorder_pca.fit(disorder_profiles)
    disorder_pca_scores = disorder_pca.transform(disorder_profiles)
    disorder_pca_df = pd.DataFrame(
        disorder_pca_scores,
        index=disorder_names,
        columns=[f"DPC{i+1}" for i in range(disorder_pca_scores.shape[1])]
    )

    logger.info(f"  Disorder PCA variance: "
                f"{disorder_pca.explained_variance_ratio_.round(3)}")

    return {
        "linkage_matrix": linkage_matrix,
        "cluster_labels": cluster_assignment,
        "optimal_k": best_k,
        "silhouette": best_sil,
        "correlation_matrix": corr_df,
        "distance_matrix": pd.DataFrame(dist_matrix, index=disorder_names,
                                        columns=disorder_names),
        "mds_embedding": embedding_df,
        "disorder_pca": disorder_pca_df,
        "disorder_pca_variance": disorder_pca.explained_variance_ratio_.tolist(),
        "ari_vs_hitop": ari,
        "nmi_vs_hitop": nmi,
        "disorder_names": disorder_names,
    }


# ======================== USER-LEVEL PCA ========================

def user_level_pca(df, feature_cols):
    """
    [FIX v1.2] User-level PCA with:
      - Training set only (prevent data leakage)
      - Subsampled t-SNE (feasible computation)
    """
    logger.info("\n--- User-Level PCA ---")

    # [FIX v1.2] Use training set only
    df_train = df[df["split"] == "train"].copy()
    logger.info(f"  Using training set: {len(df_train)} users")

    X = df_train[feature_cols].values
    valid = ~np.any(np.isnan(X), axis=1)
    X = X[valid]
    labels = df_train.loc[valid, "disorder_group"].values
    groups = df_train.loc[valid, "group"].values

    # PCA
    pca = PCA(n_components=10, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X)

    logger.info(f"  Variance explained (top 10 PCs): "
                f"{pca.explained_variance_ratio_.round(3)}")
    logger.info(f"  Cumulative: {np.cumsum(pca.explained_variance_ratio_).round(3)}")

    # Component loadings
    loading_df = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=[f"PC{i+1}" for i in range(10)]
    )

    # Embedding dataframe (for visualization)
    embed_df = pd.DataFrame(X_pca[:, :5],
                            columns=[f"PC{i+1}" for i in range(5)])
    embed_df["disorder_group"] = labels
    embed_df["group"] = groups

    # [FIX v1.2] Subsample for t-SNE
    n_tsne = min(TSNE_MAX_SAMPLES, len(X_pca))
    logger.info(f"  Running t-SNE on {n_tsne} subsampled users...")

    rng = np.random.RandomState(RANDOM_SEED)
    tsne_idx = rng.choice(len(X_pca), size=n_tsne, replace=False)

    tsne = TSNE(
        n_components=2,
        perplexity=min(30, n_tsne // 5),
        random_state=RANDOM_SEED,
        max_iter=1000,
        init="pca",
        learning_rate="auto"
    )
    X_tsne = tsne.fit_transform(X_pca[tsne_idx, :10])

    tsne_df = pd.DataFrame(X_tsne, columns=["tSNE_1", "tSNE_2"])
    tsne_df["disorder_group"] = labels[tsne_idx]
    tsne_df["group"] = groups[tsne_idx]

    logger.info(f"  t-SNE complete: {tsne_df.shape}")

    return {
        "pca_variance_explained": pca.explained_variance_ratio_.tolist(),
        "pca_cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "pca_loadings": loading_df,
        "pca_embedding": embed_df,
        "tsne_embedding": tsne_df,
        "tsne_n_samples": n_tsne,
    }


# ======================== TRANSDIAGNOSTIC vs SPECIFIC ========================

def identify_transdiagnostic_features(effect_matrix, sig_matrix,
                                       threshold_g=0.2, min_disorders=5):
    """
    [FIX v1.2] Classify features as transdiagnostic/specific/mixed.
    
    Excludes artifact features (already removed from clean matrix,
    but double-checks).
    """
    logger.info("\n--- Transdiagnostic vs Specific Feature Classification ---")

    n_disorders = effect_matrix.shape[1]
    feature_names = effect_matrix.index.tolist()
    disorder_names = effect_matrix.columns.tolist()

    # [FIX v1.2] Exclude any remaining artifact features
    artifact_present = [f for f in feature_names if f in SELECTION_ARTIFACT_FEATURES]
    if artifact_present:
        logger.warning(f"  Found {len(artifact_present)} artifact features in matrix, "
                      f"excluding from classification")
        feature_names = [f for f in feature_names if f not in SELECTION_ARTIFACT_FEATURES]

    results = []

    for feat in feature_names:
        g_values = effect_matrix.loc[feat].values
        sig_values = sig_matrix.loc[feat].values if feat in sig_matrix.index \
            else np.zeros(n_disorders, dtype=bool)

        # Count disorders where |g| > threshold AND statistically significant
        significant_large = np.array([
            (abs(g) >= threshold_g) and (s == True)
            for g, s in zip(g_values, sig_values)
        ])
        n_sig = int(np.sum(significant_large))

        # Direction consistency
        sig_g = g_values[significant_large]
        if len(sig_g) > 1:
            n_positive = np.sum(sig_g > 0)
            n_negative = np.sum(sig_g < 0)
            direction_consistency = max(n_positive, n_negative) / len(sig_g)
        else:
            direction_consistency = 1.0

        mean_abs_g_sig = float(np.mean(np.abs(sig_g))) if len(sig_g) > 0 else 0.0
        mean_abs_g_all = float(np.mean(np.abs(g_values[~np.isnan(g_values)])))

        # Classify
        if n_sig == 0:
            category = "non_significant"
        elif n_sig <= 2:
            category = "specific"
        elif n_sig < min_disorders:
            category = "mixed"
        else:
            category = "transdiagnostic"

        specific_to = [d for d, sl in zip(disorder_names, significant_large) if sl]

        results.append({
            "feature": feat,
            "n_sig_disorders": n_sig,
            "category": category,
            "direction_consistency": direction_consistency,
            "mean_abs_g_sig": mean_abs_g_sig,
            "mean_abs_g_all": mean_abs_g_all,
            "max_abs_g": float(np.nanmax(np.abs(g_values))),
            "specific_to": ";".join(specific_to),
        })

    results_df = pd.DataFrame(results)

    # Summary
    category_counts = results_df["category"].value_counts()
    logger.info(f"  Feature categories:")
    for cat, count in category_counts.items():
        logger.info(f"    {cat}: {count}")

    # Top transdiagnostic
    trans_features = results_df[results_df["category"] == "transdiagnostic"].sort_values(
        "mean_abs_g_all", ascending=False
    )
    logger.info(f"\n  Top transdiagnostic features:")
    for _, row in trans_features.head(10).iterrows():
        logger.info(f"    {row['feature']}: n_disorders={row['n_sig_disorders']}, "
                    f"mean |g|={row['mean_abs_g_all']:.3f}, "
                    f"dir_consistency={row['direction_consistency']:.2f}")

    # Top specific
    spec_features = results_df[results_df["category"] == "specific"].sort_values(
        "max_abs_g", ascending=False
    )
    if len(spec_features) > 0:
        logger.info(f"\n  Top specific features:")
        for _, row in spec_features.head(10).iterrows():
            logger.info(f"    {row['feature']}: specific to {row['specific_to']}, "
                        f"max |g|={row['max_abs_g']:.3f}")

    return results_df


# ======================== DISORDER PROFILES ========================

def build_disorder_profiles(effect_matrix, feature_meta_mapped, sig_matrix):
    """
    For each disorder, create a digital phenotype profile.
    """
    logger.info("\n--- Building Disorder Digital Phenotype Profiles ---")

    profiles = {}
    disorder_names = effect_matrix.columns.tolist()
    feature_names = effect_matrix.index.tolist()

    # Reverse mapping: feature → dimension
    feat_to_dim = {}
    for dim, cols in feature_meta_mapped.items():
        for c in cols:
            feat_to_dim[c] = dim

    for disorder in disorder_names:
        g_vals = effect_matrix[disorder].dropna()
        sorted_feats = g_vals.abs().sort_values(ascending=False)

        top_features = []
        for feat_name in sorted_feats.index[:20]:
            g = float(g_vals[feat_name])
            is_sig = bool(sig_matrix.loc[feat_name, disorder]) \
                if feat_name in sig_matrix.index else False
            dim = feat_to_dim.get(feat_name, "unknown")

            direction = "elevated" if g > 0 else ("reduced" if g < 0 else "neutral")

            top_features.append({
                "feature": feat_name,
                "hedges_g": g,
                "abs_g": abs(g),
                "direction": direction,
                "significant": is_sig,
                "dimension": dim,
            })

        # Per-dimension summary
        dim_summary = {}
        for dim in feature_meta_mapped.keys():
            dim_feats = [f for f in feature_names if feat_to_dim.get(f) == dim]
            dim_g = g_vals.reindex(dim_feats).dropna()
            if len(dim_g) > 0:
                dim_summary[dim] = {
                    "mean_g": float(dim_g.mean()),
                    "mean_abs_g": float(dim_g.abs().mean()),
                    "max_abs_g": float(dim_g.abs().max()),
                    "n_features": len(dim_g),
                    "top_feature": str(dim_g.abs().idxmax()),
                    "top_feature_g": float(dim_g[dim_g.abs().idxmax()]),
                }

        profiles[disorder] = {
            "top_features": top_features,
            "dimension_summary": dim_summary,
        }

        display_name = DISORDER_DISPLAY_NAMES.get(disorder, disorder)
        logger.info(f"\n  {display_name}:")
        for tf in top_features[:5]:
            sig_marker = "***" if tf["significant"] else ""
            logger.info(f"    {tf['feature']:40s} g={tf['hedges_g']:+.3f} "
                        f"({tf['direction']}) [{tf['dimension']}] {sig_marker}")

    return profiles


# ======================== MAIN ========================

def run_factor_analysis():
    """Main pipeline for factor analysis."""
    logger.info("=" * 80)
    logger.info("MODULE 1 - Step 05: Transdiagnostic Factor Analysis (v1.2)")
    logger.info("Fixes: clean matrix, FA orientation, varimax rotation,")
    logger.info("       consistent features, train-only PCA, subsampled t-SNE")
    logger.info("=" * 80)

    # [CRITICAL v1.2] Use clean effect size matrix (no artifacts)
    clean_path = os.path.join(STATS_DIR, "effect_size_matrix_clean.csv")
    full_path = os.path.join(STATS_DIR, "effect_size_matrix.csv")

    if os.path.exists(clean_path):
        effect_matrix = pd.read_csv(clean_path, index_col=0)
        logger.info(f"Using CLEAN effect size matrix: {effect_matrix.shape}")
    else:
        logger.warning("Clean matrix not found, falling back to full matrix. "
                      "Artifact features may dominate results!")
        effect_matrix = pd.read_csv(full_path, index_col=0)
        # Remove artifacts manually
        artifact_cols = [f for f in effect_matrix.index if f in SELECTION_ARTIFACT_FEATURES]
        if artifact_cols:
            effect_matrix = effect_matrix.drop(index=artifact_cols)
            logger.info(f"  Removed {len(artifact_cols)} artifact features manually")

    sig_matrix_path = os.path.join(STATS_DIR, "significance_matrix_clean.csv")
    if not os.path.exists(sig_matrix_path):
        sig_matrix_path = os.path.join(STATS_DIR, "significance_matrix.csv")
    sig_matrix = pd.read_csv(sig_matrix_path, index_col=0)
    sig_matrix = sig_matrix.applymap(lambda x: True if x == True or x == "True" 
                                    else (False if x == False or x == "False" else x))

    logger.info(f"Effect size matrix: {effect_matrix.shape}")
    logger.info(f"Significance matrix: {sig_matrix.shape}")

    # Load consistent features
    feature_cols = load_analysis_features()

    # Load feature metadata with log mapping (reuse s04 logic)
    feature_meta_mapped = _load_feature_meta_with_log_mapping()

    # Load user features
    df = pd.read_parquet(os.path.join(FEATURES_DIR, "user_features_standardized.parquet"))
    logger.info(f"Loaded user features: {df.shape}")

    # ====== Analysis 1: Factor Analysis on Effect Sizes ======
    fa_results = run_factor_analysis_on_effects(effect_matrix)

    # ====== Analysis 2: Disorder Clustering ======
    cluster_results = disorder_clustering(effect_matrix)

    # ====== Analysis 3: User-Level PCA ======
    pca_results = user_level_pca(df, feature_cols)

    # ====== Analysis 4: Transdiagnostic vs Specific ======
    transdiag_df = identify_transdiagnostic_features(
        effect_matrix, sig_matrix,
        threshold_g=0.2, min_disorders=5
    )

    # ====== Analysis 5: Disorder Profiles ======
    profiles = build_disorder_profiles(effect_matrix, feature_meta_mapped, sig_matrix)

    # ====== SAVE ALL RESULTS ======
    logger.info("\nSaving factor analysis results...")

    # Factor analysis
    fa_results["loading_df"].to_csv(
        os.path.join(STATS_DIR, "factor_loadings_features.csv"))
    fa_results["score_df"].to_csv(
        os.path.join(STATS_DIR, "factor_scores_disorders.csv"))

    fa_selection = fa_results["factor_selection"]
    with open(os.path.join(STATS_DIR, "factor_selection.json"), 'w') as f:
        json.dump({
            "n_factors": fa_results["n_factors"],
            "variance_explained": fa_results["variance_explained"],
            "cumulative_variance": fa_results["cumulative_variance"],
            "selection_criteria": fa_selection,
            "factor_interpretation": fa_results["factor_interpretation"],
        }, f, indent=2, default=str)

    # Clustering
    cluster_results["correlation_matrix"].to_csv(
        os.path.join(STATS_DIR, "disorder_correlation_matrix.csv"))
    cluster_results["distance_matrix"].to_csv(
        os.path.join(STATS_DIR, "disorder_distance_matrix.csv"))
    cluster_results["mds_embedding"].to_csv(
        os.path.join(STATS_DIR, "disorder_mds_embedding.csv"))
    cluster_results["disorder_pca"].to_csv(
        os.path.join(STATS_DIR, "disorder_pca_scores.csv"))

    cluster_summary = {
        "optimal_k": cluster_results["optimal_k"],
        "silhouette": cluster_results["silhouette"],
        "cluster_labels": cluster_results["cluster_labels"],
        "ari_vs_hitop": cluster_results["ari_vs_hitop"],
        "nmi_vs_hitop": cluster_results["nmi_vs_hitop"],
        "disorder_pca_variance": cluster_results["disorder_pca_variance"],
    }
    with open(os.path.join(STATS_DIR, "clustering_summary.json"), 'w') as f:
        json.dump(cluster_summary, f, indent=2, default=str)

    np.save(os.path.join(STATS_DIR, "linkage_matrix.npy"),
            cluster_results["linkage_matrix"])

    # PCA
    pca_results["pca_loadings"].to_csv(
        os.path.join(STATS_DIR, "pca_loadings.csv"))
    pca_results["pca_embedding"].to_csv(
        os.path.join(STATS_DIR, "pca_embedding.csv"), index=False)
    pca_results["tsne_embedding"].to_csv(
        os.path.join(STATS_DIR, "tsne_embedding.csv"), index=False)

    with open(os.path.join(STATS_DIR, "pca_summary.json"), 'w') as f:
        json.dump({
            "variance_explained": pca_results["pca_variance_explained"],
            "cumulative_variance": pca_results["pca_cumulative_variance"],
            "tsne_n_samples": pca_results["tsne_n_samples"],
        }, f, indent=2)

    # Transdiagnostic
    transdiag_df.to_csv(
        os.path.join(STATS_DIR, "transdiagnostic_features.csv"), index=False)

    # Profiles
    with open(os.path.join(STATS_DIR, "disorder_profiles.json"), 'w') as f:
        json.dump(profiles, f, indent=2, default=str)

    # Comprehensive report
    report = {
        "pipeline_version": "v1.2",
        "effect_matrix_source": "clean" if os.path.exists(clean_path) else "full_manual_filter",
        "n_features_analyzed": len(effect_matrix),
        "n_disorders": len(effect_matrix.columns),
        "factor_analysis": {
            "n_factors": fa_results["n_factors"],
            "variance_explained": fa_results["variance_explained"],
            "cumulative_variance": fa_results["cumulative_variance"],
            "orientation": "features_as_observations_98x10",
            "rotation": "varimax",
        },
        "clustering": cluster_summary,
        "transdiagnostic_summary": {
            "n_transdiagnostic": int((transdiag_df["category"] == "transdiagnostic").sum()),
            "n_specific": int((transdiag_df["category"] == "specific").sum()),
            "n_mixed": int((transdiag_df["category"] == "mixed").sum()),
            "n_non_significant": int((transdiag_df["category"] == "non_significant").sum()),
            "top_transdiagnostic": transdiag_df[
                transdiag_df["category"] == "transdiagnostic"
            ].nlargest(5, "mean_abs_g_all")[
                ["feature", "n_sig_disorders", "mean_abs_g_all", "direction_consistency"]
            ].to_dict("records"),
        },
        "pca": {
            "variance_top5": pca_results["pca_variance_explained"][:5],
            "cumulative_top5": pca_results["pca_cumulative_variance"][:5],
            "tsne_samples": pca_results["tsne_n_samples"],
        },
    }

    with open(os.path.join(STATS_DIR, "factor_analysis_report.json"), 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("FACTOR ANALYSIS SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  Features analyzed: {len(effect_matrix)} (artifacts excluded)")
    logger.info(f"  Factors extracted: {fa_results['n_factors']}")
    logger.info(f"  Variance explained: "
                f"{np.cumsum(fa_results['variance_explained'])[-1]:.1%}")
    logger.info(f"  Optimal clusters: {cluster_results['optimal_k']} "
                f"(silhouette={cluster_results['silhouette']:.3f})")
    logger.info(f"  Transdiagnostic features: "
                f"{(transdiag_df['category']=='transdiagnostic').sum()}")
    logger.info(f"  Specific features: "
                f"{(transdiag_df['category']=='specific').sum()}")
    if not np.isnan(cluster_results['ari_vs_hitop']):
        logger.info(f"  Data-driven vs HiTOP: ARI={cluster_results['ari_vs_hitop']:.3f}, "
                    f"NMI={cluster_results['nmi_vs_hitop']:.3f}")
    logger.info(f"{'='*80}")
    logger.info("FACTOR ANALYSIS COMPLETE")
    logger.info(f"{'='*80}")

    return fa_results, cluster_results, pca_results, transdiag_df, profiles


def _load_feature_meta_with_log_mapping():
    """Load feature metadata with log features mapped to base dimensions."""
    meta_path = os.path.join(FEATURES_DIR, "feature_metadata.json")
    if not os.path.exists(meta_path):
        return {}

    with open(meta_path, 'r') as f:
        raw_meta = json.load(f)

    feature_to_dim = {}
    log_features_set = set()
    signed_log_features_set = set()

    for dim, cols in raw_meta.items():
        if not isinstance(cols, list):
            continue
        if dim == 'log_transforms':
            log_features_set = set(cols)
            continue
        if dim == 'signed_log_transforms':
            signed_log_features_set = set(cols)
            continue
        if dim in ('quality',):
            continue
        for col in cols:
            feature_to_dim[col] = dim

    for log_feat in log_features_set:
        base_feat = log_feat.replace("_log", "")
        if base_feat in feature_to_dim:
            feature_to_dim[log_feat] = feature_to_dim[base_feat]
        else:
            feature_to_dim[log_feat] = 'temporal'

    for signed_feat in signed_log_features_set:
        base_feat = signed_feat.replace("_log_signed", "")
        if base_feat in feature_to_dim:
            feature_to_dim[signed_feat] = feature_to_dim[base_feat]
        else:
            if 'score' in base_feat or 'comment' in base_feat:
                feature_to_dim[signed_feat] = 'social'
            else:
                feature_to_dim[signed_feat] = 'temporal'

    dim_to_features = defaultdict(list)
    for feat, dim in feature_to_dim.items():
        dim_to_features[dim].append(feat)

    return dict(dim_to_features)


if __name__ == "__main__":
    run_factor_analysis()