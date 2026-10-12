"""
Module 3 - Analysis 2: Digital Phenotype Space Geometry
========================================================
RQ2: Do the 10 disorders form a structure in digital phenotype space that is
better described by dimensional (HiTOP) spectra than by discrete DSM-5 classes?

Methods:
  A2.1 User-level UMAP projection (subsampled) -> "phenotype universe" figure.
  A2.2 Disorder centroids in UMAP space; test HiTOP spectrum separation:
       within-spectrum pairs closer than between-spectrum pairs.
  A2.3 Structural fit: correlate digital pairwise distance matrix with
       (a) HiTOP spectrum co-membership, (b) DSM-5 chapter co-membership,
       (c) clinical comorbidity. Compare fit to decide dimensional vs categorical.

Scientific rigor:
  - Subsampling per disorder (n=1500) + controls (n=3000) for tractability.
  - UMAP on standardized (RobustScaler) features; fixed seed.
  - Distance matrix on user-level centroids (more stable than single UMAP coords).
  - Permutation test for within- vs between-spectrum separation.
  - Procrustes/Mantel-style correlation of structural matrices.
"""

import os
import json
import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import umap
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = "/path/to/project"
FEATURES_PATH = os.path.join(PROJECT_ROOT, "module1_results", "features", "user_features_standardized.parquet")
OUT_DIR = os.path.join(PROJECT_ROOT, "module3_results", "statistics")
FIG_DIR = os.path.join(PROJECT_ROOT, "module3_results", "figures")
RANDOM_SEED = 42
N_PER_DISORDER = 1500
N_CONTROL = 3000

# HiTOP spectra (from config_module1)
HITOP_SPECTRA = {
    "Internalizing": ["depression", "anxiety", "eating_disorders"],
    "Thought Disorder": ["schizophrenia", "bipolar"],
    "Externalizing": ["adhd"],
    "Detachment/Personality": ["bpd", "ptsd", "ocd"],
    "Neurodevelopmental": ["autism"],
}
DISORDER_TO_SPECTRUM = {d: s for s, ds in HITOP_SPECTRA.items() for d in ds}

# DSM-5 chapters (categorical structure for comparison)
# Anxiety/OCD chapter: anxiety, ocd, ptsd | Depressive: depression, bipolar
# Neurodevelopmental: adhd, autism | Schizophrenia: schizophrenia
# Feeding/Eating: eating_disorders | Personality (B cluster): bpd
DSM5_CHAPTERS = {
    "Depressive/Bipolar": ["depression", "bipolar"],
    "Anxiety/OCD/Trauma": ["anxiety", "ocd", "ptsd"],
    "Schizophrenia": ["schizophrenia"],
    "Neurodevelopmental": ["adhd", "autism"],
    "Feeding/Eating": ["eating_disorders"],
    "Personality": ["bpd"],
}
DISORDER_TO_DSM5 = {d: c for c, ds in DSM5_CHAPTERS.items() for d in ds}

# Reuse clinical comorbidity from A1
CLINICAL_COMORBIDITY = {
    ("depression", "anxiety"): 45.0, ("depression", "bipolar"): 20.0,
    ("depression", "ptsd"): 28.0, ("depression", "ocd"): 18.0,
    ("depression", "adhd"): 25.0, ("depression", "bpd"): 30.0,
    ("depression", "eating_disorders"): 24.0, ("depression", "schizophrenia"): 15.0,
    ("depression", "autism"): 20.0, ("anxiety", "bipolar"): 25.0,
    ("anxiety", "ptsd"): 35.0, ("anxiety", "ocd"): 22.0,
    ("anxiety", "adhd"): 30.0, ("anxiety", "bpd"): 32.0,
    ("anxiety", "eating_disorders"): 28.0, ("anxiety", "schizophrenia"): 12.0,
    ("anxiety", "autism"): 18.0, ("bipolar", "ptsd"): 15.0,
    ("bipolar", "ocd"): 12.0, ("bipolar", "adhd"): 20.0,
    ("bipolar", "bpd"): 18.0, ("bipolar", "eating_disorders"): 12.0,
    ("bipolar", "schizophrenia"): 10.0, ("bipolar", "autism"): 12.0,
    ("ptsd", "ocd"): 15.0, ("ptsd", "adhd"): 20.0,
    ("ptsd", "bpd"): 25.0, ("ptsd", "eating_disorders"): 22.0,
    ("ptsd", "schizophrenia"): 10.0, ("ptsd", "autism"): 15.0,
    ("ocd", "adhd"): 18.0, ("ocd", "bpd"): 20.0,
    ("ocd", "eating_disorders"): 15.0, ("ocd", "schizophrenia"): 8.0,
    ("ocd", "autism"): 15.0, ("adhd", "bpd"): 22.0,
    ("adhd", "eating_disorders"): 18.0, ("adhd", "schizophrenia"): 10.0,
    ("adhd", "autism"): 30.0, ("bpd", "eating_disorders"): 25.0,
    ("bpd", "schizophrenia"): 12.0, ("bpd", "autism"): 15.0,
    ("eating_disorders", "schizophrenia"): 10.0, ("eating_disorders", "autism"): 15.0,
    ("schizophrenia", "autism"): 12.0,
}

EXCLUDE_SUBSTR = [
    "user_id_hash", "group", "disorder_group", "is_retrospective", "t0_date",
    "analysis_timestamp", "quality_", "pct_liwc", "pct_nrc", "pct_regex",
    "primary_subreddit", "ps_score", "iptw", "eb_weight", "strat_key",
    "split", "is_temporal_structurally_imputed", "self_post_ratio",
]


def load_and_prepare():
    df = pd.read_parquet(FEATURES_PATH)
    feature_cols = [
        c for c in df.columns
        if df[c].dtype in (np.float64, np.float32, np.int64, np.int32, np.bool_)
        and not any(s in c for s in EXCLUDE_SUBSTR)
        and c not in ("user_id_hash", "group", "disorder_group")
    ]
    all_nan = [c for c in feature_cols if df[c].isna().all()]
    feature_cols = [c for c in feature_cols if c not in all_nan]
    df = df.dropna(subset=feature_cols).copy()

    # Subsample for tractability
    patients = df[df["group"] == "patient"].copy()
    patients = patients[patients["disorder_group"] != "unknown"]
    sampled = []
    for d in sorted(patients["disorder_group"].unique()):
        sub = patients[patients["disorder_group"] == d]
        n = min(N_PER_DISORDER, len(sub))
        sampled.append(sub.sample(n=n, random_state=RANDOM_SEED))
    # Controls
    controls = df[df["group"] == "control"]
    sampled.append(controls.sample(n=min(N_CONTROL, len(controls)), random_state=RANDOM_SEED))

    sub_df = pd.concat(sampled)
    logger.info(f"Subsampled: {len(sub_df)} users")
    return sub_df, feature_cols


def build_centroids(emb, labels):
    """Disorder centroids in embedding space (user-level mean)."""
    centroids = {}
    for d in sorted(set(labels) - {None}):
        mask = labels == d
        centroids[d] = emb[mask].mean(axis=0)
    return centroids


def permutation_test(vec_a, vec_b, n_perm=10000):
    """Permutation test: is |mean difference| of vec_a vs vec_b significant?"""
    obs = np.abs(vec_a.mean() - vec_b.mean())
    pooled = np.concatenate([vec_a, vec_b])
    rng = np.random.RandomState(RANDOM_SEED)
    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(pooled))
        a = pooled[:len(vec_a)]
        b = pooled[len(vec_a):]
        if np.abs(a.mean() - b.mean()) >= obs:
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)


def mantel_corr(mat_a, mat_b, n_perm=10000):
    """Correlate upper-triangle distances; permutation p-value."""
    tri_a = mat_a[np.triu_indices_from(mat_a, k=1)]
    tri_b = mat_b[np.triu_indices_from(mat_b, k=1)]
    valid = ~(np.isnan(tri_a) | np.isnan(tri_b))
    tri_a, tri_b = tri_a[valid], tri_b[valid]
    if len(tri_a) < 3:
        return np.nan, np.nan
    r_obs = np.corrcoef(tri_a, tri_b)[0, 1]
    rng = np.random.RandomState(RANDOM_SEED)
    cnt = 0
    for _ in range(n_perm):
        r_perm = np.corrcoef(tri_a, rng.permutation(tri_b))[0, 1]
        if abs(r_perm) >= abs(r_obs):
            cnt += 1
    return float(r_obs), (cnt + 1) / (n_perm + 1)


def main():
    logger.info("Loading and preparing data...")
    sub_df, feature_cols = load_and_prepare()
    logger.info(f"Features: {len(feature_cols)}")

    # Robust scaling
    X = sub_df[feature_cols].values
    X = RobustScaler().fit_transform(X)

    # PCA pre-reduction (UMAP on 108 dims is slow; 50 dims is standard practice)
    pca = PCA(n_components=50, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X)
    logger.info(f"PCA: {len(feature_cols)} -> 50 (explained {pca.explained_variance_ratio_.sum():.2%})")

    # UMAP projection
    logger.info("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2,
                        random_state=RANDOM_SEED, n_jobs=-1)
    emb = reducer.fit_transform(X_pca)
    logger.info("UMAP done.")

    # Labels: disorder for patients, 'control' for controls
    labels = np.where(sub_df["group"] == "patient", sub_df["disorder_group"].values, "control")
    sub_df["umap_x"] = emb[:, 0]
    sub_df["umap_y"] = emb[:, 1]
    sub_df["phenotype_label"] = labels

    # Save embedding
    sub_df[["user_id_hash", "group", "disorder_group", "umap_x", "umap_y", "phenotype_label"]].to_csv(
        os.path.join(OUT_DIR, "phenotype_universe_embedding.csv"), index=False)
    logger.info("Saved UMAP embedding.")

    # --- A2.2 Spectrum separation (patient centroids) ---
    disorders = sorted(DISORDER_TO_SPECTRUM.keys())
    centroids = {d: emb[labels == d].mean(axis=0) for d in disorders}

    # Pairwise distances between centroids (in UMAP space)
    dist_mat = np.zeros((len(disorders), len(disorders)))
    for i, a in enumerate(disorders):
        for j, b in enumerate(disorders):
            dist_mat[i, j] = np.linalg.norm(centroids[a] - centroids[b])

    # Within- vs between-spectrum distances
    within = []
    between = []
    for a, b in combinations(disorders, 2):
        d = dist_mat[disorders.index(a), disorders.index(b)]
        if DISORDER_TO_SPECTRUM[a] == DISORDER_TO_SPECTRUM[b]:
            within.append(d)
        else:
            between.append(d)
    within = np.array(within)
    between = np.array(between)
    obs_diff, p_val = permutation_test(within, between)
    logger.info(f"\n=== Spectrum separation (UMAP centroid distances) ===")
    logger.info(f"  within-spectrum mean dist: {within.mean():.3f} (n={len(within)})")
    logger.info(f"  between-spectrum mean dist: {between.mean():.3f} (n={len(between)})")
    logger.info(f"  observed diff: {obs_diff:.3f}, permutation p={p_val:.4f}")

    # --- A2.3 Structural fit comparison ---
    # Distance matrix from centroids (digital space)
    # HiTOP co-membership matrix (1 if same spectrum)
    hitop_mat = np.zeros((len(disorders), len(disorders)))
    dsm_mat = np.zeros((len(disorders), len(disorders)))
    comorb_mat = np.zeros((len(disorders), len(disorders)))
    for i, a in enumerate(disorders):
        for j, b in enumerate(disorders):
            if i == j:
                hitop_mat[i, j] = dsm_mat[i, j] = 1
                continue
            hitop_mat[i, j] = 1 if DISORDER_TO_SPECTRUM[a] == DISORDER_TO_SPECTRUM[b] else 0
            dsm_mat[i, j] = 1 if DISORDER_TO_DSM5[a] == DISORDER_TO_DSM5[b] else 0
            pair = tuple(sorted([a, b]))
            comorb_mat[i, j] = CLINICAL_COMORBIDITY.get(pair, np.nan)

    # Correlation of digital distance with each structure
    # Note: lower digital distance + higher co-membership -> negative corr for HiTOP/DSM
    r_hitop, p_hitop = mantel_corr(dist_mat, hitop_mat)
    r_dsm, p_dsm = mantel_corr(dist_mat, dsm_mat)
    r_comorb, p_comorb = mantel_corr(dist_mat, comorb_mat)

    logger.info(f"\n=== Structural fit (digital distance vs taxonomy) ===")
    logger.info(f"  HiTOP co-membership: r={r_hitop:.3f}, p={p_hitop:.4f}")
    logger.info(f"  DSM-5 chapter:      r={r_dsm:.3f}, p={p_dsm:.4f}")
    logger.info(f"  Clinical comorbidity: r={r_comorb:.3f}, p={p_comorb:.4f}")

    # Also user-level HiTOP separation via raw features (not just UMAP centroids)
    # For each patient, distance to own-spectrum centroid vs other-spectrum centroid
    own_d = []
    other_d = []
    for d in disorders:
        mask = labels == d
        Xd = emb[mask]
        sp = DISORDER_TO_SPECTRUM[d]
        other_centroids = [centroids[o] for o in disorders
                           if DISORDER_TO_SPECTRUM[o] != sp]
        other_centroid = np.mean(other_centroids, axis=0)
        own_centroid = centroids[d]
        for u in Xd:
            own_d.append(np.linalg.norm(u - own_centroid))
            other_d.append(np.linalg.norm(u - other_centroid))
    own_d = np.array(own_d)
    other_d = np.array(other_d)
    ratio = (own_d / (other_d + 1e-9))
    logger.info(f"\n=== User-level separation ===")
    logger.info(f"  mean dist to own centroid: {own_d.mean():.3f}")
    logger.info(f"  mean dist to other-spectrum centroid: {other_d.mean():.3f}")
    logger.info(f"  own/other ratio: {ratio.mean():.3f} (<1 means own spectrum closer)")

    result = {
        "n_users": len(sub_df),
        "n_features": len(feature_cols),
        "pca_components": 50,
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "spectrum_separation": {
            "within_mean": float(within.mean()),
            "between_mean": float(between.mean()),
            "perm_p": float(p_val),
        },
        "structural_fit": {
            "hitop_r": r_hitop, "hitop_p": p_hitop,
            "dsm5_r": r_dsm, "dsm5_p": p_dsm,
            "comorbidity_r": r_comorb, "comorbidity_p": p_comorb,
        },
        "user_separation": {
            "own_mean": float(own_d.mean()),
            "other_mean": float(other_d.mean()),
            "ratio": float(ratio.mean()),
        },
    }
    with open(os.path.join(OUT_DIR, "phenotype_geometry_report.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("DONE")


if __name__ == "__main__":
    main()
