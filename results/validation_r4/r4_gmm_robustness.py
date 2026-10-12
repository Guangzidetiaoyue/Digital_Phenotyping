"""
P1-GMM: Trajectory-subtype GMM robustness (k / init / scaling)
==============================================================
Re-runs GMM model selection on patient trajectory shape features with
variations, to confirm the reported k=2 choice is stable:
  - k = 1..5 (full covariance), BIC + silhouette
  - n_init / random seeds
  - feature scaling variants (RobustScaler default vs no-scaling)
  - bootstrap ARI for the chosen k (aligned labels)
Uses a comparable shape-feature set as s09 (baseline/post level, step,
slopes, volatility, AUC, pre-t0 change-point count) computed per patient from
reliable windows of user_trajectories.parquet.

Outputs (NAS): r4_gmm_robustness_results.json
"""
import os
import json
import logging
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

TRAJ = "/path/to/project/module2_results/trajectories/user_trajectories.parquet"
CP = "/path/to/project/module2_results/statistics/changepoint_analysis.csv"
OUT = "/path/to/project/module4_results/validation_r4"
READ = ["user_id_hash", "group", "disorder_group", "window_index",
        "days_from_t0_center", "is_reliable", "p_factor_severity_z"]
RANDOM_SEED = 2025


def main():
    pf = pq.ParquetFile(TRAJ)
    chunks = []
    for b in pf.iter_batches(batch_size=500_000, columns=READ):
        d = b.to_pandas()
        d = d[(d["group"] == "patient") & (d["is_reliable"] == True)]
        if len(d):
            chunks.append(d)
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    logger.info(f"patient reliable rows: {len(df)}")

    # pre-t0 change-point count (feature used by s09)
    pre_cp = {}
    if os.path.exists(CP):
        c = pd.read_csv(CP, usecols=["user_id_hash", "group", "is_pre_t0"])
        c = c[(c["group"] == "patient") & (c["is_pre_t0"] == True)]
        pre_cp = c.groupby("user_id_hash").size().to_dict()
        logger.info(f"users with pre-t0 cp: {len(pre_cp)}")

    recs = []
    for uid, g in df.groupby("user_id_hash"):
        g = g.sort_values("days_from_t0_center")
        s = g["p_factor_severity_z"].values
        days = g["days_from_t0_center"].values
        ok = ~np.isnan(s)
        s, days = s[ok], days[ok]
        if len(s) < 6:
            continue
        base = s[(days > -360) & (days < -270)]
        post = s[(days > 30) & (days < 150)]
        pre = s[days < -15]
        post2 = s[days > 15]
        rec = {
            "user_id_hash": uid,
            "n_pre_t0_changepoints": pre_cp.get(uid, 0),
            "baseline": float(np.mean(base)) if len(base) else np.nan,
            "post_level": float(np.mean(post)) if len(post) else np.nan,
            "pre_mean": float(np.mean(pre)) if len(pre) else np.nan,
            "post_mean": float(np.mean(post2)) if len(post2) else np.nan,
            "volatility": float(np.std(s, ddof=1)) if len(s) > 1 else np.nan,
            "auc": float(np.trapz(s, days)),
            "n_reliable": int(len(s)),
        }
        # pre/post slopes (OLS on reliable windows)
        if len(pre) >= 3:
            b = np.polyfit(days[days < -15], s[days < -15], 1)[0]
            rec["pre_slope"] = float(b)
        if len(post2) >= 3:
            b = np.polyfit(days[days > 15], s[days > 15], 1)[0]
            rec["post_slope"] = float(b)
        recs.append(rec)

    sh = pd.DataFrame(recs).dropna(subset=["baseline", "post_level"])
    logger.info(f"patients with shape features: {len(sh)}")

    feats = [c for c in sh.columns if c not in ("user_id_hash",)]
    feats = [c for c in feats if c not in ("auc", "n_reliable")]  # AUC scale-dependent; keep main shape set
    feats = [c for c in feats if sh[c].notna().mean() > 0.5]
    sh = sh.dropna(subset=feats).reset_index(drop=True)
    X = sh[feats].values
    Xs = RobustScaler().fit_transform(X)
    logger.info(f"shape matrix {X.shape}, features {feats}")

    res = {}
    # k selection on scaled features
    krows = []
    for k in range(1, 6):
        try:
            gmm = GaussianMixture(n_components=k, covariance_type="full",
                                  random_state=RANDOM_SEED, n_init=10, max_iter=300,
                                  reg_covar=1e-6)
            gmm.fit(Xs)
            lab = gmm.predict(Xs)
            sil = float(silhouette_score(Xs, lab)) if k > 1 else np.nan
            krows.append({"k": k, "bic": float(gmm.bic(Xs)),
                          "silhouette": round(sil, 4)})
            logger.info(f"k={k}: BIC={gmm.bic(Xs):.0f}, sil={sil:.3f}")
        except Exception as e:
            logger.warning(f"k={k} failed: {e}")
    res["k_selection_scaled"] = krows

    # init robustness for k=2 (a few seeds)
    seeds = [1, 7, 42, 2025]
    labels2 = {}
    for sd in seeds:
        gmm = GaussianMixture(n_components=2, covariance_type="full",
                              random_state=sd, n_init=10, max_iter=300, reg_covar=1e-6)
        gmm.fit(Xs)
        labels2[sd] = gmm.predict(Xs)
    ari_init = []
    s0 = labels2[RANDOM_SEED]
    for sd in seeds[1:]:
        ari_init.append(float(adjusted_rand_score(s0, labels2[sd])))
    res["k2_init_seed_ari"] = {"seeds": seeds, "ari_vs_2025": ari_init}

    # scaling robustness: no-scaling on raw
    kraw = []
    for k in range(1, 5):
        try:
            gmm = GaussianMixture(n_components=k, covariance_type="full",
                                  random_state=RANDOM_SEED, n_init=10, max_iter=300,
                                  reg_covar=1e-6)
            gmm.fit(X)
            lab = gmm.predict(X)
            sil = float(silhouette_score(X, lab)) if k > 1 else np.nan
            kraw.append({"k": k, "bic": float(gmm.bic(X)), "silhouette": round(sil, 4)})
        except Exception as e:
            kraw.append({"k": k, "error": str(e)})
    res["k_selection_no_scaling"] = kraw

    # bootstrap ARI for k=2 (scaled) - 30 iters to save time
    rng = np.random.RandomState(RANDOM_SEED)
    aris = []
    n_boot = 30
    for i in range(n_boot):
        idx = rng.choice(len(Xs), len(Xs), replace=True)
        gmm = GaussianMixture(n_components=2, covariance_type="diag",
                              random_state=RANDOM_SEED + i, n_init=3, max_iter=200,
                              reg_covar=1e-4)
        gmm.fit(Xs[idx])
        lb = gmm.predict(Xs)
        ari = adjusted_rand_score(labels2[RANDOM_SEED], lb)
        aris.append(float(ari))
    res["k2_bootstrap_ari"] = {"n": n_boot, "mean": float(np.mean(aris)),
                               "std": float(np.std(aris)), "min": float(np.min(aris))}

    # disorder composition of k=2 (scaled, seed 2025) vs prior
    sh2 = sh.copy()
    sh2["subtype"] = labels2[RANDOM_SEED]
    comp = sh2.groupby(["disorder_group", "subtype"]).size().unstack(fill_value=0)
    comp = comp.div(comp.sum(axis=1), axis=0).round(3)
    res["k2_disorder_composition"] = comp.to_dict()

    with open(os.path.join(OUT, "r4_gmm_robustness_results.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)
    logger.info("DONE -> %s", OUT)


if __name__ == "__main__":
    main()
