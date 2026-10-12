"""
s31b: Empirical (same-cohort) comorbidity robustness for the Mantel test
=======================================================================
The primary Mantel test (s31) uses an UNCITED hand-entered clinical
comorbidity matrix; that is exploratory until sourced externally
(Plana-Ripoll et al. 2019, Lancet Psychiatry; Kessler NCS-R/NESARC).

This script adds an INDEPENDENT, same-cohort comorbidity matrix derived from
the multi-diagnosis parser (s34) output (user_multi_disorder.parquet:
128,003 users with has_<disorder> flags), and re-runs the Mantel test against
the SAME pairwise AUROC matrix. Two co-occurrence measures are reported:
  - Jaccard  : n_both / n_either
  - phi      : Pearson phi coefficient of the two binary indicators

Caveat: phenotype classifiers and comorbidity flags share the same cohort, so
this is INTERNAL CONSISTENCY, not external convergence; report as robustness
secondary to a properly sourced external matrix.

Outputs (NAS): empirical_comorbidity_mantel.json + comorbidity_empirical_jaccard.csv
"""
import os
import json
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

STATS = "/path/to/project/module3_results/statistics"
REPORT = os.path.join(STATS, "pairwise_distinguishability_report.json")
USERS = os.path.join(STATS, "user_multi_disorder.parquet")
RANDOM_SEED = 2025


def mantel(mat_a, mat_b, n_perm=10000, seed=RANDOM_SEED):
    """Pearson correlation of upper triangles + permutation p (symmetric)."""
    n = mat_a.shape[0]
    iu = np.triu_indices(n, k=1)
    a = mat_a[iu]
    b = mat_b[iu]
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    r = float(np.corrcoef(a, b)[0, 1])
    from scipy import stats
    tau, p_param = stats.pearsonr(a, b)
    rng = np.random.RandomState(seed)
    cnt = 0
    for _ in range(n_perm):
        bp = rng.permutation(b)
        if abs(np.corrcoef(a, bp)[0, 1]) >= abs(r):
            cnt += 1
    p_perm = (cnt + 1) / (n_perm + 1)
    return r, float(p_param), float(p_perm), int(ok.sum())


def main():
    with open(REPORT) as f:
        rep = json.load(f)
    disorders = rep["disorders"]
    auroc = np.array(rep["auroc_matrix"], dtype=float)
    dist = 1.0 - auroc
    np.fill_diagonal(dist, 0.0)

    u = pd.read_parquet(USERS)
    has = {d: u[f"has_{d}"].values.astype(bool) for d in disorders}

    n = len(disorders)
    jac = np.full((n, n), np.nan)
    phi = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                jac[i, j] = 1.0
                phi[i, j] = 1.0
                continue
            a, b = has[disorders[i]], has[disorders[j]]
            inter = np.sum(a & b)
            union = np.sum(a | b)
            jac[i, j] = inter / union if union else np.nan
            phi[i, j] = np.corrcoef(a, b)[0, 1]
    # Mantel: higher comorbidity -> lower separability -> expect NEGATIVE r
    # between AUROC-distance and comorbidity similarity
    r_j, p_j, pp_j, n_j = mantel(dist, jac)
    r_p, p_p, pp_p, n_p = mantel(dist, phi)

    # externally sourced comorbidity matrix (single source) recomputed for side-by-side
    ext = None
    try:
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(__file__))
        from config_module3 import load_comorbidity_matrix
        cm_mat, cm = load_comorbidity_matrix(disorders)
        r_l, p_l, pp_l, n_l = mantel(dist, cm_mat)
        ext = {"mantel_r": r_l, "pearson_p": p_l, "perm_p": pp_l, "n_pairs": n_l,
               "measure": cm.measure, "refs": cm.refs}
    except Exception as e:
        ext = {"error": str(e)}

    out = {
        "disorders": disorders,
        "empirical_jaccard": {"mantel_r": r_j, "pearson_p": p_j, "perm_p": pp_j, "n_pairs": n_j},
        "empirical_phi": {"mantel_r": r_p, "pearson_p": p_p, "perm_p": pp_p, "n_pairs": n_p},
        "external_sourced_comorbidity": ext,
        "note": ("Empirical matrices share cohort with phenotype classifiers -> "
                 "internal consistency, not external convergence. Primary Mantel "
                 "uses the externally sourced comorbidity matrix."),
    }
    pd.DataFrame(jac, index=disorders, columns=disorders).to_csv(
        os.path.join(STATS, "comorbidity_empirical_jaccard.csv"))
    with open(os.path.join(STATS, "empirical_comorbidity_mantel.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
