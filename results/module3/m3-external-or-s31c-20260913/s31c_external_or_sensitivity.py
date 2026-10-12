"""s31c: sensitivity of the Mantel convergence test to the external OR matrix.

Reuses the deterministic AUROC matrix produced by s31 (RANDOM_SEED=42) and tests:
  - raw adjusted-OR external matrix (reproduces s31's Pearson Mantel -> cross-check)
  - log(OR) external matrix
  - rank(OR) external matrix (Spearman-equivalent)
  - the UNCITED in-code % matrix (reference only)
Reports Pearson Mantel r, permutation p (joint row/col, seed 42), Spearman r/p, n pairs.
NaN pairs (not externally sourced) are dropped inside the Mantel test.
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy import stats

STATS = "/path/to/project/module3_results/statistics"
sys.path.insert(0, "/path/to/code/Module3")
import s31_pairwise_distinguishability as s31  # reuse mantel_test + CLINICAL_COMORBIDITY

N_PERM = 10000
SEED = 42

rep = json.load(open(os.path.join(STATS, "pairwise_distinguishability_report.json")))
disorders = rep["disorders"]
idx = {d: i for i, d in enumerate(disorders)}
auroc = np.load(os.path.join(STATS, "auroc_matrix.npy"))
ext = pd.read_csv(os.path.join(STATS, "external_comorbidity.csv"))

def build(transform):
    M = np.full((len(disorders), len(disorders)), np.nan)
    for _, r in ext.iterrows():
        a, b = r["disorder_a"], r["disorder_b"]
        if a not in idx or b not in idx:
            print("  WARN unknown disorder in external csv:", a, b); continue
        v = float(r["comorbidity_pct"])
        if transform == "log":
            v = float(np.log(v))
        M[idx[a], idx[b]] = v
        M[idx[b], idx[a]] = v
    return M

def tri(M):
    return M[np.triu_indices_from(M, k=1)]

def mantel(A, B, n_perm=N_PERM):
    ta, tb = tri(A), tri(B)
    valid = ~(np.isnan(ta) | np.isnan(tb))
    a, b = ta[valid], tb[valid]
    if len(a) < 3:
        return np.nan, np.nan, int(len(a))
    r_obs = np.corrcoef(a, b)[0, 1]
    rng = np.random.RandomState(SEED)
    c = 0
    for _ in range(n_perm):
        p = rng.permutation(B.shape[0])
        tb_p = tri(B[np.ix_(p, p)])
        v = ~(np.isnan(ta) | np.isnan(tb_p))
        rp = np.corrcoef(ta[v], tb_p[v])[0, 1]
        if abs(rp) >= abs(r_obs):
            c += 1
    return float(r_obs), float((c + 1) / (n_perm + 1)), int(len(a))

results = {"disorders": disorders, "n_external_pairs_in_csv": int(len(ext))}
for name, tr in [("raw_OR", "raw"), ("log_OR", "log")]:
    M = build(tr)
    r, p, n = mantel(auroc, M)
    ta, tb = tri(auroc), tri(M)
    val = ~(np.isnan(ta) | np.isnan(tb))
    rs, ps = stats.spearmanr(ta[val], tb[val])
    results[name] = {"mantel_r_pearson": r, "mantel_p": p, "spearman_r": float(rs),
                     "spearman_p": float(ps), "n_pairs": n}
    print(f"{name}: Mantel r={r:.3f} p={p:.4f} | Spearman r={rs:.3f} p={ps:.4f} | n={n}")

# uncited in-code matrix (reference)
Munc = np.full((len(disorders), len(disorders)), np.nan)
for (x, y), val in s31.CLINICAL_COMORBIDITY.items():
    if x in idx and y in idx:
        Munc[idx[x], idx[y]] = val
        Munc[idx[y], idx[x]] = val
r, p, n = mantel(auroc, Munc)
ta, tb = tri(auroc), tri(Munc)
val = ~(np.isnan(ta) | np.isnan(tb))
rs, ps = stats.spearmanr(ta[val], tb[val])
results["uncited_in_code_pct"] = {"mantel_r_pearson": r, "mantel_p": p,
                                  "spearman_r": float(rs), "spearman_p": float(ps), "n_pairs": n}
print(f"uncited_in_code_pct: Mantel r={r:.3f} p={p:.4f} | Spearman r={rs:.3f} p={ps:.4f} | n={n}")

out = os.path.join(STATS, "s31c_external_or_sensitivity.json")
json.dump(results, open(out, "w"), indent=2)
print("WROTE", out)
