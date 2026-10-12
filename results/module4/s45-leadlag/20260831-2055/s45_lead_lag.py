"""
M4 - Step 45: Lead-Lag Analysis (RQ4 external validity)
========================================================
Does the Reddit digital signal lead / coincide / lag traditional
epidemiological indicators?

Inputs : module4_results/data/digital_signal_annual_diff.csv
         module4_results/data/external/processed/epidemiology_annual.csv
Outputs: module4_results/statistics/
  lead_lag_results.csv       : correlation by lag (digital relative to external)
  lead_lag_comparison.csv    : aligned annual series (digital + external)
  lead_lag_summary.json

Design (rigorous, sample-size aware):
  RQ4a (temporal): align annual Reddit patient-control p-factor diff with
  NSDUH MDE prevalence and CDC suicide rate. Compute cross-correlation at
  lags -1, 0, +1 (digital signal leading / coincident / lagging external).
  - Small samples (8-15 overlapping years) -> report bootstrap 95% CI and
    explicitly note limited statistical power; no over-claiming Granger.
  - NSDUH 2021 methodological break flagged; correlation computed on
    pre-break years (2015-2020) as primary, all-years as sensitivity.
  RQ4b (spatial): attempted only if state-level Reddit data existed.
    NOTE: trajectory data has NO geographic label -> spatial external
    validity not feasible; documented as limitation.
  Robustness: Spearman + Pearson; rank-based lag relation.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = "/path/to/project"
DATA_DIR = os.path.join(PROJECT_ROOT, "module4_results", "data")
STATS_DIR = os.path.join(PROJECT_ROOT, "module4_results", "statistics")
os.makedirs(STATS_DIR, exist_ok=True)

LAGS = [-1, 0, 1]  # digital leads external (-1), coincident (0), digital lags (+1)


def bootstrap_ci(x, y, n_boot=2000, seed=42):
    """Bootstrap CI for Pearson correlation."""
    rng = np.random.RandomState(seed)
    n = len(x)
    rs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if np.std(x[idx]) == 0 or np.std(y[idx]) == 0:
            continue
        rs.append(stats.pearsonr(x[idx], y[idx])[0])
    if not rs:
        return np.nan, np.nan
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def lag_correlation(digital, external, digital_year, ext_year, lag):
    """Correlation between digital[year] and external[year + lag].
    lag=-1: digital leads external (digital in year t vs external in t+1)."""
    d = pd.DataFrame({"year": digital_year, "dig": digital,
                      "ext_year": ext_year, "ext": external})
    # align: ext at year+lag
    d_align = d.merge(d[["year", "ext"]], left_on="year",
                      right_on="year", suffixes=("", "_same"))
    # simpler: build pairs where digital year t pairs with external year t+lag
    pairs = []
    for _, row in d.iterrows():
        target_ext_year = row["year"] + lag
        ext_row = d[(d["year"] == target_ext_year)]
        if len(ext_row) > 0:
            pairs.append((row["dig"], ext_row["ext"].iloc[0]))
    if len(pairs) < 5:
        return {"lag": lag, "n": len(pairs), "r": np.nan, "p": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan}
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    r, p = stats.pearsonr(x, y)
    ci_lo, ci_hi = bootstrap_ci(x, y)
    # Spearman
    rs, ps = stats.spearmanr(x, y)
    return {"lag": lag, "n": len(pairs), "r": float(r), "p": float(p),
            "ci_lo": ci_lo, "ci_hi": ci_hi, "spearman_r": float(rs), "spearman_p": float(ps)}


def main():
    # ---- Load digital annual signal ----
    dig = pd.read_csv(os.path.join(DATA_DIR, "digital_signal_annual_diff.csv"))
    dig_year = dig["year"].values
    dig_pf = dig["p_factor_severity_z_diff"].values

    # ---- Load external ----
    ext = pd.read_csv(os.path.join(DATA_DIR, "external", "processed", "epidemiology_annual.csv"))
    mde = ext[ext["indicator"] == "MDE_prevalence"][["year", "value"]].rename(
        columns={"value": "MDE"})
    sui = ext[ext["indicator"] == "suicide_rate"][["year", "value"]].rename(
        columns={"value": "suicide"})

    # ---- RQ4a: lag correlations ----
    results = []
    # MDE (2015-2020 pre-break primary; all 2015-2022 sensitivity)
    dig_2015 = dig[(dig["year"] >= 2015) & (dig["year"] <= 2022)]
    mde_all = mde[(mde["year"] >= 2015) & (mde["year"] <= 2022)]
    for label, dsel, esel, note in [
        ("MDE_all", dig_2015, mde_all, "2015-2022 (all)"),
        ("MDE_prebreak", dig_2015[dig_2015["year"] <= 2020],
         mde_all[mde_all["year"] <= 2020], "2015-2020 (pre-2021 break)"),
    ]:
        for lag in LAGS:
            r = lag_correlation(dsel["p_factor_severity_z_diff"].values,
                                esel["MDE"].values, dsel["year"].values,
                                esel["year"].values, lag)
            r["target"] = "MDE"
            r["sample"] = note
            results.append(r)

    # Suicide (overlap 2008-2018)
    dig_08 = dig[(dig["year"] >= 2008) & (dig["year"] <= 2018)]
    sui_08 = sui[(sui["year"] >= 2008) & (sui["year"] <= 2018)]
    for lag in LAGS:
        r = lag_correlation(dig_08["p_factor_severity_z_diff"].values,
                            sui_08["suicide"].values, dig_08["year"].values,
                            sui_08["year"].values, lag)
        r["target"] = "suicide_rate"
        r["sample"] = "2008-2018"
        results.append(r)

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(STATS_DIR, "lead_lag_results.csv"), index=False)
    logger.info("Saved lead_lag_results.csv")
    for _, r in res_df.iterrows():
        logger.info(f"  {r['target']} {r['sample']} lag={r['lag']}: r={r['r']:.3f} "
                    f"(p={r['p']:.3f}, CI[{r['ci_lo']:.2f},{r['ci_hi']:.2f}])")

    # ---- Aligned comparison table ----
    comp = dig.merge(mde, on="year", how="outer").merge(sui, on="year", how="outer")
    comp = comp.sort_values("year")
    comp.to_csv(os.path.join(STATS_DIR, "lead_lag_comparison.csv"), index=False)

    # ---- RQ4b spatial ----
    spatial_status = "NOT_FEASIBLE: trajectory data has no state-level geographic label; spatial external validity cannot be computed with Reddit state signal."

    # ---- Summary ----
    # Best (strongest |r|) finding per target
    best = {}
    for tgt in ["MDE", "suicide_rate"]:
        sub = res_df[res_df["target"] == tgt]
        if len(sub):
            abs_r = sub["r"].abs()
            best[tgt] = {
                "lag": int(sub.loc[abs_r.idxmax(), "lag"]),
                "r": float(sub.loc[abs_r.idxmax(), "r"]),
                "p": float(sub.loc[abs_r.idxmax(), "p"]),
                "sample": sub.loc[abs_r.idxmax(), "sample"],
            }
    summary = {
        "rq4a_temporal": {
            "method": "annual cross-correlation at lags -1/0/+1; bootstrap CI; sample-size limited",
            "best_by_target": best,
            "nsduh_break_note": "2021 methodological break; pre-2021 primary",
        },
        "rq4b_spatial": {"status": "NOT_FEASIBLE", "reason": spatial_status},
        "limitations": [
            "Annual samples (8-15 yrs) limit power for Granger/lead-lag; results descriptive",
            "CDC suicide rate overlap 2008-2018 (11 yrs); NSDUH MDE 2015-2022 (8 yrs)",
            "Population mismatch: Reddit users vs US general population",
        ],
    }
    with open(os.path.join(STATS_DIR, "lead_lag_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("DONE")


if __name__ == "__main__":
    main()
