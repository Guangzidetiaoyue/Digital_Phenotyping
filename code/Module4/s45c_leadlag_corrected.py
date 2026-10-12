"""
M4 - s45c: Corrected Lead-Lag Analysis (composition-adjusted)
==============================================================
Re-estimates the lead-lag relationship using BAND-ADJUSTED annual digital
signals (composition-controlled), which removes the event-time bias that
confounded s45. Includes the corrected p_factor (no trend) plus the real
behavioral signals (volatility, posting_intensity).

Method: annual cross-correlation at lags -1/0/+1 with bootstrap CI.
Honest reporting: small samples (8-13 yrs) -> descriptive only.

Output: module4_results/statistics/lead_lag_corrected_results.csv
        module4_results/statistics/lead_lag_corrected_summary.json
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

LAGS = [-1, 0, 1]
# band-adjusted annual has p_factor + posting + withdrawal + volatility diffs
DIG_COLS = ["p_factor_severity_z_diff", "posting_intensity_diff",
            "social_withdrawal_index_diff", "volatility_index_diff"]


def bootstrap_ci(x, y, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    rs = []
    for _ in range(n_boot):
        idx = rng.choice(len(x), len(x), replace=True)
        if np.std(x[idx]) == 0 or np.std(y[idx]) == 0:
            continue
        rs.append(stats.pearsonr(x[idx], y[idx])[0])
    if not rs:
        return np.nan, np.nan
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def lag_corr(dig, ext, dyear, eyear, lag):
    pairs = []
    for i in range(len(dig)):
        target = dyear[i] + lag
        if target in eyear:
            j = eyear.index(target)
            pairs.append((dig[i], ext[j]))
    if len(pairs) < 5:
        return {"lag": lag, "n": len(pairs), "r": np.nan, "p": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    r, p = stats.pearsonr(x, y)
    ci_lo, ci_hi = bootstrap_ci(x, y)
    return {"lag": lag, "n": len(pairs), "r": float(r), "p": float(p),
            "ci_lo": ci_lo, "ci_hi": ci_hi}


def main():
    adj = pd.read_csv(os.path.join(DATA_DIR, "digital_signal_annual_band_adjusted.csv"))
    ext = pd.read_csv(os.path.join(DATA_DIR, "external", "processed", "epidemiology_annual.csv"))
    mde = ext[ext["indicator"] == "MDE_prevalence"][["year", "value"]].rename(columns={"value": "MDE"})
    sui = ext[ext["indicator"] == "suicide_rate"][["year", "value"]].rename(columns={"value": "suicide"})

    results = []
    # MDE overlap: band-adj has 2010-2022; MDE 2015-2022
    for dcol in DIG_COLS:
        if dcol not in adj.columns:
            continue
        for tgt, extsrc in [("MDE", mde), ("suicide_rate", sui)]:
            # overlap years
            merged = adj[["year", dcol]].merge(extsrc, on="year", how="inner")
            if len(merged) < 5:
                continue
            for lag in LAGS:
                val_col = "MDE" if tgt == "MDE" else "suicide"
                r = lag_corr(merged[dcol].values, merged[val_col].values,
                             merged["year"].tolist(), merged["year"].tolist(), lag)
                r["digital_signal"] = dcol
                r["external"] = tgt
                results.append(r)

    res = pd.DataFrame(results)
    res.to_csv(os.path.join(STATS_DIR, "lead_lag_corrected_results.csv"), index=False)
    logger.info("Saved lead_lag_corrected_results.csv")
    for _, r in res.iterrows():
        logger.info(f"  {r['digital_signal'][:15]} vs {r['external']} lag={r['lag']}: "
                    f"r={r['r']:.3f} (p={r['p']:.3f}, n={r['n']})")

    # Summary: best lag per signal-target
    summary = {"method": "annual cross-correlation on band-adjusted (composition-controlled) signals",
               "samples_small_note": "8-13 years only; descriptive, not Granger",
               "best": {}}
    for (sig, tgt), g in res.groupby(["digital_signal", "external"]):
        g = g.dropna(subset=["r"])
        if len(g):
            best = g.loc[g["r"].abs().idxmax()]
            summary["best"][f"{sig}__{tgt}"] = {
                "lag": int(best["lag"]), "r": float(best["r"]), "p": float(best["p"]), "n": int(best["n"])
            }
    with open(os.path.join(STATS_DIR, "lead_lag_corrected_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("DONE")


if __name__ == "__main__":
    main()
