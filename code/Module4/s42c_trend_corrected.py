"""
M4 - s42c: Corrected long-term trend analysis (composition-adjusted)
====================================================================
Uses the band-adjusted annual diff (s41c) that removes event-time window
composition bias. Tests whether any calendar trend survives composition control.

Input : module4_results/data/digital_signal_annual_band_adjusted.csv
Output: module4_results/statistics/trend_corrected_results.json
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

METRICS = ["p_factor_severity_z_diff", "posting_intensity_diff",
           "social_withdrawal_index_diff", "volatility_index_diff"]


def mann_kendall(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    tau, p = stats.kendalltau(np.arange(len(x)), x)
    return float(tau), float(p), None


def main():
    adj = pd.read_csv(os.path.join(DATA_DIR, "digital_signal_annual_band_adjusted.csv"))
    logger.info(f"Band-adjusted annual: {len(adj)} rows, cols={adj.columns.tolist()}")

    report = {"n_years": int(len(adj)), "metrics": {}}
    for m in METRICS:
        if m not in adj.columns:
            continue
        tau, p, _ = mann_kendall(adj[m].values)
        # COVID comparison
        pre = adj.loc[adj.year.between(2015, 2019), m].mean()
        post = adj.loc[adj.year.between(2020, 2022), m].mean()
        # 2022 vs 2015
        v15 = adj.loc[adj.year == 2015, m].iloc[0] if (adj.year == 2015).any() else np.nan
        v22 = adj.loc[adj.year == 2022, m].iloc[0] if (adj.year == 2022).any() else np.nan
        report["metrics"][m] = {
            "kendall_tau": tau, "kendall_p": p,
            "mean_2015_2019": float(pre) if not pd.isna(pre) else None,
            "mean_2020_2022": float(post) if not pd.isna(post) else None,
            "value_2015": float(v15) if not pd.isna(v15) else None,
            "value_2022": float(v22) if not pd.isna(v22) else None,
        }
        logger.info(f"  {m}: tau={tau:.3f} (p={p:.4f}), "
                    f"15-19={pre:.4f} vs 20-22={post:.4f}, "
                    f"2015={v15:.4f} -> 2022={v22:.4f}")

    with open(os.path.join(STATS_DIR, "trend_corrected_results.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("DONE")


if __name__ == "__main__":
    main()
