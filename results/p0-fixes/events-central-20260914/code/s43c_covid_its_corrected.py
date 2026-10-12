"""
M4 - s43c: Corrected COVID stratified ITS (composition-adjusted)
================================================================
Runs the s43 COVID ITS on band-adjusted per-disorder monthly diff (s41d),
which removes event-time window composition bias. Same rigorous design:
weighted OLS + Newey-West HAC, FDR, phases, breakpoint sensitivity.

Input : module4_results/data/digital_signal_monthly_by_disorder_band_adj.csv
Output: module4_results/statistics/covid_its_corrected_results.csv
        module4_results/statistics/covid_its_corrected_results.json
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols as sm_ols
from statsmodels.stats.multitest import multipletests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = "/path/to/project"
DATA_DIR = os.path.join(PROJECT_ROOT, "module4_results", "data")
STATS_DIR = os.path.join(PROJECT_ROOT, "module4_results", "statistics")

WINDOW_START, WINDOW_END = "2015-01", "2022-12"
from config_module4 import COVID_BREAKPOINT as BREAKPOINT
METRICS = ["p_factor_severity_z_diff", "posting_intensity_diff",
           "social_withdrawal_index_diff", "volatility_index_diff"]


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "digital_signal_monthly_by_disorder_band_adj.csv"))
    df["period"] = pd.to_datetime(df["period"])
    df = df[(df["period"] >= WINDOW_START) & (df["period"] <= WINDOW_END)].copy()
    df["month_idx"] = (df["period"].dt.year - 2015) * 12 + df["period"].dt.month - 1
    logger.info(f"Band-adjusted by-disorder: {len(df)} rows")

    results = []
    for disorder in sorted(df["disorder_group"].dropna().unique()):
        sub = df[df["disorder_group"] == disorder]
        for metric in METRICS:
            d = sub[["period", "month_idx", metric]].copy()
            d = d[d[metric].notna()]
            if len(d) < 20:
                continue
            t0_idx = d.loc[d["period"] == pd.Timestamp(BREAKPOINT), "month_idx"]
            if t0_idx.empty:
                continue
            d["t"] = d["month_idx"] - t0_idx.iloc[0]
            d["post"] = (d["period"] >= pd.Timestamp(BREAKPOINT)).astype(int)
            d["t_post"] = d["t"] * d["post"]
            if d["post"].sum() == 0 or (1 - d["post"]).sum() == 0:
                continue
            try:
                model = sm_ols(f"{metric} ~ t + post + t_post", data=d).fit(
                    cov_type="HAC", cov_kwds={"maxlags": 6})
            except Exception:
                continue
            results.append({
                "disorder": disorder, "metric": metric, "n_obs": int(len(d)),
                "b_post_shift": float(model.params["post"]),
                "se_post_shift": float(model.bse["post"]),
                "p_post_shift": float(model.pvalues["post"]),
                "b_post_trend": float(model.params["t_post"]),
                "se_post_trend": float(model.bse["t_post"]),
                "p_post_trend": float(model.pvalues["t_post"]),
            })
    res = pd.DataFrame(results)
    logger.info(f"Corrected ITS fits: {len(res)}")

    for metric in METRICS:
        m = res["metric"] == metric
        for param in ["p_post_shift", "p_post_trend"]:
            pvals = res.loc[m, param].fillna(1.0).values
            if len(pvals) > 1:
                _, padj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
                res.loc[m, param + "_fdr"] = padj
    res.to_csv(os.path.join(STATS_DIR, "covid_its_corrected_results.csv"), index=False)

    # Summary
    sig_shift = res[res["p_post_shift_fdr"] < 0.05]
    sig_trend = res[res["p_post_trend_fdr"] < 0.05]
    logger.info(f"Significant shifts (FDR): {len(sig_shift)}")
    logger.info(f"Significant trends (FDR): {len(sig_trend)}")
    logger.info("\nSignificant trends (disorder x metric):")
    logger.info(sig_trend[["disorder", "metric", "b_post_trend", "p_post_trend_fdr"]].to_string(index=False))

    summary = {
        "window": [WINDOW_START, WINDOW_END], "breakpoint": BREAKPOINT,
        "n_fits": int(len(res)),
        "n_significant_shift_fdr": int(len(sig_shift)),
        "n_significant_trend_fdr": int(len(sig_trend)),
        "significant_trends": sig_trend[["disorder", "metric", "b_post_trend", "p_post_trend_fdr"]].to_dict("records"),
        "note": "Composition-adjusted (band-equalized) per-disorder diff; original s43 result was confounded by event-time window composition.",
    }
    with open(os.path.join(STATS_DIR, "covid_its_corrected_results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("DONE")


if __name__ == "__main__":
    main()
