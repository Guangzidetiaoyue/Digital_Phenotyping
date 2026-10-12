"""
M4 - Step 42: Long-Term Trend Analysis
======================================
Analyze long-term trends in the calendar-aggregated digital signal (s41).

Inputs : module4_results/data/digital_signal_{monthly,annual}*.csv
Outputs: module4_results/statistics/
  trend_decomposition.json      : STL trend components + summary stats
  trend_test_results.json       : Mann-Kendall, regime tests, breakpoint
  trend_comparison_external.csv : digital signal vs CDC/NSDUH (annual)
  s42 log

Design (scientific rigor):
  - STL decomposition of the monthly patient-control difference series to
    separate trend / seasonal / residual.
  - Mann-Kendall trend test (implemented via scipy; non-parametric, robust to
    non-normality and missing values) on the trend component.
  - Regime comparison: 2008-2012 vs 2013-2022 ("smartphone era" hypothesis),
    and 2015-2019 vs 2020-2022 (COVID era) using Welch's t-test.
  - Weighted by n_windows to down-weight sparse early months.
  - Per-disorder trend: Mann-Kendall on each disorder's monthly diff.
  - External comparison: overlay annual digital p-factor diff with CDC suicide
    rate and NSDUH MDE (descriptive, given population differences).
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import STL

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = "/path/to/project"
DATA_DIR = os.path.join(PROJECT_ROOT, "module4_results", "data")
STATS_DIR = os.path.join(PROJECT_ROOT, "module4_results", "statistics")
os.makedirs(STATS_DIR, exist_ok=True)

EXTERNAL_ANNUAL = os.path.join(DATA_DIR, "external", "processed", "epidemiology_annual.csv")


def mann_kendall(x):
    """Mann-Kendall trend test (non-parametric). Returns (tau, p, slope)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan
    # Kendall's tau
    tau, p_tau = stats.kendalltau(np.arange(n), x)
    # Sen's slope
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if x[j] != x[i]:
                slopes.append((x[j] - x[i]) / (j - i))
    slope = np.median(slopes) if slopes else np.nan
    return float(tau), float(p_tau), float(slope)


def welch_test(a, b):
    """Welch's t-test (unequal variance). Returns (t, p, mean_a, mean_b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan, np.nan, np.nan
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p), float(a.mean()), float(b.mean())


def main():
    # ---- Load monthly diff ----
    m_path = os.path.join(DATA_DIR, "digital_signal_monthly_diff.csv")
    mdf = pd.read_csv(m_path)
    mdf["period"] = pd.to_datetime(mdf["period"])
    mdf = mdf.sort_values("period").reset_index(drop=True)
    logger.info(f"Monthly diff: {len(mdf)} months")

    # Focus on p_factor (primary outcome)
    target = "p_factor_severity_z_diff"
    y = mdf[target].values
    n_w = mdf["n_patient"].values
    # Weight: sqrt(n) so sparse months get lower weight but not zero
    w = np.sqrt(n_w)
    w = w / w.mean()

    # ---- STL decomposition (monthly, period=12) ----
    # Need contiguous series; STL handles missing via interpolation internally
    ts = pd.Series(y, index=mdf["period"])
    ts_w = pd.Series(y, index=mdf["period"])
    stl = STL(ts_w, period=12, robust=True)
    res = stl.fit()
    trend = res.trend
    seasonal = res.seasonal
    resid = res.resid
    logger.info("STL decomposition done.")

    trend_vals = trend.dropna().values
    mk_tau, mk_p, mk_slope = mann_kendall(trend_vals)
    logger.info(f"Trend Mann-Kendall: tau={mk_tau:.3f}, p={mk_p:.4f}, slope={mk_slope:.5f}/month")

    # ---- Regime comparisons (weighted) ----
    yrs = mdf["period"].dt.year.values
    # smartphone-era: 2008-2012 vs 2013-2022
    m1 = y[(yrs >= 2008) & (yrs <= 2012)]
    m2 = y[(yrs >= 2013) & (yrs <= 2022)]
    t1, p1, mu1, mu2 = welch_test(m1, m2)
    logger.info(f"2008-12 vs 2013-22: t={t1:.3f}, p={p1:.4f}, means={mu1:.4f}/{mu2:.4f}")

    # COVID-era: 2015-2019 vs 2020-2022
    m3 = y[(yrs >= 2015) & (yrs <= 2019)]
    m4 = y[(yrs >= 2020) & (yrs <= 2022)]
    t2, p2, mu3, mu4 = welch_test(m3, m4)
    logger.info(f"2015-19 vs 2020-22: t={t2:.3f}, p={p2:.4f}, means={mu3:.4f}/{mu4:.4f}")

    # COVID peak analysis: monthly trend in 2020-2022
    covid_mask = (yrs >= 2020) & (yrs <= 2022)
    mk_covid, p_covid, slope_covid = mann_kendall(y[covid_mask])
    logger.info(f"COVID-era (2020-22) trend: tau={mk_covid:.3f}, p={p_covid:.4f}")

    # ---- Per-disorder trend ----
    dis_path = os.path.join(DATA_DIR, "digital_signal_monthly_by_disorder_diff.csv")
    ddf = pd.read_csv(dis_path)
    dis_results = {}
    for d in sorted(ddf["disorder_group"].dropna().unique()):
        sub = ddf[ddf["disorder_group"] == d].sort_values("period")
        v = sub[target].values
        tau, p, slope = mann_kendall(v)
        dis_results[d] = {"tau": tau, "p": p, "slope": slope}
        logger.info(f"  {d}: tau={tau:.3f}, p={p:.4f}, slope={slope:.5f}/mo")

    # ---- External comparison (annual, descriptive) ----
    ext_annual = None
    if os.path.exists(EXTERNAL_ANNUAL):
        ext = pd.read_csv(EXTERNAL_ANNUAL)
        # annual digital diff
        a_path = os.path.join(DATA_DIR, "digital_signal_annual_diff.csv")
        adf = pd.read_csv(a_path)
        compare = adf[["year", target]].rename(columns={target: "digital_pfactor_diff"})
        # suicide rate
        sui = ext[ext["indicator"] == "suicide_rate"][["year", "value"]].rename(
            columns={"value": "suicide_rate"})
        # NSDUH MDE
        mde = ext[ext["indicator"] == "MDE_prevalence"][["year", "value"]].rename(
            columns={"value": "MDE_pct"})
        compare = compare.merge(sui, on="year", how="left").merge(mde, on="year", how="left")
        # Pearson correlations (on overlapping years)
        corr = {}
        for col in ["suicide_rate", "MDE_pct"]:
            c = compare[["digital_pfactor_diff", col]].dropna()
            if len(c) >= 5:
                r, p_c = stats.pearsonr(c["digital_pfactor_diff"], c[col])
                corr[col] = {"r": float(r), "p": float(p_c), "n": len(c)}
            else:
                corr[col] = {"r": np.nan, "p": np.nan, "n": len(c)}
        ext_annual = compare
        logger.info(f"External correlations: {corr}")
    else:
        corr = {}
        compare = None
        logger.warning("External annual data not found; skipping external comparison")

    # ---- Save outputs ----
    trend_df = pd.DataFrame({
        "period": trend.index,
        "trend": trend.values,
        "seasonal": seasonal.values,
        "resid": resid.values,
    })
    trend_df.to_csv(os.path.join(STATS_DIR, "trend_decomposition.csv"), index=False)

    dis_df = pd.DataFrame([{"disorder": k, **v} for k, v in dis_results.items()])
    dis_df.to_csv(os.path.join(STATS_DIR, "trend_by_disorder.csv"), index=False)

    if compare is not None:
        compare.to_csv(os.path.join(STATS_DIR, "trend_comparison_external.csv"), index=False)

    report = {
        "n_months": int(len(mdf)),
        "target": target,
        "stl": {
            "trend_last": float(trend.iloc[-1]) if not pd.isna(trend.iloc[-1]) else None,
            "trend_first": float(trend.iloc[0]) if not pd.isna(trend.iloc[0]) else None,
            "trend_change": float(trend.iloc[-1] - trend.iloc[0]) if not (pd.isna(trend.iloc[-1]) or pd.isna(trend.iloc[0])) else None,
        },
        "mann_kendall_trend": {"tau": mk_tau, "p": mk_p, "sen_slope": mk_slope},
        "regime_smartphone_era": {"t": t1, "p": p1, "mean_0812": mu1, "mean_1322": mu2},
        "regime_covid": {"t": t2, "p": p2, "mean_1519": mu3, "mean_2022": mu4},
        "covid_era_trend": {"tau": mk_covid, "p": p_covid, "slope": slope_covid},
        "per_disorder": dis_results,
        "external_corr": corr,
    }
    with open(os.path.join(STATS_DIR, "trend_test_results.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("DONE")


if __name__ == "__main__":
    main()
