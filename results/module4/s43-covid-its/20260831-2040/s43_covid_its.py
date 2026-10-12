"""
M4 - Step 43: COVID-19 Stratified Interrupted Time Series (ITS)
================================================================
Core M4 analysis: does the COVID-19 pandemic differentially impact the
patient-control digital phenotype gap across the 10 disorders?

Inputs : module4_results/data/digital_signal_monthly_by_disorder_weighted.csv
Outputs: module4_results/statistics/
  covid_its_results.csv          : per-disorder ITS estimates (all metrics)
  covid_its_results.json         : summary + FDR
  covid_its_phases.csv           : phase-wise means (acute/adaptation/vaccine)
  covid_its_model_diagnostics.csv: model diagnostics

Design (rigorous, stratified ITS):
  - Window: 2015-01 .. 2022-12 (96 months, adequate samples).
  - Breakpoint: 2020-03 (WHO declares pandemic; US national emergency).
  - For EACH disorder, fit ITS on its patient-control diff series:
      Y_t = b0 + b1*t + b2*post + b3*(t_centered * post) + e
    where t_centered = months since 2020-03 (0 at breakpoint).
    b2 = immediate level shift (acute shock)
    b3 = post-break slope change (recovery/deterioration trend)
  - Weighted OLS (weights = sqrt(n_patient)) to down-weight sparse months.
  - Newey-West HAC standard errors (handles autocorrelation).
  - 10 disorders x 4 metrics; FDR (Benjamini-Hochberg) within metric.
  - Phase analysis: acute (2020-03..06), adaptation (2020-07..2021-03),
    vaccine (2021-04..2022-01), new-normal (2022-02..12).
  - Sensitivity: breakpoint at 2020-01 and 2020-04 (reported).
Hypotheses:
  H1 anxiety max acute shock; H2 depression slow deterioration;
  H3 ADHD worsens at reopening; H4 autism improves in acute phase.
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
os.makedirs(STATS_DIR, exist_ok=True)

WINDOW_START, WINDOW_END = "2015-01", "2022-12"
BREAKPOINT = "2020-03"
METRICS = ["p_factor_severity_z_diff", "posting_intensity_diff",
           "social_withdrawal_index_diff", "volatility_index_diff"]
METRIC_LABEL = {
    "p_factor_severity_z_diff": "p_factor",
    "posting_intensity_diff": "posting_intensity",
    "social_withdrawal_index_diff": "social_withdrawal",
    "volatility_index_diff": "volatility",
}
PHASES = [
    ("acute", "2020-03", "2020-06"),
    ("adaptation", "2020-07", "2021-03"),
    ("vaccine", "2021-04", "2022-01"),
    ("new_normal", "2022-02", "2022-12"),
]


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "digital_signal_monthly_by_disorder_weighted.csv"))
    df["period"] = pd.to_datetime(df["period"])
    df = df[(df["period"] >= WINDOW_START) & (df["period"] <= WINDOW_END)].copy()
    df["month_idx"] = (df["period"].dt.year - 2015) * 12 + df["period"].dt.month - 1
    df["weight"] = np.sqrt(df["n_patient"].clip(lower=1))
    return df


def fit_its(df, metric, break_date="2020-03-01"):
    """Fit ITS for one disorder-metric series. Returns dict of estimates."""
    d = df.copy()
    d["t"] = d["month_idx"] - d.loc[d["period"] == pd.Timestamp(break_date), "month_idx"].iloc[0]
    d["post"] = (d["period"] >= pd.Timestamp(break_date)).astype(int)
    d["t_post"] = d["t"] * d["post"]
    d = d[d[metric].notna()].copy()
    if len(d) < 20 or d["post"].sum() == 0 or (1 - d["post"]).sum() == 0:
        return None
    # Weighted OLS with HAC
    try:
        model = sm_ols(f"{metric} ~ t + post + t_post", data=d).fit(
            cov_type="HAC", cov_kwds={"maxlags": 6}, weights=d["weight"])
    except Exception:
        return None
    out = {
        "n_obs": int(len(d)),
        "b_intercept": float(model.params["Intercept"]),
        "b_trend": float(model.params["t"]),
        "b_post_shift": float(model.params["post"]),
        "b_post_trend": float(model.params["t_post"]),
        "se_post_shift": float(model.bse["post"]),
        "se_post_trend": float(model.bse["t_post"]),
        "p_post_shift": float(model.pvalues["post"]),
        "p_post_trend": float(model.pvalues["t_post"]),
        "rsquared": float(model.rsquared),
    }
    return out


def main():
    df = load_data()
    logger.info(f"Loaded {len(df)} rows ({df['period'].min()}..{df['period'].max()})")

    results = []
    for disorder in sorted(df["disorder_group"].dropna().unique()):
        sub = df[df["disorder_group"] == disorder]
        for metric in METRICS:
            r = fit_its(sub, metric)
            if r:
                r["disorder"] = disorder
                r["metric"] = metric
                results.append(r)
    res = pd.DataFrame(results)
    logger.info(f"ITS fits: {len(res)}")

    # FDR within each metric for the two key params
    for metric in METRICS:
        m = res["metric"] == metric
        for param in ["p_post_shift", "p_post_trend"]:
            pvals = res.loc[m, param].fillna(1.0).values
            _, padj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
            res.loc[m, param + "_fdr"] = padj

    res.to_csv(os.path.join(STATS_DIR, "covid_its_results.csv"), index=False)
    logger.info("Saved covid_its_results.csv")

    # Phase-wise means (descriptive)
    phase_rows = []
    for disorder in sorted(df["disorder_group"].dropna().unique()):
        sub = df[df["disorder_group"] == disorder]
        for (pname, pstart, pend) in PHASES:
            ps = sub[(sub["period"] >= pstart) & (sub["period"] <= pend)]
            for metric in METRICS:
                if len(ps) > 0:
                    phase_rows.append({
                        "disorder": disorder, "phase": pname, "metric": metric,
                        "mean_diff": float(ps[metric].mean()),
                        "n_patient_mean": float(ps["n_patient"].mean()),
                    })
    phases = pd.DataFrame(phase_rows)
    phases.to_csv(os.path.join(STATS_DIR, "covid_its_phases.csv"), index=False)

    # Sensitivity: alternate breakpoints
    sens_rows = []
    for bp in ["2020-01-01", "2020-04-01"]:
        for disorder in sorted(df["disorder_group"].dropna().unique()):
            sub = df[df["disorder_group"] == disorder]
            r = fit_its(sub, "p_factor_severity_z_diff", break_date=bp)
            if r:
                sens_rows.append({"breakpoint": bp, "disorder": disorder, **r})
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(os.path.join(STATS_DIR, "covid_its_sensitivity_breakpoint.csv"), index=False)

    # Summary report
    report = {
        "window": [WINDOW_START, WINDOW_END],
        "breakpoint": BREAKPOINT,
        "n_fits": len(res),
        "n_significant_shift_fdr": int((res["p_post_shift_fdr"] < 0.05).sum()),
        "n_significant_trend_fdr": int((res["p_post_trend_fdr"] < 0.05).sum()),
        "p_factor_significant": {
            d: {
                "shift": float(res[(res.disorder==d)&(res.metric=='p_factor_severity_z_diff')]['p_post_shift_fdr'].iloc[0]),
                "trend": float(res[(res.disorder==d)&(res.metric=='p_factor_severity_z_diff')]['p_post_trend_fdr'].iloc[0]),
            } for d in sorted(res["disorder"].unique())
        },
    }
    with open(os.path.join(STATS_DIR, "covid_its_results.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("DONE")


if __name__ == "__main__":
    main()
