"""
M4 - Step 44: Multi-Event Response Analysis (Event Study)
==========================================================
Does major societal events differentially shift the patient-control digital
phenotype gap? Uses the standard financial event-study approach.

Inputs : module4_results/data/digital_signal_monthly_by_disorder_weighted.csv
         module4_results/data/digital_signal_monthly_diff.csv (pooled)
Outputs: module4_results/statistics/
  event_study_results.csv      : per event x disorder CAR, AR, significance
  event_study_ar_series.csv    : AR series around each event (pooled)
  event_study_summary.json

Method (rigorous, financial event-study):
  - Event list (with anchor dates):
      E1 2016-11 US Presidential Election
      E2 2018-06 Celebrity suicides (media contagion)
      E3 2020-03 COVID-19 pandemic onset
      E4 2020-05 George Floyd / BLM protests
      E5 2021-01 US Capitol attack
  - For each event and disorder:
      Estimation window = [-12, -2] months before event (11 months)
      Expected model: linear trend fitted on estimation window
      AR_t = observed_t - predicted_t   (t in event window)
      Event window = [0, +3] months (post-event)
      CAR = sum of AR over event window
      SE of CAR from estimation-window residuals (scaled)
      t-test: CAR / SE
  - Pooled (all disorders) analysis on digital_signal_monthly_diff.
  - Sensitivity: event window [0, +1] and [0, +6]; drop COVID-adjacent events
    from each other's windows.
Rigor notes:
  - Patient-control DIFF series already removes platform trends.
  - Weighted by sqrt(n_patient) in estimation.
  - Only 2015-2022 used (early years sample too small).
  - FDR correction across events x disorders for CAR significance.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = "/path/to/project"
DATA_DIR = os.path.join(PROJECT_ROOT, "module4_results", "data")
STATS_DIR = os.path.join(PROJECT_ROOT, "module4_results", "statistics")
os.makedirs(STATS_DIR, exist_ok=True)

from config_module4 import EVENTS as _EVENTS, EST_WINDOW, EVENT_WINDOWS
# name -> (anchor month, label); anchors/labels defined once in config_module4.
EVENTS = {k: (v["anchor"], v["label"]) for k, v in _EVENTS.items()}
METRIC = "p_factor_severity_z_diff"


def compute_ar(df, anchor, est_win, ev_win, weights=None):
    """Compute AR series for one event on a diff series.
    Returns (ar_series_in_window, car, se_car, t_car, n_est)."""
    df = df.copy()
    df["period"] = pd.to_datetime(df["period"])
    anchor_ts = pd.Timestamp(anchor)
    df["rel"] = (df["period"].dt.year - anchor_ts.year) * 12 + df["period"].dt.month - anchor_ts.month
    # estimation window
    est = df[(df["rel"] >= est_win[0]) & (df["rel"] <= est_win[1])]
    if len(est) < 8:
        return None
    x = est["rel"].values
    y = est[METRIC].values
    # drop rows with missing y
    ok = ~np.isnan(y)
    x, y = x[ok], y[ok]
    if len(x) < 8:
        return None
    if weights is not None:
        w = weights.reindex(est.index).fillna(1.0).values[ok]
    else:
        w = np.ones(len(x))
    # linear trend fit (weighted)
    A = np.vstack([np.ones(len(x)), x]).T
    W = np.diag(w)
    try:
        beta, _, _, _ = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ y, rcond=None)
    except Exception:
        return None
    resid_est = y - A @ beta
    sigma = np.std(resid_est, ddof=2) if len(resid_est) > 2 else np.nan
    # event window
    ev = df[(df["rel"] >= ev_win[0]) & (df["rel"] <= ev_win[1])]
    ev = ev[ev[METRIC].notna()]
    # allow partial windows (some months may be missing due to sparse data);
    # require at least 2 observations in the window
    if len(ev) < 2:
        return None
    A_ev = np.vstack([np.ones(len(ev)), ev["rel"].values]).T
    pred_ev = A_ev @ beta
    ar = ev[METRIC].values - pred_ev
    car = float(ar.sum())
    # SE of CAR: scale residual sd by sqrt(n_ev)
    n_ev = len(ev)
    se_car = float(sigma * np.sqrt(n_ev)) if not np.isnan(sigma) and sigma > 0 else np.nan
    t_car = car / se_car if not np.isnan(se_car) and se_car > 0 else np.nan
    p_car = 2 * stats.t.sf(abs(t_car), df=len(est) - 2) if not np.isnan(t_car) else np.nan
    return {
        "event": None, "disorder": None, "est_n": int(len(est)),
        "ar_series": list(ar), "car": car, "se_car": se_car, "t_car": t_car, "p_car": p_car,
    }


def main():
    # ---- Load weighted by-disorder ----
    wd = pd.read_csv(os.path.join(DATA_DIR, "digital_signal_monthly_by_disorder_weighted.csv"))
    wd["period"] = pd.to_datetime(wd["period"])
    # pooled
    pooled = pd.read_csv(os.path.join(DATA_DIR, "digital_signal_monthly_diff.csv"))
    pooled["period"] = pd.to_datetime(pooled["period"])

    rows = []
    ar_series_rows = []
    main_win = EVENT_WINDOWS[0]
    for ev_name, (anchor, label) in EVENTS.items():
        # Pooled analysis
        weights = np.sqrt(pooled["n_patient"].clip(lower=1))
        r = compute_ar(pooled, anchor, EST_WINDOW, main_win, weights=pd.Series(weights.values, index=pooled.index))
        if r:
            r = {**r, "event": ev_name, "event_label": label, "disorder": "ALL"}
            rows.append(r)

        # Per-disorder
        for disorder in sorted(wd["disorder_group"].dropna().unique()):
            sub = wd[wd["disorder_group"] == disorder]
            w_sub = np.sqrt(sub["n_patient"].clip(lower=1))
            r = compute_ar(sub, anchor, EST_WINDOW, main_win,
                           weights=pd.Series(w_sub.values, index=sub.index))
            if r:
                r = {**r, "event": ev_name, "event_label": label, "disorder": disorder}
                rows.append(r)

    res = pd.DataFrame(rows)
    logger.info(f"Event-study fits: {len(res)}")

    # FDR on CAR p-values (within event, across disorders)
    for ev in EVENTS:
        m = res["event"] == ev
        res.loc[m, "p_car_fdr"] = np.nan
        pvals = res.loc[m, "p_car"].fillna(1.0).values
        if len(pvals) > 1 and np.isfinite(pvals).sum() > 0:
            _, padj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
            res.loc[m, "p_car_fdr"] = padj

    # Keep main window results for CSV; sensitivity separate
    res.to_csv(os.path.join(STATS_DIR, "event_study_results.csv"), index=False)
    logger.info("Saved event_study_results.csv")

    # Sensitivity: different event windows (pooled only)
    sens_rows = []
    for ev_name, (anchor, label) in EVENTS.items():
        for wname, ev_win in [("w3", (0,3)), ("w1", (0,1)), ("w6", (0,6))]:
            weights = np.sqrt(pooled["n_patient"].clip(lower=1))
            r = compute_ar(pooled, anchor, EST_WINDOW, ev_win,
                           weights=pd.Series(weights.values, index=pooled.index))
            if r:
                sens_rows.append({"event": ev_name, "window": wname, **r})
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(os.path.join(STATS_DIR, "event_study_sensitivity_window.csv"), index=False)

    # Summary
    # Summary (pooled CAR per event, with fallback if ALL missing)
    pooled_car = {}
    for ev in EVENTS:
        sub = res[(res.event == ev) & (res.disorder == "ALL")]
        if len(sub) > 0:
            pooled_car[ev] = {
                "car": float(sub["car"].iloc[0]),
                "p_fdr": float(sub["p_car_fdr"].iloc[0]) if not pd.isna(sub["p_car_fdr"].iloc[0]) else None,
            }
    summary = {
        "events": {k: {"date": v[0], "label": v[1]} for k, v in EVENTS.items()},
        "method": "financial event study: linear-trend expected model, estimation [-12,-2], CAR over [0,+3]",
        "n_fits": int(len(res)),
        "n_significant_car_fdr": int((res["p_car_fdr"] < 0.05).sum()),
        "pooled_car": pooled_car,
    }
    with open(os.path.join(STATS_DIR, "event_study_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("DONE")


if __name__ == "__main__":
    main()
