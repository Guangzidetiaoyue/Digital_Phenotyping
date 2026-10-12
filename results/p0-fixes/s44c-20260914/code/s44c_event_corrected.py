"""
M4 - s44c: Corrected Multi-Event Response Analysis (final)
==========================================================
Formal event study on COMPOSITION-ADJUSTED data (s41d band-adjusted), which
removes event-time window bias. Adopts MacKinlay (1997) standard event-study
setup, cross-validated by a window-level DID regression (control for
days_from_t0 polynomial + calendar trend).

Events (anchor, event window [0,+3] post):
  E1 2016-11 US election
  E2 2018-06 celebrity suicides
  E3 2020-03 COVID onset
  E4 2020-05 BLM
  E5 2021-01 Capitol attack

Method A (band-adjusted AR/CAR): expected model = 12-month pre-trend
  (estimation [-12,-2]); AR = obs - predicted; CAR over [0,+3].
Method B (window-level DID): regression of p_factor diff on event-window
  indicator within disorder, controlling for calendar month trend.
  (validated separately; primary evidence here is Method A on band-adj data)

Outputs:
  module4_results/statistics/event_study_corrected_results.csv
  module4_results/statistics/event_study_corrected_summary.json
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

METRICS = ["p_factor_severity_z_diff", "posting_intensity_diff",
           "social_withdrawal_index_diff", "volatility_index_diff"]
EVENTS = {
    "E1_election_2016": ("2016-11-01", "US Presidential Election 2016"),
    "E2_celebrity_2018": ("2018-06-01", "Celebrity suicides"),
    "E3_covid_2020": ("2020-03-01", "COVID-19 onset"),
    "E4_blm_2020": ("2020-05-01", "George Floyd / BLM"),
    "E5_capitol_2021": ("2021-01-01", "US Capitol attack"),
}
EST_WINDOW = (-12, -2)
EV_WINDOW = (0, 3)


def compute_car(d, anchor, metric, est_win=EST_WINDOW, ev_win=EV_WINDOW):
    """Band-adjusted CAR for one disorder-metric series around an event."""
    d = d.copy()
    d["period"] = pd.to_datetime(d["period"])
    a = pd.Timestamp(anchor)
    d["rel"] = (d["period"].dt.year - a.year) * 12 + d["period"].dt.month - a.month
    est = d[(d["rel"] >= est_win[0]) & (d["rel"] <= est_win[1])]
    est = est[est[metric].notna()]
    if len(est) < 8:
        return None
    x, y = est["rel"].values, est[metric].values
    A = np.vstack([np.ones(len(x)), x]).T
    try:
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    except Exception:
        return None
    resid = y - A @ beta
    sigma = np.std(resid, ddof=2) if len(resid) > 2 else np.nan
    ev = d[(d["rel"] >= ev_win[0]) & (d["rel"] <= ev_win[1])]
    ev = ev[ev[metric].notna()]
    if len(ev) < 2:
        return None
    A_ev = np.vstack([np.ones(len(ev)), ev["rel"].values]).T
    ar = ev[metric].values - A_ev @ beta
    car = float(ar.sum())
    se_car = float(sigma * np.sqrt(len(ev))) if not np.isnan(sigma) and sigma > 0 else np.nan
    t_car = car / se_car if not np.isnan(se_car) and se_car > 0 else np.nan
    p_car = 2 * stats.t.sf(abs(t_car), df=len(est) - 2) if not np.isnan(t_car) else np.nan
    return {"car": car, "se": se_car, "t": t_car, "p": p_car, "n_est": int(len(est)), "n_ev": int(len(ev))}


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "digital_signal_monthly_by_disorder_band_adj.csv"))
    df["period"] = pd.to_datetime(df["period"])
    logger.info(f"Band-adjusted by-disorder: {len(df)} rows")

    rows = []
    for ev_name, (anchor, label) in EVENTS.items():
        for disorder in sorted(df["disorder_group"].dropna().unique()):
            sub = df[df["disorder_group"] == disorder]
            for metric in METRICS:
                r = compute_car(sub, anchor, metric)
                if r:
                    rows.append({"event": ev_name, "event_label": label,
                                 "disorder": disorder, "metric": metric, **r})
    res = pd.DataFrame(rows)
    logger.info(f"Corrected event-study fits: {len(res)}")

    # FDR within event x metric across disorders
    res["p_fdr"] = np.nan
    for ev in EVENTS:
        for metric in METRICS:
            m = (res["event"] == ev) & (res["metric"] == metric)
            pvals = res.loc[m, "p"].fillna(1.0).values
            if len(pvals) > 1:
                _, padj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
                res.loc[m, "p_fdr"] = padj
    res.to_csv(os.path.join(STATS_DIR, "event_study_corrected_results.csv"), index=False)

    # Summary: significant per event (pooled across disorders, p_factor primary)
    sig = res[res["p_fdr"] < 0.05]
    logger.info(f"Significant CAR (FDR<0.05): {len(sig)}")
    logger.info("\nSignificant events (p_factor primary):")
    pf_sig = sig[sig["metric"] == "p_factor_severity_z_diff"]
    logger.info(pf_sig[["event", "disorder", "car", "p_fdr"]].to_string(index=False))

    # Pooled (mean across disorders) CAR per event per metric
    pooled = res.groupby(["event", "metric"])["car"].mean().reset_index()
    pooled.to_csv(os.path.join(STATS_DIR, "event_study_corrected_pooled.csv"), index=False)

    # Significant p-factor events, COMPUTED from this script's own results
    # (FDR<0.05); no pre-stated conclusion about any other analysis.
    pf_sig = res[(res["p_fdr"] < 0.05) & (res["metric"] == "p_factor_severity_z_diff")]
    pf_sig_events = sorted(pf_sig["event"].unique().tolist())

    summary = {
        "method": "MacKinlay (1997) event study on composition-adjusted (band-equalized) data",
        "events": {k: {"date": v[0], "label": v[1]} for k, v in EVENTS.items()},
        "n_fits": int(len(res)),
        "n_significant_fdr": int(len(sig)),
        "significant_pfactor_events_fdr": pf_sig_events,
        "mean_car_by_event_pfactor": {
            ev: float(res[(res.event == ev) & (res.metric == "p_factor_severity_z_diff")]["car"].mean())
            for ev in EVENTS
        },
        "note": ("Corrected for event-time composition bias (the original s44 CAR was confounded). "
                 "Significant p-factor CAR events at FDR<0.05: "
                 f"{pf_sig_events if pf_sig_events else 'none'}. "
                 "The window-level DID (s44f) is a separate analysis; see its own output."),
    }
    with open(os.path.join(STATS_DIR, "event_study_corrected_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("DONE")


if __name__ == "__main__":
    main()
