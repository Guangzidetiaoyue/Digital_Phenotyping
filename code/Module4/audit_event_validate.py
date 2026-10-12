"""
CRITICAL VALIDATION 2: Window-level event effects (does s44 CAR survive
event-time composition control?)

For each event, estimate the patient-control gap change in the event window
[0,+3] months, controlling for days_from_t0 polynomial and calendar trend.
This is the DID-style validation of the financial event-study CAR.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TRAJ = "/path/to/project/module2_results/trajectories/user_trajectories.parquet"
COLS = ["user_id_hash", "group", "disorder_group", "window_start_ts", "days_from_t0_center",
        "is_reliable", "p_factor_severity_z"]

from config_module4 import EVENTS as _EVENTS
EVENTS = {k: (v["anchor"], v["est_start"], v["ev_end"]) for k, v in _EVENTS.items()}


def main():
    pf = pq.ParquetFile(TRAJ)
    chunks = []
    for i, batch in enumerate(pf.iter_batches(batch_size=500_000, columns=COLS)):
        df = batch.to_pandas()
        df = df[df["is_reliable"] == True]
        df = df.sample(frac=0.2, random_state=42) if len(df) else df
        chunks.append(df)
        if i > 50:
            break
    tr = pd.concat(chunks, ignore_index=True)
    ts = pd.to_datetime(tr["window_start_ts"], unit="s", utc=True)
    tr["period"] = ts.dt.to_period("M").astype(str)
    tr["patient"] = (tr["group"] == "patient").astype(int)
    tr["d2"] = tr["days_from_t0_center"] ** 2
    tr["d3"] = tr["days_from_t0_center"] ** 3

    results = {}
    for ev, (anchor, est_start, ev_end) in EVENTS.items():
        # estimation pre-window (12 months) + event window
        est_end = pd.Timestamp(anchor) - pd.offsets.MonthBegin(1)
        est_start_ts = pd.Timestamp(est_start)
        ev_end_ts = pd.Timestamp(ev_end)
        d = tr[(tr["period"] >= est_start_ts.strftime("%Y-%m")) &
               (tr["period"] <= ev_end_ts.strftime("%Y-%m"))].copy()
        d["in_event"] = (d["period"] >= pd.Timestamp(anchor).strftime("%Y-%m")).astype(int)
        d["group_in_event"] = d["patient"] * d["in_event"]
        # calendar trend
        d["month_float"] = (d["period"].str[:4].astype(int) - 2010) * 12 + d["period"].str[5:7].astype(int)
        d = d[d["p_factor_severity_z"].notna()]
        X = d[["patient", "in_event", "group_in_event", "days_from_t0_center", "d2", "d3", "month_float"]]
        X = sm.add_constant(X)
        try:
            model = sm.OLS(d["p_factor_severity_z"], X).fit(
                cov_type="cluster", cov_kwds={"groups": d["user_id_hash"]})
            results[ev] = {
                "coef_group_event": float(model.params["group_in_event"]),
                "se": float(model.bse["group_in_event"]),
                "p": float(model.pvalues["group_in_event"]),
                "n": int(len(d)),
                "month_coef": float(model.params["month_float"]),
            }
            logger.info(f"  {ev}: group*event coef={model.params['group_in_event']:.4f} "
                        f"(p={model.pvalues['group_in_event']:.4g})")
        except Exception as e:
            results[ev] = {"error": str(e)}
            logger.warning(f"  {ev} FAILED: {e}")

    out = "/path/to/project/module4_results/statistics/event_windowlevel_validation.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("DONE")


if __name__ == "__main__":
    main()
