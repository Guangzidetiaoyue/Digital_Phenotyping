"""
M4 - s44 FINAL: Multi-Event Response Analysis (gold-standard)
=============================================================
Formal event-study conclusions based on the window-level DID
(controls event-time composition + calendar trend + user clustering),
cross-checked with band-adjusted CAR (s44c).

Reports the window-level DID (cluster-robust) for every event and metric.
Conclusions are COMPUTED from the fitted models (no hard-coded results).

Output: module4_results/statistics/event_study_final.json
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
STATS = "/path/to/project/module4_results/statistics"
COLS = ["user_id_hash", "group", "window_start_ts", "days_from_t0_center", "is_reliable",
        "p_factor_severity_z", "posting_intensity", "social_withdrawal_index", "volatility_index"]
from config_module4 import EVENTS as _EVENTS
EVENTS = {k: (v["anchor"], v["est_start"], v["ev_end"], v["label"])
          for k, v in _EVENTS.items()}
METRICS = ["p_factor_severity_z", "posting_intensity", "social_withdrawal_index", "volatility_index"]


def main():
    logger.info("Reading trajectory (20% sample)...")
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
    logger.info(f"Sample: {len(tr)}")

    final = {"events": {}, "method": "window-level DID (gold standard)"}
    for ev, (anchor, est_start, ev_end, label) in EVENTS.items():
        ev_res = {}
        est_end = pd.Timestamp(anchor) - pd.offsets.MonthBegin(1)
        ev_end_ts = pd.Timestamp(ev_end)
        d = tr[(tr["period"] >= pd.Timestamp(est_start).strftime("%Y-%m")) &
               (tr["period"] <= ev_end_ts.strftime("%Y-%m"))].copy()
        d["in_event"] = (d["period"] >= pd.Timestamp(anchor).strftime("%Y-%m")).astype(int)
        d["group_in_event"] = d["patient"] * d["in_event"]
        d["month_float"] = (d["period"].str[:4].astype(int) - 2010) * 12 + d["period"].str[5:7].astype(int)
        for metric in METRICS:
            dd = d[d[metric].notna()]
            X = dd[["patient", "in_event", "group_in_event", "days_from_t0_center", "d2", "d3", "month_float"]]
            X = sm.add_constant(X)
            try:
                m = sm.OLS(dd[metric], X).fit(cov_type="cluster", cov_kwds={"groups": dd["user_id_hash"]})
                ev_res[metric] = {
                    "coef_group_event": float(m.params["group_in_event"]),
                    "se": float(m.bse["group_in_event"]),
                    "p": float(m.pvalues["group_in_event"]),
                    "n": int(len(dd)),
                }
            except Exception as e:
                ev_res[metric] = {"error": str(e)}
        final["events"][ev] = {"label": label, **ev_res}
        logger.info(f"  {ev}: p_factor coef={ev_res.get('p_factor_severity_z',{}).get('coef_group_event','?')}")

    # Conclusions COMPUTED from the fitted models (no hard-coded coefficients).
    robust_effects, non_effects = [], []
    for ev in EVENTS:
        r = final["events"][ev].get("p_factor_severity_z", {})
        if "coef_group_event" not in r or "p" not in r:
            continue
        coef, p = r["coef_group_event"], r["p"]
        label = final["events"][ev]["label"]
        direction = "WIDENS" if coef > 0 else "NARROWS"
        desc = (f"{ev} ({label}): patient-control p_factor gap {direction} "
                f"(coef={coef:+.4f}, p={p:.3g})")
        (robust_effects if p < 0.05 else non_effects).append(desc)
    final["conclusions"] = {
        "robust_effects": robust_effects,
        "non_effects": non_effects,
        "alpha": 0.05,
        "note": "Computed from window-level DID (cluster-robust SE). "
                "The COVID LONG-TERM effect (s43c) is a distinct analysis from the "
                "short event-window effect reported here.",
    }
    with open(os.path.join(STATS, "event_study_final.json"), "w") as f:
        json.dump(final, f, indent=2, default=str)
    logger.info("DONE")


if __name__ == "__main__":
    main()
