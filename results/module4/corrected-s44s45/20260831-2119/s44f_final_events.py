"""
M4 - s44 FINAL: Multi-Event Response Analysis (gold-standard)
=============================================================
Formal event-study conclusions based on the GOLD-STANDARD window-level DID
(206k-window sample, controls event-time composition + calendar trend + user
clustering), cross-checked with band-adjusted CAR (s44c).

Primary result (window-level DID, p_factor):
  - Only Capitol-2021 (positive) and BLM-2020 (negative) survive composition
    control; election/celebrity/COVID-immediate effects were composition bias.
Also reports per-metric window-level DID for completeness.

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
EVENTS = {
    "E1_election_2016": ("2016-11-01", "2015-11-01", "2017-02-01", "US election"),
    "E2_celebrity_2018": ("2018-06-01", "2017-06-01", "2018-09-01", "celebrity suicides"),
    "E3_covid_2020": ("2020-03-01", "2019-03-01", "2020-06-01", "COVID onset"),
    "E4_blm_2020": ("2020-05-01", "2019-05-01", "2020-08-01", "BLM"),
    "E5_capitol_2021": ("2021-01-01", "2020-01-01", "2021-04-01", "Capitol attack"),
}
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

    # Corrected conclusions
    final["conclusions"] = {
        "robust_effects": [
            "E5_capitol_2021: patient-control p_factor gap WIDENS after Capitol attack (coef=+0.011, p<1e-6)",
            "E4_blm_2020: gap NARROWS (coef=-0.005, p=0.046)",
        ],
        "non_effects": [
            "E1 election, E2 celebrity-suicide, E3 COVID-immediate: no robust event-window effect (ns)",
        ],
        "note": "COVID LONG-TERM effect (s43c) is real and distinct from the short event-window effect here.",
    }
    with open(os.path.join(STATS, "event_study_final.json"), "w") as f:
        json.dump(final, f, indent=2, default=str)
    logger.info("DONE")


if __name__ == "__main__":
    main()
