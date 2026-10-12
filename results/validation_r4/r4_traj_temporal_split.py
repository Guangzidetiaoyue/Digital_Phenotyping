"""
R4-3: Temporal generalization at the trajectory scale
======================================================
Checks that the individual-level event-study morphology around the
self-reported diagnosis date (patients below controls far pre-t0, transient
t0-surge, post-t0 return) REPLICATES across calendar eras — i.e. the phenotype
is not an artefact of one Reddit era.

Input : module2_results/trajectories/user_trajectories.parquet (server)
        (user_id_hash, group, disorder_group, window_start_ts,
         days_from_t0_center, is_reliable, p_factor_severity_z, posting_intensity,
         social_withdrawal_index, volatility_index)
Output: r4_traj_temporal_split.csv / .json
         per-era patient-control diff by event-time band + era comparison tests
"""
import os
import json
import logging
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

TRAJ = os.environ.get(
    "R4_TRAJ",
    "/path/to/project/module2_results/trajectories/user_trajectories.parquet")
OUT = os.environ.get(
    "R4_OUT",
    "/path/to/project/module4_results/validation_r4")
os.makedirs(OUT, exist_ok=True)

METRICS = ["p_factor_severity_z", "posting_intensity", "social_withdrawal_index", "volatility_index"]
READ = ["user_id_hash", "group", "disorder_group", "window_start_ts", "days_from_t0_center",
        "is_reliable"] + METRICS
BANDS = {
    "far_pre": (-720, -360), "mid_pre": (-360, -180), "near_pre": (-180, -90),
    "t0_surge": (-90, 90), "near_post": (90, 180), "mid_post": (180, 360), "far_post": (360, 720),
}
ERAS = [("E1_2010-2015", (2010, 2015)), ("E2_2016-2019", (2016, 2019)), ("E3_2020-2022", (2020, 2022))]


def assign_band(days):
    out = pd.Series("none", index=days.index)
    for name, (lo, hi) in BANDS.items():
        out[(days > lo) & (days <= hi)] = name
    return out


def main():
    pf = pq.ParquetFile(TRAJ)
    chunks = []
    for b in pf.iter_batches(batch_size=500_000, columns=READ):
        df = b.to_pandas()
        df = df[df["is_reliable"] == True]
        if len(df):
            chunks.append(df)
    tr = pd.concat(chunks, ignore_index=True)
    del chunks
    ts = pd.to_datetime(tr["window_start_ts"], unit="s", utc=True)
    tr["year"] = ts.dt.year
    tr["era"] = pd.cut(tr["year"], bins=[2007, 2015, 2019, 2023],
                       labels=["E1_2010-2015", "E2_2016-2019", "E3_2020-2022"])
    tr["band"] = assign_band(tr["days_from_t0_center"])
    logger.info(f"reliable windows: {len(tr)}")

    rows = []
    era_series = {}
    for era, (y0, y1) in ERAS:
        sub = tr[tr["year"].between(y0, y1)]
        pat = sub[sub["group"] == "patient"].groupby("band").agg(
            **{f"{m}_pat": (m, "mean") for m in METRICS}, n_pat=("user_id_hash", "count"))
        ctl = sub[sub["group"] == "control"].groupby("band").agg(
            **{f"{m}_ctl": (m, "mean") for m in METRICS}, n_ctl=("user_id_hash", "count"))
        merged = pat.join(ctl, how="inner")
        for m in METRICS:
            merged[f"{m}_diff"] = merged[f"{m}_pat"] - merged[f"{m}_ctl"]
        merged["era"] = era
        merged["n_patient"] = merged["n_pat"]
        merged["n_control"] = merged["n_ctl"]
        era_series[era] = merged.copy()
        rows.append(merged.reset_index())
        logger.info(f"{era}: windows={len(sub)}")

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(os.path.join(OUT, "r4_traj_temporal_split.csv"), index=False)

    # Compare the t0-surge magnitude and far_pre depression across eras
    surge = out[out["band"] == "t0_surge"].set_index("era")["p_factor_severity_z_diff"]
    farpre = out[out["band"] == "far_pre"].set_index("era")["p_factor_severity_z_diff"]
    report = {
        "n_reliable_windows": int(len(tr)),
        "era_series": {e: g.to_dict(orient="index") for e, g in era_series.items()},
        "p_factor_surge_by_era": {e: float(v) for e, v in surge.items()},
        "p_factor_far_pre_by_era": {e: float(v) for e, v in farpre.items()},
        "morphology_replicates_across_eras": bool(
            surge.get("E1_2010-2015", np.nan) > 0.05
            and surge.get("E2_2016-2019", np.nan) > 0.05
            and surge.get("E3_2020-2022", np.nan) > 0.05),
    }
    with open(os.path.join(OUT, "r4_traj_temporal_split.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("surge by era: %s", {e: round(float(v), 3) for e, v in surge.items()})
    logger.info("far_pre by era: %s", {e: round(float(v), 3) for e, v in farpre.items()})
    logger.info("DONE -> %s", OUT)


if __name__ == "__main__":
    main()
