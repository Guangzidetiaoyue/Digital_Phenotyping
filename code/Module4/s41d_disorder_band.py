"""
M4 - s41d: Per-disorder band-adjusted monthly diff
==================================================
Extends s41c to per-disorder: for each disorder, patient-control diff by
calendar month AND event-time band, plus band-adjusted (equal-weight across
stable bands) series. This removes composition bias from s43/s44.

Outputs:
  digital_signal_monthly_by_disorder_band.csv      : month x disorder x band diff
  digital_signal_monthly_by_disorder_band_adj.csv  : month x disorder band-adjusted diff
  digital_signal_quarterly_by_disorder_band_adj.csv
"""
import os
import logging
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = "/path/to/project"
TRAJ_PATH = os.path.join(PROJECT_ROOT, "module2_results", "trajectories", "user_trajectories.parquet")
OUT_DIR = os.path.join(PROJECT_ROOT, "module4_results", "data")

METRICS = ["p_factor_severity_z", "posting_intensity", "social_withdrawal_index", "volatility_index"]
READ_COLS = ["user_id_hash", "group", "disorder_group", "window_start_ts", "days_from_t0_center",
             "is_reliable"] + METRICS

BANDS = {
    "far_pre": (-720, -360), "mid_pre": (-360, -180), "near_pre": (-180, -90),
    "t0_surge": (-90, 90), "near_post": (90, 180), "mid_post": (180, 360), "far_post": (360, 720),
}
STABLE_BANDS = ["far_pre", "mid_pre", "near_pre", "near_post", "mid_post", "far_post"]


def assign_band(days):
    out = pd.Series("none", index=days.index)
    for name, (lo, hi) in BANDS.items():
        out[(days > lo) & (days <= hi)] = name
    return out


def main():
    logger.info("Reading trajectory...")
    pf = pq.ParquetFile(TRAJ_PATH)
    chunks = []
    for batch in pf.iter_batches(batch_size=500_000, columns=READ_COLS):
        df = batch.to_pandas()
        df = df[df["is_reliable"] == True]
        if len(df):
            chunks.append(df)
    tr = pd.concat(chunks, ignore_index=True)
    ts = pd.to_datetime(tr["window_start_ts"], unit="s", utc=True)
    tr["year"] = ts.dt.year
    tr["period"] = ts.dt.to_period("M").astype(str)
    tr["quarter"] = ts.dt.to_period("Q").astype(str)
    tr["band"] = assign_band(tr["days_from_t0_center"])
    tr = tr[(tr["year"] >= 2010) & (tr["year"] <= 2022)]
    logger.info(f"Reliable windows 2010-2022: {len(tr)}")

    pat = tr[tr["group"] == "patient"]
    ctrl = tr[tr["group"] == "control"]
    # control by period+band (pooled across disorders; same for all disorders)
    ctrl_g = ctrl.groupby(["period", "band"]).agg(
        **{f"{m}_ctrl": (m, "mean") for m in METRICS}).reset_index()

    # per disorder: patient by period+band
    pat_g = pat.groupby(["period", "disorder_group", "band"]).agg(
        **{f"{m}_pat": (m, "mean") for m in METRICS},
        n_patient=("user_id_hash", "count")).reset_index()

    merged = pat_g.merge(ctrl_g, on=["period", "band"], how="left")
    for m in METRICS:
        merged[f"{m}_diff"] = merged[f"{m}_pat"] - merged[f"{m}_ctrl"]
    merged.to_csv(os.path.join(OUT_DIR, "digital_signal_monthly_by_disorder_band.csv"), index=False)
    logger.info(f"By-disorder band monthly: {len(merged)} rows")

    # Band-adjusted per disorder (equal weight over stable bands per month)
    stable = merged[merged["band"].isin(STABLE_BANDS)]
    adj_rows = []
    for (period, disorder), g in stable.groupby(["period", "disorder_group"]):
        row = {"period": period, "disorder_group": disorder}
        for m in METRICS:
            row[f"{m}_diff"] = g[f"{m}_diff"].mean()
        row["n_bands"] = g["band"].nunique()
        adj_rows.append(row)
    adj = pd.DataFrame(adj_rows)
    adj.to_csv(os.path.join(OUT_DIR, "digital_signal_monthly_by_disorder_band_adj.csv"), index=False)
    logger.info(f"By-disorder band-adjusted monthly: {len(adj)} rows")

    # Quarterly band-adjusted
    merged["period_dt"] = pd.to_datetime(merged["period"])
    merged["quarter"] = merged["period_dt"].dt.to_period("Q").astype(str)
    stable_q = merged[merged["band"].isin(STABLE_BANDS)]
    adj_q_rows = []
    for (quarter, disorder), g in stable_q.groupby(["quarter", "disorder_group"]):
        row = {"quarter": quarter, "disorder_group": disorder}
        for m in METRICS:
            row[f"{m}_diff"] = g[f"{m}_diff"].mean()
        adj_q_rows.append(row)
    adj_q = pd.DataFrame(adj_q_rows)
    adj_q.to_csv(os.path.join(OUT_DIR, "digital_signal_quarterly_by_disorder_band_adj.csv"), index=False)
    logger.info(f"By-disorder band-adjusted quarterly: {len(adj_q)} rows")

    logger.info("DONE")


if __name__ == "__main__":
    main()
