"""
M4 - s41c: Corrected calendar aggregation (control for event-time composition)
=============================================================================
FIX for window-composition bias: M2 windows are t0-aligned; aggregating by
calendar year mixes different days_from_t0 windows (early years = pre-t0,
late years = post-t0), creating spurious trends (t0±45d p-factor surge).

Correction: aggregate calendar series STRATIFIED by event-time band, and
report the composition-adjusted signal. Two strategies:
  (1) Event-time band stratification: separate series for pre-t0, near-t0,
      post-t0 windows per calendar period.
  (2) Balanced-composition estimate: for each calendar period, compute the
      patient-control diff using ONLY windows in a fixed event-time band
      (e.g. t0-360..-90 "stable pre" and t0+90..+360 "stable post"), so the
      calendar comparison is apples-to-apples within a band.

Outputs (module4_results/data/):
  digital_signal_monthly_by_band.csv      : month x band x group means
  digital_signal_monthly_by_band_diff.csv : month x band patient-control diff
  digital_signal_annual_by_band_diff.csv  : annual x band diff (for s42)
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

# Event-time bands (days_from_t0_center)
BANDS = {
    "far_pre": (-720, -360),
    "mid_pre": (-360, -180),
    "near_pre": (-180, -90),
    "t0_surge": (-90, 90),     # includes the transient t0 surge
    "near_post": (90, 180),
    "mid_post": (180, 360),
    "far_post": (360, 720),
}
# Stable bands (exclude t0 surge) for primary corrected analysis
STABLE_BANDS = ["far_pre", "mid_pre", "near_pre", "near_post", "mid_post", "far_post"]


def assign_band(days):
    """Vectorized band assignment."""
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
    tr["band"] = assign_band(tr["days_from_t0_center"])
    tr = tr[(tr["year"] >= 2010) & (tr["year"] <= 2022)]  # 2010+ adequate samples
    logger.info(f"Reliable windows 2010-2022: {len(tr)}")

    # Monthly by band (patient mean, control mean)
    pat = tr[tr["group"] == "patient"]
    ctrl = tr[tr["group"] == "control"]
    pat_g = pat.groupby(["period", "band"]).agg(
        **{f"{m}_pat": (m, "mean") for m in METRICS}, n_patient=("user_id_hash", "count"))
    ctrl_g = ctrl.groupby(["period", "band"]).agg(
        **{f"{m}_ctrl": (m, "mean") for m in METRICS}, n_control=("user_id_hash", "count"))
    out = pat_g.reset_index().merge(ctrl_g.reset_index(), on=["period", "band"], how="outer")
    for m in METRICS:
        out[f"{m}_diff"] = out[f"{m}_pat"] - out[f"{m}_ctrl"]
    out.to_csv(os.path.join(OUT_DIR, "digital_signal_monthly_by_band.csv"), index=False)
    logger.info(f"Monthly by band: {len(out)} rows")

    # Annual by band (for s42 corrected)
    pat_a = pat.groupby(["year", "band"]).agg(
        **{f"{m}_pat": (m, "mean") for m in METRICS}, n_patient=("user_id_hash", "count"))
    ctrl_a = ctrl.groupby(["year", "band"]).agg(
        **{f"{m}_ctrl": (m, "mean") for m in METRICS}, n_control=("user_id_hash", "count"))
    out_a = pat_a.reset_index().merge(ctrl_a.reset_index(), on=["year", "band"], how="outer")
    for m in METRICS:
        out_a[f"{m}_diff"] = out_a[f"{m}_pat"] - out_a[f"{m}_ctrl"]
    out_a.to_csv(os.path.join(OUT_DIR, "digital_signal_annual_by_band.csv"), index=False)
    logger.info(f"Annual by band: {len(out_a)} rows")

    # Composition-adjusted annual diff: average over stable bands (equal weight),
    # i.e. what the diff would be if each band contributed equally each year.
    stable = out_a[out_a["band"].isin(STABLE_BANDS)]
    adj = stable.groupby("year").apply(
        lambda g: pd.Series({
            f"{m}_diff": g[f"{m}_diff"].mean() for m in METRICS  # equal-weight across bands
        }), include_groups=False).reset_index()
    adj.to_csv(os.path.join(OUT_DIR, "digital_signal_annual_band_adjusted.csv"), index=False)
    logger.info("Annual band-adjusted diff:")
    cols_show = [c for c in [f"{m}_diff" for m in METRICS] if c in adj.columns]
    logger.info(adj[["year"] + cols_show].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
