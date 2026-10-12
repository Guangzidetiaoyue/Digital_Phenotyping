"""
M4 - Step 41: Calendar-Time Digital Signal Aggregation
=======================================================
Aggregate Reddit digital phenotype signals by CALENDAR time (absolute dates),
as the macro-epidemiology foundation of M4. This differs from M2's
event-time (t0-aligned) analysis.

Inputs : module2_results/trajectories/user_trajectories.parquet
Outputs: module4_results/data/
  digital_signal_{monthly,quarterly,annual}_pooled.csv   (group means + counts)
  digital_signal_{monthly,quarterly,annual}_diff.csv     (patient - control)
  digital_signal_monthly_by_disorder.csv                 (per-disorder patient means)
  digital_signal_summary.json

Design (scientific rigor):
  - Use ONLY reliable windows (is_reliable==True) to avoid MNAR-imputed pseudo
    values contaminating population-level trends.
  - Calendar month from window_start_ts (absolute epoch -> UTC).
  - Patient - control DIFFERENCE per period removes platform/user-structure
    trends (control = platform baseline), same logic as M2 event study.
  - Main window 2008-2022 (post-2022 reliable rate collapses).
  - n_windows reported for downstream weighting / low-sample exclusion.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = "/path/to/project"
TRAJ_PATH = os.path.join(PROJECT_ROOT, "module2_results", "trajectories", "user_trajectories.parquet")
OUT_DIR = os.path.join(PROJECT_ROOT, "module4_results", "data")
os.makedirs(OUT_DIR, exist_ok=True)

MAIN_YEAR_START = 2008
MAIN_YEAR_END = 2022
METRICS = ["p_factor_severity_z", "posting_intensity",
           "social_withdrawal_index", "volatility_index"]
READ_COLS = ["user_id_hash", "group", "disorder_group", "window_start_ts",
             "is_reliable"] + METRICS


def load_reliable():
    logger.info("Reading trajectory (reliable windows only)...")
    pf = pq.ParquetFile(TRAJ_PATH)
    chunks = []
    for batch in pf.iter_batches(batch_size=500_000, columns=READ_COLS):
        df = batch.to_pandas()
        df = df[df["is_reliable"] == True]
        if len(df) > 0:
            chunks.append(df)
    tr = pd.concat(chunks, ignore_index=True)
    del chunks
    logger.info(f"Reliable windows: {len(tr)}")
    ts = pd.to_datetime(tr["window_start_ts"], unit="s", utc=True)
    tr["year"] = ts.dt.year
    tr["period"] = ts.dt.to_period("M").astype(str)
    tr["quarter"] = ts.dt.to_period("Q").astype(str)
    return tr


def pool_and_diff(tr, period_col, prefix):
    """Group means by (period, group); compute patient-control diff."""
    pooled = tr.groupby([period_col, "group"])[METRICS].agg(["mean", "count"])
    pooled.columns = ["_".join(c).strip("_") for c in pooled.columns]
    pooled = pooled.reset_index()
    pooled_path = os.path.join(OUT_DIR, f"digital_signal_{prefix}_pooled.csv")
    pooled.to_csv(pooled_path, index=False)

    piv = pooled.pivot(index=period_col, columns="group")
    diff = pd.DataFrame(index=piv.index)
    for m in METRICS:
        col = (f"{m}_mean", "patient")
        ccol = (f"{m}_mean", "control")
        if col in piv.columns and ccol in piv.columns:
            diff[f"{m}_diff"] = piv[col] - piv[ccol]
    # sample size per period (from p_factor count col)
    npc = (f"{METRICS[0]}_count", "patient")
    ncc = (f"{METRICS[0]}_count", "control")
    if npc in piv.columns:
        diff["n_patient"] = piv[npc]
    if ncc in piv.columns:
        diff["n_control"] = piv[ncc]
    diff = diff.reset_index()
    diff_path = os.path.join(OUT_DIR, f"digital_signal_{prefix}_diff.csv")
    diff.to_csv(diff_path, index=False)
    logger.info(f"Saved {prefix} pooled ({len(pooled)}) + diff ({len(diff)})")
    return diff


def main():
    tr = load_reliable()
    tr = tr[(tr["year"] >= MAIN_YEAR_START) & (tr["year"] <= MAIN_YEAR_END)]
    logger.info(f"Main window {MAIN_YEAR_START}-{MAIN_YEAR_END}: {len(tr)} reliable windows")

    # Monthly (all disorders pooled)
    diff_m = pool_and_diff(tr, "period", "monthly")

    # Per-disorder monthly (patient means + patient-control diff per disorder)
    pat_dis = tr[tr["group"] == "patient"].groupby(["period", "disorder_group"])[METRICS].mean().reset_index()
    pat_dis.to_csv(os.path.join(OUT_DIR, "digital_signal_monthly_by_disorder.csv"), index=False)
    logger.info(f"Per-disorder monthly: {len(pat_dis)} rows")

    # Per-disorder patient-control diff: for each disorder, control = all controls
    ctrl = tr[tr["group"] == "control"].groupby("period")[METRICS].mean()
    dis_rows = []
    for d in sorted(tr[tr["group"] == "patient"]["disorder_group"].dropna().unique()):
        pd_ = tr[(tr["group"] == "patient") & (tr["disorder_group"] == d)] \
            .groupby("period")[METRICS].mean()
        for per in pd_.index:
            if per in ctrl.index:
                row = {"period": per, "disorder_group": d}
                for m in METRICS:
                    row[f"{m}_diff"] = pd_.loc[per, m] - ctrl.loc[per, m]
                dis_rows.append(row)
    dis_diff = pd.DataFrame(dis_rows)
    dis_diff.to_csv(os.path.join(OUT_DIR, "digital_signal_monthly_by_disorder_diff.csv"), index=False)
    logger.info(f"Per-disorder diff: {len(dis_diff)} rows")

    # Quarterly & annual pooled + diff
    pool_and_diff(tr, "quarter", "quarterly")
    pool_and_diff(tr, "year", "annual")

    summary = {
        "n_reliable_windows": int(len(tr)),
        "main_window": [MAIN_YEAR_START, MAIN_YEAR_END],
        "n_months": int(tr["period"].nunique()),
        "n_quarters": int(tr["quarter"].nunique()),
        "n_years": int(tr["year"].nunique()),
        "output_dir": OUT_DIR,
    }
    with open(os.path.join(OUT_DIR, "digital_signal_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("DONE")


if __name__ == "__main__":
    main()
