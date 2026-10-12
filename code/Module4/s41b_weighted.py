"""
s41b: Regenerate per-disorder monthly with window-count weights.
(s41 produced by_disorder_diff without sample sizes; s43 needs weights.)
Output: digital_signal_monthly_by_disorder_weighted.csv
"""
import os
import logging
import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = "/path/to/project"
TRAJ_PATH = os.path.join(PROJECT_ROOT, "module2_results", "trajectories", "user_trajectories.parquet")
OUT_DIR = os.path.join(PROJECT_ROOT, "module4_results", "data")

READ_COLS = ["user_id_hash", "group", "disorder_group", "window_start_ts", "is_reliable",
             "p_factor_severity_z", "posting_intensity", "social_withdrawal_index", "volatility_index"]
METRICS = ["p_factor_severity_z", "posting_intensity", "social_withdrawal_index", "volatility_index"]

def main():
    logger.info("Reading trajectory...")
    pf = pq.ParquetFile(TRAJ_PATH)
    chunks = []
    for batch in pf.iter_batches(batch_size=500_000, columns=READ_COLS):
        df = batch.to_pandas()
        df = df[df["is_reliable"] == True]
        if len(df): chunks.append(df)
    tr = pd.concat(chunks, ignore_index=True)
    ts = pd.to_datetime(tr["window_start_ts"], unit="s", utc=True)
    tr["period"] = ts.dt.to_period("M").astype(str)
    tr = tr[(tr["period"] >= "2015-01") & (tr["period"] <= "2022-12")]
    logger.info(f"Reliable windows 2015-2022: {len(tr)}")

    # Per disorder: patient mean + n_patient + control mean + n_control
    pat = tr[tr["group"] == "patient"]
    ctrl = tr[tr["group"] == "control"]
    pat_g = pat.groupby(["period", "disorder_group"]).agg(
        **{f"{m}_pat_mean": (m, "mean") for m in METRICS},
        n_patient=("user_id_hash", "count"))
    ctrl_g = ctrl.groupby("period").agg(
        **{f"{m}_ctrl_mean": (m, "mean") for m in METRICS},
        n_control=("user_id_hash", "count"))
    out = pat_g.reset_index().merge(ctrl_g.reset_index(), on="period", how="left")
    for m in METRICS:
        out[f"{m}_diff"] = out[f"{m}_pat_mean"] - out[f"{m}_ctrl_mean"]
    out.to_csv(os.path.join(OUT_DIR, "digital_signal_monthly_by_disorder_weighted.csv"), index=False)
    logger.info(f"Saved weighted by-disorder monthly: {len(out)} rows, cols={out.columns.tolist()}")

if __name__ == "__main__":
    main()
