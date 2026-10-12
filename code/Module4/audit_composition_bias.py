"""
CRITICAL AUDIT: Check if calendar aggregation introduces composition bias.
M2 windows are t0-aligned (days_from_t0 -720..+720). When aggregated by
calendar year, if the t0 (diagnosis) date distribution is non-uniform across
years, then early years = mostly pre-diagnosis windows, late years = mostly
post-diagnosis windows. This would CREATE spurious calendar trends.
"""
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

TRAJ = "/path/to/project/module2_results/trajectories/user_trajectories.parquet"
COLS = ["user_id_hash","group","disorder_group","window_start_ts","days_from_t0_center",
        "is_reliable","p_factor_severity_z"]

pf = pq.ParquetFile(TRAJ)
chunks = []
for batch in pf.iter_batches(batch_size=500_000, columns=COLS):
    chunks.append(batch.to_pandas())
tr = pd.concat(chunks, ignore_index=True)
ts = pd.to_datetime(tr["window_start_ts"], unit="s", utc=True)
tr["year"] = ts.dt.year
rel = tr[tr["is_reliable"] == True]

print("=== A. t0 (diagnosis) date distribution by year (patient) ===")
# t0 = window_start_ts - days_from_t0 (approx, since days_from_t0 is window START)
# better: t0 is per-user constant. t0_timestamp col not loaded; approximate:
# t0 ≈ window_start_ts - days_from_t0*86400
t0_est = tr["window_start_ts"] - tr["days_from_t0_center"]*86400
t0_est_ts = pd.to_datetime(t0_est, unit="s", utc=True)
pat = tr[tr["group"]=="patient"]
t0_pat = t0_est_ts[tr["group"]=="patient"]
t0_year = t0_pat.dt.year
print(t0_year.value_counts().sort_index().loc[2008:2022].to_string())

print("\n=== B. Window composition by calendar year (patient, reliable) ===")
rel_pat = rel[rel["group"]=="patient"]
# average days_from_t0_center per calendar year
comp = rel_pat.groupby("year")["days_from_t0_center"].agg(["mean","median","count"])
print(comp.loc[2008:2022].round(1).to_string())
print("\n  -> If mean days_from_t0 shifts over years, composition bias exists.")

print("\n=== C. Reliable rate by year (potential bias) ===")
yr = tr.groupby("year")["is_reliable"].mean()
print(yr.loc[2008:2022].round(3).to_string())

print("\n=== D. Control: days_from_t0 by year (should be ~0, t0-matched) ===")
rel_ctrl = rel[rel["group"]=="control"]
compc = rel_ctrl.groupby("year")["days_from_t0_center"].agg(["mean","median"])
print(compc.loc[2008:2022].round(1).to_string())
