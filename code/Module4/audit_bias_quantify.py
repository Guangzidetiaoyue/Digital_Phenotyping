"""
CRITICAL AUDIT 2: Quantify how much of the M4 calendar trend is composition bias.
If p_factor differs by days_from_t0 (which M2 event study shows: pre<post),
then a calendar-year with more post-diagnosis windows will show higher p_factor
diff, purely from composition, not calendar-time change.
"""
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

TRAJ = "/path/to/project/module2_results/trajectories/user_trajectories.parquet"
COLS = ["user_id_hash","group","disorder_group","window_start_ts","days_from_t0_center",
        "is_reliable","p_factor_severity_z","posting_intensity","social_withdrawal_index"]

pf = pq.ParquetFile(TRAJ)
chunks = []
for batch in pf.iter_batches(batch_size=500_000, columns=COLS):
    chunks.append(batch.to_pandas())
tr = pd.concat(chunks, ignore_index=True)
ts = pd.to_datetime(tr["window_start_ts"], unit="s", utc=True)
tr["year"] = ts.dt.year
rel = tr[tr["is_reliable"] == True]

print("=== A. p_factor by days_from_t0 (event-time shape, M2 known) ===")
bins = [-720,-360,-180,-90,-45,0,45,90,180,360,720]
rel["t0_bin"] = pd.cut(rel["days_from_t0_center"], bins=bins)
shape = rel.groupby(["t0_bin","group"])["p_factor_severity_z"].mean().unstack()
shape["diff"] = shape.get("patient",0) - shape.get("control",0)
print(shape.round(4).to_string())

print("\n=== B. p_factor diff by calendar year (as s42 computed) ===")
# patient - control within year
pat = rel[rel["group"]=="patient"].groupby("year")["p_factor_severity_z"].mean()
ctrl = rel[rel["group"]=="control"].groupby("year")["p_factor_severity_z"].mean()
diff = (pat - ctrl)
print(diff.loc[2008:2022].round(4).to_string())

print("\n=== C. p_factor diff by year, RESTRICTED to same days_from_t0 band ===")
# e.g. only post-diagnosis windows (days_from_t0 > 90): does year trend persist?
post = rel[rel["days_from_t0_center"] > 90]
pre = rel[rel["days_from_t0_center"] < -90]
for label, sub in [("post-only", post), ("pre-only", pre)]:
    pat_s = sub[sub["group"]=="patient"].groupby("year")["p_factor_severity_z"].mean()
    ctrl_s = sub[sub["group"]=="control"].groupby("year")["p_factor_severity_z"].mean()
    d = (pat_s - ctrl_s)
    print(f"\n  {label}: year diff 2015={d.get(2015):.4f}, 2019={d.get(2019):.4f}, 2020={d.get(2020):.4f}, 2022={d.get(2022):.4f}")
    print(f"  {label} mean overall: {d.mean():.4f}")
