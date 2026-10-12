import pandas as pd
import numpy as np

res = pd.read_csv('results/module4/s44-events/20260831-2049/event_study_results.csv')
print("=== Pooled CAR by event (all disorders) ===")
pool = res[res['disorder']=='ALL'][['event','event_label','car','se_car','t_car','p_car','p_car_fdr']]
print(pool.to_string(index=False))

print("\n=== Significant per-disorder CAR (FDR<0.05) ===")
sig = res[res['p_car_fdr']<0.05][['event','disorder','car','se_car','t_car','p_car_fdr']]
print(sig.to_string(index=False))

print("\n=== Event x disorder CAR matrix (pooled excluded) ===")
piv = res[res['disorder']!='ALL'].pivot(index='disorder', columns='event', values='car')
piv_sig = res[res['disorder']!='ALL'].pivot(index='disorder', columns='event', values='p_car_fdr')
print("CAR:")
print(piv.round(4).to_string())
print("\nFDR p-values (<0.05 marked *):")
print(piv_sig.round(3).applymap(lambda x: '*' if x<0.05 else '.').to_string())
