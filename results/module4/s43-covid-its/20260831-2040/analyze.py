import pandas as pd
import numpy as np

res = pd.read_csv('results/module4/s43-covid-its/20260831-2040/covid_its_results.csv')
print("=== COVID ITS: p_factor (patient-control diff) ===")
pf = res[res['metric']=='p_factor_severity_z_diff'].copy()
pf = pf.sort_values('b_post_shift', ascending=False)
print(pf[['disorder','b_post_shift','p_post_shift_fdr','b_post_trend','p_post_trend_fdr','b_trend','rsquared']].to_string(index=False))

print("\n=== COVID ITS: all metrics - significant shifts (FDR<0.05) ===")
sig = res[res['p_post_shift_fdr']<0.05][['disorder','metric','b_post_shift','p_post_shift_fdr']]
print(sig.to_string(index=False))

print("\n=== Significant post-trends (FDR<0.05) ===")
sig_t = res[res['p_post_trend_fdr']<0.05][['disorder','metric','b_post_trend','p_post_trend_fdr']]
print(sig_t.to_string(index=False))

print("\n=== Phase means (p_factor diff, by disorder) ===")
ph = pd.read_csv('results/module4/s43-covid-its/20260831-2040/covid_its_phases.csv')
ph_pf = ph[ph['metric']=='p_factor_severity_z_diff'].pivot(index='disorder', columns='phase', values='mean_diff')
print(ph_pf.round(4).to_string())
