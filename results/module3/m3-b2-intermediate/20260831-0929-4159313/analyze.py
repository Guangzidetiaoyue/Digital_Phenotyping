import json
rep = json.load(open('results/module3/m3-b2-intermediate/20260831-0929-4159313/comorbidity_intermediate_report.json'))
print('=== Aggregate ===')
for k, v in rep['aggregate'].items():
    print(f'  {k}: {v}')
print()
print('=== Significant smaller-residual pairs (FDR<0.05) ===')
for r in rep['per_pair']:
    if r.get('p_res_mw_fdr', 1) < 0.05:
        print(f"  {r['disorder_a']}-{r['disorder_b']}: frac_int={r['frac_intermediate']:.3f}, res_ratio={r['res_ratio']:.3f}, p_res_fdr={r['p_res_mw_fdr']:.4f}")
print()
print('=== Pairs with frac_intermediate > 0.4 ===')
for r in rep['per_pair']:
    if r['frac_intermediate'] > 0.4:
        print(f"  {r['disorder_a']}-{r['disorder_b']}: {r['frac_intermediate']:.3f}, n={r['n_dual']}")
