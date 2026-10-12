# s31 Mantel with externally sourced comorbidity (executed 2026-09-13)

## What ran
- Input: `references_docs/external_comorbidity_or_subset.csv` (21 disorder-level,
  term-mappable **adjusted/pooled ORs**; eating pairs fixed to BED).
  Converted to s31's input `external_comorbidity.csv`
  (`disorder_a,disorder_b,comorbidity_pct,source,ref` + provenance columns;
  `comorbidity_pct` holds the OR, documented by the `measure` column).
- `run_s31_external_or.sh` → run **run_f52ba8c45c08281b** (full s31; deterministic
  AUROC, seed 42; 45 pairs, 108 features; 127,168 patients).
- `run_s31c.sh` → run **run_338f0488a9eb8a34** (log-OR / rank sensitivity over the
  saved AUROC matrix; also recomputes the uncited matrix and cross-checks s31).

## Headline result
| matrix | pairs | Mantel r | Mantel p | Spearman r | Spearman p |
|---|---|---|---|---|---|
| **External, cited OR-only (primary)** | 21 | **−0.247** | **0.297** | **−0.095** | **0.683** |
| log(OR), same 21 pairs | 21 | −0.244 | 0.302 | −0.095 | 0.683 |
| Uncited in-code % matrix | 45 | −0.363 | 0.084 | −0.412 | 0.005 |
| Empirical same-cohort Jaccard (s31b) | 45 | +0.401 (1−AUROC) | 0.008 | — | — |

Direction is as hypothesised (higher comorbidity ↔ lower AUROC) but **not
significant** on the cited subset. AUROC matrix is identical to the pre-run baseline
(mean AUROC 0.7409844), so the change is entirely the external matrix. The s31 main
result and s31c's independent recomputation agree to 4 decimals.

## Files
- `log`, `exit_code`, `env_s31ext.txt`, `run_s31_external_or.sh`
- `s31_pairwise_distinguishability.py` (exact server code used)
- `pairwise_distinguishability_report.json`, `pairwise_distinguishability.csv`
- `auroc_matrix.npy`, `auprc_matrix.npy`
- `external_comorbidity.csv` (input as read)
- sensitivity dir: `../m3-external-or-s31c-20260913/` (JSON, log, env, both scripts)

## Coverage
21/45 pairs have a cited OR. 5 pairs missing entirely; autism and BPD pairs exist
only as prevalence/co-occurrence %; bipolar pairs partially use the NCS-R
manic/hypomanic (MHE) proxy. See `references_docs/comorbidity_extraction_report.md`
for the full exclusion table and verification status.
