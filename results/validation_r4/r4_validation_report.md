# §R4 Validation Programme — Execution Report

> 2026-09-01 · local run (pandas 2.3.3, numpy 2.0.2, scipy 1.13.1, statsmodels 0.14.6) + remote run (remote-host, conda my_py311, 24 cores)
> Purpose: run the reviewer-mandated validity checks (paper_outline_EN.md §R4).
> **Server layout (M1–M4-consistent):** code at `/path/to/code/validation_r4/` (sibling of Module1–4); results on NAS at `/path/to/project/module4_results/validation_r4/`. Quick jobs re-run on server (run 159bb887e31a4deb); Leakage A launched as long detached job.

---

## 1. R4-6 · Composition-Induced Temporal Bias (CITB) simulation ✅ (local)

`simulate_citb.py` → `citb_simulation_report.json`, `citb_simulation_results.csv`, series `citb_series_*_*.csv`

Generative model mirrors the observed cohort: 150k matched pairs, K=40 windows/pair, `days_from_t0 ~ N(0,300)` clipped ±720, t0-year composition from real 2008–2022 patient counts (2200× growth; simulated mean `days_from_t0` drifts −282 → +140, close to the observed −283 → +213). Window-level diff = event-time shape (far-pre below → surge near t0 → post) + `γ·calendar` + pair effect + noise. Three estimators: naive calendar mean, band-adjusted (6 stable bands, equal weight; excl. t0-surge), window-level OLS (cubic days + calendar).

| Scenario | true γ | naive slope (τ, p) | band-adjusted (τ, p) | window-reg γ̂ (bias) |
|---|---|---|---|---|
| A · true trend + drift | +0.030 | +0.0302 (τ=+0.98) | +0.0279 (τ=+0.98) | +0.0300 (0.0000) |
| **B · no trend, only drift** | 0 | **+0.0016 (τ=+0.77, p=1.0e−5) ✗ spurious-significant** | +0.0003 (τ=+0.16, ns) | +0.0000 (0.0000) |
| C · small trend + drift | +0.010 | +0.0120 (≈+20% inflated) | +0.0108 | +0.0100 (0.0000) |

**Interpretation.** Band-adjusted and window-level regression recover the ground-truth trend in all three scenarios; naive calendar aggregation produces a *spurious, statistically significant* trend under pure composition drift (B). With the real cohort's steeper t0 growth + event-time shape, this bias is strong enough to reverse the sign of a conclusion (observed: naive τ +0.31 → band-adjusted τ −0.31). → **The audit is a general method, not data-fitting** (reviewer point §7).

---

## 2. R4-7 · External-validity robustness suite ✅ (run on BOTH the naive and the corrected/band-adjusted series)

`external_robustness.py [signal_path]` → `external_robustness_report_{naive,corrected}.json`, `external_robustness_results_{naive,corrected}.csv`
The corrected series `digital_signal_annual_band_adjusted.csv` was fetched from the server (`s41c` output) and the full suite re-run on it.

**(a) Corrected (band-adjusted) series — the headline series:**

| signal vs external | n | raw r (p) | first-diff r | detrended r | HAC p | block-perm p | LOYO range | boot CI |
|---|---|---|---|---|---|---|---|---|
| **volatility vs suicide** | 13 | **+0.906 (<1e−3)** | **+0.511** | **+0.473** | <1e−3 | <0.001 | 0.029 | [0.77, 0.97] |
| posting vs suicide | 13 | −0.922 (<1e−3) | −0.079 | −0.391 | <1e−3 | 0.001 | 0.035 | [−0.97, −0.63] |
| withdrawal vs suicide | 13 | +0.485 (0.093) | +0.032 | +0.102 | 0.083 | 0.038 | 0.66 | [−0.10, 0.93] |
| withdrawal vs MDE | 8 | +0.719 (0.044) | −0.090 | +0.046 | <1e−3 | 0.084 | 0.27 | [−0.42, 0.93] |
| volatility vs MDE | 8 | +0.747 (0.033) | −0.326 | −0.340 | <1e−3 | 0.122 | 0.35 | [−0.26, 0.99] |
| posting vs MDE | 8 | −0.905 (0.002) | +0.073 | +0.250 | <1e−3 | 0.041 | 0.08 | [−1.00, −0.83] |

**Reading (corrected series).** The **volatility–suicide association SURVIVES common-trend control**: first-difference r=+0.51, detrended r=+0.47, block-permutation p<0.001, leave-one-year-out range 0.029 (no single year drives it), bootstrap CI excludes 0. This is the flagship, defensible external-validation result. The **posting–suicide** association is directionally robust (detrended −0.39, permutation p=0.001) but its magnitude is partly common-trend (first-diff r≈0). The **MDE pairs and withdrawal–suicide** are largely common-trend: after detrending they weaken to ≈0 → must be downplayed in the paper.

**(b) Naive (composition-unadjusted) series — methodological demonstration:**

| signal vs external | n | raw r (p) | first-diff r | detrended r |
|---|---|---|---|---|
| posting vs suicide | 15 | −0.561 (0.030) | +0.102 | −0.053 |
| withdrawal vs suicide | 15 | +0.614 (0.015) | −0.124 | +0.230 |
| volatility vs suicide | 15 | +0.396 (0.144) | +0.141 | −0.075 |
| volatility vs MDE | 8 | +0.813 (0.014) | −0.372 | −0.300 |

The naive series' raw correlations are largely trend artefacts (first-diff/detrended collapse) — confirming why the corrected series + trend control are mandatory.

---

## 2b. R4-3 · Temporal generalization at the trajectory scale ✅ (remote, run 49f17b530f5deacc)

`r4_traj_temporal_split.py` (run on server `remote-host`, my_py311) → `remote_trajsplit-20260901-160612/r4_traj_temporal_split.{csv,json}`
Event-study morphology (patient–control p-factor diff by event-time band) computed separately for three calendar eras from `user_trajectories.parquet` (17.45M reliable windows):

| Era | t0-surge diff | far-pre diff | volatility surge |
|---|---|---|---|
| E1 · 2010–2015 | +0.105 | −0.016 | +0.136 |
| E2 · 2016–2019 | +0.107 | −0.014 | +0.200 |
| E3 · 2020–2022 | +0.109 | −0.016 | +0.229 |

**Reading.** The individual-level phenotype around the self-report date is **not an era artefact**: the far-pre below-control pattern and the t0-surge replicate almost identically across three eras (surge 0.105/0.107/0.109). Notably, the *volatility* surge grows across eras (0.136 → 0.229), consistent with the M4 population finding that the volatility gap widens over calendar time — a coherent cross-module story.

### 2b-ii. Classifier-level temporal generalization ✅ (remote, run fc5d2b5236a737b4)

`r4_temporal_split_classifier.py`: single-disorder vs-control logistic (L2, balanced, C=1) **trained on users whose t0 falls in an earlier era**, evaluated out-of-era on later-era patients + full later-era controls (98 features, atlas feature space).

| Direction | mean Δ vs atlas | worst case |
|---|---|---|
| **train ≤2015 → test 2016–2019** | −0.005 | ocd −0.014; eating −0.010; autism −0.006 |
| **train 2016–2019 → test 2020–2022** | +0.010 | schizophrenia −0.006 (only negative); autism +0.021, adhd +0.020 |

**Reading.** A classifier trained only on pre-2016 users achieves atlas-level AUROC on 2016–2019 users (mean Δ −0.005; worst −0.014), and training on 2016–2019 transfers to 2020–2022 at or above atlas (mean Δ +0.010). **Digital phenotypes generalised across major Reddit eras at both the trajectory and classifier levels, arguing against an explanation based on era-specific platform artefacts** — this hedged wording is used in the paper (the era experiments rule out era-specific classifier/trajectory dependence, not all conceivable temporal confounding). Closes R4 #3.

---

## 2c. R4-1 · Leakage A ✅ (remote, run 3812cf57884e0224)

`r4_leakage_a_classify.py` (s04-matching protocol): single-disorder vs-control AUROC recomputed from user features built **excluding diagnosis-triggering posts** (`is_diagnosis_post == False`); controls subsampled 3×n_case, 5 trials, LogisticRegressionCV(lbfgs-L2); 91 features (of 93 in atlas; 2 undefined without diagnosis posts); 690,964 users after NaN-row drop.

| Disorder | n_case | nodiag AUROC | original AUROC | Δ |
|---|---|---|---|---|
| depression | 29,730 | 0.847 | 0.862 | **−0.015** |
| anxiety | 14,001 | 0.876 | 0.893 | **−0.017** |
| ocd | 5,575 | 0.883 | 0.884 | −0.001 |
| ptsd | 8,028 | 0.901 | 0.905 | −0.004 |
| bipolar | 9,714 | 0.908 | 0.903 | +0.006 |
| bpd | 5,891 | 0.924 | 0.922 | +0.002 |
| eating_disorders | 1,653 | 0.926 | 0.923 | +0.003 |
| schizophrenia | 2,385 | 0.880 | 0.878 | +0.002 |
| adhd | 36,430 | 0.836 | 0.835 | +0.001 |
| autism | 9,990 | 0.857 | 0.851 | +0.006 |

**Reading.** All ten disorders remain strongly discriminable (AUROC 0.84–0.93, mean 0.884) **without** any diagnosis-adjacent posts. Mean |Δ| = 0.006; the only meaningful drops are **anxiety (−0.017) and depression (−0.015)** (internalizing disorders whose discourse is most diagnosis-saturated); the remaining eight are unchanged or slightly higher. **Conclusion: classification is robust to diagnosis-language leakage** — part of the discriminability derives from explicit mental-health discourse, concentrated in depression/anxiety (~0.015–0.02 AUROC), but the core detectability claim survives leakage control. This directly de-risks the reviewer's top Module-1 validity threat.

---

## 3. Existing-evidence consolidation ✅ (local; re-used, no recompute)

`validation_summary.py` → `r4_existing_evidence_summary.json`

| Check | Result |
|---|---|
| **Leakage B / feature-family ablation** (individual level, `dimension_ablation.csv`) | all-dim AUROC 0.886; only-affective 0.825; only-cognitive 0.742; only-behavioral **0.666 (−0.219)**; only-social 0.653; only-temporal **0.597 (−0.288)** → no single family reproduces full AUROC; affective language is the strongest single signal, behaviour/temporal alone are markedly weaker |
| **R4-9 confounder adjustment** | correlation with original g = 0.989; mean abs diff 0.033 |
| **R4-9 sample-size stability** | AUROC at 25→100% subsample: max change ≤ 0.003 (all 10 disorders) |
| **R4-9 temporal stability** | 271/500 (54.2%) feature×period cells strongly monotone (|ρ|=1, p<1e−3); no direction flips |
| **R4-9 cross-disorder heterogeneity** | median I² = 97.3%; 100% classified "high" → magnitudes differ, directions consistent |
| **R4-5 negative control (M2 placebo)** | true DID 0.0161; placebo mean 0.0061 (sd 0.0437); 100/100 exceed true; p=1.0 → INCONCLUSIVE (already in summary; repackaged as negative-control/placebo) |
| **Multi-class separation** | macro AUROC 0.719; balanced acc 0.260 (weak; supports "shared/weakly separable" claim) |

---

## 4. Status of the full §R4 checklist

| # | Check | Status |
|---|---|---|
| 1 | Leakage A (strip diagnosis terms, retrain) | ✅ **done** (run 3812cf57884e0224; all 10 disorders AUROC 0.84–0.93, mean \|Δ\|=0.006; only anxiety/depression drop ~0.015–0.017) |
| 2 | Leakage B (behaviour+temporal+social only) | ✅ individual-level (`dimension_ablation.csv`: only-behavioural 0.666, only-temporal 0.597 vs 0.886 all-dim) |
| 3 | Leakage C / temporal generalization | ✅ **trajectory level** (surge 0.105/0.107/0.109 across eras) **+ classifier level** (train-era→test-era, mean Δ −0.005 / +0.010; run fc5d2b5236a737b4, §2b-ii) |
| 4 | Leakage D (diagnosis-discussion-free cohort) | 🅿️ deferred (SI-optional; heaviest) |
| 5 | Negative controls | ✅ M2 100-placebo DID = pseudo-T0 negative control (true 0.0161 vs placebo mean 0.0061, p=1.0 INCONCLUSIVE) — repackaged as negative control |
| 6 | **CITB simulation (A/B/C)** | ✅ done (local) |
| 7 | External robustness (detrend/first-diff/LOYO/permutation/lag/bootstrap) | ✅ **run on both naive and corrected (band-adjusted) series**; corrected-series findings in §2 |
| 8 | Feature-family decomposition at population scale | 🟡 by construction: corrected p-factor (severity/language composite) shows NO trend/external signal, while behavioural metrics (volatility, posting) carry the population signal → supports "behaviour > language at population scale" (no new compute needed) |
| 9 | Effect-size reproducibility (subsampling/confounder/temporal/meta) | ✅ done (existing sensitivity files consolidated) |

**Server jobs completed:** corrected-series external robustness, trajectory temporal split (run 49f17b530f5deacc), quick-run set (run 159bb887e31a4deb), Leakage A feature extraction + classifier stage (run 3812cf57884e0224), **classifier temporal split (run fc5d2b5236a737b4)**. **Deferred/optional:** Leakage D (diagnosis-discussion-free cohort, SI-optional).

---

## 5. Bottom line for the paper (rev.3 outline)

1. **CITB audit is now simulation-validated** (R4-6) → can be presented as a general methodological contribution, not data-fitting (supports §R3.1, Fig. 6).
2. **External validity on the corrected series** (R4-7): the **volatility–suicide association survives first-differencing (r=+0.51), detrending (+0.47), block-permutation, and leave-one-year-out** → this is the flagship, defensible external-validation result. **posting–suicide** is directionally robust but partly trend; **MDE and withdrawal pairs are largely common-trend** and must be hedged/downplayed. The headline claim in §R3.4 is therefore updated: volatility is the robust population monitor; raw r=0.91 survives trend control (first-diff +0.51, detrended +0.47).
3. **Temporal generalization at both scales** (R4-3): trajectory morphology (t0-surge +0.105–0.109) replicates across eras, and a classifier trained on pre-2016 users achieves atlas-level AUROC on later eras (mean Δ −0.005; worst −0.014 ocd) → digital phenotypes are not a single-Reddit-era artefact; supports the "reproducible" claim and the can/cannot table (§2b, §2b-ii).
4. **Leakage A is now directly measured**: excluding diagnosis-triggering posts leaves AUROC 0.84–0.93 (mean 0.884 vs 0.886); only anxiety/depression drop ~0.015–0.017 → the paper can state the discriminability claim survives leakage control, with a modest diagnosis-adjacent-language contribution concentrated in internalizing disorders (§2c). Leakage B (feature-family) shows affective language dominates and behaviour/temporal families are weaker (AUROC 0.60–0.67), consistent with "partly driven by diagnosis-adjacent language".
5. **Population feature-family** (R4-8, by construction): corrected p-factor (language/severity composite) has no trend and no robust external correlation, while behavioural metrics (volatility↑, posting↓) carry the population signal → **"behaviour > language at population scale"** is already supported by existing corrected outputs.
6. Reproducibility evidence (confounder r=0.989, sample-size ≤0.003, temporal 54% strong-monotone, I² 97%) is consolidated for the "small but systematic" framing.
