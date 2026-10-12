"""
R4: consolidate existing evidence for the validity programme
=============================================================
Sources (all local, already computed):
  - dimension_ablation.csv   : per-disorder AUROC by feature family (leakage B)
  - sensitivity_report.json  : confounder adjustment + sample-size stability (R4-9)
  - sensitivity_temporal.csv : temporal-split stability (R4-9)
  - sensitivity_heterogeneity.csv : cross-disorder meta I^2 (R4-9)
  - placebo_test_results.json     : DID placebo / negative control (R4-5, M2)
  - multiclass_results.json / discrimination_report.json : multi-class weak separation

Output: r4_existing_evidence_summary.json + console summary
"""
import json
import os
import numpy as np
import pandas as pd

OUT_DIR = os.environ.get("R4_OUT", os.path.dirname(os.path.abspath(__file__)))
os.makedirs(OUT_DIR, exist_ok=True)
M1_STAT = os.environ.get(
    "R4_M1_STAT",
    os.path.normpath(os.path.join(OUT_DIR, "..", "module1_results", "statistics")))
M2_STAT = os.environ.get(
    "R4_M2_STAT",
    os.path.normpath(os.path.join(OUT_DIR, "..", "module2_results", "statistics")))


def load(p):
    return os.path.join(M1_STAT, p)


def main():
    out = {}

    # ---- Leakage B / feature-family ablation (individual level) ----
    abl = pd.read_csv(load("dimension_ablation.csv"), index_col=0)
    cols = ["all_dimensions", "only_affective", "only_pronoun", "only_cognitive",
            "only_behavioral", "only_readability", "only_temporal", "only_social"]
    fam = abl[[c for c in cols if c in abl.columns]].copy()
    out["feature_family_auroc"] = {
        "mean_across_disorders": {c: float(fam[c].mean()) for c in fam.columns},
        "delta_from_all": {c: float(fam[c].mean() - fam["all_dimensions"].mean()) for c in fam.columns},
        "per_disorder": fam.to_dict(orient="index"),
    }
    print("[feature-family] mean AUROC across 10 disorders:")
    for c in fam.columns:
        print(f"   {c:20s} {fam[c].mean():.3f}  (delta {fam[c].mean()-fam['all_dimensions'].mean():+.3f})")

    # ---- Effect-size reproducibility (R4-9) ----
    with open(load("sensitivity_report.json")) as f:
        sen = json.load(f)
    conf = sen["confound_adjustment"]
    out["confounder_adjustment"] = conf
    print(f"[confounder] corr_with_original={conf['correlation_with_original']:.3f} "
          f"mean_abs_diff={conf['mean_abs_difference']:.3f}")

    sss = sen["sample_size_stability"]
    sss_sum = {d: {fr: round(v["mean_auroc"], 3) for fr, v in dd.items()} for d, dd in sss.items()}
    out["sample_size_stability_auroc"] = sss_sum
    print("[sample-size] AUROC at 0.25/0.5/0.75/1.0:")
    for d, dd in sss_sum.items():
        print(f"   {d:10s} {dd}")

    tmp = pd.read_csv(load("sensitivity_temporal.csv"))
    # % features with strong monotone trend (|rho|=1, p<0.001)
    strong = tmp[(tmp["trend_rho"].abs() == 1) & (tmp["trend_p"] < 0.001)]
    out["temporal_stability"] = {
        "n_rows": int(len(tmp)), "n_strong_monotone": int(len(strong)),
        "frac_strong_monotone": round(float(len(strong) / len(tmp)), 3),
    }
    print(f"[temporal] {len(strong)}/{len(tmp)} ({len(strong)/len(tmp):.1%}) "
          f"feature-periods strong monotone (|rho|=1, p<1e-3)")

    het = pd.read_csv(load("sensitivity_heterogeneity.csv"))
    out["heterogeneity"] = {
        "median_I2": float(het["I_squared"].median()),
        "pct_high_heterogeneity": float((het["heterogeneity_class"] == "high").mean()),
        "n_features": int(len(het)),
    }
    print(f"[heterogeneity] median I^2={het['I_squared'].median():.1f}%, "
          f"{(het['heterogeneity_class']=='high').mean():.0%} classified high")

    # ---- Negative control / placebo (M2) ----
    with open(os.path.join(M2_STAT, "placebo_test_results.json")) as f:
        plc = json.load(f)
    out["did_placebo_negative_control"] = plc
    print(f"[placebo/negative-control] true DID={plc['true_did']:.4f}, "
          f"placebo mean={plc['placebo_did_mean']:.4f} sd={plc['placebo_did_std']:.4f}, "
          f"n_exceeding={plc['n_exceeding_true']}, p={plc['placebo_p_value']}, "
          f"conclusion={plc['conclusion']}")

    # ---- Multi-class weak separation ----
    with open(load("multiclass_results.json")) as f:
        mc = json.load(f)
    out["multiclass"] = {"macro_auroc": mc.get("macro_auroc_mean"), "balanced_acc": mc.get("balanced_accuracy_mean")}
    print(f"[multiclass] macro AUROC={mc.get('macro_auroc_mean'):.3f}, "
          f"bal acc={mc.get('balanced_accuracy_mean'):.3f}")

    with open(os.path.join(OUT_DIR, "r4_existing_evidence_summary.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("DONE ->", os.path.join(OUT_DIR, "r4_existing_evidence_summary.json"))


if __name__ == "__main__":
    main()
