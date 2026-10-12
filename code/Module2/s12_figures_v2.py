"""
Module 2 Publication Figures (v2) - Nature/Science-grade
==========================================================
Main figures -> module2_results/figures/main/
SI figures    -> module2_results/figures/ed/

Consistent design system:
  - Arial sans-serif, base 7pt, titles 8pt, ticks 6pt
  - Thin 0.8pt axes, no top/right spines
  - Okabe-Ito / Nature color palette
  - PDF (vector) + 300dpi PNG
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
from scipy import stats

# ---------- paths ----------
PROJECT_ROOT = "/path/to/project"
M2_STATS = os.path.join(PROJECT_ROOT, "module2_results", "statistics")
M2_TRAJ = os.path.join(PROJECT_ROOT, "module2_results", "trajectories")
M2_FIGS = os.path.join(PROJECT_ROOT, "module2_results", "figures")
MAIN_DIR = os.path.join(M2_FIGS, "main")
ED_DIR = os.path.join(M2_FIGS, "ed")
os.makedirs(MAIN_DIR, exist_ok=True)
os.makedirs(ED_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------- style ----------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#333333",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 3,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

PAT_CTRL = {"patient": "#E64B35", "control": "#4DBBD5"}
SUB_COLORS = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F"]
DISPLAY = {
    "depression": "Depression", "bipolar": "Bipolar", "anxiety": "Anxiety",
    "ocd": "OCD", "ptsd": "PTSD", "bpd": "BPD", "eating_disorders": "Eating D.",
    "schizophrenia": "Schizophrenia", "adhd": "ADHD", "autism": "Autism",
    "control": "Control", "all": "All",
}

def _save(fig, name, subdir="main"):
    d = MAIN_DIR if subdir == "main" else ED_DIR
    for fmt in ("pdf", "png"):
        fig.savefig(os.path.join(d, f"{name}.{fmt}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {subdir}/{name} (pdf+png)")

def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
        ax.tick_params(**{s: False})
    ax.tick_params(direction="out")

def add_panel(ax, label, x=-0.13, y=1.06):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="top", ha="right")

def _load_csv(name, **kw):
    p = os.path.join(M2_STATS, name)
    if not os.path.exists(p):
        logger.warning(f"  Missing: {name}")
        return pd.DataFrame()
    return pd.read_csv(p, **kw)

def _load_json(name):
    p = os.path.join(M2_STATS, name)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


# =====================================================================
#  FIGURE M2-1: Trajectory + Event Study (core narrative)
# =====================================================================
def fig1_trajectory_eventstudy():
    """Main Fig 1: (a) p-factor trajectories patient vs control,
    (b) event-study coefficients with FDR shading, (c) per-disorder
    signal lead time."""
    agg = pd.read_csv(os.path.join(M2_TRAJ, "trajectory_aggregated_by_window.csv"))
    es = _load_csv("event_study_results.csv")
    lead = _load_csv("signal_lead_time.csv")

    fig = plt.figure(figsize=(7.2, 6.5))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.3, 1], hspace=0.45, wspace=0.4)

    # (a) Trajectories
    ax = fig.add_subplot(gs[0, :])
    for grp, c in PAT_CTRL.items():
        d = agg[(agg["group"] == grp) & (agg["window_index"].between(-24, 24))]
        ax.plot(d["days_from_t0_center"], d["p_factor_severity_z_mean"], color=c,
                label=grp.capitalize(), lw=1.8)
        ax.fill_between(d["days_from_t0_center"],
                        d["p_factor_severity_z_mean"] - d["p_factor_severity_z_se"],
                        d["p_factor_severity_z_mean"] + d["p_factor_severity_z_se"],
                        color=c, alpha=0.15, lw=0)
    ax.axvline(0, color="#333", lw=1, ls="--")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xlim(-730, 730)
    ax.set_xlabel("Days from t$_{0}$")
    ax.set_ylabel("P-factor severity (z)")
    ax.legend(frameon=False, loc="upper left")
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.text(0, ax.get_ylim()[1], "t$_{0}$", ha="center", fontsize=6.5)
    _despine(ax)
    add_panel(ax, "a")

    # (b) Event study
    ax = fig.add_subplot(gs[1, 0])
    ax.axhline(0, color="#888", lw=0.8)
    ax.axvline(0, color="#888", lw=0.8, ls="--")
    ref = es["coef"] == 0
    pre = es["days_from_t0"] < -45
    post = es["days_from_t0"] > 45
    ax.fill_between(es.loc[pre & (es["p_fdr"] < 0.05), "days_from_t0"], -0.3, 0.3,
                    color="#E64B35", alpha=0.08)
    ax.fill_between(es.loc[post & (es["p_fdr"] < 0.05), "days_from_t0"], -0.3, 0.3,
                    color="#4DBBD5", alpha=0.08)
    ax.errorbar(es.loc[pre, "days_from_t0"], es.loc[pre, "coef"],
                yerr=1.96 * es.loc[pre, "se"], fmt="o", ms=3, capsize=1.5,
                color="#4DBBD5", ecolor="#4DBBD5", elinewidth=0.8)
    ax.errorbar(es.loc[post, "days_from_t0"], es.loc[post, "coef"],
                yerr=1.96 * es.loc[post, "se"], fmt="o", ms=3, capsize=1.5,
                color="#E64B35", ecolor="#E64B35", elinewidth=0.8)
    sig = es["p_fdr"] < 0.05
    ax.scatter(es.loc[sig, "days_from_t0"], es.loc[sig, "coef"], s=18,
               color="none", edgecolors="black", lw=0.5, zorder=5)
    ax.set_xlim(-730, 730)
    ax.set_ylim(-0.15, 0.15)
    ax.set_xlabel("Days from t$_{0}$")
    ax.set_ylabel("Event-study coef.")
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    _despine(ax)
    add_panel(ax, "b")

    # (c) Signal lead time
    ax = fig.add_subplot(gs[1, 1])
    if not lead.empty:
        disorders = [r for r in lead["disorder"].tolist() if r != "all"]
        vals = [lead.loc[lead["disorder"] == d, "median_lead_days"].iloc[0]
                for d in disorders if d in lead["disorder"].values]
        lbls = [DISPLAY.get(d, d) for d in disorders]
        order = np.argsort(vals)
        ax.barh([lbls[i] for i in order], [vals[i] for i in order],
                color="#E64B35", edgecolor="none", height=0.7)
        ax.set_xlabel("Median signal lead (days)")
        ax.set_ylabel("")
        ax.set_xlim(0, max(vals) * 1.15)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    _despine(ax)
    add_panel(ax, "c")

    _save(fig, "fig1_trajectory_eventstudy")


# =====================================================================
#  FIGURE M2-2: Changepoint analysis
# =====================================================================
def fig2_changepoints():
    """Main Fig 2: (a) PELT sensitivity, (b) changepoint distribution
    patient vs control, (c) per-disorder pre-t0 rate."""
    sens = _load_csv("pelt_sensitivity_analysis.csv")
    cp = _load_csv("changepoint_analysis.csv")
    lead = _load_csv("signal_lead_time.csv")

    fig = plt.figure(figsize=(7.2, 3.2))
    gs = gridspec.GridSpec(1, 3, wspace=0.45)

    # (a) PELT sensitivity
    ax = fig.add_subplot(gs[0])
    if not sens.empty:
        ax.plot(sens["penalty_multiplier"], sens["avg_cps_per_user"], "o-",
                color="#4DBBD5", ms=4)
        ax.set_xlabel("Penalty multiplier")
        ax.set_ylabel("Avg changepoints/user")
        ax.set_yscale("log")
    _despine(ax)
    add_panel(ax, "a")

    # (b) Changepoint day distribution (patient vs control)
    ax = fig.add_subplot(gs[1])
    if not cp.empty:
        for grp, c in PAT_CTRL.items():
            days = cp.loc[cp["group"] == grp, "changepoint_days_from_t0"]
            sns.kdeplot(days, ax=ax, color=c, label=grp.capitalize(), lw=1.2, fill=True, alpha=0.2)
        ax.axvline(0, color="#333", lw=0.8, ls="--")
        ax.set_xlabel("Changepoint day (rel. t$_{0}$)")
        ax.set_ylabel("Density")
        ax.legend(frameon=False)
    _despine(ax)
    add_panel(ax, "b")

    # (c) Pre-t0 changepoint fraction by disorder (from signal_lead_time)
    ax = fig.add_subplot(gs[2])
    if not cp.empty:
        pat = cp[cp["group"] == "patient"]
        fracs = pat.groupby("disorder_group")["is_pre_t0"].mean() * 100
        fracs = fracs.sort_values()
        colors = ["#E64B35" if v > 55 else "#4DBBD5" for v in fracs]
        # Dot plot with value labels (variety: dots instead of horizontal bars)
        ys = np.arange(len(fracs))
        ax.scatter(fracs.values, ys, s=40, c=colors, edgecolors="white", lw=0.5, zorder=3)
        for yi, (d, v) in zip(ys, fracs.items()):
            ax.text(v + 2, yi, f"{v:.0f}%", va="center", fontsize=5)
        ax.axvline(50, color="#888", lw=0.8, ls="--")
        ax.set_yticks(ys)
        ax.set_yticklabels([DISPLAY.get(d, d) for d in fracs.index], fontsize=5.5)
        ax.set_xlabel("Pre-t$_{0}$ changepoints (%)")
        ax.set_xlim(0, 110)
    _despine(ax)
    add_panel(ax, "c")

    _save(fig, "fig2_changepoints")


# =====================================================================
#  FIGURE M2-3: Heterogeneity / subtypes
# =====================================================================
def fig3_subtypes():
    """Main Fig 3: (a) subtype trajectories, (b) subtype x disorder
    distribution, (c) GMM model selection (BIC/Silhouette)."""
    sub_agg = _load_csv("subtype_aggregated_trajectories.csv")
    gmm = _load_csv("gmm_model_selection.csv")
    dis_analysis = _load_json("disorder_subtype_analysis.json")

    fig = plt.figure(figsize=(7.2, 4.6))
    gs = gridspec.GridSpec(1, 3, wspace=0.5)

    # (a) Subtype trajectories
    ax = fig.add_subplot(gs[0])
    if not sub_agg.empty:
        sub_agg["subtype_label"] = sub_agg["subtype_label"].astype(int)
        for i in sorted(sub_agg["subtype_label"].unique()):
            d = sub_agg[sub_agg["subtype_label"] == i]
            ax.plot(d["days_from_t0_center"], d["p_factor_severity_z_mean"],
                    color=SUB_COLORS[i], lw=1.5, label=f"S{i}")
            ax.fill_between(d["days_from_t0_center"],
                            d["p_factor_severity_z_mean"] - d["p_factor_severity_z_se"],
                            d["p_factor_severity_z_mean"] + d["p_factor_severity_z_se"],
                            color=SUB_COLORS[i], alpha=0.12, lw=0)
        ax.axvline(0, color="#333", lw=0.8, ls="--")
        ax.set_xlim(-730, 730)
        ax.set_xlabel("Days from t$_{0}$")
        ax.set_ylabel("P-factor severity (z)")
        ax.legend(frameon=False)
    _despine(ax)
    add_panel(ax, "a")

    # (b) Subtype x disorder stacked
    ax = fig.add_subplot(gs[1])
    if dis_analysis:
        rows = []
        for dis, info in dis_analysis.items():
            for sub_k, pct in info["subtype_percentages"].items():
                rows.append({"disorder": DISPLAY.get(dis, dis),
                             "subtype": f"S{int(float(sub_k))}",
                             "pct": float(pct.rstrip("%"))})
        if rows:
            df = pd.DataFrame(rows)
            piv = df.pivot(index="disorder", columns="subtype", values="pct").fillna(0)
            piv = piv.loc[piv.sum(axis=1).sort_values().index]
            piv.plot(kind="barh", stacked=True, ax=ax,
                     color=[SUB_COLORS[int(c[1:])] for c in piv.columns],
                     edgecolor="white", lw=0.5, legend=False)
            ax.set_xlabel("Proportion (%)")
            ax.set_ylabel("")
            ax.set_xlim(0, 100)
    _despine(ax)
    add_panel(ax, "b")

    # (c) GMM selection
    ax = fig.add_subplot(gs[2])
    if not gmm.empty:
        ax2 = ax.twinx()
        ax.plot(gmm["n_components"], gmm["silhouette"], "o-", color="#E64B35", ms=4)
        ax.set_xlabel("k components")
        ax.set_ylabel("Silhouette", color="#E64B35")
        ax.tick_params(axis="y", labelcolor="#E64B35")
        ax2.plot(gmm["n_components"], gmm["bic"] / 1e5, "s--", color="#4DBBD5", ms=4)
        ax2.set_ylabel("BIC (×10$^{5}$)", color="#4DBBD5")
        ax2.tick_params(axis="y", labelcolor="#4DBBD5")
        ax.set_xticks(gmm["n_components"].values)
        ax2.spines["top"].set_visible(False)
    _despine(ax)
    add_panel(ax, "c")

    _save(fig, "fig3_subtypes")


# =====================================================================
#  FIGURE M2-4: Causal attribution
# =====================================================================
def fig4_causal():
    """Main Fig 4: (a) DID estimates by variable/method, (b) placebo
    distribution, (c) dose-response."""
    did = _load_csv("did_results.csv")
    placebo = _load_csv("placebo_did_distribution.csv")
    ptest = _load_json("placebo_test_results.json")
    dose = pd.read_csv(os.path.join(M2_STATS, "dose_response_scatter_data.csv"))

    fig = plt.figure(figsize=(7.2, 4.2))
    gs = gridspec.GridSpec(1, 3, wspace=0.5)

    # (a) DID estimates
    ax = fig.add_subplot(gs[0])
    if not did.empty:
        vnames = {"p_factor_severity_z": "P-factor", "volatility_index": "Volatility",
                  "social_withdrawal_index": "Withdrawal", "vader_compound_mean": "VADER"}
        for i, var in enumerate(did["variable"].unique()):
            row = did[did["variable"] == var]
            row = row[row["method"] == "regression_clustered"]
            if row.empty:
                continue
            est = row["did_estimate"].iloc[0]
            se = row["se"].iloc[0]
            ax.errorbar(est, i, xerr=1.96 * se, fmt="o", color="#3C5488",
                        ms=5, capsize=2)
            ax.text(est + 0.005, i, f"{est:.3f}", va="center", fontsize=5.5)
        ax.axvline(0, color="#888", lw=0.8)
        ax.set_yticks(range(len(did["variable"].unique())))
        ax.set_yticklabels([vnames.get(v, v) for v in did["variable"].unique()])
        ax.set_xlabel("DID estimate (SD)")
        ax.invert_yaxis()
    _despine(ax)
    add_panel(ax, "a")

    # (b) Placebo
    ax = fig.add_subplot(gs[1])
    if not placebo.empty:
        sns.histplot(placebo["placebo_did"], ax=ax, color="#4DBBD5", alpha=0.5, bins=30)
        td = ptest.get("true_did", np.nan)
        if not np.isnan(td):
            ax.axvline(td, color="#E64B35", lw=2)
            ax.text(td, ax.get_ylim()[1] * 0.9, f"True={td:.3f}", color="#E64B35",
                    fontsize=6, ha="center")
        ax.set_xlabel("Placebo DID")
        ax.set_ylabel("Count")
    _despine(ax)
    add_panel(ax, "b")

    # (c) Dose-response
    ax = fig.add_subplot(gs[2])
    if not dose.empty:
        ax.scatter(dose["dose_auc"], dose["response_post_mean"], s=4, color="#4DBBD5",
                   alpha=0.4, edgecolors="none")
        if "predicted_response" in dose.columns:
            dd = dose.dropna(subset=["dose_auc", "predicted_response"]).sort_values("dose_auc")
            ax.plot(dd["dose_auc"], dd["predicted_response"], color="#E64B35", lw=2)
        ax.set_xlabel("Pre-t$_{0}$ cumulative (AUC)")
        ax.set_ylabel("Post-t$_{0}$ severity")
    _despine(ax)
    add_panel(ax, "c")

    _save(fig, "fig4_causal")


# =====================================================================
#  FIGURE M2-5: Cross-disorder DID
# =====================================================================
def fig5_crossdisorder_did():
    """Main Fig 5: cross-disorder DID forest plot."""
    cd = _load_csv("cross_disorder_did.csv")

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    if not cd.empty:
        cd = cd.sort_values("did_estimate")
        ys = range(len(cd))
        for y, (_, r) in zip(ys, cd.iterrows()):
            c = "#E64B35" if r["p_value"] < 0.05 else "#AAAAAA"
            ax.errorbar(r["did_estimate"], y, xerr=1.96 * r["se"], fmt="o",
                        color=c, ms=5, capsize=2)
            ax.text(r["did_estimate"] + 0.002, y,
                    f"{r['did_estimate']:.3f}" + ("*" if r["p_value"] < 0.05 else ""),
                    va="center", fontsize=5.5)
        ax.axvline(0, color="#888", lw=0.8)
        ax.set_yticks(list(ys))
        ax.set_yticklabels([DISPLAY.get(d, d) for d in cd["disorder"]])
        ax.set_xlabel("DID estimate (SD)")
        ax.invert_yaxis()
    _despine(ax)
    _save(fig, "fig5_crossdisorder_did")


# =====================================================================
#  SI FIGURES
# =====================================================================
def ed_fig1_quality_report():
    """SI: trajectory quality metrics."""
    q = _load_json(os.path.join(M2_TRAJ, "trajectory_quality_report.json"))
    fig, ax = plt.subplots(figsize=(6, 2.6))
    labels = ["sufficient_pre", "sufficient_post", "both"]
    pats = [q.get("patient_with_sufficient_pre", 0), q.get("patient_with_sufficient_post", 0),
            q.get("n_users_with_both", 0)]
    ctrls = [q.get("control_with_sufficient_pre", 0), q.get("control_with_sufficient_post", 0),
             q.get("n_users_with_both", 0)]
    x = np.arange(3)
    ax.bar(x - 0.2, pats, 0.4, label="Patient", color="#E64B35")
    ax.bar(x + 0.2, ctrls, 0.4, label="Control", color="#4DBBD5")
    ax.set_xticks(x)
    ax.set_xticklabels(["Sufficient pre", "Sufficient post", "Both"])
    ax.set_ylabel("Users")
    ax.legend(frameon=False)
    _despine(ax)
    _save(fig, "ed_fig1_quality", "ed")


def ed_fig2_multivariate_changepoints():
    """SI: multivariate changepoint consensus."""
    mv = _load_csv("multivariate_changepoints.csv")
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    if not mv.empty:
        pre = mv["is_pre_t0"].astype(int)
        ax.bar([0, 1], [pre.value_counts().get(1, 0), pre.value_counts().get(0, 0)],
               color=["#E64B35", "#4DBBD5"])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pre-t$_{0}$", "Post-t$_{0}$"])
        ax.set_ylabel("Consensus changepoints")
    _despine(ax)
    _save(fig, "ed_fig2_multivariate", "ed")


def ed_fig3_predictors():
    """SI: subtype predictor permutation importance."""
    pred = _load_csv("subtype_predictors.csv")
    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    if not pred.empty:
        top = pred.head(15)
        ax.barh([p[:25] for p in top["feature"]], top["importance_mean"],
                xerr=top["importance_std"], color="#3C5488", height=0.7,
                error_kw={"elinewidth": 0.6})
        ax.set_xlabel("Permutation importance")
        ax.invert_yaxis()
    _despine(ax)
    _save(fig, "ed_fig3_predictors", "ed")


def ed_fig4_t0_validation():
    """SI: t0 validation (linguistic shift)."""
    meta = _load_json(os.path.join(M2_TRAJ, "trajectory_metadata.json"))
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    tv = meta.get("t0_validation", {})
    grps = [g for g in ("patient", "control") if g in tv]
    shifts = [tv[g]["mean_shift"] for g in grps]
    ds = [tv[g]["cohen_d"] for g in grps]
    x = np.arange(len(grps))
    ax.bar(x, ds, 0.5, color=["#E64B35", "#4DBBD5"])
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([g.capitalize() for g in grps])
    ax.set_ylabel("Cohen's d (t$_{0}$ shift)")
    for xi, (g, d) in zip(x, zip(grps, ds)):
        ax.text(xi, d + 0.01, f"{d:.3f}", ha="center", fontsize=6)
    _despine(ax)
    _save(fig, "ed_fig4_t0validation", "ed")


def ed_fig5_did_by_method():
    """SI: DID across methods (regression/pair/IPTW)."""
    did = _load_csv("did_results.csv")
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    if not did.empty:
        vnames = {"p_factor_severity_z": "P-factor", "volatility_index": "Volatility",
                  "social_withdrawal_index": "Withdrawal", "vader_compound_mean": "VADER"}
        mcolors = {"regression_clustered": "#E64B35", "pair_fe": "#4DBBD5",
                   "iptw_weighted": "#00A087"}
        x = np.arange(len(did["variable"].unique()))
        for j, var in enumerate(did["variable"].unique()):
            for i, (m, c) in enumerate(mcolors.items()):
                row = did[(did["variable"] == var) & (did["method"] == m)]
                if row.empty:
                    continue
                est, se = row["did_estimate"].iloc[0], row["se"].iloc[0]
                ax.errorbar(j + (i - 1) * 0.25, est, yerr=1.96 * se, fmt="o",
                            color=c, ms=3.5, capsize=1.5)
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([vnames.get(v, v) for v in did["variable"].unique()], rotation=20)
        ax.set_ylabel("DID estimate (SD)")
        import matplotlib.patches as mpatches
        ax.legend(handles=[mpatches.Patch(color=c, label=m) for m, c in mcolors.items()],
                  frameon=False, fontsize=5.5)
    _despine(ax)
    _save(fig, "ed_fig5_did_by_method", "ed")


def ed_fig6_subtype_profiles():
    """SI: subtype feature profiles (median features)."""
    prof = _load_json("subtype_profiles.json")
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    if prof:
        feats = ["p_factor_severity_z_step_height", "p_factor_severity_z_slope_pre",
                 "p_factor_severity_z_slope_post", "p_factor_severity_z_volatility_pre",
                 "p_factor_severity_z_turning_angle", "p_factor_severity_z_recovery_pct"]
        flabels = ["Step", "Slope pre", "Slope post", "Volatility", "Turning", "Recovery"]
        x = np.arange(len(feats))
        for i, k in enumerate(sorted(prof.keys())):
            meds = [prof[k]["feature_medians"].get(f, np.nan) for f in feats]
            ax.plot(x, meds, "o-", color=SUB_COLORS[i], ms=3, label=f"S{i}")
        ax.set_xticks(x)
        ax.set_xticklabels(flabels, rotation=20)
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_ylabel("Median feature")
        ax.legend(frameon=False)
    _despine(ax)
    _save(fig, "ed_fig6_subtype_profiles", "ed")


def ed_fig7_eventstudy_disorder():
    """SI: changepoint lead-time by disorder with IQR."""
    lead = _load_csv("signal_lead_time.csv")
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    if not lead.empty:
        d = lead[lead["disorder"] != "all"]
        for i, r in d.iterrows():
            ax.errorbar(r["median_lead_days"], r["disorder"],
                        xerr=[[r["median_lead_days"] - r["q25_lead_days"]],
                              [r["q75_lead_days"] - r["median_lead_days"]]],
                        fmt="o", color="#3C5488", ms=4, capsize=2)
        ax.set_xlabel("Signal lead (days)")
        ax.set_ylabel("")
    _despine(ax)
    _save(fig, "ed_fig7_leadtime_disorder", "ed")


def run_all():
    logger.info("=== M2 figures v2 ===")
    fig1_trajectory_eventstudy()
    fig2_changepoints()
    fig3_subtypes()
    fig4_causal()
    fig5_crossdisorder_did()
    ed_fig1_quality_report()
    ed_fig2_multivariate_changepoints()
    ed_fig3_predictors()
    ed_fig4_t0_validation()
    ed_fig5_did_by_method()
    ed_fig6_subtype_profiles()
    ed_fig7_eventstudy_disorder()
    logger.info("DONE")


if __name__ == "__main__":
    run_all()
