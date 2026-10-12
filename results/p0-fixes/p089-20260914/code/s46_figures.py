"""
M4 - Step 46: Publication Figures (Nature/Science grade, M1-M3 consistent)
==========================================================================
Figures for macro digital epidemiology (M4), based on CORRECTED
(composition-adjusted) analyses.

Main figures -> module4_results/figures/main/
SI figures    -> module4_results/figures/ed/

Design system (shared with M1/M2/M3):
  Arial sans-serif, base 7pt, titles 8pt, ticks 6pt, annotations 6pt
  thin 0.8pt axes, no top/right spines, Okabe-Ito / Nature palette
  PDF (vector) + 300dpi PNG
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patheffects import withStroke

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = "/path/to/project"
DATA_DIR = os.path.join(PROJECT_ROOT, "module4_results", "data")
STATS_DIR = os.path.join(PROJECT_ROOT, "module4_results", "statistics")
FIG_ROOT = os.path.join(PROJECT_ROOT, "module4_results", "figures")
MAIN_DIR = os.path.join(FIG_ROOT, "main")
ED_DIR = os.path.join(FIG_ROOT, "ed")
os.makedirs(MAIN_DIR, exist_ok=True)
os.makedirs(ED_DIR, exist_ok=True)

FS_TITLE, FS_AXLABEL, FS_TICK, FS_ANNOT, FS_SUPTITLE = 8, 7, 6, 6, 9
DOUBLE_COL = 7.2
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": FS_TITLE, "axes.labelsize": FS_AXLABEL,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": 6,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#333333", "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "lines.linewidth": 1.2, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

DISPLAY = {
    "adhd": "ADHD", "anxiety": "Anxiety", "autism": "Autism", "bipolar": "Bipolar",
    "bpd": "BPD", "depression": "Depression", "eating_disorders": "Eating D.",
    "ocd": "OCD", "ptsd": "PTSD", "schizophrenia": "Schizophrenia",
}
C_PAT = "#E64B35"; C_CTRL = "#4DBBD5"; C_EXT = "#3C5488"; C_ADJ = "#00A087"


def _save(fig, name, subdir="main"):
    d = MAIN_DIR if subdir == "main" else ED_DIR
    for fmt in ("pdf", "png"):
        fig.savefig(os.path.join(d, f"{name}.{fmt}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {subdir}/{name}")

def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
        ax.tick_params(**{s: False})

def add_panel(ax, label, x=-0.13, y=1.06):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="top", ha="right")

def _csv(path, **kw):
    if not os.path.exists(path):
        logger.warning(f"  Missing: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, **kw)


# =====================================================================
# FIG 1: Calendar signal overview (corrected vs raw + external overlay)
# =====================================================================
def fig1_overview():
    raw = _csv(os.path.join(DATA_DIR, "digital_signal_annual_diff.csv"))
    adj = _csv(os.path.join(DATA_DIR, "digital_signal_annual_band_adjusted.csv"))
    ext = _csv(os.path.join(DATA_DIR, "external", "processed", "epidemiology_annual.csv"))
    sui = ext[ext["indicator"] == "suicide_rate"][["year", "value"]] if not ext.empty else pd.DataFrame()

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.8))
    # raw vs adjusted p_factor diff
    ax.plot(raw["year"], raw["p_factor_severity_z_diff"], "o-", color="#AAAAAA",
            ms=3, lw=1, label="Raw (confounded)")
    ax.plot(adj["year"], adj["p_factor_severity_z_diff"], "o-", color=C_ADJ,
            ms=4, lw=1.8, label="Composition-adjusted")
    ax.axhline(0, color="#888", lw=0.8)
    ax.axvspan(2020, 2022, color="#E64B35", alpha=0.08)
    ax.text(2021, ax.get_ylim()[1]*0.95, "COVID", ha="center", fontsize=6, color="#E64B35")
    ax.set_xlabel("Year")
    ax.set_ylabel("Patient−control gap (z)")
    ax.legend(frameon=False, loc="upper left", fontsize=5.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    _despine(ax)
    fig.suptitle("Digital Phenotype Signal Over Calendar Time", fontsize=FS_SUPTITLE,
                 fontweight="bold", y=0.98)
    _save(fig, "fig1_calendar_overview")


# =====================================================================
# FIG 2: Corrected long-term trends (3 metrics)
# =====================================================================
def fig2_trends():
    adj = _csv(os.path.join(DATA_DIR, "digital_signal_annual_band_adjusted.csv"))
    metrics = [("p_factor_severity_z_diff", "P-factor gap", "#4DBBD5"),
               ("volatility_index_diff", "Volatility gap", "#E64B35"),
               ("posting_intensity_diff", "Posting-intensity gap", "#00A087")]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.6), sharex=True)
    for ax, (m, label, c) in zip(axes, metrics):
        if m not in adj.columns:
            continue
        ax.plot(adj["year"], adj[m], "o-", color=c, ms=4, lw=1.8)
        ax.axhline(0, color="#888", lw=0.8)
        ax.axvspan(2020, 2022, color="#E64B35", alpha=0.08)
        ax.set_title(label, fontsize=FS_TITLE, fontweight="bold")
        ax.set_xlabel("Year")
        if ax is axes[0]:
            ax.set_ylabel("Patient−control gap (z)")
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        _despine(ax)
    fig.suptitle("Composition-Adjusted Long-Term Trends", fontsize=FS_SUPTITLE,
                 fontweight="bold", y=0.98)
    _save(fig, "fig2_corrected_trends")


# =====================================================================
# FIG 3: COVID ITS (forest plot: shift + trend)
# =====================================================================
def fig3_covid_its():
    its = _csv(os.path.join(STATS_DIR, "covid_its_corrected_results.csv"))
    pf = its[its["metric"] == "p_factor_severity_z_diff"] if not its.empty else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(DOUBLE_COL*0.6, 3.2))
    if not pf.empty:
        pf = pf.sort_values("b_post_shift")
        ys = np.arange(len(pf))
        colors = ["#E64B35" if p < 0.05 else "#BBBBBB" for p in pf["p_post_shift_fdr"].fillna(1)]
        ax.errorbar(pf["b_post_shift"], ys, xerr=1.96*pf["se_post_shift"], fmt="none",
                    ecolor="#999999", capsize=2, elinewidth=0.8)
        ax.scatter(pf["b_post_shift"], ys, c=colors, s=28, zorder=3, edgecolors="none")
        ax.axvline(0, color="#888", lw=0.8)
        ax.set_yticks(ys)
        ax.set_yticklabels([DISPLAY.get(d, d) for d in pf["disorder"]], fontsize=FS_TICK)
        ax.set_xlabel("COVID immediate shift (level jump)")
        ax.invert_yaxis()
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    _despine(ax)
    fig.suptitle("COVID-19 Immediate Shift on Patient−Control Gap", fontsize=FS_SUPTITLE,
                 fontweight="bold", y=0.98)
    _save(fig, "fig3_covid_its")


# =====================================================================
# FIG 4: Event study (window-level DID, robust effects)
# =====================================================================
def fig4_events():
    ev_final = json.load(open(os.path.join(STATS_DIR, "event_study_final.json")))
    evs = ["E1_election_2016", "E2_celebrity_2018", "E3_covid_2020",
           "E4_blm_2020", "E5_capitol_2021"]
    labels = [ev_final["events"][e]["label"] for e in evs if e in ev_final["events"]]
    labels = ["Election", "Celebrity", "COVID", "BLM", "Capitol"]
    coefs = []
    ses = []
    for e in evs:
        if e in ev_final["events"] and "p_factor_severity_z" in ev_final["events"][e]:
            d = ev_final["events"][e]["p_factor_severity_z"]
            if "coef_group_event" in d:
                coefs.append(d["coef_group_event"])
                ses.append(d["se"])
            else:
                coefs.append(np.nan); ses.append(np.nan)
        else:
            coefs.append(np.nan); ses.append(np.nan)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL*0.6, 2.6))
    xs = np.arange(len(labels))
    cols = ["#4DBBD5", "#4DBBD5", "#4DBBD5", "#E64B35", "#E64B35"]
    ax.errorbar(xs, coefs, yerr=[1.96*s if not np.isnan(s) else 0 for s in ses],
                fmt="o", color="#3C5488", ms=6, capsize=3, elinewidth=1.2, mec="white")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=FS_TICK)
    ax.set_ylabel("Event effect on gap (z)")
    # annotate significance
    for xi, (c, e) in enumerate(zip(coefs, ["E1_election_2016","E2_celebrity_2018","E3_covid_2020","E4_blm_2020","E5_capitol_2021"])):
        if not np.isnan(c):
            pv = ev_final["events"][e]["p_factor_severity_z"].get("p", 1)
            if pv < 0.05:
                ax.text(xi, c + (0.003 if c >= 0 else -0.006), "*", ha="center",
                        fontsize=10, fontweight="bold", color="#E64B35")
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    _despine(ax)
    fig.suptitle("Major Societal Events: Window-Level Effect on Gap", fontsize=FS_SUPTITLE,
                 fontweight="bold", y=0.98)
    _save(fig, "fig4_events")


# =====================================================================
# FIG 5: Lead-lag (volatility & posting vs suicide rate)
# =====================================================================
def fig5_leadlag():
    ll = _csv(os.path.join(STATS_DIR, "lead_lag_corrected_results.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.6))
    for ax, sig in zip(axes, ["volatility_index_diff", "posting_intensity_diff"]):
        sub = ll[(ll["digital_signal"] == sig) & (ll["external"] == "suicide_rate")]
        if sub.empty:
            continue
        sub = sub.sort_values("lag")
        ax.plot(sub["lag"], sub["r"], "o-", color=C_PAT if sig.startswith("vol") else C_ADJ,
                ms=5, lw=1.8)
        ax.fill_between(sub["lag"], sub["ci_lo"], sub["ci_hi"], alpha=0.15,
                        color=C_PAT if sig.startswith("vol") else C_ADJ)
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(["Digital lags\n1yr", "Concurrent", "Digital leads\n1yr"], fontsize=5)
        ax.set_ylim(-1, 1)
        title = "Volatility gap" if sig.startswith("vol") else "Posting-intensity gap"
        ax.set_title(title, fontsize=FS_TITLE, fontweight="bold")
        if ax is axes[0]:
            ax.set_ylabel("Correlation with suicide rate")
        _despine(ax)
    fig.suptitle("Lead-Lag: Digital Signal vs Suicide Rate", fontsize=FS_SUPTITLE,
                 fontweight="bold", y=0.98)
    _save(fig, "fig5_leadlag")


# =====================================================================
# SI FIGURES
# =====================================================================
def ed_fig1_composition_bias():
    """SI: show raw vs adjusted to illustrate the composition bias fix."""
    raw = _csv(os.path.join(DATA_DIR, "digital_signal_annual_diff.csv"))
    adj = _csv(os.path.join(DATA_DIR, "digital_signal_annual_band_adjusted.csv"))
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    ax.plot(raw["year"], raw["p_factor_severity_z_diff"], "o--", color="#AAAAAA",
            ms=3, lw=1, label="Raw (mixed t0 windows)")
    ax.plot(adj["year"], adj["p_factor_severity_z_diff"], "o-", color=C_ADJ,
            ms=4, lw=1.8, label="Band-adjusted (composition-equal)")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xlabel("Year"); ax.set_ylabel("Patient−control p-factor gap (z)")
    ax.legend(frameon=False, fontsize=5.5)
    ax.set_title("Composition-Bias Correction", fontsize=FS_TITLE, fontweight="bold")
    _despine(ax)
    _save(fig, "ed_fig1_composition_bias", "ed")


def ed_fig2_disorder_trends():
    """SI: per-disorder band-adjusted p_factor trend (2015 vs 2022)."""
    adj_m = _csv(os.path.join(DATA_DIR, "digital_signal_monthly_by_disorder_band_adj.csv"))
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    if not adj_m.empty:
        adj_m["period"] = pd.to_datetime(adj_m["period"])
        piv = adj_m.pivot_table(index="disorder_group", columns="period",
                                values="p_factor_severity_z_diff", aggfunc="mean")
        piv = piv.sort_index()
        x = np.arange(piv.shape[1])
        for i, (d, row) in enumerate(piv.iterrows()):
            ax.plot(x, row.values, "o-", ms=2, lw=0.7, alpha=0.7, label=DISPLAY.get(d, d))
        n_ticks = 7
        tick_pos = np.linspace(0, piv.shape[1]-1, n_ticks).astype(int)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([pd.Timestamp(piv.columns[t]).year for t in tick_pos], fontsize=5)
        ax.set_xlabel("Month (2010-2022)")
        ax.set_ylabel("P-factor gap (z)")
        ax.legend(frameon=False, fontsize=4.5, ncol=2, loc="upper left")
    _despine(ax)
    _save(fig, "ed_fig2_disorder_trends", "ed")


def ed_fig3_disorder_covid():
    """SI: per-disorder COVID ITS trend effects (all metrics)."""
    its = _csv(os.path.join(STATS_DIR, "covid_its_corrected_results.csv"))
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    if not its.empty:
        pf = its[its["metric"] == "p_factor_severity_z_diff"].sort_values("b_post_trend")
        ys = np.arange(len(pf))
        ax.errorbar(pf["b_post_trend"], ys, xerr=1.96*pf["se_post_trend"], fmt="none",
                    ecolor="#999999", capsize=1.5, elinewidth=0.8)
        ax.scatter(pf["b_post_trend"], ys, color="#E64B35", s=26, zorder=3)
        ax.axvline(0, color="#888", lw=0.8)
        ax.set_yticks(ys)
        ax.set_yticklabels([DISPLAY.get(d, d) for d in pf["disorder"]], fontsize=5.5)
        ax.set_xlabel("COVID post-trend (slope change)")
        ax.invert_yaxis()
    _despine(ax)
    _save(fig, "ed_fig3_disorder_covid", "ed")


def ed_fig4_leadlag_matrix():
    """SI: lead-lag correlation matrix (signals x externals x lags)."""
    ll = _csv(os.path.join(STATS_DIR, "lead_lag_corrected_results.csv"))
    if ll.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    sigs = ll["digital_signal"].unique()
    exts = ll["external"].unique()
    lags = [-1, 0, 1]
    mat = np.full((len(sigs), len(exts)*len(lags)), np.nan)
    for i, s in enumerate(sigs):
        for j, e in enumerate(exts):
            for k, lag in enumerate(lags):
                r = ll[(ll.digital_signal==s)&(ll.external==e)&(ll.lag==lag)]
                if not r.empty and not np.isnan(r["r"].iloc[0]):
                    mat[i, j*3+k] = r["r"].iloc[0]
    cmap = plt.get_cmap("RdBu_r")
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(exts)*len(lags)))
    ax.set_xticklabels([f"{e}\n{['L','C','Lag'][k]}" for e in exts for k in range(3)], fontsize=5)
    ax.set_yticks(range(len(sigs)))
    ax.set_yticklabels([s[:14] for s in sigs], fontsize=5.5)
    for i in range(len(sigs)):
        for j in range(len(exts)*len(lags)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=5,
                        color="white" if abs(mat[i,j])>0.6 else "black")
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.15)
    fig.colorbar(im, cax=cax)
    _save(fig, "ed_fig4_leadlag_matrix", "ed")


def run_all():
    logger.info("=== M4 figures (corrected analyses) ===")
    fig1_overview()
    fig2_trends()
    fig3_covid_its()
    fig4_events()
    fig5_leadlag()
    ed_fig1_composition_bias()
    ed_fig2_disorder_trends()
    ed_fig3_disorder_covid()
    ed_fig4_leadlag_matrix()
    logger.info("DONE")


if __name__ == "__main__":
    run_all()
