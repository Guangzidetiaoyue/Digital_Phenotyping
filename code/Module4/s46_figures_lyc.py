"""
Module 4 Publication Figures (lyc) — Nature/Science-grade
=========================================================
Main -> module4_results/figures_lyc/main/
SI   -> module4_results/figures_lyc/ed/

CLI: python s46_figures_lyc.py -p fig1 edfig3
      python s46_figures_lyc.py -p all
"""

import os
import json
import logging
import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import AutoMinorLocator

from config_module4 import DATA_DIR, STATS_DIR, FIGURES_DIR, DISORDER_DISPLAY_NAMES

MAIN_DIR = os.path.join(FIGURES_DIR, "main")
ED_DIR = os.path.join(FIGURES_DIR, "ed")
os.makedirs(MAIN_DIR, exist_ok=True)
os.makedirs(ED_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(FIGURES_DIR, "visualization.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ---- global style (设计参考.md / Module1 lyc) ----


def mm2inch(mm: float) -> float:
    return mm / 25.4


SINGLE_COL = mm2inch(89)
ONE_HALF_COL = mm2inch(120)
DOUBLE_COL = mm2inch(183)

FS_TITLE = 8
FS_AXLABEL = 7
FS_TICK = 6
FS_ANNOT = 6
FS_LEGEND = 6.5
FS_SUPTITLE = 9

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": FS_TITLE,
    "axes.titleweight": "bold",
    "axes.labelsize": FS_AXLABEL,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND,
    "figure.titlesize": FS_SUPTITLE,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#333333",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "lines.linewidth": 1.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    # Layout is set explicitly per figure; do not let tight cropping change
    # the designed canvas or cut labels/colorbars.
    "savefig.bbox": None,
    "savefig.pad_inches": 0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

DISPLAY = dict(DISORDER_DISPLAY_NAMES)
C_PAT = "#E53935"
C_CTRL = "#1E88E5"
C_EXT = "#3C5488"
C_ADJ = "#009E73"
C_RAW = "#BDBDBD"
C_COVID_SHADE = "#E53935"
C_NONSIG = "#BDBDBD"

# ED heatmap layout follows Module1's shared heatmap implementation.
HM_ED_WIDTH = 1920 / 300
HM_ED_HEIGHT = HM_ED_WIDTH * 0.90
HM_ED_MARGINS = dict(left=0.18, right=0.92, top=0.90, bottom=0.18)
HM_CBAR_W = "4%"
HM_CBAR_PAD = 0.15
HM_REF_VMAX = 0.6
HM_TEXT_WHITE_AT = 0.35


def _save(fig, name, subdir="main"):
    d = MAIN_DIR if subdir == "main" else ED_DIR
    for fmt in ("pdf", "png"):
        fig.savefig(
            os.path.join(d, f"{name}.{fmt}"),
            dpi=300,
            bbox_inches=None,
            pad_inches=0,
        )
    plt.close(fig)
    logger.info(f"  Saved: {subdir}/{name} (pdf+png)")


def _despine(ax, top=True, right=True, left=False, bottom=False):
    if top:
        ax.spines["top"].set_visible(False)
    if right:
        ax.spines["right"].set_visible(False)
    if left:
        ax.spines["left"].set_visible(False)
    if bottom:
        ax.spines["bottom"].set_visible(False)


def _csv(path, **kw):
    if not os.path.exists(path):
        logger.warning(f"  Missing: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, **kw)


def _require_columns(df, columns, figure_name):
    """Fail loudly instead of silently drawing a misleading partial figure."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{figure_name}: missing required columns {missing}")


def _set_year_axis(ax, start=2010, end=2022, step=2):
    ax.set_xlim(start - 0.35, end + 0.35)
    ax.set_xticks(np.arange(start, end + 1, step))
    ax.set_xlabel("Year")


def _finish_single_panel(
    fig,
    ax,
    *,
    left=0.20,
    right=0.95,
    bottom=0.16,
    top=0.84,
):
    """Standard single-panel canvas with a square internal plotting area."""
    ax.set_box_aspect(1)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)


def _heatmap_cell_white_threshold(scale_max):
    """Scale the Module1 cell-text threshold to the active color range."""
    return HM_TEXT_WHITE_AT * (float(scale_max) / HM_REF_VMAX)


def _ed_heatmap_figure():
    """Module1 reference canvas: non-square figure, square plotting area."""
    fig, ax = plt.subplots(figsize=(HM_ED_WIDTH, HM_ED_HEIGHT))
    fig.subplots_adjust(**HM_ED_MARGINS)
    return fig, ax


def _ed_heatmap_imshow(ax, data, *, scale_max):
    norm = TwoSlopeNorm(vmin=-scale_max, vcenter=0, vmax=scale_max)
    im = ax.imshow(
        data,
        cmap="RdBu_r",
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )
    n_rows, n_cols = data.shape
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_box_aspect(1)
    return im


def _ed_heatmap_annotate(ax, data, text_white_threshold):
    """Module1-style two-decimal annotations, including em dashes for NaN."""
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data[row, col]
            if not np.isfinite(value):
                ax.text(
                    col, row, "—", ha="center", va="center",
                    fontsize=FS_ANNOT, color="#AAAAAA", zorder=5,
                )
                continue
            display_value = float(np.round(value, 2))
            text_color = (
                "white"
                if abs(display_value) > text_white_threshold
                else "black"
            )
            ax.text(
                col, row, f"{display_value:.2f}",
                ha="center", va="center",
                fontsize=FS_ANNOT, color=text_color, zorder=5,
            )


def _ed_heatmap_style_frame(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#222222")
        spine.set_linewidth(1.0)
    ax.tick_params(labelsize=FS_TICK)


def _ed_heatmap_colorbar(fig, ax, im, label):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=HM_CBAR_W, pad=HM_CBAR_PAD)
    colorbar = fig.colorbar(im, cax=cax)
    colorbar.set_label(label, fontsize=FS_AXLABEL)
    colorbar.ax.tick_params(labelsize=FS_TICK)
    colorbar.outline.set_visible(True)
    colorbar.outline.set_color("#222222")
    colorbar.outline.set_linewidth(1.0)
    n_rows, n_cols = im.get_array().shape
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_box_aspect(1)
    return colorbar


def _ed_heatmap_title(ax, title):
    ax.set_title(
        title,
        fontsize=FS_TITLE,
        fontweight="bold",
        loc="center",
        pad=20,
    )


# =====================================================================
# FIG 1: Calendar signal overview (corrected vs raw)
# =====================================================================
def plot_fig1_overview():
    raw = _csv(os.path.join(DATA_DIR, "digital_signal_annual_diff.csv"))
    adj = _csv(os.path.join(DATA_DIR, "digital_signal_annual_band_adjusted.csv"))
    required = ["year", "p_factor_severity_z_diff"]
    _require_columns(raw, required, "fig1 raw series")
    _require_columns(adj, required, "fig1 adjusted series")

    # Raw starts in 2008 whereas the corrected estimator is defined from 2010.
    # Restrict both curves to common years so the visual comparison is like-for-like.
    common_years = np.intersect1d(raw["year"].unique(), adj["year"].unique())
    raw = raw[raw["year"].isin(common_years)].sort_values("year")
    adj = adj[adj["year"].isin(common_years)].sort_values("year")

    fig, ax = plt.subplots(figsize=(ONE_HALF_COL, ONE_HALF_COL * 0.90))
    ax.plot(
        raw["year"], raw["p_factor_severity_z_diff"], "o-",
        color=C_RAW, ms=3, lw=1, label="Raw (confounded)",
    )
    ax.plot(
        adj["year"], adj["p_factor_severity_z_diff"], "o-",
        color=C_ADJ, ms=4, lw=1.5, label="Composition-adjusted",
    )
    ax.axhline(0, color="#888888", lw=0.8)
    ax.axvspan(2020, 2022, color=C_COVID_SHADE, alpha=0.08)
    ax.text(
        2021, 0.96, "COVID", transform=ax.get_xaxis_transform(),
        ha="center", va="top", fontsize=FS_ANNOT, color=C_COVID_SHADE,
    )
    _set_year_axis(ax)
    ax.set_ylabel("Patient−control gap (z)")
    ax.legend(
        frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.19),
        ncol=2, fontsize=FS_LEGEND, handlelength=2.2, columnspacing=1.4,
    )
    _despine(ax)
    ax.set_title(
        "Digital Phenotype Signal Over Calendar Time",
        fontsize=FS_TITLE, fontweight="bold", pad=20,
    )
    _finish_single_panel(fig, ax, left=0.18, bottom=0.25, top=0.84)
    _save(fig, "fig1_calendar_overview")


# =====================================================================
# FIG 2: Corrected long-term trends (3 metrics)
# =====================================================================
def plot_fig2_trends():
    adj = _csv(os.path.join(DATA_DIR, "digital_signal_annual_band_adjusted.csv"))
    _require_columns(
        adj,
        [
            "year",
            "p_factor_severity_z_diff",
            "volatility_index_diff",
            "posting_intensity_diff",
        ],
        "fig2",
    )
    metrics = [
        ("p_factor_severity_z_diff", "P-factor gap", C_CTRL),
        ("volatility_index_diff", "Volatility gap", C_PAT),
        ("posting_intensity_diff", "Posting-intensity gap", C_ADJ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, DOUBLE_COL * 0.43), sharex=True)
    for ax, (m, label, c) in zip(axes, metrics):
        ax.plot(adj["year"], adj[m], "o-", color=c, ms=4, lw=1.5)
        ax.axhline(0, color="#888888", lw=0.8)
        ax.axvspan(2020, 2022, color=C_COVID_SHADE, alpha=0.08)
        ax.set_title(label, fontsize=FS_TITLE, fontweight="bold", pad=12)
        _set_year_axis(ax, step=4)
        if ax is axes[0]:
            ax.set_ylabel("Patient−control gap (z)")
        ax.set_box_aspect(1)
        _despine(ax)
    fig.suptitle(
        "Composition-Adjusted Long-Term Trends",
        fontsize=FS_SUPTITLE, fontweight="bold", y=0.97,
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.16, top=0.76, wspace=0.28)
    _save(fig, "fig2_corrected_trends")


# =====================================================================
# FIG 3: COVID ITS (forest plot: shift + trend)
# =====================================================================
def plot_fig3_covid_its():
    its = _csv(os.path.join(STATS_DIR, "covid_its_corrected_results.csv"))
    required = [
        "disorder", "metric", "b_post_shift", "se_post_shift",
        "p_post_shift_fdr",
    ]
    _require_columns(its, required, "fig3")
    pf = its[its["metric"] == "p_factor_severity_z_diff"].copy()
    valid = (
        np.isfinite(pf["b_post_shift"])
        & np.isfinite(pf["se_post_shift"])
        & (pf["se_post_shift"] >= 0)
    )
    if (~valid).any():
        logger.warning(f"  fig3: omitted {(~valid).sum()} rows with invalid estimates/SE")
    pf = pf.loc[valid].sort_values("b_post_shift")

    fig, ax = plt.subplots(figsize=(ONE_HALF_COL, ONE_HALF_COL * 0.90))
    ys = np.arange(len(pf))
    colors = [
        C_PAT if p < 0.05 else C_NONSIG
        for p in pf["p_post_shift_fdr"].fillna(1)
    ]
    ax.errorbar(
        pf["b_post_shift"], ys, xerr=1.96 * pf["se_post_shift"], fmt="none",
        ecolor="#999999", capsize=2, elinewidth=0.8,
    )
    ax.scatter(pf["b_post_shift"], ys, c=colors, s=25, zorder=3, edgecolors="none")
    ax.axvline(0, color="#888888", lw=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels(
        [DISPLAY.get(d, d) for d in pf["disorder"]],
        fontsize=FS_TICK,
        fontweight="bold",
    )
    ax.set_xlabel("Immediate shift in patient−control gap (95% CI)")
    ax.margins(x=0.10, y=0.06)
    ax.invert_yaxis()
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    _despine(ax)
    ax.set_title(
        "COVID-19 Immediate Shift on Patient−Control Gap",
        fontsize=FS_TITLE, fontweight="bold", pad=20,
    )
    _finish_single_panel(fig, ax, left=0.27, right=0.95, bottom=0.17, top=0.84)
    _save(fig, "fig3_covid_its")


# =====================================================================
# FIG 4: Event study (window-level DID, robust effects)
# =====================================================================
def plot_fig4_events():
    path = os.path.join(STATS_DIR, "event_study_final.json")
    if not os.path.exists(path):
        logger.warning(f"  Missing: {path}")
        return
    with open(path) as f:
        ev_final = json.load(f)

    from config_module4 import EVENTS as _EVENTS
    evs = list(_EVENTS.keys())
    labels = [_EVENTS[e]["label"] for e in evs]
    coefs = []
    ses = []
    for e in evs:
        if e in ev_final["events"] and "p_factor_severity_z" in ev_final["events"][e]:
            d = ev_final["events"][e]["p_factor_severity_z"]
            if "coef_group_event" in d:
                coefs.append(d["coef_group_event"])
                ses.append(d["se"])
            else:
                coefs.append(np.nan)
                ses.append(np.nan)
        else:
            coefs.append(np.nan)
            ses.append(np.nan)

    coefs = np.asarray(coefs, dtype=float)
    ses = np.asarray(ses, dtype=float)
    valid = np.isfinite(coefs) & np.isfinite(ses) & (ses >= 0)
    if (~valid).any():
        logger.warning(f"  fig4: omitted {(~valid).sum()} events with invalid estimates/SE")
    labels = [label for label, keep in zip(labels, valid) if keep]
    valid_evs = [event for event, keep in zip(evs, valid) if keep]
    coefs = coefs[valid]
    ses = ses[valid]

    fig, ax = plt.subplots(figsize=(ONE_HALF_COL, ONE_HALF_COL * 0.90))
    ys = np.arange(len(labels))
    ax.errorbar(
        coefs, ys, xerr=1.96 * ses,
        fmt="o", color=C_EXT, ms=5, capsize=2, elinewidth=0.8, mec="white",
    )
    ax.axvline(0, color="#888888", lw=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=FS_TICK, fontweight="bold")
    ax.set_xlabel("Window-level DID estimate (95% CI)")
    ax.invert_yaxis()

    ci_lo = coefs - 1.96 * ses
    ci_hi = coefs + 1.96 * ses
    span = max(np.nanmax(ci_hi) - np.nanmin(ci_lo), 1e-6)
    for yi, (coef, hi, event) in enumerate(zip(coefs, ci_hi, valid_evs)):
        p_value = ev_final["events"][event]["p_factor_severity_z"].get("p", 1)
        label = f"{coef:+.3f}" + (" *" if p_value < 0.05 else "")
        ax.text(
            hi + 0.035 * span, yi, label,
            va="center", ha="left", fontsize=FS_ANNOT,
            color=C_PAT if p_value < 0.05 else "#333333",
        )
    ax.set_xlim(
        min(np.nanmin(ci_lo), 0) - 0.08 * span,
        max(np.nanmax(ci_hi), 0) + 0.28 * span,
    )
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    _despine(ax)
    ax.set_title(
        "Major Societal Events: Window-Level Effect on Gap",
        fontsize=FS_TITLE, fontweight="bold", pad=20,
    )
    _finish_single_panel(fig, ax, left=0.30, right=0.96, bottom=0.17, top=0.84)
    _save(fig, "fig4_events")


# =====================================================================
# FIG 5: Lead-lag (volatility & posting vs suicide rate)
# =====================================================================
def plot_fig5_leadlag():
    ll = _csv(os.path.join(STATS_DIR, "lead_lag_corrected_results.csv"))
    _require_columns(
        ll,
        ["lag", "r", "ci_lo", "ci_hi", "digital_signal", "external"],
        "fig5",
    )
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.48))
    for ax, sig in zip(axes, ["volatility_index_diff", "posting_intensity_diff"]):
        sub = ll[(ll["digital_signal"] == sig) & (ll["external"] == "suicide_rate")]
        if sub.empty:
            raise ValueError(f"fig5: no suicide-rate rows for {sig}")
        sub = sub.sort_values("lag")
        if not np.array_equal(sub["lag"].to_numpy(), np.array([-1, 0, 1])):
            raise ValueError(f"fig5: expected lags [-1, 0, 1] for {sig}")
        color = C_PAT if sig.startswith("vol") else C_ADJ
        ax.plot(sub["lag"], sub["r"], "o-", color=color, ms=5, lw=1.5)
        ax.fill_between(sub["lag"], sub["ci_lo"], sub["ci_hi"], alpha=0.15, color=color)
        ax.axhline(0, color="#888888", lw=0.8)
        ax.set_xticks([-1, 0, 1])
        # Upstream s45c pairs external_year = digital_year + lag.
        ax.set_xticklabels(
            ["Digital lags\n1 year", "Concurrent", "Digital leads\n1 year"],
            fontsize=FS_ANNOT,
        )
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1, 1)
        title = "Volatility gap" if sig.startswith("vol") else "Posting-intensity gap"
        ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", pad=12)
        if ax is axes[0]:
            ax.set_ylabel("Correlation with suicide rate")
        ax.set_box_aspect(1)
        _despine(ax)
    fig.suptitle(
        "Lead-Lag: Digital Signal vs Suicide Rate",
        fontsize=FS_SUPTITLE, fontweight="bold", y=0.97,
    )
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.21, top=0.76, wspace=0.24)
    _save(fig, "fig5_leadlag")


# =====================================================================
# ED FIGURES
# =====================================================================
def plot_ed_fig1_composition_bias():
    raw = _csv(os.path.join(DATA_DIR, "digital_signal_annual_diff.csv"))
    adj = _csv(os.path.join(DATA_DIR, "digital_signal_annual_band_adjusted.csv"))
    required = ["year", "p_factor_severity_z_diff"]
    _require_columns(raw, required, "edfig1 raw series")
    _require_columns(adj, required, "edfig1 adjusted series")
    common_years = np.intersect1d(raw["year"].unique(), adj["year"].unique())
    raw = raw[raw["year"].isin(common_years)].sort_values("year")
    adj = adj[adj["year"].isin(common_years)].sort_values("year")

    fig, ax = plt.subplots(figsize=(ONE_HALF_COL, ONE_HALF_COL * 0.90))
    ax.plot(
        raw["year"], raw["p_factor_severity_z_diff"], "o--",
        color=C_RAW, ms=3, lw=1, label="Raw (mixed t0 windows)",
    )
    ax.plot(
        adj["year"], adj["p_factor_severity_z_diff"], "o-",
        color=C_ADJ, ms=4, lw=1.5, label="Band-adjusted (composition-equal)",
    )
    ax.axhline(0, color="#888888", lw=0.8)
    _set_year_axis(ax)
    ax.set_ylabel("Patient−control p-factor gap (z)")
    ax.legend(
        frameon=False, fontsize=FS_LEGEND, loc="upper center",
        bbox_to_anchor=(0.5, -0.19), ncol=2, handlelength=2.2,
    )
    ax.set_title(
        "Composition-Bias Correction",
        fontsize=FS_TITLE, fontweight="bold", pad=20,
    )
    _despine(ax)
    _finish_single_panel(fig, ax, left=0.18, bottom=0.25, top=0.84)
    _save(fig, "ed_fig1_composition_bias", "ed")


def plot_ed_fig2_disorder_trends():
    adj_m = _csv(os.path.join(DATA_DIR, "digital_signal_monthly_by_disorder_band_adj.csv"))
    _require_columns(
        adj_m,
        ["period", "disorder_group", "p_factor_severity_z_diff", "n_bands"],
        "edfig2",
    )
    adj_m["period"] = pd.to_datetime(adj_m["period"])
    piv = adj_m.pivot_table(
        index="disorder_group", columns="period",
        values="p_factor_severity_z_diff", aggfunc="mean",
    ).sort_index()

    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.78, DOUBLE_COL * 0.70))
    for disorder, row in piv.iterrows():
        ax.plot(
            piv.columns, row.values, "o-", ms=2, lw=0.7, alpha=0.70,
            label=DISPLAY.get(disorder, disorder),
        )
    from matplotlib.dates import YearLocator, DateFormatter

    ax.xaxis.set_major_locator(YearLocator(2))
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    date_span = piv.columns.max() - piv.columns.min()
    ax.set_xlim(
        piv.columns.min() - date_span * 0.02,
        piv.columns.max() + date_span * 0.02,
    )
    ax.set_xlabel("Calendar month")
    ax.set_ylabel("P-factor patient−control gap (z)")
    ax.set_title(
        "Disorder-Specific Monthly Trajectories",
        fontsize=FS_TITLE, fontweight="bold", pad=20,
    )
    ax.legend(
        frameon=False, fontsize=FS_LEGEND, ncol=5,
        loc="upper center", bbox_to_anchor=(0.5, -0.17),
        handlelength=1.8, columnspacing=1.2,
    )
    ax.axhline(0, color="#888888", lw=0.6)
    ax.set_box_aspect(1)
    _despine(ax)
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.24, top=0.86)
    _save(fig, "ed_fig2_disorder_trends", "ed")


def plot_ed_fig3_disorder_covid():
    its = _csv(os.path.join(STATS_DIR, "covid_its_corrected_results.csv"))
    required = [
        "disorder", "metric", "b_post_trend", "se_post_trend",
        "p_post_trend_fdr",
    ]
    _require_columns(its, required, "edfig3")
    pf = its[its["metric"] == "p_factor_severity_z_diff"].copy()
    valid = (
        np.isfinite(pf["b_post_trend"])
        & np.isfinite(pf["se_post_trend"])
        & (pf["se_post_trend"] >= 0)
    )
    if (~valid).any():
        logger.warning(f"  edfig3: omitted {(~valid).sum()} rows with invalid estimates/SE")
    pf = pf.loc[valid].sort_values("b_post_trend")

    fig, ax = plt.subplots(figsize=(ONE_HALF_COL, ONE_HALF_COL * 0.90))
    ys = np.arange(len(pf))
    ax.errorbar(
        pf["b_post_trend"], ys, xerr=1.96 * pf["se_post_trend"], fmt="none",
        ecolor="#999999", capsize=2, elinewidth=0.8,
    )
    ax.scatter(pf["b_post_trend"], ys, color=C_PAT, s=25, zorder=3)
    ax.axvline(0, color="#888888", lw=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels(
        [DISPLAY.get(d, d) for d in pf["disorder"]],
        fontsize=FS_TICK,
        fontweight="bold",
    )
    ax.set_xlabel("Monthly post-COVID slope change (95% CI)")
    ax.margins(x=0.10, y=0.06)
    ax.invert_yaxis()
    ax.set_title(
        "Disorder-Specific Post-COVID Trend Changes",
        fontsize=FS_TITLE, fontweight="bold", pad=20,
    )
    _despine(ax)
    _finish_single_panel(fig, ax, left=0.27, right=0.95, bottom=0.17, top=0.84)
    _save(fig, "ed_fig3_disorder_covid", "ed")


def plot_ed_fig4_leadlag_matrix():
    ll = _csv(os.path.join(STATS_DIR, "lead_lag_corrected_results.csv"))
    _require_columns(ll, ["lag", "r", "digital_signal", "external"], "edfig4")
    sigs = [
        "p_factor_severity_z_diff",
        "posting_intensity_diff",
        "social_withdrawal_index_diff",
        "volatility_index_diff",
    ]
    exts = ["MDE", "suicide_rate"]
    lags = [-1, 0, 1]
    mat = np.full((len(sigs), len(exts) * len(lags)), np.nan)
    for i, s in enumerate(sigs):
        for j, e in enumerate(exts):
            for k, lag in enumerate(lags):
                r = ll[(ll.digital_signal == s) & (ll.external == e) & (ll.lag == lag)]
                if not r.empty and not np.isnan(r["r"].iloc[0]):
                    mat[i, j * 3 + k] = r["r"].iloc[0]

    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        raise ValueError("edfig4: no finite correlation values")
    if np.any(np.abs(finite) > 1 + 1e-12):
        raise ValueError("edfig4: correlation outside the valid [-1, 1] range")

    # Correlations have a meaningful fixed full range, unlike an arbitrary
    # effect matrix whose limits should be inferred from observed values.
    scale_max = 1.0
    text_threshold = _heatmap_cell_white_threshold(scale_max)
    fig, ax = _ed_heatmap_figure()
    im = _ed_heatmap_imshow(ax, mat, scale_max=scale_max)
    _ed_heatmap_annotate(ax, mat, text_threshold)

    ax.set_xticks(range(len(exts) * len(lags)))
    ax.set_xticklabels(["−1", "0", "+1"] * len(exts), fontsize=FS_TICK, fontweight="bold")
    ax.set_xlabel("Lag (external year − digital year)", fontsize=FS_AXLABEL)
    ax.set_yticks(range(len(sigs)))
    signal_labels = [
        "P-factor severity gap",
        "Posting-intensity gap",
        "Social-withdrawal gap",
        "Volatility gap",
    ]
    ax.set_yticklabels(signal_labels, fontsize=FS_TICK, fontweight="bold")
    top_axis = ax.secondary_xaxis("top")
    top_axis.set_xticks([1, 4])
    top_axis.set_xticklabels(["MDE prevalence", "Suicide rate"], fontweight="bold")
    top_axis.tick_params(axis="x", length=0, pad=6, labelsize=FS_TICK)

    _ed_heatmap_style_frame(ax)
    _ed_heatmap_colorbar(fig, ax, im, "Pearson r")
    _ed_heatmap_title(
        ax,
        "Lead-Lag Correlations with External Indicators",
    )
    _save(fig, "ed_fig4_leadlag_matrix", "ed")


PLOT_REGISTRY = {
    "fig1": plot_fig1_overview,
    "fig2": plot_fig2_trends,
    "fig3": plot_fig3_covid_its,
    "fig4": plot_fig4_events,
    "fig5": plot_fig5_leadlag,
    "edfig1": plot_ed_fig1_composition_bias,
    "edfig2": plot_ed_fig2_disorder_trends,
    "edfig3": plot_ed_fig3_disorder_covid,
    "edfig4": plot_ed_fig4_leadlag_matrix,
}


def run_visualizations(selected_plots):
    logger.info("=" * 70)
    logger.info("Module 4 Publication Visualizations (lyc — Modular CLI)")
    logger.info("=" * 70)

    if "all" in selected_plots:
        targets = list(PLOT_REGISTRY.keys())
    else:
        targets = [p for p in selected_plots if p in PLOT_REGISTRY]
        invalid = [p for p in selected_plots if p not in PLOT_REGISTRY and p != "all"]
        if invalid:
            logger.warning(f"Unrecognized plot commands: {invalid}")
            logger.info(f"Available: {list(PLOT_REGISTRY.keys())}")

    if not targets:
        logger.error("No valid plots selected. Exiting.")
        sys.exit(1)

    logger.info(f"Targets: {targets}\n")
    failed = []
    for target in targets:
        func = PLOT_REGISTRY[target]
        try:
            func()
        except Exception as e:
            logger.error(f"Error executing {target}: {e}")
            failed.append(target)

    logger.info("\n" + "=" * 70)
    if failed:
        logger.error(f"FAILED PLOTS: {failed}")
    else:
        logger.info("TARGETED VISUALIZATIONS COMPLETE")
    logger.info(f"  Main: {MAIN_DIR}")
    logger.info(f"  ED:   {ED_DIR}")
    logger.info("=" * 70)
    if failed:
        raise RuntimeError(f"Failed to generate: {', '.join(failed)}")


def run_all():
    run_visualizations(["all"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Modular CLI for Module 4 visualizations.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-p", "--plots",
        nargs="+",
        default=["all"],
        help=(
            "Plots to generate.\n"
            "Example: python s46_figures_lyc.py -p fig1 fig3 edfig2\n"
            "Options: 'all', 'fig1'-'fig5', 'edfig1'-'edfig4'"
        ),
    )
    args = parser.parse_args()
    run_visualizations(args.plots)
