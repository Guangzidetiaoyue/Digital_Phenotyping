"""
Module 2 Publication Figures (v2_lyc) — Nature/Science-grade
=============================================================
Main figures -> module2_results/figures_lyc/main/
SI figures    -> module2_results/figures_lyc/ed/

CLI: python s12_figures_v2_lyc.py -p fig1 edfig3
      python s12_figures_v2_lyc.py -p all
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
from matplotlib.ticker import AutoMinorLocator
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats

from config_module2 import (
    M2_OUTPUT_DIR, STATS_DIR, TRAJECTORY_DIR,
    DISORDER_DISPLAY_NAMES, HITOP_SPECTRA, HITOP_COLORS,
)

# LYC figures always go to figures_lyc/ (config_module2 uses figures/ for legacy s12).
M2_FIGURES_LYC_DIR = os.path.join(M2_OUTPUT_DIR, "figures_lyc")
MAIN_DIR = os.environ.get("M2_MAIN_DIR", os.path.join(M2_FIGURES_LYC_DIR, "main"))
ED_DIR = os.environ.get("M2_ED_DIR", os.path.join(M2_FIGURES_LYC_DIR, "ed"))
os.makedirs(MAIN_DIR, exist_ok=True)
os.makedirs(ED_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.environ.get("VIZ_LOG", os.path.join(M2_FIGURES_LYC_DIR, "visualization.log"))),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ---------- global style ----------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.titleweight": "bold",
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6.5,
    "figure.titlesize": 9,
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
    "axes.labelpad": 4.0,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def mm2inch(mm: float) -> float:
    return mm / 25.4


SINGLE_COL = mm2inch(89)
DOUBLE_COL = mm2inch(183)

FS_TITLE = 8
FS_AXLABEL = 7
FS_TICK = 6
FS_ANNOT = 6
FS_LEGEND = 6.5
FS_SUPTITLE = 9

# Three-panel main figures: fixed drawing areas and physical whitespace.
# Canvas dimensions are derived from these values rather than standardized.
THREE_PANEL_W = 2.35
THREE_PANEL_H = 2.05
THREE_HERO_H = 2.35
THREE_COL_GAP = 0.75
THREE_ROW_GAP = 1.10
THREE_OUTER_WHITESPACE = 0.35

# ED 柱状/条形图 — 详见 lyc/柱状图规范.md（与 Module1 常量一致）
BAR_EDGE_COLOR = "#222222"
BAR_EDGE_LW = 0.6
BAR_W_SIMPLE = 0.55
BAR_W_GROUP = 0.4
BAR_GROUP_OFF = 0.2
BAR_H_HORZ = 0.7
BAR_Y_PAD = 1.12
BAR_ERR_KW = dict(elinewidth=0.6, capsize=1.5)
BAR_ED_W = 1920 / 300
BAR_ED_H = BAR_ED_W * 0.90
BAR_MARGIN_LEFT = 0.14
BAR_MARGIN_RIGHT = 0.94
BAR_MARGIN_TOP = 0.90
BAR_MARGIN_BOTTOM = 0.12
BAR_MARGIN_BOTTOM_LEGEND = 0.20
BAR_MARGIN_LEFT_LONG_LABEL = 0.22

# Single-panel line / estimate figures — see lyc/折线轨迹图规范.md and
# lyc/误差棒森林图规范.md
LINE_WIDTH = 1.5
LINE_MARKER_SIZE = 4
EST_MARKER_SIZE = 5
EST_LINE_WIDTH = 0.8
EST_CAPSIZE = 2
PLOT_ZERO_COLOR = "#BBBBBB"
PLOT_ZERO_LW = 0.6

PAT_CTRL = {"patient": "#E53935", "control": "#1E88E5"}
SUB_COLORS = ["#E53935", "#1E88E5", "#009E73", "#FB8C00", "#8E24AA"]
DISPLAY = {
    "depression": "Depression", "bipolar": "Bipolar Disorder", "anxiety": "Anxiety Disorders",
    "ocd": "OCD", "ptsd": "PTSD", "bpd": "Borderline PD", "eating_disorders": "Eating Disorders",
    "schizophrenia": "Schizophrenia", "adhd": "ADHD", "autism": "Autism Spectrum",
    "control": "Control", "all": "All",
}


def get_disorder_color(disorder):
    for spectrum, disorders in HITOP_SPECTRA.items():
        if disorder in disorders:
            return HITOP_COLORS.get(spectrum, "#808080")
    return "#808080"


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


def _bar_style_ax(ax, *, ylabel=None, xlabel=None):
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS_AXLABEL)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FS_AXLABEL)
    ax.tick_params(labelsize=FS_TICK, direction="out")
    ax.set_box_aspect(1)


def _bar_figure(*, left=BAR_MARGIN_LEFT, bottom=BAR_MARGIN_BOTTOM):
    """Compact canvas with a 1:1 plotting area for bar charts."""
    fig = plt.figure(figsize=(BAR_ED_W, BAR_ED_H))
    fig.subplots_adjust(
        left=left,
        right=BAR_MARGIN_RIGHT,
        top=BAR_MARGIN_TOP,
        bottom=bottom,
    )
    ax = fig.add_subplot(111)
    ax.set_box_aspect(1)
    return fig, ax


def _bar_title(ax, title):
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold",
                 loc="center", pad=20)


def _bar_ylim_from_zero(ax, ymax):
    if ymax is not None and ymax > 0:
        ax.set_ylim(0, ymax * BAR_Y_PAD)


def _bar_vert_grouped_two(ax, x, series_a, series_b, *, colors=None):
    """Two-series grouped vertical bars at each x."""
    colors = colors or (PAT_CTRL["patient"], PAT_CTRL["control"])
    ax.bar(x - BAR_GROUP_OFF, series_a, BAR_W_GROUP, color=colors[0],
           edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_LW, label="_nolegend_")
    ax.bar(x + BAR_GROUP_OFF, series_b, BAR_W_GROUP, color=colors[1],
           edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_LW, label="_nolegend_")
    ax.set_xticks(x)
    ya = max(max(series_a, default=0), max(series_b, default=0))
    _bar_ylim_from_zero(ax, ya)


def _bar_vert_simple(ax, x, heights, colors, *, width=BAR_W_SIMPLE):
    ax.bar(x, heights, width, color=colors, edgecolor=BAR_EDGE_COLOR,
           linewidth=BAR_EDGE_LW, label="_nolegend_")
    ax.set_xticks(x if hasattr(x, "__len__") else [x])


def _bar_horz_simple(ax, labels, values, *, color="#3C5488", xerr=None):
    ax.barh(labels, values, height=BAR_H_HORZ, color=color,
            edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_LW, xerr=xerr,
            error_kw=BAR_ERR_KW)
    ax.invert_yaxis()
    if xerr is not None:
        xmax = (np.asarray(values) + np.asarray(xerr)).max()
    else:
        xmax = np.max(values) if len(values) else 0
    if np.isfinite(xmax) and xmax > 0:
        ax.set_xlim(0, xmax * BAR_Y_PAD)


def _bar_annotate_values_vert(ax, x, values, *, pad_frac=0.35, fmt="{:.3f}"):
    vals = list(values)
    if not vals:
        return
    d_min, d_max = min(vals), max(vals)
    pad = max(0.02, (d_max - d_min) * 0.15) if vals else 0.05
    for xi, d in zip(x, vals):
        y_off = pad * pad_frac if d >= 0 else -pad * pad_frac
        va = "bottom" if d >= 0 else "top"
        ax.text(xi, d + y_off, fmt.format(d), ha="center", va=va, fontsize=FS_ANNOT, zorder=5)


def _bar_annotate_grouped(ax, x, series_a, series_b, *, fmt="{:,.0f}"):
    ymax = max(max(series_a, default=0), max(series_b, default=0))
    pad = ymax * 0.018 if ymax > 0 else 0.02
    for xpos, values in ((np.asarray(x) - BAR_GROUP_OFF, series_a),
                         (np.asarray(x) + BAR_GROUP_OFF, series_b)):
        for xi, value in zip(xpos, values):
            ax.text(xi, value + pad, fmt.format(value), ha="center",
                    va="bottom", fontsize=FS_ANNOT)


def _line_figure(*, bottom=BAR_MARGIN_BOTTOM_LEGEND):
    """Compact canvas with a 1:1 plotting area for a multi-series line chart."""
    return _bar_figure(bottom=bottom)


def _line_style_ax(ax, *, xlabel=None, ylabel=None, title=None):
    _bar_style_ax(ax, xlabel=xlabel, ylabel=ylabel)
    _despine(ax)
    if title:
        _bar_title(ax, title)


def _estimate_figure(*, left=BAR_MARGIN_LEFT_LONG_LABEL,
                     bottom=BAR_MARGIN_BOTTOM):
    """Compact canvas with a 1:1 plotting area for errorbar / forest charts."""
    return _bar_figure(left=left, bottom=bottom)


def _estimate_style_ax(ax, *, xlabel=None, ylabel=None, title=None):
    _bar_style_ax(ax, xlabel=xlabel, ylabel=ylabel)
    _despine(ax)
    if title:
        _bar_title(ax, title)


def _forest_errorbar_h(ax, x, y, lo, hi, *, color):
    ax.errorbar(
        x, y, xerr=[[lo], [hi]], fmt="o", color=color,
        markersize=EST_MARKER_SIZE, elinewidth=EST_LINE_WIDTH,
        capsize=EST_CAPSIZE, zorder=3,
    )


def _add_panel_label(ax, label, x=-0.12, y=1.04, fs=FS_TITLE):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fs, fontweight="bold",
            va="top", ha="right")


def _load_csv(name, **kw):
    p = os.path.join(STATS_DIR, name)
    if not os.path.exists(p):
        logger.warning(f"  Missing: {name}")
        return pd.DataFrame()
    return pd.read_csv(p, **kw)


def _load_json(name):
    p = os.path.join(STATS_DIR, name)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


# =====================================================================
# FIGURE M2-1: Trajectory + Event Study (core narrative)
# =====================================================================
def _add_panel_label_topleft(ax, label, title, x_offset=0.0, y=1.05, fs=FS_TITLE):
    """Panel label at the top-left corner of the axes boundary.

    Uses a blended transform (figure-x × axes-y) so that the **x**
    position is in *figure* coordinates — identical absolute pixel
    position for every panel regardless of axes width — while **y**
    stays relative to each axes (text sits just above the axes top).
    This guarantees that panels a (full width) and b (half width)
    have perfectly aligned labels.
    """
    from matplotlib.transforms import blended_transform_factory
    # Compute the figure-fraction x of the axes left edge, then shift left.
    _ax_left = ax.get_position().x0
    _x_fig = _ax_left + x_offset
    _trans = blended_transform_factory(ax.figure.transFigure, ax.transAxes)
    ax.text(_x_fig, y, f"{label}  {title}", transform=_trans, fontsize=fs,
            fontweight="bold", va="bottom", ha="left")


def _legend_below(ax, handles, labels, ncol=2, pad=0.18):
    """Place legend below the axes with 3:1 (L:W) color rectangles (small)."""
    leg = ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -pad),
                    ncol=ncol, frameon=False, fontsize=FS_LEGEND,
                    handlelength=1.2, handleheight=0.4, handletextpad=0.4,
                    columnspacing=0.8)
    return leg


def _three_panel_figure(*, left_text=0.40, right_text=0.20,
                        top_text=0.35, bottom_text=0.40):
    """Create fixed-size hero/left/right axes with fixed outer whitespace."""
    hero_w = THREE_PANEL_W * 2 + THREE_COL_GAP
    left_space = THREE_OUTER_WHITESPACE + left_text
    right_space = THREE_OUTER_WHITESPACE + right_text
    top_space = THREE_OUTER_WHITESPACE + top_text
    bottom_space = THREE_OUTER_WHITESPACE + bottom_text
    fig_w = left_space + hero_w + right_space
    fig_h = (
        bottom_space + THREE_PANEL_H + THREE_ROW_GAP
        + THREE_HERO_H + top_space
    )
    fig = plt.figure(figsize=(fig_w, fig_h))

    def _axes_at(x, y, width, height):
        return fig.add_axes([
            x / fig_w, y / fig_h, width / fig_w, height / fig_h
        ])

    y_hero = bottom_space + THREE_PANEL_H + THREE_ROW_GAP
    ax_hero = _axes_at(left_space, y_hero, hero_w, THREE_HERO_H)
    ax_left = _axes_at(left_space, bottom_space, THREE_PANEL_W, THREE_PANEL_H)
    ax_right = _axes_at(
        left_space + THREE_PANEL_W + THREE_COL_GAP,
        bottom_space,
        THREE_PANEL_W,
        THREE_PANEL_H,
    )
    return fig, ax_hero, ax_left, ax_right


def _save_main_layout(fig, name):
    """Save a derived main-figure canvas without changing its margins."""
    original_bbox = plt.rcParams.get("savefig.bbox")
    plt.rcParams["savefig.bbox"] = None
    try:
        for fmt in ("pdf", "png"):
            fig.savefig(os.path.join(MAIN_DIR, f"{name}.{fmt}"),
                        dpi=300, bbox_inches=None, pad_inches=0)
    finally:
        plt.rcParams["savefig.bbox"] = original_bbox
        plt.close(fig)
    logger.info(f"  Saved: main/{name} (pdf+png, fixed drawing areas)")


# ED figures: unified 1920 px width, 4:3 aspect (single-panel supplement layout).
ED_FIG_W = 1920 / 300
ED_FIG_H = ED_FIG_W * 0.75


def _ed_figure(has_legend=True, left_text=0.45, bottom_extra=None):
    """Create a single-panel ED figure with margins tuned for bottom legends."""
    fig = plt.figure(figsize=(ED_FIG_W, ED_FIG_H))
    _vm = 0.30
    if bottom_extra is None:
        bottom_extra = 0.35 if has_legend else 0.22
    fig.subplots_adjust(
        left=(_vm + left_text) / ED_FIG_W,
        right=1 - _vm / ED_FIG_W,
        top=1 - (_vm + 0.08) / ED_FIG_H,
        bottom=(_vm + bottom_extra) / ED_FIG_H,
    )
    ax = fig.add_subplot(111)
    return fig, ax


def _save_ed(fig, name):
    """Save ED figure at fixed 1920 px canvas width (no tight crop)."""
    _orig_bbox = plt.rcParams.get("savefig.bbox")
    plt.rcParams["savefig.bbox"] = None
    for fmt in ("pdf", "png"):
        fig.savefig(os.path.join(ED_DIR, f"{name}.{fmt}"), dpi=300, pad_inches=0)
    plt.rcParams["savefig.bbox"] = _orig_bbox
    plt.close(fig)
    logger.info(f"  Saved: ed/{name} (pdf+png, 1920px width)")


def fig1_trajectory_eventstudy():
    """Main Fig 1: (a) p-factor trajectories patient vs control,
    (b) event-study coefficients with FDR shading, (c) per-disorder
    signal lead time."""
    agg = pd.read_csv(os.path.join(TRAJECTORY_DIR, "trajectory_aggregated_by_window.csv"))
    es = _load_csv("event_study_results.csv")
    lead = _load_csv("signal_lead_time.csv")

    fig, ax_hero, ax_left, ax_right = _three_panel_figure()

    # (a) Trajectories — spans full width
    ax = ax_hero
    for grp, c in PAT_CTRL.items():
        d = agg[(agg["group"] == grp) & (agg["window_index"].between(-24, 24))]
        ax.plot(d["days_from_t0_center"], d["p_factor_severity_z_mean"], color=c,
                lw=1.5)
        ax.fill_between(d["days_from_t0_center"],
                        d["p_factor_severity_z_mean"] - d["p_factor_severity_z_se"],
                        d["p_factor_severity_z_mean"] + d["p_factor_severity_z_se"],
                        color=c, alpha=0.15, lw=0)
    ax.axvline(0, color="#333", lw=0.8, ls="--")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xlim(-730, 730)
    ax.set_xlabel("Days from t$_{0}$", fontsize=FS_AXLABEL)
    ax.set_ylabel("P-factor severity (z)", fontsize=FS_AXLABEL)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(labelsize=FS_TICK)
    _add_panel_label_topleft(ax, "a", "P-factor trajectories")
    _despine(ax)
    # Legend below with 3:1 rectangles
    h_a = [mpatches.Patch(color="#E53935", label="Patient"),
           mpatches.Patch(color="#1E88E5", label="Control")]
    _legend_below(ax, h_a, ["Patient", "Control"], ncol=2, pad=0.15)

    # (b) Event study — bottom-left, y-axis aligned with (a)
    ax = ax_left
    ax.axhline(0, color="#888", lw=0.8)
    ax.axvline(0, color="#888", lw=0.8, ls="--")
    pre = es["days_from_t0"] < -45
    post = es["days_from_t0"] > 45
    ax.fill_between(es.loc[pre & (es["p_fdr"] < 0.05), "days_from_t0"], -0.3, 0.3,
                    color="#E53935", alpha=0.08)
    ax.fill_between(es.loc[post & (es["p_fdr"] < 0.05), "days_from_t0"], -0.3, 0.3,
                    color="#1E88E5", alpha=0.08)
    ax.errorbar(es.loc[pre, "days_from_t0"], es.loc[pre, "coef"],
                yerr=1.96 * es.loc[pre, "se"], fmt="o", ms=3, capsize=1.5,
                color="#1E88E5", ecolor="#1E88E5", elinewidth=0.8)
    ax.errorbar(es.loc[post, "days_from_t0"], es.loc[post, "coef"],
                yerr=1.96 * es.loc[post, "se"], fmt="o", ms=3, capsize=1.5,
                color="#E53935", ecolor="#E53935", elinewidth=0.8)
    sig = es["p_fdr"] < 0.05
    ax.scatter(es.loc[sig, "days_from_t0"], es.loc[sig, "coef"], s=18,
               color="none", edgecolors="black", lw=0.5, zorder=5)
    ax.set_xlim(-730, 730)
    ax.set_ylim(-0.15, 0.15)
    ax.set_xlabel("Days from t$_{0}$", fontsize=FS_AXLABEL)
    ax.set_ylabel("Event-study coef.", fontsize=FS_AXLABEL)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(labelsize=FS_TICK)
    _add_panel_label_topleft(ax, "b", "Event-study coefficients")
    _despine(ax)
    # Legend below
    h_b = [mpatches.Patch(color="#1E88E5", label="Pre-t$_{0}$"),
           mpatches.Patch(color="#E53935", label="Post-t$_{0}$"),
           mpatches.Patch(facecolor="none", edgecolor="black", label="FDR < 0.05")]
    _legend_below(ax, h_b, ["Pre-t$_{0}$", "Post-t$_{0}$", "FDR < 0.05"], ncol=3, pad=0.15)

    # (c) Signal lead time — bottom-right, right edge aligned with (a)
    ax = ax_right
    if not lead.empty:
        disorders = [r for r in lead["disorder"].tolist() if r != "all"]
        vals = [lead.loc[lead["disorder"] == d, "median_lead_days"].iloc[0]
                for d in disorders if d in lead["disorder"].values]
        lbls = [DISPLAY.get(d, d) for d in disorders]
        order = np.argsort(vals)
        bar_labels = [lbls[i] for i in order]
        bar_vals = [vals[i] for i in order]
        # Distinct color per bar
        cmap = plt.get_cmap("tab10")
        bar_colors = [cmap(i % 10) for i in range(len(bar_labels))]
        ax.barh(bar_labels, bar_vals, color=bar_colors, edgecolor="none", height=0.7)
        ax.set_xlabel("Median signal lead (days)", fontsize=FS_AXLABEL)
        ax.set_ylabel("")
        ax.set_xlim(0, max(vals) * 1.15)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(labelsize=FS_TICK)
    _add_panel_label_topleft(ax, "c", "Signal lead time by disorder")
    _despine(ax)

    _save_main_layout(fig, "fig1_trajectory_eventstudy")


# =====================================================================
# FIGURE M2-2: Changepoint analysis
# =====================================================================
def fig2_changepoints():
    """Main Fig 2: (a) changepoint distribution patient vs control (hero),
    (b) PELT sensitivity, (c) per-disorder pre-t0 rate.

    Layout mirrors fig1: one wide hero panel on top (a), two panels
    side-by-side on the bottom (b, c).
    """
    sens = _load_csv("pelt_sensitivity_analysis.csv")
    cp = _load_csv("changepoint_analysis.csv")

    fig, ax_hero, ax_left, ax_right = _three_panel_figure()

    # ===================== (a) Changepoint distribution (hero) ==========
    ax = ax_hero
    if not cp.empty:
        for grp, c in PAT_CTRL.items():
            days = cp.loc[cp["group"] == grp, "changepoint_days_from_t0"]
            sns.kdeplot(days, ax=ax, color=c, label=grp.capitalize(),
                        lw=1.5, fill=True, alpha=0.2)
        ax.axvline(0, color="#333", lw=0.8, ls="--")
    ax.set_xlabel("Changepoint day (rel. t$_{0}$)", fontsize=FS_AXLABEL)
    ax.set_ylabel("Density", fontsize=FS_AXLABEL)
    ax.tick_params(labelsize=FS_TICK)
    _add_panel_label_topleft(ax, "a", "Changepoint distribution")
    _despine(ax)
    if not cp.empty:
        h_a = [mpatches.Patch(color=PAT_CTRL["patient"], label="Patient"),
               mpatches.Patch(color=PAT_CTRL["control"], label="Control")]
        _legend_below(ax, h_a, ["Patient", "Control"], ncol=2, pad=0.15)

    # ===================== (b) PELT sensitivity ========================
    ax = ax_left
    if not sens.empty:
        # 只保留 penalty_multiplier <= 4.0 的数据点
        sens_plot = sens[sens["penalty_multiplier"] <= 4.0]
        ax.plot(sens_plot["penalty_multiplier"], sens_plot["avg_cps_per_user"], "o-",
                color="#1E88E5", ms=4)
        ax.set_xlabel("Penalty multiplier", fontsize=FS_AXLABEL)
        ax.set_ylabel("Avg changepoints / user", fontsize=FS_AXLABEL)
        ax.set_xlim(0, 4.25)
        # 主刻度: 0~4.0 每 0.5 一个，均匀易读
        ax.set_xticks(np.arange(0, 4.1, 0.5))
        # 次刻度: 对应实际数据点(0.25,0.75,1.25,1.75)，无标签
        from matplotlib.ticker import FixedLocator, ScalarFormatter
        ax.xaxis.set_minor_locator(FixedLocator(
            [0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75]))
        # y 轴: 线性刻度 0~0.1 均分，不使用科学计数法
        ax.set_ylim(0, 0.11)
        ax.set_yticks(np.arange(0, 0.11, 0.02))
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.tick_params(labelsize=FS_TICK)
    _add_panel_label_topleft(ax, "b", "PELT sensitivity")
    _despine(ax)

    # ===================== (c) Pre-t0 changepoint fraction ============
    ax = ax_right
    if not cp.empty:
        pat = cp[cp["group"] == "patient"]
        fracs = pat.groupby("disorder_group")["is_pre_t0"].mean() * 100
        fracs = fracs.sort_values()
        colors = ["#E53935" if v > 55 else "#1E88E5" for v in fracs]
        ys = np.arange(len(fracs))
        ax.scatter(fracs.values, ys, s=40, c=colors, edgecolors="white",
                   lw=0.5, zorder=3)
        for yi, (d, v) in zip(ys, fracs.items()):
            ax.text(v + 2, yi, f"{v:.0f}%", va="center", fontsize=FS_ANNOT)
        ax.axvline(50, color="#888", lw=0.8, ls="--")
        ax.set_yticks(ys)
        ax.set_yticklabels([DISPLAY.get(d, d) for d in fracs.index],
                           fontsize=FS_TICK)
        ax.set_xlabel("Pre-t$_{0}$ changepoints (%)", fontsize=FS_AXLABEL)
        ax.set_xlim(0, 110)
        ax.tick_params(labelsize=FS_TICK)
    _add_panel_label_topleft(ax, "c", "Pre-t$_{0}$ changepoint fraction")
    _despine(ax)

    _save_main_layout(fig, "fig2_changepoints")


# =====================================================================
# FIGURE M2-3: Heterogeneity / subtypes
# =====================================================================
def fig3_subtypes():
    """Main Fig 3: (a) subtype trajectories, (b) subtype x disorder
    distribution, (c) GMM model selection (BIC/Silhouette).

    Layout mirrors fig1/fig2: one wide hero panel on top (a), two panels
    side-by-side on the bottom (b, c).
    """
    sub_agg = _load_csv("subtype_aggregated_trajectories.csv")
    gmm = _load_csv("gmm_model_selection.csv")
    dis_analysis = _load_json("disorder_subtype_analysis.json")

    # Keep the same drawing areas as the other three-panel figures.
    # Extra text allowances enlarge only the canvas around those areas.
    fig, ax_hero, ax_left, ax_right = _three_panel_figure(
        left_text=0.80,
        right_text=0.55,
    )

    # ===================== (a) Subtype trajectories (hero) ============
    ax = ax_hero
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
        ax.set_xlabel("Days from t$_{0}$", fontsize=FS_AXLABEL)
        ax.set_ylabel("P-factor severity (z)", fontsize=FS_AXLABEL)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(labelsize=FS_TICK)
    _add_panel_label_topleft(ax, "a", "Subtype trajectories")
    _despine(ax)
    if not sub_agg.empty:
        h_a = [mpatches.Patch(color=SUB_COLORS[i], label=f"S{i}")
               for i in sorted(sub_agg["subtype_label"].unique())]
        l_a = [f"S{i}" for i in sorted(sub_agg["subtype_label"].unique())]
        _legend_below(ax, h_a, l_a, ncol=len(h_a), pad=0.15)

    # ===================== (b) Subtype × disorder stacked =============
    ax = ax_left
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
            ax.set_xlabel("Proportion (%)", fontsize=FS_AXLABEL)
            ax.set_ylabel("")
            ax.set_xlim(0, 100)
            ax.tick_params(labelsize=FS_TICK)
    _add_panel_label_topleft(ax, "b", "Subtype × disorder")
    _despine(ax)

    # ===================== (c) GMM model selection ====================
    ax = ax_right
    if not gmm.empty:
        ax2 = ax.twinx()
        ax.plot(gmm["n_components"], gmm["silhouette"], "o-", color="#E53935", ms=4)
        ax.set_xlabel("k components", fontsize=FS_AXLABEL)
        ax.set_ylabel("Silhouette", color="#E53935", fontsize=FS_AXLABEL)
        ax.tick_params(axis="y", labelcolor="#E53935", labelsize=FS_TICK)
        ax2.plot(gmm["n_components"], gmm["bic"] / 1e5, "s--", color="#1E88E5", ms=4)
        ax2.set_ylabel("BIC (×10$^{5}$)", color="#1E88E5", fontsize=FS_AXLABEL)
        ax2.tick_params(axis="y", labelcolor="#1E88E5", labelsize=FS_TICK)
        ax.set_xticks(gmm["n_components"].values)
        ax2.spines["top"].set_visible(False)
        ax.tick_params(axis="x", labelsize=FS_TICK)
    _add_panel_label_topleft(ax, "c", "GMM model selection")
    _despine(ax)
    if not gmm.empty:
        h_c = [mpatches.Patch(color="#E53935", label="Silhouette"),
               mpatches.Patch(color="#1E88E5", label="BIC")]
        _legend_below(ax, h_c, ["Silhouette", "BIC"], ncol=2, pad=0.15)

    _save_main_layout(fig, "fig3_subtypes")


# =====================================================================
# FIGURE M2-4: Causal attribution
# =====================================================================
def fig4_causal():
    """Main Fig 4: (a) dose-response scatter with OLS fit, (b) DID estimates
    (3 methods), (c) placebo distribution.

    Layout mirrors fig1/fig2: one wide hero panel on top (a), two panels
    side-by-side on the bottom (b, c).
    """
    did = _load_csv("did_results.csv")
    placebo = _load_csv("placebo_did_distribution.csv")
    ptest = _load_json("placebo_test_results.json")
    dose = pd.read_csv(os.path.join(STATS_DIR, "dose_response_scatter_data.csv"))

    fig, ax_hero, ax_left, ax_right = _three_panel_figure()

    # ===================== (a) Dose-response scatter (hero) ==========
    ax = ax_hero
    if not dose.empty:
        # Color scatter by disorder group
        disorder_cols = [c for c in dose.columns if c.startswith("disorder_")]
        disorder_names = [c.replace("disorder_", "") for c in disorder_cols]
        cmap = plt.get_cmap("tab10")
        disorder_colors = {d: cmap(i % 10) for i, d in enumerate(disorder_names)}

        for dcol, dname in zip(disorder_cols, disorder_names):
            mask = dose[dcol] == 1
            if mask.sum() > 0:
                ax.scatter(dose.loc[mask, "dose_auc"],
                           dose.loc[mask, "response_post_mean"],
                           s=3, color=disorder_colors[dname], alpha=0.3,
                           edgecolors="none", label=DISPLAY.get(dname, dname))

        # OLS fitted line (compute in-place since predicted_response is missing)
        from sklearn.linear_model import LinearRegression
        valid = dose.dropna(subset=["dose_auc", "response_post_mean"])
        if len(valid) > 50:
            # Trim outliers for axis limits (1st–99th percentile)
            x_lo, x_hi = valid["dose_auc"].quantile(0.01), valid["dose_auc"].quantile(0.99)
            y_lo, y_hi = valid["response_post_mean"].quantile(0.01), valid["response_post_mean"].quantile(0.99)
            ax.set_xlim(x_lo * 1.2, x_hi * 1.2)
            ax.set_ylim(y_lo * 1.2, y_hi * 1.2)

            X = valid[["dose_auc"]].values
            y = valid["response_post_mean"].values
            model = LinearRegression().fit(X, y)
            x_fit = np.linspace(max(x_lo * 1.1, valid["dose_auc"].min()),
                                 min(x_hi * 1.1, valid["dose_auc"].max()), 200)
            y_fit = model.predict(x_fit.reshape(-1, 1))
            ax.plot(x_fit, y_fit, color="#E53935", lw=1.5, ls="-",
                    label=f"OLS fit (slope={model.coef_[0]:.2e})")

            # Correlation
            r = np.corrcoef(valid["dose_auc"], valid["response_post_mean"])[0, 1]
            ax.text(0.02, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
                    fontsize=FS_ANNOT, va="top", fontweight="bold")

        ax.set_xlabel("Pre-t$_{0}$ cumulative severity (AUC)",
                      fontsize=FS_AXLABEL)
        ax.set_ylabel("Post-t$_{0}$ severity (z)", fontsize=FS_AXLABEL)
        ax.tick_params(labelsize=FS_TICK)
    _add_panel_label_topleft(ax, "a", "Dose-response")
    _despine(ax)
    if not dose.empty:
        h_a = [mpatches.Patch(color=disorder_colors[d], label=DISPLAY.get(d, d))
               for d in disorder_names if dose[f"disorder_{d}"].sum() > 0]
        _legend_below(ax, h_a, [DISPLAY.get(d, d) for d in disorder_names
                      if dose[f"disorder_{d}"].sum() > 0],
                      ncol=min(6, len(h_a)), pad=0.15)

    # ===================== (b) DID estimates (3 methods) ==============
    ax = ax_left
    if not did.empty:
        vnames = {"p_factor_severity_z": "P-factor",
                  "volatility_index": "Volatility",
                  "social_withdrawal_index": "Withdrawal",
                  "vader_compound_mean": "VADER"}
        method_styles = {
            "regression_clustered": {"marker": "o", "color": "#E53935",
                                     "label": "Regression (clustered SE)"},
            "pair_fe": {"marker": "s", "color": "#1E88E5",
                        "label": "Pair FE"},
            "iptw_weighted": {"marker": "^", "color": "#009E73",
                              "label": "IPTW weighted"},
        }
        variables = list(did["variable"].unique())
        offsets = [-0.2, 0, 0.2]  # vertical offset for 3 methods
        for vi, var in enumerate(variables):
            for mi, (method, style) in enumerate(method_styles.items()):
                row = did[(did["variable"] == var) & (did["method"] == method)]
                if row.empty:
                    continue
                est = row["did_estimate"].iloc[0]
                se = row["se"].iloc[0]
                p = row["p_value"].iloc[0]
                y_pos = vi + offsets[mi]
                ax.errorbar(est, y_pos, xerr=1.96 * se, fmt=style["marker"],
                            color=style["color"], ms=4, capsize=1.5, lw=1.0)
                # Significance marker
                if p < 0.05:
                    ax.text(est + 1.96 * se + 0.002, y_pos, "*",
                            fontsize=FS_ANNOT, va="center", ha="left",
                            color=style["color"], fontweight="bold")
        ax.axvline(0, color="#888", lw=0.8)
        ax.set_yticks(range(len(variables)))
        ax.set_yticklabels([vnames.get(v, v) for v in variables],
                           fontsize=FS_TICK)
        ax.set_xlabel("DID estimate (SD)", fontsize=FS_AXLABEL)
        ax.invert_yaxis()
        ax.tick_params(labelsize=FS_TICK)
    _add_panel_label_topleft(ax, "b", "DID estimates")
    _despine(ax)
    if not did.empty:
        h_b = [mpatches.Patch(color=s["color"], label=s["label"])
               for s in method_styles.values()]
        _legend_below(ax, h_b, [s["label"] for s in method_styles.values()],
                      ncol=3, pad=0.15)

    # ===================== (c) Placebo distribution ===================
    ax = ax_right
    if not placebo.empty:
        sns.histplot(placebo["placebo_did"], ax=ax, color="#1E88E5",
                     alpha=0.4, bins=30, stat="density")
        # KDE overlay
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(placebo["placebo_did"].dropna())
        x_kde = np.linspace(placebo["placebo_did"].min(),
                            placebo["placebo_did"].max(), 200)
        ax.plot(x_kde, kde(x_kde), color="#1E88E5", lw=1.5)
        td = ptest.get("true_did", np.nan)
        if not np.isnan(td):
            ax.axvline(td, color="#E53935", lw=2, ls="--")
            # Place True DID label above the axes, not overlapping the line
            ax.text(td, 1.02, f"True DID = {td:.4f}",
                    transform=ax.get_xaxis_transform(),
                    color="#E53935", fontsize=FS_ANNOT, ha="center",
                    va="bottom", fontweight="bold")
        ax.set_xlabel("Placebo DID estimate", fontsize=FS_AXLABEL)
        ax.set_ylabel("Density", fontsize=FS_AXLABEL)
        ax.tick_params(labelsize=FS_TICK)
    _add_panel_label_topleft(ax, "c", "Placebo test")
    _despine(ax)
    if not placebo.empty:
        h_c = [mpatches.Patch(color="#1E88E5", label="Placebo DID"),
               mpatches.Patch(color="#E53935", label="True DID")]
        _legend_below(ax, h_c, ["Placebo DID", "True DID"], ncol=2, pad=0.15)

    _save_main_layout(fig, "fig4_causal")


# =====================================================================
# FIGURE M2-5: Cross-disorder DID
# =====================================================================
def fig5_crossdisorder_did():
    """Main Fig 5: cross-disorder DID forest plot."""
    cd = _load_csv("cross_disorder_did.csv")

    fig, ax = _estimate_figure(bottom=BAR_MARGIN_BOTTOM_LEGEND)
    if not cd.empty:
        valid = (
            np.isfinite(cd["did_estimate"]) &
            np.isfinite(cd["se"]) &
            (cd["se"] >= 0)
        )
        cd = cd.loc[valid].copy()
        cd = cd.sort_values("did_estimate")
        ys = np.arange(len(cd))
        ax.axvline(0, color=PLOT_ZERO_COLOR, lw=PLOT_ZERO_LW, zorder=1)
        ci_low = cd["did_estimate"] - 1.96 * cd["se"]
        ci_high = cd["did_estimate"] + 1.96 * cd["se"]
        span = max(float(ci_high.max() - ci_low.min()), 0.01)
        for y, (_, r) in zip(ys, cd.iterrows()):
            c = "#E53935" if r["p_value"] < 0.05 else "#AAAAAA"
            err = 1.96 * r["se"]
            _forest_errorbar_h(ax, r["did_estimate"], y, err, err, color=c)
            label = f"{r['did_estimate']:.3f}" + ("*" if r["p_value"] < 0.05 else "")
            ax.text(r["did_estimate"] + err + span * 0.025, y, label,
                    ha="left", va="center", fontsize=FS_ANNOT, color=c)
        ax.set_xlim(float(ci_low.min()) - span * 0.08,
                    float(ci_high.max()) + span * 0.18)
        ax.set_yticks(ys)
        ax.set_yticklabels(
            [DISPLAY.get(d, d) for d in cd["disorder"]],
            fontsize=FS_TICK, fontweight="bold",
        )
        for label, disorder in zip(ax.get_yticklabels(), cd["disorder"]):
            label.set_color(get_disorder_color(disorder))
        ax.invert_yaxis()
    handles = [
        Line2D([0], [0], marker="o", color="#E53935", lw=0,
               markersize=4, label="p < 0.05"),
        Line2D([0], [0], marker="o", color="#AAAAAA", lw=0,
               markersize=4, label="Not significant"),
    ]
    _legend_below(ax, handles, ["p < 0.05", "Not significant"],
                  ncol=2, pad=0.12)
    _estimate_style_ax(
        ax,
        xlabel="DID estimate (SD)",
        title="Figure 5 | Cross-disorder DID Estimates",
    )
    _save(fig, "fig5_crossdisorder_did")


# =====================================================================
# ED FIGURES
# =====================================================================
def ed_fig1_quality_report():
    """ED Fig 1: trajectory quality metrics."""
    q = _load_json(os.path.join(TRAJECTORY_DIR, "trajectory_quality_report.json"))
    fig, ax = _bar_figure(bottom=BAR_MARGIN_BOTTOM_LEGEND)
    # The report has only an overall n_users_with_both, not group-specific
    # values. Do not duplicate that total into both grouped series.
    pats = [q.get("patient_with_sufficient_pre", 0),
            q.get("patient_with_sufficient_post", 0)]
    ctrls = [q.get("control_with_sufficient_pre", 0),
             q.get("control_with_sufficient_post", 0)]
    x = np.arange(2)
    _bar_vert_grouped_two(ax, x, pats, ctrls)
    _bar_annotate_grouped(ax, x, pats, ctrls)
    ax.set_xticklabels(["Sufficient pre", "Sufficient post"],
                       fontsize=FS_TICK, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:,.0f}"))
    _bar_style_ax(ax, ylabel="Users")
    _despine(ax)
    h = [mpatches.Patch(color="#E53935", label="Patient"),
         mpatches.Patch(color="#1E88E5", label="Control")]
    _legend_below(ax, h, ["Patient", "Control"], ncol=2, pad=0.12)
    _bar_title(ax, "ED Fig 1 | Trajectory Data Availability")
    _save_ed(fig, "ed_fig1_quality")


def ed_fig2_multivariate_changepoints():
    """ED Fig 2: multivariate changepoint consensus."""
    mv = _load_csv("multivariate_changepoints.csv")
    fig, ax = _bar_figure()
    if not mv.empty:
        pre = mv["is_pre_t0"].astype(int)
        n_pre = pre.value_counts().get(1, 0)
        n_post = pre.value_counts().get(0, 0)
        bx = np.array([0, 1])
        _bar_vert_simple(ax, bx, [n_pre, n_post], ["#E53935", "#1E88E5"], width=BAR_W_SIMPLE)
        _bar_annotate_values_vert(ax, bx, [n_pre, n_post],
                                  pad_frac=0.20, fmt="{:,.0f}")
        ax.set_xticklabels(["Pre-t$_{0}$", "Post-t$_{0}$"],
                           fontsize=FS_TICK, fontweight="bold")
        _bar_ylim_from_zero(ax, max(n_pre, n_post) * 1.04)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:,.0f}"))
    _bar_style_ax(ax, ylabel="Consensus changepoints")
    _despine(ax)
    _bar_title(ax, "ED Fig 2 | Multivariate Changepoint Consensus")
    _save_ed(fig, "ed_fig2_multivariate")


def ed_fig3_predictors():
    """ED Fig 3: subtype predictor permutation importance."""
    pred = _load_csv("subtype_predictors.csv")
    fig, ax = _bar_figure(left=BAR_MARGIN_LEFT_LONG_LABEL)
    if not pred.empty:
        top = pred.sort_values("importance_mean", ascending=False).head(15)
        labels = [p[:28] for p in top["feature"]]
        _bar_horz_simple(ax, labels, top["importance_mean"].values,
                         xerr=top["importance_std"].values)
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")
    _bar_style_ax(ax, xlabel="Permutation importance")
    _despine(ax)
    _bar_title(ax, "ED Fig 3 | Subtype Predictor Importance")
    _save_ed(fig, "ed_fig3_predictors")


def ed_fig4_t0_validation():
    """ED Fig 4: t0 validation (linguistic shift)."""
    meta = _load_json(os.path.join(TRAJECTORY_DIR, "trajectory_metadata.json"))
    fig, ax = _bar_figure()
    tv = meta.get("t0_validation", {})
    grps = [g for g in ("patient", "control") if g in tv]
    ds = [tv[g]["cohen_d"] for g in grps]
    x = np.arange(len(grps))
    colors = ["#E53935", "#1E88E5"][: len(grps)]
    _bar_vert_simple(ax, x, ds, colors, width=BAR_W_SIMPLE)
    ax.axhline(0, color="#BBBBBB", lw=0.6, zorder=1)
    ax.set_xticklabels([g.capitalize() for g in grps],
                       fontsize=FS_TICK, fontweight="bold")
    if ds:
        d_min, d_max = min(ds), max(ds)
        pad = max(0.02, (d_max - d_min) * 0.15)
        ax.set_ylim(min(0, d_min) - pad, max(0, d_max) + pad * 2.5)
        _bar_annotate_values_vert(ax, x, ds)
    _bar_style_ax(ax, ylabel="Cohen's d (t$_{0}$ shift)")
    _despine(ax)
    _bar_title(ax, "ED Fig 4 | t$_{0}$ Validation")
    _save_ed(fig, "ed_fig4_t0validation")


def ed_fig5_did_by_method():
    """ED Fig 5: DID across methods (regression/pair/IPTW)."""
    did = _load_csv("did_results.csv")
    fig, ax = _estimate_figure(
        left=BAR_MARGIN_LEFT, bottom=BAR_MARGIN_BOTTOM_LEGEND)
    method_styles = {
        "regression_clustered": ("Regression (clustered SE)", "#E53935"),
        "pair_fe": ("Pair FE", "#1E88E5"),
        "iptw_weighted": ("IPTW weighted", "#009E73"),
    }
    if not did.empty:
        vnames = {"p_factor_severity_z": "P-factor", "volatility_index": "Volatility",
                  "social_withdrawal_index": "Withdrawal", "vader_compound_mean": "VADER"}
        variables = list(did["variable"].unique())
        x = np.arange(len(variables))
        ax.axhline(0, color=PLOT_ZERO_COLOR, lw=PLOT_ZERO_LW, zorder=1)
        for j, var in enumerate(variables):
            for i, (m, (mlab, c)) in enumerate(method_styles.items()):
                row = did[(did["variable"] == var) & (did["method"] == m)]
                if row.empty:
                    continue
                est, se = row["did_estimate"].iloc[0], row["se"].iloc[0]
                ax.errorbar(j + (i - 1) * 0.25, est, yerr=1.96 * se, fmt="o",
                            color=c, ms=EST_MARKER_SIZE,
                            elinewidth=EST_LINE_WIDTH,
                            capsize=EST_CAPSIZE, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([vnames.get(v, v) for v in variables], rotation=25,
                           ha="right", fontsize=FS_TICK, fontweight="bold")
        ax.margins(x=0.08)
    h = [mpatches.Patch(color=c, label=mlab) for mlab, c in method_styles.values()]
    labs = [mlab for mlab, _ in method_styles.values()]
    _legend_below(ax, h, labs, ncol=3, pad=0.12)
    _estimate_style_ax(
        ax,
        ylabel="DID estimate (SD)",
        title="ED Fig 5 | DID Estimates by Method",
    )
    _save_ed(fig, "ed_fig5_did_by_method")


def ed_fig6_subtype_profiles():
    """ED Fig 6: subtype feature profiles (median features)."""
    prof = _load_json("subtype_profiles.json")
    fig, ax = _line_figure()
    if prof:
        feats = ["p_factor_severity_z_step_height", "p_factor_severity_z_slope_pre",
                 "p_factor_severity_z_slope_post", "p_factor_severity_z_volatility_pre",
                 "p_factor_severity_z_turning_angle", "p_factor_severity_z_recovery_pct"]
        flabels = ["Step", "Slope pre", "Slope post", "Volatility", "Turning", "Recovery"]
        x = np.arange(len(feats))
        keys = sorted(prof.keys())
        raw = np.array([
            [prof[k]["feature_medians"].get(f, np.nan) for f in feats]
            for k in keys
        ], dtype=float)
        # Features have incompatible units (per-day slopes, volatility,
        # recovery percentage). Scale each feature by its maximum absolute
        # subtype median before connecting them.
        scale = np.nanmax(np.abs(raw), axis=0)
        scale[~np.isfinite(scale) | (scale == 0)] = 1.0
        relative = raw / scale
        for i, values in enumerate(relative):
            ax.plot(
                x, values, "o-", color=SUB_COLORS[i],
                ms=LINE_MARKER_SIZE, lw=LINE_WIDTH, label=f"S{i}",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            flabels, rotation=25, ha="right",
            fontsize=FS_TICK, fontweight="bold",
        )
        ax.axhline(0, color=PLOT_ZERO_COLOR, lw=PLOT_ZERO_LW, zorder=1)
        ax.set_ylim(-1.12, 1.12)
        ax.margins(x=0.08)
    if prof:
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.16),
            ncol=min(4, len(keys)), frameon=False, fontsize=FS_LEGEND,
            handlelength=1.5, columnspacing=1.0,
        )
    _line_style_ax(
        ax,
        ylabel="Relative median (scaled within feature)",
        title="ED Fig 6 | Subtype Feature Profiles",
    )
    _save_ed(fig, "ed_fig6_subtype_profiles")


def ed_fig7_eventstudy_disorder():
    """ED Fig 7: changepoint lead-time by disorder with IQR."""
    lead = _load_csv("signal_lead_time.csv")
    fig, ax = _estimate_figure()
    if not lead.empty:
        d = lead[lead["disorder"] != "all"].copy()
        valid = (
            np.isfinite(d["q25_lead_days"]) &
            np.isfinite(d["median_lead_days"]) &
            np.isfinite(d["q75_lead_days"]) &
            (d["q25_lead_days"] <= d["median_lead_days"]) &
            (d["median_lead_days"] <= d["q75_lead_days"])
        )
        d = d.loc[valid].copy()
        d = d.sort_values("median_lead_days")
        ys = np.arange(len(d))
        for yi, (_, r) in zip(ys, d.iterrows()):
            c = get_disorder_color(r["disorder"])
            lo = r["median_lead_days"] - r["q25_lead_days"]
            hi = r["q75_lead_days"] - r["median_lead_days"]
            _forest_errorbar_h(
                ax, r["median_lead_days"], yi, lo, hi, color=c)
            ax.text(r["q75_lead_days"] + 5, yi,
                    f"{r['median_lead_days']:.0f}",
                    ha="left", va="center", fontsize=FS_ANNOT, color=c)
        ax.set_yticks(ys)
        ax.set_yticklabels(
            [DISPLAY.get(dis, dis) for dis in d["disorder"]],
            fontsize=FS_TICK, fontweight="bold",
        )
        for label, disorder in zip(ax.get_yticklabels(), d["disorder"]):
            label.set_color(get_disorder_color(disorder))
        xmax = d["q75_lead_days"].max()
        if np.isfinite(xmax) and xmax > 0:
            ax.set_xlim(0, xmax * 1.15)
        ax.invert_yaxis()
    _estimate_style_ax(
        ax,
        xlabel="Signal lead (days; median and IQR)",
        title="ED Fig 7 | Signal Lead Time by Disorder",
    )
    _save_ed(fig, "ed_fig7_leadtime_disorder")


# =====================================================================
# MASTER RUNNER & CLI
# =====================================================================
PLOT_REGISTRY = {
    "fig1": fig1_trajectory_eventstudy,
    "fig2": fig2_changepoints,
    "fig3": fig3_subtypes,
    "fig4": fig4_causal,
    "fig5": fig5_crossdisorder_did,
    "edfig1": ed_fig1_quality_report,
    "edfig2": ed_fig2_multivariate_changepoints,
    "edfig3": ed_fig3_predictors,
    "edfig4": ed_fig4_t0_validation,
    "edfig5": ed_fig5_did_by_method,
    "edfig6": ed_fig6_subtype_profiles,
    "edfig7": ed_fig7_eventstudy_disorder,
}


def run_visualizations(selected_plots):
    logger.info("=" * 70)
    logger.info("Module 2 Publication Visualizations (v2_lyc - Modular CLI)")
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
    for target in targets:
        func = PLOT_REGISTRY[target]
        try:
            func()
        except Exception as e:
            logger.error(f"Error executing {target}: {e}")

    logger.info("\n" + "=" * 70)
    logger.info("TARGETED VISUALIZATIONS COMPLETE")
    logger.info(f"  Main: {MAIN_DIR}")
    logger.info(f"  ED:   {ED_DIR}")
    logger.info("=" * 70)


def run_all():
    run_visualizations(["all"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Modular CLI for Module 2 visualizations.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-p", "--plots",
        nargs="+",
        default=["all"],
        help=(
            "Plots to generate.\n"
            "Example: python s12_figures_v2_lyc.py -p fig1 fig3 edfig5\n"
            "Options: 'all', 'fig1'-'fig5', 'edfig1'-'edfig7'"
        ),
    )
    args = parser.parse_args()
    run_visualizations(args.plots)
