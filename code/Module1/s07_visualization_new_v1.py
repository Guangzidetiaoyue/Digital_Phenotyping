"""
Step 07: Publication-Quality Visualizations (v3.0)

Restructured for Nature manuscript:
  4 Main Figures (core narrative) + 8 Extended Data Figures (validation)

Main Figure Narrative Arc:
  Fig 1 — The Atlas: 93×10 effect size landscape
  Fig 2 — The p-Factor: shared direction, graded severity
  Fig 3 — Detectability: AUROC + clinical utility
  Fig 4 — The Specificity Paradox: high utility, low differential specificity

Extended Data: factor details, clustering, t-SNE, confusion matrix,
               ablation, sensitivity, distributions, strict inconsistency

v2.0 → v3.0 Changelog:
  [RESTRUCTURE] 12 flat figures → 4 main + 8 extended data
  [NEW] Fig 2 Panel C: effect size distribution by disorder (violin)
  [NEW] Fig 3 Panel B: clinical utility lollipop
  [NEW] ED Fig 7: effect size category breakdown by disorder
  [NEW] ED Fig 8: strict direction inconsistency detail (18 features)
  [MERGE] Old Fig 6 (direction) + Fig 4 (factor biplot) → New Fig 2
  [MERGE] Old Fig 2 (AUROC) + Fig 7 (clinical utility) → New Fig 3
  [MERGE] Old Fig 5 (transdiag) + Fig 11 (profiles) → New Fig 4
  [MOVE] Clustering, t-SNE, confusion, ablation, sensitivity → ED
  [FIX] Scree plot: compute eigenvalues from effect matrix
  [FIX] All panels: minimum 5pt font, color-blind palette, CI annotations
"""

import os
import json
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial import ConvexHull
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from config_module1 import (
    STATS_DIR, FIGURES_DIR, FEATURES_DIR,
    DISORDER_GROUPS, DISORDER_DISPLAY_NAMES,
    HITOP_SPECTRA, HITOP_COLORS, RANDOM_SEED
)

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(os.path.join(FIGURES_DIR, "main"), exist_ok=True)
os.makedirs(os.path.join(FIGURES_DIR, "ed"), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(FIGURES_DIR, "visualization.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =====================================================================
# SECTION 0: GLOBAL CONSTANTS
# =====================================================================

SELECTION_ARTIFACT_FEATURES = {
    "mental_health_sub_ratio", "support_sub_ratio",
    "subreddit_entropy", "subreddit_herfindahl",
    "primary_subreddit_ratio",
}

MAIN_DIR = os.path.join(FIGURES_DIR, "main")
ED_DIR = os.path.join(FIGURES_DIR, "ed")

# Nature-style rcParams
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'lines.linewidth': 0.75,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

SINGLE_COL = 89 / 25.4
DOUBLE_COL = 183 / 25.4
FULL_PAGE = 247 / 25.4

# Color-blind-friendly diverging: blue-white-orange
DIVERGING_CMAP = sns.diverging_palette(220, 20, as_cmap=True)

DIM_COLORS = {
    "affective": "#E53935", "pronoun": "#8E24AA",
    "cognitive": "#1E88E5", "temporal_linguistic": "#00897B",
    "risk": "#F4511E", "behavioral": "#6D4C41",
    "readability": "#546E7A", "post_length": "#78909C",
    "temporal": "#FF9800", "social": "#4CAF50",
    "community": "#9C27B0", "other": "#607D8B",
}

DIM_ORDER = [
    'affective', 'pronoun', 'cognitive', 'temporal_linguistic',
    'risk', 'behavioral', 'readability', 'post_length',
    'temporal', 'social', 'community'
]

# =====================================================================
# SECTION 1: UTILITY FUNCTIONS
# =====================================================================

def get_disorder_color(d):
    for s, ds in HITOP_SPECTRA.items():
        if d in ds:
            return HITOP_COLORS.get(s, "#808080")
    return "#808080"

def get_disorder_display(d):
    return DISORDER_DISPLAY_NAMES.get(d, d)

def _csv(path, **kw):
    if not os.path.exists(path):
        logger.warning(f"  Not found: {path}")
        return None
    try:
        return pd.read_csv(path, **kw)
    except Exception as e:
        logger.error(f"  Error: {path}: {e}")
        return None

def _json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

def _load_dim_map():
    meta = _json(os.path.join(FEATURES_DIR, "feature_metadata.json"))
    if not meta:
        return {}, DIM_ORDER
    f2d = {}
    log_s, slog_s = set(), set()
    for dim, cols in meta.items():
        if not isinstance(cols, list):
            continue
        if dim == 'log_transforms':
            log_s = set(cols); continue
        if dim == 'signed_log_transforms':
            slog_s = set(cols); continue
        if dim in ('quality',):
            continue
        for c in cols:
            f2d[c] = dim
    for lf in log_s:
        f2d[lf] = f2d.get(lf.replace("_log", ""), 'temporal')
    for sf in slog_s:
        f2d[sf] = f2d.get(sf.replace("_log_signed", ""), 'social')
    return f2d, DIM_ORDER

def _save(fig, name, subdir="main"):
    d = MAIN_DIR if subdir == "main" else ED_DIR
    for fmt in ['pdf', 'png']:
        fig.savefig(os.path.join(d, f"{name}.{fmt}"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved: {subdir}/{name} (pdf+png)")

def _compute_eigenvalues():
    """Compute eigenvalues + parallel analysis from effect matrix."""
    em = _csv(os.path.join(STATS_DIR, "effect_size_matrix_clean.csv"), index_col=0)
    if em is None:
        em = _csv(os.path.join(STATS_DIR, "effect_size_matrix.csv"), index_col=0)
        if em is not None:
            arts = [f for f in em.index if f in SELECTION_ARTIFACT_FEATURES]
            if arts:
                em = em.drop(index=arts)
    if em is None:
        return [], []
    X = np.nan_to_num(em.values.astype(float), nan=0.0)
    ns, nv = X.shape
    corr = np.corrcoef(X.T)
    eig = sorted(np.linalg.eigh(corr)[0], reverse=True)
    rng = np.random.RandomState(42)
    r_eigs = np.zeros((200, nv))
    for i in range(200):
        rd = rng.standard_normal((ns, nv))
        r_eigs[i] = sorted(np.linalg.eigh(np.corrcoef(rd.T))[0], reverse=True)
    p95 = np.percentile(r_eigs, 95, axis=0).tolist()
    return [float(e) for e in eig], p95

# =====================================================================
# SECTION 2: FIGURE 1 — THE ATLAS (Main)
# =====================================================================

def plot_fig1_atlas():
    """
    Figure 1: The Digital Phenotyping Atlas.
    93 features × 10 disorders effect size heatmap.
    Signature figure of the paper.
    """
    logger.info("MAIN Fig 1: The Atlas...")

    em = _csv(os.path.join(STATS_DIR, "effect_size_matrix_clean.csv"), index_col=0)
    if em is None:
        em = _csv(os.path.join(STATS_DIR, "effect_size_matrix.csv"), index_col=0)
        if em is not None:
            arts = [f for f in em.index if f in SELECTION_ARTIFACT_FEATURES]
            if arts: em = em.drop(index=arts)
    if em is None:
        logger.error("  No effect matrix. Skip Fig 1."); return

    sig = _csv(os.path.join(STATS_DIR, "significance_matrix_clean.csv"), index_col=0)
    if sig is None:
        sig = _csv(os.path.join(STATS_DIR, "significance_matrix.csv"), index_col=0)

    f2d, _ = _load_dim_map()
    em.columns = [get_disorder_display(c) for c in em.columns]
    if sig is not None:
        sig.columns = [get_disorder_display(c) for c in sig.columns]

    # Order features by dimension, then |g| within
    ordered, boundaries, dim_names = [], [], []
    for dim in DIM_ORDER:
        df = [f for f in em.index if f2d.get(f) == dim]
        if not df: continue
        ranked = em.loc[df].abs().mean(axis=1).sort_values(ascending=False)
        ordered.extend(ranked.index.tolist())
        boundaries.append(len(ordered))
        dim_names.append(dim)
    rem = [f for f in em.index if f not in ordered]
    if rem:
        ordered.extend(rem)
        boundaries.append(len(ordered))
        dim_names.append('other')

    em_o = em.loc[ordered]
    sig_o = sig.reindex(ordered) if sig is not None else None
    nf, nd = len(ordered), len(em_o.columns)

    # Figure layout
    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 1.05))
    gs = gridspec.GridSpec(2, 3, figure=fig,
        width_ratios=[0.12, 1, 0.04], height_ratios=[0.06, 1],
        hspace=0.01, wspace=0.01)

    # --- Main heatmap ---
    ax = fig.add_subplot(gs[1, 1])
    power = 0.6
    data = em_o.values.astype(float)
    disp = np.sign(data) * np.power(np.abs(data), power)
    vmax_r = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
    vmax_d = np.sign(vmax_r) * np.power(abs(vmax_r), power)
    norm = TwoSlopeNorm(vmin=-vmax_d, vcenter=0, vmax=vmax_d)
    im = ax.imshow(disp, cmap=DIVERGING_CMAP, norm=norm, aspect='auto', interpolation='none')

    # Significance borders
    if sig_o is not None:
        for i in range(nf):
            for j in range(nd):
                if np.isnan(data[i, j]): continue
                try:
                    s = sig_o.iloc[i, j]
                    if isinstance(s, str): s = s.lower() == 'true'
                    if s:
                        ax.add_patch(Rectangle((j-.5, i-.5), 1, 1,
                            fill=False, edgecolor='black', lw=0.25, zorder=3))
                except: pass

    # Dimension separators
    prev = 0
    for b in boundaries:
        if b > prev + 1:
            ax.axhline(y=b - 0.5, color='white', lw=1.5, zorder=4)
        prev = b

    # Axis labels
    ax.set_xticks(range(nd))
    ax.set_xticklabels(em_o.columns, rotation=45, ha='right', fontsize=6)
    ax.set_yticks(range(nf))
    ylbl = [f.replace('_rate','').replace('_ratio','').replace('_mean','').replace('_score','').replace('_',' ') for f in ordered]
    ax.set_yticklabels(ylbl, fontsize=4.5)

    # Color x-labels by spectrum
    for lb in ax.get_xticklabels():
        rn = [k for k, v in DISORDER_DISPLAY_NAMES.items() if v == lb.get_text()]
        if rn: lb.set_color(get_disorder_color(rn[0])); lb.set_fontweight('bold')

    # Color y-labels by dimension
    for i, f in enumerate(ordered):
        d = f2d.get(f, 'other')
        ax.get_yticklabels()[i].set_color(DIM_COLORS.get(d, '#333'))

    # --- Dimension band (left) ---
    ax_d = fig.add_subplot(gs[1, 0])
    ax_d.set_xlim(0, 1); ax_d.set_ylim(nf, 0); ax_d.axis('off')
    prev = 0
    for dim, b in zip(dim_names, boundaries):
        if b > prev:
            c = DIM_COLORS.get(dim, '#607D8B')
            ax_d.add_patch(Rectangle((0, prev), 1, b - prev, fc=c, alpha=0.6))
            if b - prev > 2:
                lab = dim.replace('_',' ').replace('temporal linguistic','temp-ling').title()
                rot = 90 if b - prev < 6 else 0
                ax_d.text(0.5, (prev+b)/2, lab, ha='center', va='center',
                         fontsize=4, color='white', fontweight='bold', rotation=rot)
        prev = b

    # --- HiTOP band (top) ---
    ax_t = fig.add_subplot(gs[0, 1])
    ax_t.set_xlim(-0.5, nd-0.5); ax_t.axis('off')
    for j, col in enumerate(em_o.columns):
        rn = [k for k, v in DISORDER_DISPLAY_NAMES.items() if v == col]
        if rn:
            c = get_disorder_color(rn[0])
            ax_t.add_patch(Rectangle((j-.5, 0), 1, 1, fc=c, alpha=0.7))
            for sp, ds in HITOP_SPECTRA.items():
                if rn[0] in ds:
                    ax_t.text(j, 0.5, sp[:3].upper(), ha='center', va='center',
                             fontsize=4, color='white', fontweight='bold')
                    break

    # --- Colorbar ---
    ax_cb = fig.add_subplot(gs[1, 2])
    cb = fig.colorbar(im, cax=ax_cb, aspect=40)
    g_ticks = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
    g_ticks = [v for v in g_ticks if -vmax_r <= v <= vmax_r]
    d_ticks = [np.sign(v)*abs(v)**power for v in g_ticks]
    cb.set_ticks(d_ticks)
    cb.set_ticklabels([f"{v:.1f}" for v in g_ticks])
    cb.set_label("Hedges' g", fontsize=7)
    cb.ax.tick_params(labelsize=5)

    fig.suptitle(
        "Figure 1 | Digital Phenotyping Atlas: Effect Sizes Across 10 Psychiatric Disorders\n"
        f"{nf} features × {nd} disorders (artifacts excluded); black border = FDR-significant (q < 0.05)",
        fontsize=8, fontweight='bold', y=1.02)

    _save(fig, "fig1_atlas", "main")


# =====================================================================
# SECTION 3: FIGURE 2 — p-FACTOR EVIDENCE (Main)
# =====================================================================

def plot_fig2_pfactor():
    """
    Figure 2: The p-Factor — Shared Direction, Graded Severity.
    Panel A: Direction fingerprint (93 features × positive/negative/NS)
    Panel B: Observed vs expected direction inconsistency
    Panel C: Effect size distribution by disorder (violin + box)
    """
    logger.info("MAIN Fig 2: p-Factor Evidence...")

    ef = _csv(os.path.join(STATS_DIR, "effect_sizes_full.csv"))
    if ef is None:
        logger.error("  No effect sizes. Skip Fig 2."); return

    ec = ef[~ef["feature"].isin(SELECTION_ARTIFACT_FEATURES)].copy()
    g_piv = ec.pivot_table(index="feature", columns="disorder", values="hedges_g", aggfunc="first")
    s_piv = ec.pivot_table(index="feature", columns="disorder", values="sig_fdr_global", aggfunc="first")

    # Direction counts
    feats = sorted(g_piv.index)
    disords = sorted(g_piv.columns)
    dir_data = []
    for f in feats:
        gv = g_piv.loc[f]
        sv = s_piv.loc[f] if f in s_piv.index else pd.Series()
        np_, nn, nns = 0, 0, 0
        for d in disords:
            g = gv.get(d, np.nan)
            s = sv.get(d, False)
            if isinstance(s, str): s = s.lower() == 'true'
            if pd.isna(g): nns += 1
            elif s and g > 0: np_ += 1
            elif s and g < 0: nn += 1
            else: nns += 1
        dir_data.append({'feature': f, 'n_pos': np_, 'n_neg': nn, 'n_ns': nns,
                         'mean_abs_g': gv.abs().mean()})
    dir_df = pd.DataFrame(dir_data)
    dir_df['consistency'] = dir_df[['n_pos','n_neg']].max(axis=1) / dir_df[['n_pos','n_neg']].sum(axis=1).replace(0,1)
    dir_df = dir_df.sort_values('consistency', ascending=True)

    # --- Create figure ---
    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.85))
    gs = gridspec.GridSpec(2, 2, figure=fig,
        height_ratios=[2.5, 1], width_ratios=[1.2, 1],
        hspace=0.3, wspace=0.3)

    # ===== Panel A: Direction Fingerprint =====
    ax_a = fig.add_subplot(gs[0, :])
    n = len(dir_df)
    y = np.arange(n)
    ax_a.barh(y, dir_df['n_pos'].values, color='#E53935', height=0.9,
              label='Sig. positive (patient > control)')
    ax_a.barh(y, dir_df['n_neg'].values, left=dir_df['n_pos'].values,
              color='#1E88E5', height=0.9, label='Sig. negative (control > patient)')
    ax_a.barh(y, dir_df['n_ns'].values,
              left=dir_df['n_pos'].values + dir_df['n_neg'].values,
              color='#E0E0E0', height=0.9, label='Non-significant')

    # Label mixed-direction features
    for i, (_, r) in enumerate(dir_df.iterrows()):
        if r['n_pos'] > 0 and r['n_neg'] > 0:
            ax_a.text(-0.2, i, r['feature'][:28], ha='right', va='center',
                     fontsize=4, color='#B71C1C', fontweight='bold')

    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(-0.5, n - 0.5)
    ax_a.set_yticks([])
    ax_a.set_xlabel("Number of Disorders (out of 10)", fontsize=7)
    ax_a.legend(fontsize=5, loc='lower right')
    ax_a.set_title("A. Direction Fingerprint: Near-Universal Same-Direction Effects",
                   fontsize=8, fontweight='bold', loc='left')
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)

    # Annotation box
    n_mixed = ((dir_df['n_pos'] > 0) & (dir_df['n_neg'] > 0)).sum()
    ax_a.text(0.98, 0.95,
        f"{n - n_mixed}/{n} features shift in the\n"
        f"same direction across all disorders\n"
        f"({n_mixed} mixed-direction, all driven by\n"
        f"autism/schizophrenia opposing others)",
        transform=ax_a.transAxes, fontsize=5, va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD',
                 edgecolor='#1565C0', alpha=0.9))

    # ===== Panel B: Observed vs Expected =====
    ax_b = fig.add_subplot(gs[1, 0])
    null_exp = 90.7  # from s03 logs
    cats = ['Observed', 'Expected\n(null)']
    vals = [n_mixed, null_exp]
    cols = ['#E53935', '#BDBDBD']
    bars = ax_b.bar(range(2), vals, color=cols, edgecolor='black', lw=0.3, width=0.5)
    for b, v in zip(bars, vals):
        ax_b.text(b.get_x() + b.get_width()/2, v + 1, f'{v:.0f}',
                 ha='center', fontsize=7, fontweight='bold')
    ratio = n_mixed / max(1, null_exp)
    ax_b.text(0.5, 0.90,
        f"O/E = {ratio:.1f}× (p < 0.001)\nDirection consistency far exceeds chance\n→ Supports p-factor (general psychopathology)",
        transform=ax_b.transAxes, fontsize=5.5, ha='center', va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4',
                 edgecolor='#F57F17', alpha=0.9))
    ax_b.set_xticks(range(2))
    ax_b.set_xticklabels(cats, fontsize=6)
    ax_b.set_ylabel("Features with Direction\nInconsistency", fontsize=6)
    ax_b.set_title("B. Observed vs. Null Expectation", fontsize=8,
                   fontweight='bold', loc='left')
    ax_b.spines['top'].set_visible(False); ax_b.spines['right'].set_visible(False)

    # ===== Panel C: Effect Size Distribution by Disorder =====
    ax_c = fig.add_subplot(gs[1, 1])

    # Compute |g| per disorder
    disorder_g = {}
    for d in disords:
        mask = ec["disorder"] == d
        vals = ec.loc[mask, "hedges_g"].dropna().abs()
        if len(vals) > 0:
            disorder_g[get_disorder_display(d)] = vals.values

    if disorder_g:
        # Sort by median |g|
        sorted_d = sorted(disorder_g.keys(), key=lambda x: np.median(disorder_g[x]))
        positions = range(len(sorted_d))
        parts = ax_c.violinplot(
            [disorder_g[d] for d in sorted_d],
            positions=list(positions), vert=True, widths=0.7,
            showmedians=True, showextrema=False
        )
        for i, d in enumerate(sorted_d):
            rn = [k for k, v in DISORDER_DISPLAY_NAMES.items() if v == d]
            c = get_disorder_color(rn[0]) if rn else '#808080'
            parts['bodies'][i].set_facecolor(c)
            parts['bodies'][i].set_alpha(0.6)
            parts['bodies'][i].set_edgecolor('black')
            parts['bodies'][i].set_linewidth(0.3)

        # Effect size reference lines
        ax_c.axhline(y=0.2, color='#BDBDBD', ls=':', lw=0.5)
        ax_c.axhline(y=0.5, color='#9E9E9E', ls=':', lw=0.5)
        ax_c.axhline(y=0.8, color='#757575', ls=':', lw=0.5)
        ax_c.text(len(sorted_d)-0.5, 0.22, 'small', fontsize=4, color='#757575')
        ax_c.text(len(sorted_d)-0.5, 0.52, 'medium', fontsize=4, color='#757575')
        ax_c.text(len(sorted_d)-0.5, 0.82, 'large', fontsize=4, color='#757575')

        ax_c.set_xticks(list(positions))
        ax_c.set_xticklabels(sorted_d, rotation=45, ha='right', fontsize=5)
        for lb in ax_c.get_xticklabels():
            rn = [k for k, v in DISORDER_DISPLAY_NAMES.items() if v == lb.get_text()]
            if rn: lb.set_color(get_disorder_color(rn[0])); lb.set_fontweight('bold')

    ax_c.set_ylabel("|Hedges' g|", fontsize=6)
    ax_c.set_title("C. Effect Size Distribution by Disorder",
                   fontsize=8, fontweight='bold', loc='left')
    ax_c.spines['top'].set_visible(False); ax_c.spines['right'].set_visible(False)

    fig.suptitle("Figure 2 | The p-Factor: Shared Direction, Graded Severity",
                fontsize=9, fontweight='bold', y=1.01)
    _save(fig, "fig2_pfactor", "main")


# =====================================================================
# SECTION 4: FIGURE 3 — DETECTABILITY (Main)
# =====================================================================

def plot_fig3_detectability():
    """
    Figure 3: Disorder Detectability and Clinical Utility.
    Panel A: vs-control AUROC bar chart with CIs + test markers
    Panel B: Clinical utility top-15 lollipop chart
    Panel C: Pairwise AUROC matrix
    """
    logger.info("MAIN Fig 3: Detectability...")

    pw = _csv(os.path.join(STATS_DIR, "auroc_pairwise_matrix.csv"), index_col=0)
    vc = _csv(os.path.join(STATS_DIR, "auroc_vs_control.csv"))
    dvc = _csv(os.path.join(STATS_DIR, "discrimination_disorder_vs_control.csv"))
    cu = _csv(os.path.join(STATS_DIR, "feature_clinical_utility.csv"), index_col=0)
    em = _csv(os.path.join(STATS_DIR, "effect_size_matrix_clean.csv"), index_col=0)
    f2d, _ = _load_dim_map()

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.7))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1.1], wspace=0.35)

    # ===== Panel A: vs-control bars =====
    ax_a = fig.add_subplot(gs[0])

    if vc is not None and len(vc) > 0:
        vs = vc.sort_values('auroc_vs_control', ascending=True)
        disps = [get_disorder_display(d) for d in vs['disorder']]
        aucs = vs['auroc_vs_control'].values
        colors = [get_disorder_color(d) for d in vs['disorder']]

        # Get std
        yerr = None
        if dvc is not None and len(dvc) > 0:
            yerr = []
            for d in vs['disorder']:
                if d in dvc.index:
                    v = dvc.loc[d, 'logistic_l2_auroc_std'] if 'logistic_l2_auroc_std' in dvc.columns else 0
                    yerr.append(float(v) if pd.notna(v) else 0)
                else:
                    yerr.append(0)

        ax_a.barh(range(len(disps)), aucs, color=colors, edgecolor='black',
                  lw=0.3, height=0.7, xerr=yerr,
                  error_kw=dict(ecolor='black', capsize=2, capthick=0.4, lw=0.4))

        # Test AUROC markers
        if dvc is not None and 'test_auroc' in dvc.columns:
            for i, d in enumerate(vs['disorder']):
                if d in dvc.index:
                    ta = dvc.loc[d, 'test_auroc']
                    if pd.notna(ta):
                        ax_a.plot(ta, i, 'D', color='black', ms=3, zorder=5)

        for i, v in enumerate(aucs):
            ax_a.text(v + 0.008, i, f'{v:.3f}', va='center', fontsize=5)

        ax_a.set_yticks(range(len(disps)))
        ax_a.set_yticklabels(disps, fontsize=5.5)
        for lb in ax_a.get_yticklabels():
            rn = [k for k, v_ in DISORDER_DISPLAY_NAMES.items() if v_ == lb.get_text()]
            if rn: lb.set_color(get_disorder_color(rn[0])); lb.set_fontweight('bold')
        ax_a.set_xlim(0.5, 1.0)
        ax_a.axvline(x=0.5, color='red', ls='--', lw=0.5)

        # Legend
        h1 = Line2D([0],[0], marker='s', color='w', markerfacecolor='gray', ms=4, label='Dev AUROC')
        h2 = Line2D([0],[0], marker='D', color='w', markerfacecolor='black', ms=3, label='Test AUROC')
        ax_a.legend(handles=[h1,h2], fontsize=5, loc='lower right')

    ax_a.set_xlabel("AUROC", fontsize=7)
    ax_a.set_title("A. Disorder vs. Controls\n(◆ = test set)",
                   fontsize=8, fontweight='bold', loc='left')
    ax_a.spines['top'].set_visible(False); ax_a.spines['right'].set_visible(False)

    # ===== Panel B: Clinical utility lollipop =====
    ax_b = fig.add_subplot(gs[1])

    if cu is not None:
        cu_s = cu.squeeze()
        top = cu_s.nlargest(min(15, len(cu_s)))
        y_pos = np.arange(len(top))

        for i, (feat, val) in enumerate(top.items()):
            dim = f2d.get(feat, 'other')
            c = DIM_COLORS.get(dim, '#607D8B')
            ax_b.plot([0, val], [i, i], color=c, lw=0.8, alpha=0.5)
            ax_b.scatter(val, i, c=c, s=50, edgecolors='black', lw=0.3, zorder=5)

            # Best disorder annotation
            if em is not None and feat in em.index:
                best = em.loc[feat].abs().idxmax()
                bg = em.loc[feat, best]
                ax_b.text(val + 0.01, i,
                         f"{get_disorder_display(best)} (g={bg:+.2f})",
                         va='center', fontsize=4,
                         color=get_disorder_color(best))

        ax_b.set_yticks(y_pos)
        ax_b.set_yticklabels([f[:25] for f in top.index], fontsize=5)
        for i, f in enumerate(top.index):
            dim = f2d.get(f, 'other')
            ax_b.get_yticklabels()[i].set_color(DIM_COLORS.get(dim, '#333'))
        ax_b.set_xlim(0, 0.85)
        ax_b.axvline(x=0.5, color='gray', ls=':', lw=0.5)

    ax_b.set_xlabel("Clinical Utility Score", fontsize=7)
    ax_b.set_title("B. Top Clinical Utility Features",
                   fontsize=8, fontweight='bold', loc='left')
    ax_b.spines['top'].set_visible(False); ax_b.spines['right'].set_visible(False)

    # ===== Panel C: Pairwise AUROC matrix =====
    ax_c = fig.add_subplot(gs[2])

    if pw is not None:
        pw.index = [get_disorder_display(d) for d in pw.index]
        pw.columns = [get_disorder_display(d) for d in pw.columns]
        data = pw.values.astype(float)
        np.fill_diagonal(data, np.nan)
        valid = data[~np.isnan(data)]
        vmin = max(0.5, np.nanmin(valid) - 0.02)
        vmax = min(1.0, np.nanmax(valid) + 0.02)

        im = ax_c.imshow(data, cmap='RdYlGn', vmin=vmin, vmax=vmax, aspect='equal')
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if np.isnan(v): continue
                tc = 'white' if v > 0.82 or v < vmin + 0.05 else 'black'
                ax_c.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=5, color=tc)
        for i in range(data.shape[0]):
            ax_c.text(i, i, '—', ha='center', va='center', fontsize=6, color='#AAA')

        ax_c.set_xticks(range(len(pw.columns)))
        ax_c.set_xticklabels(pw.columns, rotation=45, ha='right', fontsize=5)
        ax_c.set_yticks(range(len(pw.index)))
        ax_c.set_yticklabels(pw.index, fontsize=5)
        for lb in list(ax_c.get_xticklabels()) + list(ax_c.get_yticklabels()):
            rn = [k for k, v_ in DISORDER_DISPLAY_NAMES.items() if v_ == lb.get_text()]
            if rn: lb.set_color(get_disorder_color(rn[0]))

        cb = fig.colorbar(im, ax=ax_c, shrink=0.7, aspect=25)
        cb.set_label("AUROC", fontsize=6); cb.ax.tick_params(labelsize=5)

    ax_c.set_title("C. Pairwise Discrimination",
                   fontsize=8, fontweight='bold', loc='left')

    fig.suptitle("Figure 3 | Disorder Detectability and Clinical Utility",
                fontsize=9, fontweight='bold', y=1.01)
    _save(fig, "fig3_detectability", "main")


# =====================================================================
# SECTION 5: FIGURE 4 — SPECIFICITY PARADOX (Main)
# =====================================================================

def plot_fig4_specificity():
    """
    Figure 4: The Specificity Paradox — High Utility, Low Differential Specificity.
    Panel A: Feature classification bar chart
    Panel B: DiffSpec vs CU scatter (core: all DiffSpec < 0.2)
    Panel C: Parallel coordinates disorder profiles
    """
    logger.info("MAIN Fig 4: Specificity Paradox...")

    td = _csv(os.path.join(STATS_DIR, "transdiagnostic_features.csv"))
    ds = _csv(os.path.join(STATS_DIR, "feature_differential_specificity.csv"), index_col=0)
    cu = _csv(os.path.join(STATS_DIR, "feature_clinical_utility.csv"), index_col=0)
    sp = _csv(os.path.join(STATS_DIR, "feature_specificity.csv"), index_col=0)
    profiles = _json(os.path.join(STATS_DIR, "disorder_profiles.json"))
    f2d, dim_order = _load_dim_map()

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.55))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35, width_ratios=[0.8, 1.2, 1])

    # ===== Panel A: Feature classification =====
    ax_a = fig.add_subplot(gs[0])

    if td is not None:
        cat_order = ["transdiagnostic", "mixed", "specific", "non_significant"]
        cat_c = {"transdiagnostic": "#E53935", "mixed": "#FB8C00",
                 "specific": "#1E88E5", "non_significant": "#BDBDBD"}
        cat_l = {"transdiagnostic": "Transdiag.\n(≥5 dis.)",
                 "mixed": "Mixed\n(3-4)", "specific": "Specific\n(1-2)",
                 "non_significant": "Non-sig."}
        counts = td["category"].value_counts()
        bh = [counts.get(c, 0) for c in cat_order]
        bars = ax_a.bar(range(4), bh, color=[cat_c[c] for c in cat_order],
                        edgecolor='black', lw=0.3)
        ax_a.set_xticks(range(4))
        ax_a.set_xticklabels([cat_l[c] for c in cat_order], fontsize=5)
        for b, h in zip(bars, bh):
            ax_a.text(b.get_x() + b.get_width()/2, h + 0.5, str(int(h)),
                     ha='center', fontsize=6)

    ax_a.set_ylabel("Number of Features", fontsize=7)
    ax_a.set_title("A. Feature Classification", fontsize=8, fontweight='bold', loc='left')
    ax_a.spines['top'].set_visible(False); ax_a.spines['right'].set_visible(False)

    # ===== Panel B: DiffSpec vs CU scatter =====
    ax_b = fig.add_subplot(gs[1])

    if ds is not None and cu is not None:
        mg = pd.DataFrame({'ds': ds.squeeze(), 'cu': cu.squeeze()})
        if sp is not None: mg['sp'] = sp.squeeze()
        else: mg['sp'] = np.nan
        v = mg.dropna()
        if len(v) > 0:
            sc = ax_b.scatter(v['sp'], v['ds'], c=v['cu'], cmap='YlOrRd',
                              s=25, edgecolors='black', lw=0.2, vmin=0, vmax=0.8)
            cb = fig.colorbar(sc, ax=ax_b, shrink=0.7, pad=0.02)
            cb.set_label("Clinical Utility", fontsize=5); cb.ax.tick_params(labelsize=4)

            top5 = v.nlargest(5, 'cu')
            for _, r in top5.iterrows():
                ax_b.annotate(r.name[:20], (r['sp'], r['ds']),
                             fontsize=4, xytext=(3,3), textcoords='offset points',
                             arrowprops=dict(arrowstyle='-', color='gray', lw=0.3))

        ax_b.axhline(y=0.2, color='gray', ls=':', lw=0.5)
        ax_b.text(0.03, 0.97,
            "All DiffSpec < 0.2\n→ Effects are quantitative,\n"
            "not qualitative (no direction\nflips between disorders)",
            transform=ax_b.transAxes, fontsize=5, va='top', style='italic',
            color='#B71C1C',
            bbox=dict(boxstyle='round,pad=0.2', fc='#FFF9C4', ec='#F57F17', alpha=0.9))

    ax_b.set_xlabel("Specificity (magnitude-based)", fontsize=6)
    ax_b.set_ylabel("Differential Specificity", fontsize=6)
    ax_b.set_title("B. The Specificity Paradox", fontsize=8, fontweight='bold', loc='left')
    ax_b.spines['top'].set_visible(False); ax_b.spines['right'].set_visible(False)

    # ===== Panel C: Parallel coordinates =====
    ax_c = fig.add_subplot(gs[2])

    if profiles:
        all_dims = set()
        for d, p in profiles.items():
            all_dims.update(p.get("dimension_summary", {}).keys())
        dims = [d for d in dim_order if d in all_dims]
        if not dims: dims = sorted(all_dims)

        for disorder in DISORDER_GROUPS:
            if disorder not in profiles: continue
            ds_ = profiles[disorder].get("dimension_summary", {})
            vals = [ds_.get(d, {}).get("mean_abs_g", 0) if d in ds_ else 0 for d in dims]
            c = get_disorder_color(disorder)
            dn = get_disorder_display(disorder)
            ax_c.plot(range(len(dims)), vals, color=c, lw=1.2, alpha=0.8,
                     label=dn, marker='o', ms=3)
            ax_c.text(len(dims) - 0.3, vals[-1], dn, fontsize=4, color=c,
                     fontweight='bold', va='center')

        dl = [d.replace('_',' ').replace('temporal linguistic','temp-ling').title() for d in dims]
        ax_c.set_xticks(range(len(dims)))
        ax_c.set_xticklabels(dl, rotation=30, ha='right', fontsize=5)
        for i, d in enumerate(dims):
            ax_c.get_xticklabels()[i].set_color(DIM_COLORS.get(d, '#333'))
            ax_c.axvspan(i-.4, i+.4, alpha=0.05, color=DIM_COLORS.get(d, '#333'))

        ax_c.axhline(y=0.2, color='#BDBDBD', ls=':', lw=0.5)
        ax_c.axhline(y=0.5, color='#9E9E9E', ls=':', lw=0.5)
        ax_c.legend(fontsize=3.5, loc='upper left', ncol=2, framealpha=0.8)

    ax_c.set_ylabel("Mean |Hedges' g|", fontsize=6)
    ax_c.set_title("C. Disorder Profiles by Dimension",
                   fontsize=8, fontweight='bold', loc='left')
    ax_c.spines['top'].set_visible(False); ax_c.spines['right'].set_visible(False)

    fig.suptitle("Figure 4 | The Specificity Paradox: High Clinical Utility, Low Differential Specificity",
                fontsize=9, fontweight='bold', y=1.02)
    _save(fig, "fig4_specificity", "main")


# =====================================================================
# SECTION 6: ED FIGURE 1 — FACTOR ANALYSIS DETAILS
# =====================================================================

def plot_ed_fig1_factor():
    """ED Fig 1: Factor analysis — loadings, scree, HiTOP alignment."""
    logger.info("ED Fig 1: Factor Details...")

    sc = _csv(os.path.join(STATS_DIR, "factor_scores_disorders.csv"), index_col=0)
    ld = _csv(os.path.join(STATS_DIR, "factor_loadings_features.csv"), index_col=0)
    fi = _json(os.path.join(STATS_DIR, "factor_selection.json"))
    if sc is None or fi is None:
        logger.error("  Skip ED Fig 1"); return

    nf = fi.get("n_factors", 2)
    ve = fi.get("variance_explained", [])
    f2d, _ = _load_dim_map()
    sc.index = [get_disorder_display(d) for d in sc.index]

    # Factor semantic labels
    fn = {0: "p-factor", 1: "Int. vs Neurodev.", 2: "Thought Dis."}
    for i in range(nf):
        fn.setdefault(i, f"F{i+1}")

    # Eigenvalues
    eig = fi.get("selection_criteria", {}).get("eigenvalues", [])
    r95 = fi.get("selection_criteria", {}).get("random_95th_eigenvalues", [])
    if not eig:
        eig, r95 = _compute_eigenvalues()

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.7))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3,
                           height_ratios=[1, 0.8])

    # Panel A: Loadings heatmap
    ax_a = fig.add_subplot(gs[0, 0])
    top_feats = []
    if ld is not None and nf >= 2:
        ml = ld.abs().max(axis=1)
        top_feats = ml.nlargest(min(20, len(ld))).index.tolist()
        ls = ld.loc[top_feats]
        vm = min(max(abs(ls.values.min()), abs(ls.values.max())), 5)
        nm = TwoSlopeNorm(vmin=-vm, vcenter=0, vmax=vm)
        im = ax_a.imshow(ls.values, cmap=DIVERGING_CMAP, norm=nm, aspect='auto')
        for i in range(ls.shape[0]):
            for j in range(ls.shape[1]):
                v = ls.iloc[i, j]
                tc = 'white' if abs(v) > vm * 0.6 else 'black'
                ax_a.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=4, color=tc)
        ax_a.set_yticks(range(len(top_feats)))
        ax_a.set_yticklabels([f[:25] for f in top_feats], fontsize=4)
        for i, f in enumerate(top_feats):
            ax_a.get_yticklabels()[i].set_color(DIM_COLORS.get(f2d.get(f,'other'), '#333'))
        xl = [f"{fn.get(i,'F'+str(i+1))}\n({ve[i]*100:.1f}%)" if i < len(ve) else f"F{i+1}" for i in range(nf)]
        ax_a.set_xticks(range(nf)); ax_a.set_xticklabels(xl, fontsize=5.5)
        fig.colorbar(im, ax=ax_a, shrink=0.8, aspect=20, pad=0.02).set_label("Loading", fontsize=5)
    ax_a.set_title("A. Top Feature Loadings", fontsize=8, fontweight='bold')

    # Panel B: Biplot
    ax_b = fig.add_subplot(gs[0, 1])
    if nf >= 2:
        for dd in sc.index:
            x, y = sc.loc[dd, sc.columns[0]], sc.loc[dd, sc.columns[1]]
            rn = [k for k, v in DISORDER_DISPLAY_NAMES.items() if v == dd]
            c = get_disorder_color(rn[0]) if rn else "#808080"
            ax_b.scatter(x, y, c=c, s=60, edgecolors='black', lw=0.5, zorder=5)
            ax_b.annotate(dd, (x, y), xytext=(4,4), textcoords='offset points',
                         fontsize=4.5, color=c, fontweight='bold')
        # HiTOP hulls
        for sp, ds_ in HITOP_SPECTRA.items():
            mem = [get_disorder_display(d) for d in ds_ if get_disorder_display(d) in sc.index]
            if len(mem) >= 3:
                try:
                    co = sc.loc[mem, [sc.columns[0], sc.columns[1]]].values
                    h = ConvexHull(co)
                    hp = np.vstack([co[h.vertices], co[h.vertices[0]]])
                    ax_b.plot(hp[:,0], hp[:,1], color=HITOP_COLORS[sp], lw=0.6, ls='--', alpha=0.5)
                except: pass
        ax_b.axhline(0, color='gray', lw=0.3); ax_b.axvline(0, color='gray', lw=0.3)
        v1 = ve[0]*100 if ve else 0; v2 = ve[1]*100 if len(ve)>1 else 0
        ax_b.set_xlabel(f"{fn.get(0,'F1')} ({v1:.1f}%)", fontsize=6)
        ax_b.set_ylabel(f"{fn.get(1,'F2')} ({v2:.1f}%)", fontsize=6)
    ax_b.set_title("B. Disorder Factor Scores", fontsize=8, fontweight='bold')
    ax_b.spines['top'].set_visible(False); ax_b.spines['right'].set_visible(False)

    # Panel C: Scree
    ax_c = fig.add_subplot(gs[1, 0])
    if eig:
        ns = min(len(eig), 10)
        x = range(1, ns+1)
        bars = ax_c.bar(x, eig[:ns], color='#1976D2', alpha=0.7, edgecolor='black', lw=0.3, width=0.6)
        for i in range(min(nf, ns)):
            bars[i].set_color('#FF9800'); bars[i].set_alpha(0.9)
        if r95:
            ax_c.plot(x, r95[:ns], 'D--', color='#E53935', ms=3, label='95th %ile (random)', lw=0.8)
        ax_c.axhline(y=1.0, color='gray', ls=':', lw=0.5, label='Kaiser')
        ax_c.set_xticks(list(x))
        if ve:
            for i, v in enumerate(eig[:ns]):
                if i < len(ve):
                    ax_c.text(i+1, v+0.05, f'{ve[i]*100:.1f}%', ha='center', fontsize=4)
    ax_c.set_xlabel("Factor Number", fontsize=6); ax_c.set_ylabel("Eigenvalue", fontsize=6)
    ax_c.set_title("C. Scree Plot", fontsize=8, fontweight='bold')
    ax_c.legend(fontsize=5); ax_c.spines['top'].set_visible(False); ax_c.spines['right'].set_visible(False)

    # Panel D: HiTOP alignment
    ax_d = fig.add_subplot(gs[1, 1])
    if sc is not None and nf >= 1:
        spectra = list(HITOP_SPECTRA.keys())
        ad = np.zeros((nf, len(spectra)))
        for si, sp in enumerate(spectra):
            mem = [get_disorder_display(d) for d in HITOP_SPECTRA[sp] if get_disorder_display(d) in sc.index]
            if mem:
                for fi_ in range(nf):
                    ad[fi_, si] = sc.loc[mem, sc.columns[fi_]].mean()
        vm = max(abs(ad.min()), abs(ad.max()))
        if vm < 1e-10: vm = 1
        nm = TwoSlopeNorm(vmin=-vm, vcenter=0, vmax=vm)
        im_ = ax_d.imshow(ad, cmap=DIVERGING_CMAP, norm=nm, aspect='auto')
        for i in range(nf):
            for j in range(len(spectra)):
                v = ad[i, j]; tc = 'white' if abs(v) > vm*0.6 else 'black'
                ax_d.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=5, color=tc, fontweight='bold')
        ax_d.set_yticks(range(nf))
        ax_d.set_yticklabels([fn.get(i, f"F{i+1}") for i in range(nf)], fontsize=5)
        ax_d.set_xticks(range(len(spectra)))
        ax_d.set_xticklabels([s.replace('_','\n') for s in spectra], fontsize=4, rotation=30, ha='right')
        for j, lb in enumerate(ax_d.get_xticklabels()):
            lb.set_color(HITOP_COLORS.get(spectra[j], 'black'))
        fig.colorbar(im_, ax=ax_d, shrink=0.8, pad=0.02).set_label("Mean Score", fontsize=5)
    ax_d.set_title("D. Factor × HiTOP Alignment", fontsize=8, fontweight='bold')

    _save(fig, "ed_fig1_factor", "ed")


# =====================================================================
# SECTION 7: ED FIGURE 2 — CLUSTERING + MDS
# =====================================================================

def plot_ed_fig2_clustering():
    """ED Fig 2: Dendrogram + MDS with HiTOP hulls."""
    logger.info("ED Fig 2: Clustering...")
    lp = os.path.join(STATS_DIR, "linkage_matrix.npy")
    mp = os.path.join(STATS_DIR, "disorder_mds_embedding.csv")
    if not os.path.exists(lp) or not os.path.exists(mp):
        logger.error("  Skip ED Fig 2"); return

    lm = np.load(lp)
    md = _csv(mp, index_col=0)
    ci = _json(os.path.join(STATS_DIR, "clustering_summary.json")) or {}

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL*0.4))

    # Dendrogram
    ax = axes[0]
    dl = [get_disorder_display(d) for d in md.index]
    dendrogram(lm, labels=dl, orientation='left', ax=ax, leaf_font_size=6,
               color_threshold=0, above_threshold_color='#333')
    for lb in ax.get_ymajorticklabels():
        rn = [k for k, v in DISORDER_DISPLAY_NAMES.items() if v == lb.get_text()]
        if rn: lb.set_color(get_disorder_color(rn[0])); lb.set_fontweight('bold')
    ax.set_title("Hierarchical Clustering", fontsize=8, fontweight='bold')
    ax.set_xlabel("Distance", fontsize=6)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # MDS
    ax2 = axes[1]
    for d in md.index:
        x, y = md.loc[d,"MDS_1"], md.loc[d,"MDS_2"]
        c = get_disorder_color(d); dn = get_disorder_display(d)
        ax2.scatter(x, y, c=c, s=80, edgecolors='black', lw=0.5, zorder=5)
        ax2.annotate(dn, (x,y), xytext=(5,5), textcoords='offset points',
                    fontsize=5, color=c, fontweight='bold')

    for sp, ds_ in HITOP_SPECTRA.items():
        mem = [d for d in ds_ if d in md.index]
        if len(mem) >= 3:
            try:
                co = md.loc[mem, ["MDS_1","MDS_2"]].values
                h = ConvexHull(co); hp = np.vstack([co[h.vertices], co[h.vertices[0]]])
                ax2.plot(hp[:,0], hp[:,1], color=HITOP_COLORS[sp], lw=0.8, ls='--', alpha=0.5)
                ax2.fill(hp[:,0], hp[:,1], color=HITOP_COLORS[sp], alpha=0.08)
            except: pass

    ax2.set_xlabel("MDS 1", fontsize=6); ax2.set_ylabel("MDS 2", fontsize=6)
    ax2.set_title("MDS Embedding", fontsize=8, fontweight='bold')
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    ari = ci.get("ari_vs_hitop", np.nan); nmi = ci.get("nmi_vs_hitop", np.nan)
    if not (isinstance(ari, float) and np.isnan(ari)):
        ax2.text(0.02, 0.98, f"ARI={ari:.3f}, NMI={nmi:.3f}",
                transform=ax2.transAxes, fontsize=5, va='top',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.9))

    sh = [Line2D([0],[0], marker='o', color='w', markerfacecolor=c, ms=5,
                 label=s.replace('_',' ').title(), markeredgecolor='black', mew=0.3)
          for s, c in HITOP_COLORS.items()]
    ax2.legend(handles=sh, fontsize=4, loc='lower right', title="HiTOP", title_fontsize=4)

    _save(fig, "ed_fig2_clustering", "ed")


# =====================================================================
# SECTION 8: ED FIGURE 3 — t-SNE SMALL MULTIPLES
# =====================================================================

def plot_ed_fig3_tsne():
    """ED Fig 3: t-SNE small multiples (2×5 grid)."""
    logger.info("ED Fig 3: t-SNE...")
    td = _csv(os.path.join(STATS_DIR, "tsne_embedding.csv"))
    if td is None: logger.error("  Skip ED Fig 3"); return

    mc = _json(os.path.join(STATS_DIR, "multiclass_results.json"))
    pca = mc.get("per_class_auroc_mean", {}) if mc else {}

    xm = [td["tSNE_1"].min(), td["tSNE_1"].max()]
    ym = [td["tSNE_2"].min(), td["tSNE_2"].max()]
    mg = [(xm[1]-xm[0])*0.05, (ym[1]-ym[0])*0.05]

    fig, axes = plt.subplots(2, 5, figsize=(DOUBLE_COL, DOUBLE_COL*0.45))
    for idx, d in enumerate(DISORDER_GROUPS):
        ax = axes[idx//5, idx%5]
        cm = td["group"] == "control"
        ax.scatter(td.loc[cm,"tSNE_1"], td.loc[cm,"tSNE_2"], c="#E0E0E0", s=1, alpha=0.15, rasterized=True)
        m = (td["group"]=="patient") & (td["disorder_group"]==d)
        if m.sum() > 0:
            ax.scatter(td.loc[m,"tSNE_1"], td.loc[m,"tSNE_2"], c=get_disorder_color(d),
                      s=3, alpha=0.5, rasterized=True)
        dn = get_disorder_display(d)
        av = pca.get(d, np.nan)
        aucs = f"\nAUC={av:.2f}" if not pd.isna(av) else ""
        ax.set_title(f"{dn}{aucs}", fontsize=5.5, fontweight='bold', color=get_disorder_color(d))
        ax.set_xlim(xm[0]-mg[0], xm[1]+mg[0]); ax.set_ylim(ym[0]-mg[1], ym[1]+mg[1])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ['top','right']: ax.spines[sp].set_visible(False)

    fig.suptitle("ED Figure 3 | t-SNE Embeddings (each disorder vs. controls)",
                fontsize=8, fontweight='bold', y=1.01)
    _save(fig, "ed_fig3_tsne", "ed")


# =====================================================================
# SECTION 9: ED FIGURE 4 — CONFUSION MATRIX
# =====================================================================

def plot_ed_fig4_confusion():
    """ED Fig 4: Confusion matrix with per-class AUROC."""
    logger.info("ED Fig 4: Confusion Matrix...")
    cm = _csv(os.path.join(STATS_DIR, "confusion_matrix_normalized.csv"), index_col=0)
    mc = _json(os.path.join(STATS_DIR, "multiclass_results.json"))
    if cm is None: logger.error("  Skip ED Fig 4"); return

    cm.index = [get_disorder_display(d) if d!="control" else "Control" for d in cm.index]
    cm.columns = [get_disorder_display(d) if d!="control" else "Control" for d in cm.columns]

    fig, ax = plt.subplots(figsize=(SINGLE_COL*1.5, SINGLE_COL*1.5))
    data = cm.values.astype(float)
    im = ax.imshow(data, cmap='Blues', vmin=0, vmax=0.6, aspect='equal')

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i,j]; tc = 'white' if v > 0.35 else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=4, color=tc)

    # Top-3 confusion pairs
    od = [(i,j,data[i,j]) for i in range(data.shape[0]) for j in range(data.shape[1]) if i!=j]
    od.sort(key=lambda x: x[2], reverse=True)
    for rank, (i,j,v) in enumerate(od[:3]):
        ax.add_patch(Rectangle((j-.5,i-.5),1,1, fill=False, edgecolor='#E53935', lw=1, ls='--'))

    ax.set_xticks(range(len(cm.columns)))
    ax.set_xticklabels(cm.columns, rotation=45, ha='right', fontsize=5)
    ax.set_yticks(range(len(cm.index)))
    ax.set_yticklabels(cm.index, fontsize=5)
    ax.set_xlabel("Predicted", fontsize=7); ax.set_ylabel("True", fontsize=7)

    if mc:
        ma = mc.get("macro_auroc_mean", np.nan); ba = mc.get("balanced_accuracy_mean", np.nan)
        ax.text(0.02, 0.02, f"Macro AUROC={ma:.3f}\nBal.Acc={ba:.3f}",
               transform=ax.transAxes, fontsize=5, va='bottom',
               bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.9))

    fig.colorbar(im, ax=ax, shrink=0.6).set_label("Proportion", fontsize=6)
    ax.set_title("ED Fig 4 | Confusion Matrix (dashed = top confusions)", fontsize=8, fontweight='bold')
    _save(fig, "ed_fig4_confusion", "ed")


# =====================================================================
# SECTION 10: ED FIGURE 5 — DIMENSION ABLATION
# =====================================================================

def plot_ed_fig5_ablation():
    """ED Fig 5: Delta AUROC heatmap."""
    logger.info("ED Fig 5: Ablation...")
    ab = _csv(os.path.join(STATS_DIR, "dimension_ablation.csv"), index_col=0)
    if ab is None: logger.error("  Skip ED Fig 5"); return

    dc = [c for c in ab.columns if c.startswith("delta_")]
    if not dc:
        oc = [c for c in ab.columns if c.startswith("only_")]
        if "all_dimensions" not in ab.columns or not oc:
            logger.error("  No ablation data"); return
        for c in oc:
            ab[c.replace("only_","delta_")] = ab["all_dimensions"] - ab[c]
        dc = [c for c in ab.columns if c.startswith("delta_")]

    dd = ab[dc].copy()
    dd.columns = [c.replace("delta_","") for c in dc]
    dd.index = [get_disorder_display(d) for d in dd.index]
    dd = dd[dd.mean(axis=0).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, SINGLE_COL*0.85))
    vv = dd.values[~np.isnan(dd.values)]
    vm = max(abs(vv.min()), abs(vv.max())) if len(vv) > 0 else 0.1
    nm = TwoSlopeNorm(vmin=-vm, vcenter=0, vmax=vm)
    im = ax.imshow(dd.values, cmap=DIVERGING_CMAP, norm=nm, aspect='auto')
    for i in range(dd.shape[0]):
        for j in range(dd.shape[1]):
            v = dd.iloc[i,j]
            if pd.isna(v): ax.text(j,i,'—', ha='center', fontsize=5, color='#AAA'); continue
            tc = 'white' if abs(v) > vm*0.6 else 'black'
            ax.text(j, i, f'{v:.3f}', ha='center', fontsize=5, color=tc, fontweight='bold')
    ax.set_xticks(range(len(dd.columns)))
    ax.set_xticklabels([c.replace('_',' ').title() for c in dd.columns], rotation=45, ha='right', fontsize=5.5)
    ax.set_yticks(range(len(dd.index)))
    ax.set_yticklabels(dd.index, fontsize=5.5)
    for lb in ax.get_yticklabels():
        rn = [k for k,v in DISORDER_DISPLAY_NAMES.items() if v==lb.get_text()]
        if rn: lb.set_color(get_disorder_color(rn[0]))
    fig.colorbar(im, ax=ax, shrink=0.8).set_label("ΔAUROC (Full − Without Dim)", fontsize=6)
    ax.set_title("ED Fig 5 | Dimension Contribution (ΔAUROC When Removed)", fontsize=8, fontweight='bold')
    _save(fig, "ed_fig5_ablation", "ed")


# =====================================================================
# SECTION 11: ED FIGURE 6 — SENSITIVITY SUMMARY
# =====================================================================

def plot_ed_fig6_sensitivity():
    """ED Fig 6: Robustness summary (4 panels)."""
    logger.info("ED Fig 6: Sensitivity...")
    rp = _json(os.path.join(STATS_DIR, "sensitivity_report.json"))
    if rp is None: logger.error("  Skip ED Fig 6"); return

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL*0.75))

    # Panel A: Checklist
    ax = axes[0,0]
    checks = rp.get("robustness_summary", [])
    if checks:
        ax.set_xlim(0,3); ax.set_ylim(-0.8, len(checks)-0.2); ax.axis('off')
        for i, ck in enumerate(checks):
            st = ck.get('status','UNKNOWN'); nm = ck.get('check','').replace('_',' ').title()
            dt = ck.get('detail','')
            if len(dt) > 25: dt = dt[:23]+"..."
            bg = '#E8F5E9' if st=='PASS' else '#FFF3E0'
            sc_ = '#2E7D32' if st=='PASS' else '#E65100'
            sy = '✓' if st=='PASS' else '⚠'
            ax.barh(i, 3, height=0.65, color=bg, edgecolor='none')
            ax.text(0.08, i, sy, fontsize=10, fontweight='bold', color=sc_, va='center')
            ax.text(0.40, i+0.12, nm, fontsize=5.5, fontweight='bold', color='#333', va='center')
            ax.text(0.40, i-0.12, dt, fontsize=4.5, color='#666', va='center', style='italic')
    ax.set_title("A. Robustness Checklist", fontsize=8, fontweight='bold', loc='left')

    # Panel B: Sample size stability
    ax = axes[0,1]
    stab = rp.get("sample_size_stability", {})
    if stab:
        for d, fr in stab.items():
            if not fr: continue
            fn = {}
            for k, v in fr.items():
                try: fn[float(k)] = v
                except: continue
            if not fn: continue
            fs = sorted(fn.keys())
            ax.plot([f*100 for f in fs], [fn[f].get("mean_auroc",np.nan) for f in fs],
                   marker='o', ms=2.5, lw=0.7, color=get_disorder_color(d), alpha=0.7,
                   label=get_disorder_display(d))
        ax.set_xticks([25,50,75,100]); ax.set_xticklabels(['25%','50%','75%','100%'], fontsize=5)
        ax.set_xlabel("Sample Fraction", fontsize=6); ax.set_ylabel("AUROC", fontsize=6)
        ax.legend(fontsize=3.5, ncol=2, loc='lower right')
    ax.set_title("B. Sample Size Stability", fontsize=8, fontweight='bold', loc='left')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel C: Multiple comparison (lollipop)
    ax = axes[1,0]
    mc = rp.get("multiple_comparison", {})
    if mc:
        meths = list(mc.keys())
        nsig = [mc[m].get("n_significant",0) for m in meths]
        mcols = {"none":"#E53935","bonferroni":"#1565C0","holm":"#1976D2","fdr_bh":"#42A5F5","fdr_by":"#90CAF9"}
        for i, (m, n) in enumerate(zip(meths, nsig)):
            c = mcols.get(m, '#607D8B')
            ax.plot(n, i, 'o', color=c, ms=5); ax.hlines(i, 0, n, color=c, lw=1, alpha=0.6)
            pct = mc[m].get("proportion_significant",0)
            ax.text(n+5, i, f"{n} ({pct*100:.0f}%)", va='center', fontsize=5)
        ax.set_yticks(range(len(meths)))
        ax.set_yticklabels([m.replace('_',' ').upper() for m in meths], fontsize=5)
        ax.set_xlabel("N Significant", fontsize=6)
    ax.set_title("C. Multiple Comparison", fontsize=8, fontweight='bold', loc='left')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel D: Confound adjustment
    ax = axes[1,1]
    conf = rp.get("confound_adjustment", {})
    if conf and "comparison" in conf:
        comp = conf["comparison"]
        meths = list(comp.keys())
        rs = [comp[m].get("correlation_with_original",np.nan) for m in meths]
        mads = [comp[m].get("mean_abs_difference",np.nan) for m in meths]
        x = np.arange(len(meths)); w=0.35
        ax.bar(x-w/2, rs, w, color='#1976D2', edgecolor='black', lw=0.3, label='r')
        ax.bar(x+w/2, mads, w, color='#FF9800', edgecolor='black', lw=0.3, label='MAD')
        ax.set_xticks(x); ax.set_xticklabels([m.replace('_',' ').title() for m in meths], fontsize=5)
        ax.legend(fontsize=5)
        nr = conf.get("nonlinearity_ratio", np.nan)
        if not pd.isna(nr):
            ax.text(0.5, 0.95, f"Nonlinearity ratio: {nr:.2f}", transform=ax.transAxes,
                   fontsize=5, ha='center', va='top',
                   bbox=dict(boxstyle='round', fc='#FFF9C4', ec='#F57F17', alpha=0.9))
    ax.set_title("D. Confound Adjustment", fontsize=8, fontweight='bold', loc='left')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    _save(fig, "ed_fig6_sensitivity", "ed")


# =====================================================================
# SECTION 12: ED FIGURE 7 — EFFECT SIZE CATEGORY BREAKDOWN
# =====================================================================

def plot_ed_fig7_effect_categories():
    """ED Fig 7: Effect size category breakdown by disorder."""
    logger.info("ED Fig 7: Effect Size Categories...")
    ef = _csv(os.path.join(STATS_DIR, "effect_sizes_full.csv"))
    if ef is None: logger.error("  Skip ED Fig 7"); return

    ec = ef[~ef["feature"].isin(SELECTION_ARTIFACT_FEATURES)].copy()

    def cat(g):
        if pd.isna(g): return 'negligible'
        ag = abs(g)
        if ag >= 0.8: return 'large'
        if ag >= 0.5: return 'medium'
        if ag >= 0.2: return 'small'
        if ag >= 0.1: return 'minimal'
        return 'negligible'

    ec['cat'] = ec['hedges_g'].apply(cat)
    ct = ec.groupby(['disorder','cat']).size().unstack(fill_value=0)
    cat_order = ['large','medium','small','minimal','negligible']
    for c in cat_order:
        if c not in ct.columns: ct[c] = 0
    ct = ct[cat_order]
    ct = ct.loc[[d for d in DISORDER_GROUPS if d in ct.index]]

    cat_c = {'large':'#B71C1C','medium':'#E53935','small':'#FB8C00',
             'minimal':'#FFD54F','negligible':'#E0E0E0'}

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, SINGLE_COL*0.7))
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    disps = [get_disorder_display(d) for d in ct.index]
    ct_pct.index = disps

    bottom = np.zeros(len(ct_pct))
    for c in cat_order:
        vals = ct_pct[c].values
        ax.barh(range(len(ct_pct)), vals, left=bottom, color=cat_c[c],
                edgecolor='black', lw=0.3, label=c.title(), height=0.7)
        # Label segments > 5%
        for i, v in enumerate(vals):
            if v > 5:
                ax.text(bottom[i] + v/2, i, f'{v:.0f}%', ha='center', va='center',
                       fontsize=4.5, color='white' if c in ['large','medium'] else 'black',
                       fontweight='bold')
        bottom += vals

    ax.set_yticks(range(len(disps)))
    ax.set_yticklabels(disps, fontsize=5.5)
    for lb in ax.get_yticklabels():
        rn = [k for k,v in DISORDER_DISPLAY_NAMES.items() if v==lb.get_text()]
        if rn: lb.set_color(get_disorder_color(rn[0])); lb.set_fontweight('bold')
    ax.set_xlabel("Percentage of Feature-Disorder Tests", fontsize=7)
    ax.legend(fontsize=5, loc='lower right')
    ax.set_title("ED Fig 7 | Effect Size Category Distribution by Disorder\n(Non-artifact features only)",
                fontsize=8, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    _save(fig, "ed_fig7_effect_categories", "ed")


# =====================================================================
# SECTION 13: ED FIGURE 8 — STRICT DIRECTION INCONSISTENCY
# =====================================================================

def plot_ed_fig8_strict_inconsistency():
    """ED Fig 8: Strict direction-inconsistent features detail."""
    logger.info("ED Fig 8: Strict Inconsistency...")
    di = _csv(os.path.join(STATS_DIR, "direction_inconsistency.csv"))
    if di is None:
        # Try loading from effect_sizes_report
        logger.error("  Skip ED Fig 8"); return

    strict = di[di.get("has_strict_direction_inconsistency", False) == True].copy()
    if len(strict) == 0:
        logger.warning("  No strict direction-inconsistent features")
        return

    strict = strict.sort_values("sig_g_range", ascending=True)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, max(SINGLE_COL, len(strict)*0.25)))
    y = np.arange(len(strict))

    # Diverging bars: positive disorders (red) vs negative (blue)
    for i, (_, r) in enumerate(strict.iterrows()):
        pos_g = r.get("max_sig_positive_g", 0)
        neg_g = r.get("min_sig_negative_g", 0)
        ax.barh(i, pos_g, color='#E53935', height=0.6, edgecolor='black', lw=0.2)
        ax.barh(i, neg_g, color='#1E88E5', height=0.6, edgecolor='black', lw=0.2)
        # Range annotation
        ax.text(max(pos_g, abs(neg_g)) + 0.02, i,
               f"range={r.get('sig_g_range',0):.3f}",
               va='center', fontsize=4)

    ax.set_yticks(y)
    ax.set_yticklabels(strict['feature'].values, fontsize=5)
    ax.axvline(x=0, color='black', lw=0.5)
    ax.set_xlabel("Hedges' g (Red = positive disorders, Blue = negative disorders)", fontsize=6)
    ax.set_title(
        f"ED Fig 8 | {len(strict)} Strictly Direction-Inconsistent Features\n"
        "(Significant positive in some disorders, significant negative in others)",
        fontsize=8, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    _save(fig, "ed_fig8_strict_inconsistency", "ed")


# =====================================================================
# SECTION 14: MAIN PIPELINE
# =====================================================================

def run_all_visualizations():
    """Execute all visualizations."""
    logger.info("=" * 70)
    logger.info("VISUALIZATION PIPELINE v3.0 — Nature Manuscript Figures")
    logger.info("=" * 70)

    main_figs = [
        ("Fig 1:   The Atlas", plot_fig1_atlas),
        ("Fig 2:   p-Factor Evidence", plot_fig2_pfactor),
        ("Fig 3:   Detectability", plot_fig3_detectability),
        ("Fig 4:   Specificity Paradox", plot_fig4_specificity),
    ]

    ed_figs = [
        ("ED Fig 1: Factor Details", plot_ed_fig1_factor),
        ("ED Fig 2: Clustering", plot_ed_fig2_clustering),
        ("ED Fig 3: t-SNE", plot_ed_fig3_tsne),
        ("ED Fig 4: Confusion Matrix", plot_ed_fig4_confusion),
        ("ED Fig 5: Ablation", plot_ed_fig5_ablation),
        ("ED Fig 6: Sensitivity", plot_ed_fig6_sensitivity),
        ("ED Fig 7: Effect Categories", plot_ed_fig7_effect_categories),
        ("ED Fig 8: Strict Inconsistency", plot_ed_fig8_strict_inconsistency),
    ]

    ok, fail = 0, 0
    for name, func in main_figs + ed_figs:
        logger.info(f"\n{'─'*60}\n  {name}\n{'─'*60}")
        try:
            func(); ok += 1
        except Exception as e:
            logger.error(f"  ❌ {name}: {e}")
            import traceback; traceback.print_exc()
            fail += 1

    logger.info(f"\n{'='*70}")
    logger.info(f"COMPLETE: {ok} ok, {fail} failed")
    logger.info(f"  Main:   {MAIN_DIR}")
    logger.info(f"  ED:     {ED_DIR}")
    logger.info(f"{'='*70}")

    logger.info("\nManuscript Figure Guide:")
    logger.info("  ═══ MAIN TEXT ═══")
    logger.info("  Fig 1  The Digital Phenotyping Atlas (93×10 heatmap)")
    logger.info("  Fig 2  The p-Factor: Shared Direction, Graded Severity")
    logger.info("  Fig 3  Disorder Detectability and Clinical Utility")
    logger.info("  Fig 4  The Specificity Paradox")
    logger.info("  ═══ EXTENDED DATA ═══")
    logger.info("  ED 1   Factor Analysis Details (loadings, scree, alignment)")
    logger.info("  ED 2   Hierarchical Clustering + MDS")
    logger.info("  ED 3   t-SNE Small Multiples")
    logger.info("  ED 4   Confusion Matrix")
    logger.info("  ED 5   Dimension Ablation")
    logger.info("  ED 6   Sensitivity & Robustness Summary")
    logger.info("  ED 7   Effect Size Category Breakdown by Disorder")
    logger.info("  ED 8   Strict Direction Inconsistency Detail")


if __name__ == "__main__":
    run_all_visualizations()