"""
Step 07: Publication-Quality Visualizations.
"""

import os
import json
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial import ConvexHull
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import argparse
import sys
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

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
# GLOBAL CONSTANTS
# =====================================================================

SELECTION_ARTIFACT_FEATURES = {
    "mental_health_sub_ratio", "support_sub_ratio",
    "subreddit_entropy", "subreddit_herfindahl",
    "primary_subreddit_ratio",
}

MAIN_DIR = os.path.join(FIGURES_DIR, "main")
ED_DIR = os.path.join(FIGURES_DIR, "ed")

# =====================================================================
# GLOBAL CONSTANTS & STYLING (Updated per Nature/Science Guidelines)
# =====================================================================

plt.rcParams.update({
    # ── 1. 字体规范 (Nature 指定无衬线体) ──
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
    'font.size': 7,                  # 全局基准字号
    'axes.titlesize': 8,             # 子图标题 (原为9，偏大)
    'axes.labelsize': 7,             # 轴标签 (原为8)
    'xtick.labelsize': 6,            # 刻度标签保持6
    'ytick.labelsize': 6,
    'legend.fontsize': 7,            # 图例字号 (原为6，偏小)
    'figure.titlesize': 9,           # 全图大标题
    'axes.titleweight': 'bold',      # 子图标题加粗 (顶级期刊标准)
    
    # ── 2. 轴线与刻度规范 ──
    'axes.linewidth': 0.8,           # 轴线变粗，提升灰度打印可见度 (原为0.5)
    'axes.spines.top': False,        # 全局自动隐藏上边框！
    'axes.spines.right': False,      # 全局自动隐藏右边框！
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3.0,         # 增加刻度线长度
    'ytick.major.size': 3.0,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    
    # ── 3. 线条与标记 ──
    'lines.linewidth': 1.5,          # 强调主数据线 (原为0.75，太细不符合Nature风格)
    'lines.markersize': 4.0,
    'lines.markeredgewidth': 0.5,
    'errorbar.capsize': 3.0,         # 规范误差棒横线宽度
    
    # ── 4. 矢量导出与格式 ──
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'pdf.fonttype': 42,              # 确保能在 Illustrator 中编辑文本
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
})

def mm2inch(mm: float) -> float:
    """转换毫米到英寸，适配期刊出版要求"""
    return mm / 25.4

# Nature/Science 系列标准尺寸 (英寸)
SINGLE_COL = mm2inch(89)    # 单栏 (3.5 inch)
ONE_HALF_COL = mm2inch(120) # 1.5 栏 (4.7 inch) - 新增，适合含有大图例的图表
DOUBLE_COL = mm2inch(183)   # 双栏全宽 (7.2 inch)

DIVERGING_CMAP = 'RdBu_r'

WONG_PALETTE = [
    '#0072B2', # Blue
    '#D55E00', # Vermillion
    '#009E73', # Bluish Green
    '#E69F00', # Orange
    '#56B4E9', # Sky Blue
    '#CC79A7', # Reddish Purple
    '#F0E442', # Yellow
    '#000000'  # Black
]

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

DIM_SHORT = {
    "affective": "Affect", "pronoun": "Pronoun",
    "cognitive": "Cognit.", "temporal_linguistic": "Temp-Ling",
    "risk": "Risk", "behavioral": "Behav.",
    "readability": "Readab.", "post_length": "PostLen",
    "temporal": "Temporal", "social": "Social",
    "community": "Commun.", "other": "Other",
}

# =====================================================================
# 统一方向/分类配色 (Fig2, Fig4 共用，确保同概念同色)
# =====================================================================
DIRECTION_COLORS = {
    "positive":      "#E53935",  # Patient > Control (红)
    "negative":      "#1E88E5",  # Control > Patient (蓝)
    "mixed":         "#FB8C00",  # Mixed Direction (橙)
    "non_sig":       "#BDBDBD",  # Non-significant (灰)
    "transdiagnostic": "#E53935",  # 跨诊断 → 与 positive 同色 (均为患者 > 对照)
    "specific":      "#1E88E5",  # 特异性 → 与 negative 同色
}

# 统一字号常量 (遵循 ref.md 规范)
FS_TITLE    = 8     # 子图标题
FS_AXLABEL  = 7     # 轴标签
FS_TICK     = 6     # 刻度标签
FS_ANNOT    = 6     # 图内标注
FS_LEGEND   = 6.5   # 图例文字
FS_SUPTITLE = 9     # 整图标题

# ED 热力图统一规范 — 详见 lyc/热力图规范.md
HM_ED_WIDTH = 1920 / 300
HM_ED_HEIGHT = HM_ED_WIDTH * 0.90
HM_ED_MARGINS = dict(left=0.18, right=0.92, top=0.90, bottom=0.18)
HM_CBAR_W = "4%"
HM_CBAR_PAD = 0.15
HM_REF_VMAX = 0.6          # 参照 ED Fig 4 比例色标满量程
HM_TEXT_WHITE_AT = 0.35    # 在该参照量程上 ≥ 此值用白字

# ED 柱状/条形图统一规范 — 详见 lyc/柱状图规范.md
BAR_ED_WIDTH = SINGLE_COL * 1.5
BAR_ED_HEIGHT = BAR_ED_WIDTH * 0.90
BAR_EDGE_COLOR = "#222222"
BAR_EDGE_LW = 0.6
BAR_STACK_W = 0.65
BAR_Y_PAD = 1.12
BAR_MARGIN_LEFT = 0.14
BAR_MARGIN_RIGHT = 0.94
BAR_MARGIN_TOP = 0.90
BAR_MARGIN_BOTTOM = 0.12
BAR_MARGIN_BOTTOM_ROTATED = 0.27

# =====================================================================
# UTILITY FUNCTIONS
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

def _clean_name(f):
    """Shorten feature name for compact display."""
    return (f.replace('_rate', '').replace('_ratio', '')
             .replace('_mean', '').replace('_score', '')
             .replace('_index', '').replace('_log_signed', '†')
             .replace('_log', '*').replace('_', ' '))

def _save(fig, name, subdir="main"):
    d = MAIN_DIR if subdir == "main" else ED_DIR
    for fmt in ['pdf', 'png']:
        fig.savefig(os.path.join(d, f"{name}.{fmt}"),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved: {subdir}/{name} (pdf+png)")


def _save_ed_fixed(fig, name):
    """Save an ED figure without tight crop; preserve its configured canvas."""
    original_bbox = plt.rcParams.get("savefig.bbox")
    plt.rcParams["savefig.bbox"] = None
    try:
        for fmt in ("pdf", "png"):
            fig.savefig(os.path.join(ED_DIR, f"{name}.{fmt}"),
                        dpi=300, bbox_inches=None, pad_inches=0)
    finally:
        plt.rcParams["savefig.bbox"] = original_bbox
        plt.close(fig)
    logger.info(f"  Saved: ed/{name} (pdf+png, fixed canvas)")


def _save_main_fixed(fig, name):
    """Save a main figure without tight crop; preserve its configured canvas."""
    original_bbox = plt.rcParams.get("savefig.bbox")
    plt.rcParams["savefig.bbox"] = None
    try:
        for fmt in ("pdf", "png"):
            fig.savefig(os.path.join(MAIN_DIR, f"{name}.{fmt}"),
                        dpi=300, bbox_inches=None, pad_inches=0)
    finally:
        plt.rcParams["savefig.bbox"] = original_bbox
        plt.close(fig)
    logger.info(f"  Saved: main/{name} (pdf+png, fixed canvas)")


def _despine(ax, top=True, right=True, left=False, bottom=False):
    """
    移除指定轴框，Nature 风格标准操作。
    默认隐藏顶部和右侧，可根据需求隐藏左侧和底部。
    """
    if top:
        ax.spines['top'].set_visible(False)
        ax.tick_params(top=False)
    if right:
        ax.spines['right'].set_visible(False)
        ax.tick_params(right=False)
    if left:
        ax.spines['left'].set_visible(False)
        ax.tick_params(left=False)
    if bottom:
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(bottom=False)


def _heatmap_cell_white_threshold(scale_max):
    """Cell text color threshold, scaled from HM_REF_VMAX / HM_TEXT_WHITE_AT."""
    return HM_TEXT_WHITE_AT * (float(scale_max) / HM_REF_VMAX)


def _heatmap_value_interval(finite):
    """Classify values for colormap: round 2 dp (-0.00 → 0)."""
    rounded = np.round(finite, 2)
    all_nonneg = bool(np.all(rounded >= 0))
    all_nonpos = bool(np.all(rounded <= 0))
    return all_nonneg, all_nonpos, (all_nonneg or all_nonpos)


def _prepare_heatmap_layers(data, finite, *, fixed_sequential_vmax=None):
    """
    Return (plot_data, imshow_kw, text_white_thresh, cmap_log).
    imshow_kw: cmap + (vmin/vmax) or norm for ax.imshow(..., aspect='auto').
    """
    if fixed_sequential_vmax is not None:
        vmax = float(fixed_sequential_vmax)
        plot = np.where(np.isfinite(data), data, np.nan)
        kw = dict(cmap="Blues", vmin=0, vmax=vmax)
        return plot, kw, _heatmap_cell_white_threshold(vmax), "Blues 0–fixed"

    all_nonneg, all_nonpos, single = _heatmap_value_interval(finite)
    if single and all_nonneg:
        plot = np.where(np.isfinite(data), np.maximum(data, 0.0), np.nan)
        vmax = max(float(np.nanmax(plot)) * 1.08, 0.01)
        kw = dict(cmap="Blues", vmin=0, vmax=vmax)
        return plot, kw, _heatmap_cell_white_threshold(vmax), "Blues 0–max"
    if single and all_nonpos:
        plot = np.where(np.isfinite(data), np.minimum(data, 0.0), np.nan)
        vmin = min(float(np.nanmin(plot)) * 1.08, -0.01)
        kw = dict(cmap="Blues_r", vmin=vmin, vmax=0)
        return plot, kw, _heatmap_cell_white_threshold(abs(vmin)), "Blues_r min–0"
    plot = data
    vm = max(abs(float(np.nanmin(finite))), abs(float(np.nanmax(finite))), 0.1)
    kw = dict(cmap=DIVERGING_CMAP, norm=TwoSlopeNorm(vmin=-vm, vcenter=0, vmax=vm))
    return plot, kw, _heatmap_cell_white_threshold(vm), "RdBu diverging"


def _ed_heatmap_figure():
    """Non-square canvas with a 1:1 heatmap plotting area."""
    fig, ax = plt.subplots(figsize=(HM_ED_WIDTH, HM_ED_HEIGHT))
    margins = HM_ED_MARGINS.copy()
    fig.subplots_adjust(**margins)
    return fig, ax


def _ed_heatmap_imshow(ax, plot_data, imshow_kw):
    im = ax.imshow(plot_data, aspect="auto", interpolation="nearest", **imshow_kw)
    nr, nc = plot_data.shape
    ax.set_xlim(-0.5, nc - 0.5)
    ax.set_ylim(nr - 0.5, -0.5)
    ax.set_box_aspect(1)
    return im


def _ed_heatmap_annotate(ax, data, text_white_thresh, *, empty_at=0.01):
    """Annotate cells. empty_at=None: only NaN → '—'; else |v| rounded < empty_at → '—'."""
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "\u2014", ha="center", va="center",
                        fontsize=FS_ANNOT, color="#AAAAAA", zorder=5)
                continue
            v_disp = float(np.round(v, 2))
            if empty_at is not None and abs(v_disp) < empty_at:
                ax.text(j, i, "\u2014", ha="center", va="center",
                        fontsize=FS_ANNOT, color="#AAAAAA", zorder=5)
                continue
            tc = "white" if abs(v_disp) > text_white_thresh else "black"
            ax.text(j, i, f"{v_disp:.2f}", ha="center", va="center",
                    fontsize=FS_ANNOT, color=tc, zorder=5)


def _ed_heatmap_style_frame(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#222222")
        spine.set_linewidth(1.0)
    ax.tick_params(labelsize=FS_TICK)


def _ed_heatmap_colorbar(fig, ax, im, label):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=HM_CBAR_W, pad=HM_CBAR_PAD)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=FS_AXLABEL)
    cb.ax.tick_params(labelsize=FS_TICK)
    cb.outline.set_visible(True)
    cb.outline.set_color("#222222")
    cb.outline.set_linewidth(1.0)
    nr, nc = im.get_array().shape
    ax.set_xlim(-0.5, nc - 0.5)
    ax.set_ylim(nr - 0.5, -0.5)
    ax.set_box_aspect(1)
    return cb


def _ed_heatmap_title(ax, title):
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", loc="center", pad=20)


def _ed_bar_figure(*, left=BAR_MARGIN_LEFT, right=BAR_MARGIN_RIGHT,
                   bottom=BAR_MARGIN_BOTTOM):
    """Non-square canvas with a 1:1 plotting area for ED bar charts."""
    fig, ax = plt.subplots(figsize=(BAR_ED_WIDTH, BAR_ED_HEIGHT))
    fig.subplots_adjust(
        left=left, right=right, top=BAR_MARGIN_TOP, bottom=bottom)
    ax.set_box_aspect(1)
    return fig, ax


def _ed_bar_style(ax, *, xlabel=None, ylabel=None, title=None):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FS_AXLABEL)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS_AXLABEL)
    if title:
        ax.set_title(title, fontsize=FS_TITLE, fontweight="bold",
                     loc="center", pad=20)
    ax.tick_params(labelsize=FS_TICK, direction="out")
    ax.set_box_aspect(1)
    _despine(ax)


def _compute_eigenvalues():
    em = _csv(os.path.join(STATS_DIR, "effect_size_matrix_clean.csv"),
              index_col=0)
    if em is None:
        em = _csv(os.path.join(STATS_DIR, "effect_size_matrix.csv"),
                  index_col=0)
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
        r_eigs[i] = sorted(
            np.linalg.eigh(np.corrcoef(rd.T))[0], reverse=True)
    p95 = np.percentile(r_eigs, 95, axis=0).tolist()
    return [float(e) for e in eig], p95


# =====================================================================
# FIGURE 1 — THE ATLAS (Redesigned)
# =====================================================================

import matplotlib.ticker as ticker
from matplotlib.patheffects import withStroke

def plot_fig1_atlas():
    """Figure 1: Digital Phenotyping Atlas."""
    logger.info("MAIN Fig 1: The Atlas...")

    em = _csv(os.path.join(STATS_DIR, "effect_size_matrix_clean.csv"), index_col=0)
    if em is None:
        em = _csv(os.path.join(STATS_DIR, "effect_size_matrix.csv"), index_col=0)
        if em is not None:
            arts = [f for f in em.index if f in SELECTION_ARTIFACT_FEATURES]
            if arts: em = em.drop(index=arts)
    if em is None: return

    sig = _csv(os.path.join(STATS_DIR, "significance_matrix_clean.csv"), index_col=0)
    if sig is None:
        sig = _csv(os.path.join(STATS_DIR, "significance_matrix.csv"), index_col=0)

    f2d, _ = _load_dim_map() if hasattr(sys.modules[__name__], '_load_dim_map') else ({}, [])
    em.columns = [get_disorder_display(c) for c in em.columns]
    if sig is not None:
        sig.columns = [get_disorder_display(c) for c in sig.columns]
    
    abs_max = 1.6 
    cmap = plt.get_cmap('RdBu_r')
    norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    _dim_order = globals().get('DIM_ORDER', sorted(list(set(f2d.values()))))
    y_positions = np.arange(len(_dim_order))
    _dim_short = globals().get('DIM_SHORT', {})
    _dim_colors = globals().get('DIM_COLORS', {})
    dim_labels = [_dim_short.get(d, str(d)[:8]) for d in _dim_order]
    rng = np.random.RandomState(RANDOM_SEED if 'RANDOM_SEED' in globals() else 42)

    global_max_info = {}
    for dim in _dim_order:
        best_abs_val = -1
        best_dis, best_feat = None, None
        
        for disorder in em.columns:
            g_series = em[disorder]
            feats = [f for f in g_series.index if f2d.get(f) == dim]
            if not feats: continue
            
            vals = g_series[feats].values
            valid_mask = ~np.isnan(vals)
            if not valid_mask.any(): continue
            
            v_valid, f_valid = vals[valid_mask], np.array(feats)[valid_mask]
            max_idx = np.argmax(np.abs(v_valid))
            curr_max_abs = np.abs(v_valid[max_idx])
            
            if curr_max_abs > best_abs_val:
                best_abs_val = curr_max_abs
                best_dis, best_feat = disorder, f_valid[max_idx]
                
        global_max_info[dim] = (best_dis, best_feat)

    n_rows, n_cols = 2, 5
    fig_w = DOUBLE_COL * 1.45 
    fig_h = DOUBLE_COL * 0.85
    fig = plt.figure(figsize=(fig_w, fig_h)) 
    
    gs = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        hspace=0.35, wspace=0.25, 
        left=0.06, right=0.98, top=0.90, bottom=0.15
    )

    axes = []
    
    for idx, disorder in enumerate(em.columns):
        if idx >= n_rows * n_cols: break
        
        row, col = divmod(idx, n_cols)

        bg_ax = fig.add_subplot(gs[row, col], zorder=1)
        bg_ax.axis('off')  
        bg_ax.set_xscale('symlog', linthresh=0.25)
        bg_ax.set_xlim(-2.5, 2.5)
        bg_ax.set_ylim(len(_dim_order), -1)

        for y_idx in y_positions:
            bg_color = '#F7F9FA' if y_idx % 2 == 0 else '#FFFFFF'
            bg_ax.axhspan(y_idx - 0.5, y_idx + 0.5, color=bg_color, ec='none', zorder=0)
            bg_ax.axhline(y_idx + 0.5, color='#E0E0E0', lw=0.5, zorder=1)
        bg_ax.axvline(x=0, color='#333333', lw=0.8, ls='--', zorder=2)

        ax = fig.add_subplot(gs[row, col], zorder=100)
        ax.set_facecolor('none')
        axes.append(ax)
        
        ax.set_xscale('symlog', linthresh=0.25)

        g_series = em[disorder]
        sig_series = sig[disorder] if sig is not None else pd.Series(False, index=g_series.index)

        for y_idx, dim in enumerate(_dim_order):
            feats = [f for f in g_series.index if f2d.get(f) == dim]
            if not feats: continue
            
            vals = g_series[feats].values
            sigs = sig_series[feats].values
            valid_mask = ~np.isnan(vals)
            if not valid_mask.any(): continue
            
            vals, sigs = vals[valid_mask], sigs[valid_mask]
            valid_feats = np.array(feats)[valid_mask]
            
            ys = np.zeros_like(vals)
            colors, sizes, edgecolors, linewidths = [], [], [], []
            
            for i, (v, s) in enumerate(zip(vals, sigs)):
                is_sig = s if isinstance(s, bool) else (str(s).lower() == 'true')
                c = list(cmap(norm(v)))
                
                if is_sig:
                    ys[i] = y_idx + rng.uniform(-0.35, 0.35)
                    c[3] = 0.90      
                    sizes.append(28)  
                    edgecolors.append('#222222')
                    linewidths.append(0.5)
                else:
                    ys[i] = y_idx + rng.uniform(-0.12, 0.12)
                    c = list(mcolors.to_rgba('#CCCCCC'))
                    c[3] = 0.65         
                    sizes.append(12.0)  
                    edgecolors.append('#FFFFFF') 
                    linewidths.append(0.4)       

                colors.append(c)

            ax.scatter(vals, ys, s=sizes, c=colors, edgecolors=edgecolors, linewidths=linewidths, zorder=5)

            best_dis, best_feat = global_max_info.get(dim, (None, None))
            if best_dis == disorder:
                idx_in_valid = np.where(valid_feats == best_feat)[0]
                if len(idx_in_valid) > 0:
                    idx_in_valid = idx_in_valid[0]
                    max_val, max_y = vals[idx_in_valid], ys[idx_in_valid]
                    
                    if hasattr(sys.modules[__name__], '_clean_name'):
                        raw_name = _clean_name(best_feat)
                    else:
                        raw_name = str(best_feat).replace('_', ' ')
                    
                    lookup_name = str(raw_name).lower()
                    
                    if "anxi" in lookup_name: 
                        f_name = "anx."
                    elif "first person" in lookup_name: 
                        f_name = "1st-pers."
                    elif "mattr" in lookup_name: 
                        f_name = "MATTR"
                    elif "posts per day" in lookup_name: 
                        f_name = "PPD*"
                    elif "other sub" in lookup_name: 
                        f_name = "other-sub"
                    elif "death mortal" in lookup_name: 
                        f_name = "death-mort."
                    elif "exclam" in lookup_name: 
                        f_name = "exclam."
                    elif "neg" in lookup_name: 
                        f_name = "neg."
                    elif "insight" in lookup_name: 
                        f_name = "insght."
                    elif "past focus" in lookup_name: 
                        f_name = "past-foc."
                    elif "post length" in lookup_name: 
                        f_name = "post-len."
                    else:
                        f_name = str(raw_name).replace('*', '').strip()[:8]

                    label_adjustments = {
                        "anx.":        (0, -10, 'left', 'top'),       
                        "1st-pers.":   (-2, -10, 'left', 'top'),       
                        "MATTR":       (5, 20, 'right', 'top'),   
                        "PPD*":        (16, 12, 'left', 'bottom'),     
                        "other-sub":   (2, 15, 'right', 'top'),     
                        "death-mort.": (5, 10, 'right', 'bottom'),   
                        "exclam.":     (8, 14, 'left', 'top'),       
                        "neg.":        (-15, 15, 'right', 'bottom'),   
                        "insght.":     (6, -16, 'left', 'top'),       
                        "past-foc.":   (0, -20, 'left', 'bottom'),     
                        "post-len.":   (0, -14, 'left', 'top'),       
                    }

                    if f_name in label_adjustments:
                        offset_x, offset_y, halign, valign = label_adjustments[f_name]
                    else:
                        offset_x = 15 if max_val >= 0 else -15
                        offset_y = 0
                        halign = 'left' if max_val >= 0 else 'right'
                        valign = 'center'
                    
                    dim_color = _dim_colors.get(dim, '#666666')
                    
                    txt = ax.annotate(
                        f_name, xy=(max_val, max_y), xytext=(offset_x, offset_y),
                        textcoords='offset points', ha=halign, va=valign,
                        fontsize=FS_ANNOT, color=dim_color, fontweight='bold', zorder=999,
                        clip_on=False,
                        arrowprops=dict(arrowstyle='-', color=dim_color, lw=0.8, alpha=0.8, zorder=999)
                    )
                    txt.set_path_effects([withStroke(linewidth=2.5, foreground='white')])
                    
                    if offset_x > 0 or offset_y > 0:
                        ax.set_zorder(200 - idx)
                    else:
                        ax.set_zorder(200 + idx)

        _disorder_names = globals().get('DISORDER_DISPLAY_NAMES', {})
        rn = [k for k, v in _disorder_names.items() if v == disorder]
        ax.set_title(disorder, fontsize=FS_TITLE, fontweight='bold', color=get_disorder_color(rn[0]) if rn else '#333', pad=6)
        
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(len(_dim_order), -1) 
        
        _despine(ax, top=True, right=True, left=True, bottom=False)
        ax.tick_params(left=False, bottom=True)
        
        if row == n_rows - 1:
            ax.set_xlabel("Hedges' g (SymLog)", fontsize=FS_AXLABEL, labelpad=4)
            ax.tick_params(axis='x', labelsize=FS_TICK)
            ticks = [-2.5, -0.25, 0, 0.25, 2.5]
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:g}"))
        else:
            ax.set_xticklabels([])
            ax.set_xlabel("")

        if col == 0: 
            ax.set_yticks(y_positions)
            ax.set_yticklabels(dim_labels, fontsize=FS_TICK, fontweight='bold')
            for i, d in enumerate(_dim_order):
                ax.get_yticklabels()[i].set_color(_dim_colors.get(d, '#333333'))
        else:
            ax.set_yticks([])

        fig.text(0.5, 0.985, "Figure 1 | Digital Phenotyping Atlas: Feature Distributions by Disorder",
             fontsize=FS_SUPTITLE, fontweight='bold', ha='center', va='top')
    
    legend_ax = fig.add_axes([0.15, 0.02, 0.7, 0.04]) 
    legend_ax.axis('off')
    
    legend_ax.scatter([0.05], [0.5], s=28, c='#D55E00', edgecolors='#222', lw=0.5)
    legend_ax.text(0.08, 0.5, "Significant (Patient > Control)", fontsize=FS_LEGEND, va='center')
    
    legend_ax.scatter([0.42], [0.5], s=28, c='#0072B2', edgecolors='#222', lw=0.5)
    legend_ax.text(0.45, 0.5, "Significant (Control > Patient)", fontsize=FS_LEGEND, va='center')
    
    legend_ax.scatter([0.79], [0.5], s=12, c='#CCCCCC', alpha=0.65, edgecolors='#FFFFFF', lw=0.4)
    legend_ax.text(0.82, 0.5, "Not Significant", fontsize=FS_LEGEND, va='center')

    _save(fig, "fig1_atlas", "main")


# =====================================================================
# FIGURE 2 — p-FACTOR EVIDENCE (Redesigned)
# =====================================================================

def plot_fig2_pfactor():
    """Figure 2: The p-Factor."""
    logger.info("MAIN Fig 2: p-Factor Evidence...")

    ef = _csv(os.path.join(STATS_DIR, "effect_sizes_full.csv"))
    if ef is None:
        logger.error("  No effect sizes. Skip Fig 2.")
        return

    ec = ef[~ef["feature"].isin(SELECTION_ARTIFACT_FEATURES)].copy()
    g_piv = ec.pivot_table(
        index="feature", columns="disorder",
        values="hedges_g", aggfunc="first")
    s_piv = ec.pivot_table(
        index="feature", columns="disorder",
        values="sig_fdr_global", aggfunc="first")

    feats = sorted(g_piv.index)
    disords = sorted(g_piv.columns)
    feat_info = []
    for f in feats:
        gv = g_piv.loc[f]
        sv = s_piv.loc[f] if f in s_piv.index else pd.Series()
        n_pos, n_neg = 0, 0
        for d in disords:
            g = gv.get(d, np.nan)
            s = sv.get(d, False)
            if isinstance(s, str): s = s.lower() == 'true'
            if pd.notna(g) and s:
                if g > 0: n_pos += 1
                elif g < 0: n_neg += 1
        feat_info.append({
            'feature': f, 'n_pos': n_pos, 'n_neg': n_neg,
            'is_mixed': n_pos > 0 and n_neg > 0,
            'direction': 'positive' if n_pos > 0 and n_neg == 0
                         else ('negative' if n_neg > 0 and n_pos == 0
                               else 'mixed' if n_pos > 0 and n_neg > 0 else 'none')
        })
    fi_df = pd.DataFrame(feat_info)

    rpt = _json(os.path.join(STATS_DIR, "effect_sizes_report.json"))
    ne_meta = rpt.get("direction_inconsistency_null_expectation") if isinstance(rpt, dict) else None
    ne = ne_meta.get("expected_exploratory") if isinstance(ne_meta, dict) else None
    if not isinstance(ne, (int, float)):
        raise FileNotFoundError(
            "ED Fig (direction inconsistency) requires "
            f"'direction_inconsistency_null_expectation.expected_exploratory' in "
            f"{os.path.join(STATS_DIR, 'effect_sizes_report.json')}; "
            "refusing to plot a hard-coded fallback."
        )
    null_exp = float(ne)

    COLOR_POS = '#D6604D'   # muted red (Patient > Control)
    COLOR_NEG = '#4393C3'   # muted blue (Control > Patient)
    COLOR_MIX = '#F4A736'   # muted orange (Mixed Direction)
    COLOR_NS  = '#F5F5F5'

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.95))
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        height_ratios=[1, 1],
        width_ratios=[1, 1.15],
        hspace=0.50, wspace=0.35
    )

    # Panel A: Donut
    ax_a1 = fig.add_subplot(gs[0, 0])
    dir_counts = fi_df['direction'].value_counts()
    labels, sizes, colors = [], [], []
    
    color_map = {
        'positive': ('Patient > Control', COLOR_POS),
        'mixed': ('Mixed Direction', COLOR_MIX),
        'negative': ('Control > Patient', COLOR_NEG),
        'none': ('Non-significant', '#E0E0E0')
    }
    
    for cat, (lab, col) in color_map.items():
        n = dir_counts.get(cat, 0)
        if n > 0:
            labels.append(f'{lab} (n={n})')
            sizes.append(n)
            colors.append(col)

    wedges, texts, autotexts = ax_a1.pie(
        sizes, labels=None, colors=colors,
        autopct='%1.0f%%', pctdistance=0.75,
        startangle=90, wedgeprops=dict(width=0.45, edgecolor='white', linewidth=1.5)
    )
    for t in autotexts:
        t.set_fontsize(FS_ANNOT)
        t.set_fontweight('bold')
        t.set_color('white' if t.get_text() != '0%' else 'none')

    ax_a1.legend(labels, fontsize=FS_LEGEND, loc='upper center',
                 bbox_to_anchor=(0.5, -0.05), frameon=False)
    ax_a1.set_title("A. Direction Consistency", fontsize=FS_TITLE, fontweight='bold', loc='left', pad=20)

    # Panel B: Mixed direction heatmap
    ax_a2 = fig.add_subplot(gs[0, 1])
    mixed_feats = fi_df[fi_df['is_mixed']].copy()
    
    if len(mixed_feats) > 10:
        mixed_feats = mixed_feats.sample(n=10, random_state=42).sort_values('n_neg', ascending=False)
    
    n_mixed_show = len(mixed_feats)
    total_mixed = fi_df['is_mixed'].sum()

    if n_mixed_show > 0 and g_piv is not None:
        dir_mat = np.zeros((n_mixed_show, len(disords)))
        for i, (_, row) in enumerate(mixed_feats.iterrows()):
            f = row['feature']
            if f in g_piv.index:
                for j, d in enumerate(disords):
                    g = g_piv.loc[f, d]
                    s = s_piv.loc[f, d] if f in s_piv.index else False
                    if isinstance(s, str): s = s.lower() == 'true'
                    if pd.notna(g) and s:
                        dir_mat[i, j] = 1 if g > 0 else -1

        from matplotlib.colors import ListedColormap
        dir_cmap = ListedColormap([COLOR_NEG, COLOR_NS, COLOR_POS])
        
        im = ax_a2.imshow(
            dir_mat, cmap=dir_cmap, vmin=-1, vmax=1, 
            aspect='equal', interpolation='nearest', rasterized=True
        )

        ax_a2.set_xticks(np.arange(len(disords)) - 0.5, minor=True)
        ax_a2.set_yticks(np.arange(n_mixed_show) - 0.5, minor=True)
        ax_a2.grid(which="minor", color="white", linestyle='-', linewidth=0.8, alpha=0.95)
        ax_a2.tick_params(which="minor", bottom=False, left=False)

        ax_a2.set_xticks(range(len(disords)))
        ax_a2.set_xticklabels([get_disorder_display(d) for d in disords], 
                              rotation=45, ha='right', rotation_mode="anchor", fontsize=FS_TICK)
        
        for lb in ax_a2.get_xticklabels():
            rn = [k for k, v in DISORDER_DISPLAY_NAMES.items() if v == lb.get_text()]
            if rn:
                lb.set_color(get_disorder_color(rn[0]))
                lb.set_fontweight('bold')

        ax_a2.set_yticks(range(n_mixed_show))
        clean_ylabels = [_clean_name(f)[:22] for f in mixed_feats['feature'].values]
        ax_a2.set_yticklabels(clean_ylabels, fontsize=FS_TICK, fontweight='bold')
        
        f2d, _ = _load_loadings_or_dim() if hasattr(sys.modules[__name__], '_load_loadings_or_dim') else _load_dim_map()
        for i, f in enumerate(mixed_feats['feature'].values):
            dim = f2d[0].get(f, 'other') if isinstance(f2d, tuple) else f2d.get(f, 'other')
            ax_a2.get_yticklabels()[i].set_color(DIM_COLORS.get(dim, '#333333'))

        ax_a2.text(1.02, 0, f"+ {total_mixed - 10} more features", transform=ax_a2.transAxes, 
                   fontsize=FS_ANNOT, ha='left', va='bottom', color='#777')

    h_pos = Rectangle((0, 0), 1, 1, fc=COLOR_POS, ec='none', label='Patient > Control')
    h_neg = Rectangle((0, 0), 1, 1, fc=COLOR_NEG, ec='none', label='Control > Patient')
    h_ns = Rectangle((0, 0), 1, 1, fc=COLOR_NS, ec='#CCC', label='Non-significant')
    
    ax_a2.legend(handles=[h_pos, h_neg, h_ns], fontsize=FS_LEGEND, loc='lower center', 
                 bbox_to_anchor=(0.5, 1.02), ncol=3, framealpha=0)
                 
    ax_a2.set_title("B. Direction-Inconsistent Features", fontsize=FS_TITLE, fontweight='bold', loc='left', pad=20)

    # Panel C: Bar chart
    ax_c = fig.add_subplot(gs[1, 0])
    cats = ['Observed', 'Expected\n(null)']
    vals = [total_mixed, null_exp]
    cols = [COLOR_POS, '#BDBDBD']
    
    bars = ax_c.bar(range(2), vals, color=cols, edgecolor='black', lw=0.5, width=0.5)
    
    for b, v in zip(bars, vals):
        ax_c.text(b.get_x() + b.get_width() / 2, v + 2, f'{v:.0f}', 
                  ha='center', fontsize=FS_AXLABEL, fontweight='bold')
                  
    ratio = total_mixed / max(1, null_exp)
    
    ax_c.set_ylim(0, max(vals) * 1.65)
    ax_c.text(0.5, 0.98, f"O/E = {ratio:.1f}×\nFar fewer than chance\n→ p-factor structure",
              transform=ax_c.transAxes, fontsize=FS_ANNOT, ha='center', va='top',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF9C4', edgecolor='#F57F17', alpha=0.9))
              
    ax_c.set_xticks(range(2))
    ax_c.set_xticklabels(cats, fontsize=FS_TICK)
    ax_c.set_ylabel("Direction-Inconsistent Features", fontsize=FS_AXLABEL)
    ax_c.set_title("C. Observed vs. Null", fontsize=FS_TITLE, fontweight='bold', loc='left')
    _despine(ax_c)

    # Panel D: Violin
    ax_d = fig.add_subplot(gs[1, 1])
    disorder_g = {}
    for d in disords:
        mask = ec["disorder"] == d
        vals_ = ec.loc[mask, "hedges_g"].dropna().abs()
        if len(vals_) > 0:
            disorder_g[get_disorder_display(d)] = vals_.values

    if disorder_g:
        sorted_d = sorted(disorder_g.keys(), key=lambda x: np.median(disorder_g[x]))
        positions = list(range(len(sorted_d)))
        
        ax_d.set_xlim(-0.8, len(sorted_d) + 1.2)
        
        parts = ax_d.violinplot(
            [disorder_g[d] for d in sorted_d],
            positions=positions, vert=True, widths=0.75,
            showmedians=True, showextrema=False)
            
        parts['cmedians'].set_color('#222222')
        parts['cmedians'].set_linewidth(1.2)
            
        for i, d in enumerate(sorted_d):
            rn = [k for k, v in DISORDER_DISPLAY_NAMES.items() if v == d]
            c = get_disorder_color(rn[0]) if rn else '#808080'
            parts['bodies'][i].set_facecolor(c)
            parts['bodies'][i].set_alpha(0.7)
            parts['bodies'][i].set_edgecolor('black')
            parts['bodies'][i].set_linewidth(0.4)

        for thresh, label in [(0.2, 'Small'), (0.5, 'Medium'), (0.8, 'Large')]:
            ax_d.plot([-0.5, len(sorted_d)-0.5], [thresh, thresh], color='#BDBDBD', ls=':', lw=0.8, zorder=0)
            ax_d.text(len(sorted_d) - 0.1, thresh, f'{label} ({thresh})', 
                      fontsize=FS_ANNOT, color='#757575', ha='left', va='center')

        ax_d.set_xticks(positions)
        ax_d.set_xticklabels(sorted_d, rotation=45, ha='right', rotation_mode="anchor", fontsize=FS_TICK)
        
        for lb in ax_d.get_xticklabels():
            rn = [k for k, v in DISORDER_DISPLAY_NAMES.items() if v == lb.get_text()]
            if rn:
                lb.set_color(get_disorder_color(rn[0]))
                lb.set_fontweight('bold')

    ax_d.set_ylabel("|Hedges' g|", fontsize=FS_AXLABEL)
    ax_d.set_title("D. Effect Size Distribution by Disorder", fontsize=FS_TITLE, fontweight='bold', loc='left')
    _despine(ax_d)

    fig.suptitle("Figure 2 | The p-Factor: Shared Direction, Graded Severity", 
                 fontsize=FS_SUPTITLE, fontweight='bold', y=1.02)
                 
    _save(fig, "fig2_pfactor", "main")


# =====================================================================
# FIGURE 3 — DETECTABILITY (Redesigned)
# =====================================================================

from matplotlib.gridspec import GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_fig3_detectability():
    """Figure 3: Detectability and Clinical Utility."""
    logger.info("MAIN Fig 3: Detectability...")

    pw = _csv(os.path.join(STATS_DIR, "auroc_pairwise_matrix.csv"), index_col=0)
    vc = _csv(os.path.join(STATS_DIR, "auroc_vs_control.csv"))
    dvc = _csv(os.path.join(STATS_DIR, "discrimination_disorder_vs_control.csv"), index_col=0)
    cu = _csv(os.path.join(STATS_DIR, "feature_clinical_utility.csv"), index_col=0)
    em = _csv(os.path.join(STATS_DIR, "effect_size_matrix_clean.csv"), index_col=0)
    
    f2d, _ = _load_loadings_or_dim() if hasattr(sys.modules[__name__], '_load_loadings_or_dim') else _load_dim_map()

    if dvc is not None:
        dvc.index = [str(i).lower().strip() for i in dvc.index]
    if vc is not None and "disorder" in vc.columns:
        vc["disorder_clean"] = vc["disorder"].astype(str).str.lower().str.strip()

    fig = plt.figure(figsize=(DOUBLE_COL * 2, DOUBLE_COL * 0.8))
    
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 0.85, 1.3],
                           wspace=0.38, left=0.06, right=0.97, top=0.88, bottom=0.15)

    TEST_AUROC_COLOR = '#78909C'  # muted steel blue-grey

    # Panel A: vs-control AUROC
    ax_a = fig.add_subplot(gs[0])

    if vc is not None and len(vc) > 0:
        vs = vc.sort_values('auroc_vs_control', ascending=True)
        disps = [get_disorder_display(d) for d in vs['disorder']]
        dev_aucs = vs['auroc_vs_control'].values
        colors = [get_disorder_color(d) for d in vs['disorder']]

        yerr = []
        for d in vs['disorder']:
            d_clean = str(d).lower().strip()
            if dvc is not None and d_clean in dvc.index:
                col = 'logistic_l2_auroc_std'
                v = dvc.loc[d_clean, col] if col in dvc.columns else 0
                yerr.append(float(v) if pd.notna(v) else 0)
            else:
                yerr.append(0)

        test_aucs = []
        has_test = False
        
        for d in vs['disorder']:
            d_clean = str(d).lower().strip()
            ta = np.nan
            
            if dvc is not None and d_clean in dvc.index:
                for col in dvc.columns:
                    if 'test' in col.lower() and 'auroc' in col.lower():
                        ta = dvc.loc[d_clean, col]
                        break
            if pd.isna(ta) and "disorder_clean" in vc.columns:
                d_row = vc[vc['disorder_clean'] == d_clean]
                for col in d_row.columns:
                    if 'test' in col.lower() and 'auroc' in col.lower():
                        ta = d_row[col].values[0]
                        break
            try:
                ta = float(ta)
                if pd.notna(ta): has_test = True
            except:
                ta = np.nan
            test_aucs.append(ta)

        y_pos = np.arange(len(disps))
        
        if has_test:
            bar_h = 0.36
            ax_a.barh(y_pos + bar_h/2, dev_aucs, height=bar_h, color=colors, 
                      edgecolor='white', lw=0.4, zorder=3)
            ax_a.barh(y_pos - bar_h/2, test_aucs, height=bar_h, color=TEST_AUROC_COLOR, 
                      edgecolor='white', lw=0.4, zorder=3)
            
            for i, (d_v, t_v) in enumerate(zip(dev_aucs, test_aucs)):
                ax_a.text(d_v + 0.004, i + bar_h/2, f'{d_v:.3f}', va='center', fontsize=FS_TICK, color='#222')
                if pd.notna(t_v):
                    ax_a.text(t_v + 0.004, i - bar_h/2, f'{t_v:.3f}', va='center', fontsize=FS_TICK, color=TEST_AUROC_COLOR)
        else:
            ax_a.barh(y_pos, dev_aucs, height=0.6, color=colors, edgecolor='white', lw=0.5, zorder=3)
            for i, v in enumerate(dev_aucs):
                ax_a.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=FS_TICK)

        ax_a.set_yticks(y_pos)
        ax_a.set_yticklabels(disps, fontsize=FS_TICK, fontweight='bold')
        for lb in ax_a.get_yticklabels():
            rn = [k for k, v_ in DISORDER_DISPLAY_NAMES.items() if v_ == lb.get_text()]
            if rn:
                lb.set_color(get_disorder_color(rn[0]))

        ax_a.set_xlim(0.70, 1.0)
        ax_a.set_xticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
        ax_a.axvline(x=0.5, color='#E53935', ls='--', lw=0.8, alpha=0.5, zorder=1)
        ax_a.grid(True, axis='x', color='#EEEEEE', linestyle='-', linewidth=0.5, zorder=0)

        if has_test:
            h1 = Rectangle((0,0),1,1, fc='#BDBDBD', ec='white', label='Dev AUROC')
            h2 = Rectangle((0,0),1,1, fc=TEST_AUROC_COLOR, ec='white', label='Test AUROC')
            ax_a.legend(handles=[h1, h2], fontsize=FS_LEGEND, loc='lower left', 
                        bbox_to_anchor=(0, 1.02), ncol=2, frameon=False, handletextpad=0.4)

    ax_a.set_xlabel("AUROC", fontsize=FS_AXLABEL)
    ax_a.set_title("A. Disorder vs. Controls", fontsize=FS_TITLE, fontweight='bold', loc='center', pad=20)
    _despine(ax_a)

    # Panel B: Clinical utility bar chart
    ax_b = fig.add_subplot(gs[1])

    if cu is not None:
        cu_s = cu.squeeze()
        top = cu_s.nlargest(min(12, len(cu_s))).iloc[::-1]
        y_pos = np.arange(len(top))

        for i, (feat, val) in enumerate(top.items()):
            dim = f2d[0].get(feat, 'other') if isinstance(f2d, tuple) else f2d.get(feat, 'other')
            c = DIM_COLORS.get(dim, '#607D8B')
            
            ax_b.barh(i, val, color=c, height=0.6, edgecolor='white', lw=0.5, alpha=0.85, zorder=3)

            best_dis_text = ""
            best_color = '#555555'
            if em is not None and feat in em.index:
                best = em.loc[feat].abs().idxmax()
                bg = em.loc[feat, best]
                best_dis_text = f"{get_disorder_display(best)} (g={bg:+.2f})"
                best_color = get_disorder_color(best)

            ax_b.text(val + 0.02, i, best_dis_text, va='center', ha='left', 
                      fontsize=FS_ANNOT, color=best_color, fontweight='bold')

        ax_b.set_yticks(y_pos)
        clean_ylabels = [_clean_name(f)[:20] for f in top.index]
        ax_b.set_yticklabels(clean_ylabels, fontsize=FS_TICK, fontweight='bold')
        
        for i, f in enumerate(top.index):
            dim = f2d[0].get(f, 'other') if isinstance(f2d, tuple) else f2d.get(f, 'other')
            ax_b.get_yticklabels()[i].set_color(DIM_COLORS.get(dim, '#333333'))
            
        ax_b.set_xlim(0, max(top) * 1.65)
        ax_b.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
        ax_b.axvline(x=0.5, color='#BDBDBD', ls=':', lw=0.8, zorder=0)

    ax_b.set_xlabel("Clinical Utility Score", fontsize=FS_AXLABEL)
    ax_b.set_title("B. Top Clinical Utility Features", fontsize=FS_TITLE, fontweight='bold', loc='center', pad=20)
    _despine(ax_b)

    # Panel C: Discrimination matrix
    ax_c = fig.add_subplot(gs[2])

    if pw is not None:
        pw.index = [get_disorder_display(d) for d in pw.index]
        pw.columns = [get_disorder_display(d) for d in pw.columns]
        data = pw.values.astype(float)
        np.fill_diagonal(data, np.nan)

        cmap = plt.get_cmap('RdBu_r')
        norm = TwoSlopeNorm(vmin=0.5, vcenter=0.75, vmax=1.0)
        
        im = ax_c.imshow(data, cmap=cmap, norm=norm, aspect='auto')

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if np.isnan(v):
                    ax_c.text(j, i, '\u2014', ha='center', va='center', fontsize=FS_TICK, color='#AAA')
                    continue
                tc = 'white' if (v > 0.85 or v < 0.60) else 'black'
                v_str = f'{v:.2f}'.lstrip('0') if v < 1.0 else '1.0'
                ax_c.text(j, i, v_str, ha='center', va='center', fontsize=FS_ANNOT, color=tc)

        ax_c.set_xticks(range(len(pw.columns)))
        ax_c.set_xticklabels(pw.columns, rotation=45, ha='right', rotation_mode='anchor', fontsize=FS_TICK, fontweight='bold')
        ax_c.set_yticks(range(len(pw.index)))
        ax_c.set_yticklabels(pw.index, fontsize=FS_TICK, fontweight='bold')
        
        for lb in list(ax_c.get_xticklabels()) + list(ax_c.get_yticklabels()):
            rn = [k for k, v_ in DISORDER_DISPLAY_NAMES.items() if v_ == lb.get_text()]
            if rn:
                lb.set_color(get_disorder_color(rn[0]))

        divider = make_axes_locatable(ax_c)
        cax = divider.append_axes("right", size="3%", pad=0.15)
        cb = fig.colorbar(im, cax=cax)
        cb.set_ticks([0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0])
        cb.set_label("AUROC", fontsize=FS_AXLABEL)
        cb.ax.tick_params(labelsize=FS_TICK)

    ax_c.set_title("C. Pairwise Discrimination", fontsize=FS_TITLE, fontweight='bold', loc='center', pad=20)
    for spine in ax_c.spines.values():
        spine.set_visible(True)
        spine.set_color('#222222')
        spine.set_linewidth(1.0)
    ax_c.tick_params(labelsize=FS_TICK)

    fig.suptitle("Figure 3 | Disorder Detectability and Clinical Utility", 
                 fontsize=FS_SUPTITLE, fontweight='bold', y=0.98)
                 
    _save(fig, "fig3_detectability", "main")


# =====================================================================
# FIGURE 4 — SPECIFICITY PARADOX
# =====================================================================

def plot_fig4_specificity():
    """Figure 4: The Specificity Paradox (one hero + two support panels)."""
    logger.info("MAIN Fig 4: Specificity Paradox...")

    td = _csv(os.path.join(STATS_DIR, "transdiagnostic_features.csv"))
    ds = _csv(os.path.join(STATS_DIR, "feature_differential_specificity.csv"), index_col=0)
    cu = _csv(os.path.join(STATS_DIR, "feature_clinical_utility.csv"), index_col=0)
    sp = _csv(os.path.join(STATS_DIR, "feature_specificity.csv"), index_col=0)
    profiles = _json(os.path.join(STATS_DIR, "disorder_profiles.json"))
    
    f2d, dim_order = {}, []
    if hasattr(sys.modules[__name__], '_load_dim_map'):
        f2d, dim_order = _load_dim_map()
    elif hasattr(sys.modules[__name__], '_load_loadings_or_dim'):
        res = _load_loadings_or_dim()
        f2d = res[0] if isinstance(res, tuple) else res
        dim_order = res[1] if isinstance(res, tuple) and len(res) > 1 else []

    # Three-panel layout: one full-width hero above two support panels.
    # Define axes and whitespace in physical inches, then derive the canvas;
    # this fixes the drawing areas without imposing a shared canvas size.
    panel_w, panel_h = 2.35, 2.05
    hero_h = 2.35
    col_gap, row_gap = 0.75, 1.10
    margin_left, margin_right = 0.75, 0.55
    margin_top, margin_bottom = 0.70, 0.75
    cbar_pad, cbar_w = 0.08, 0.10
    hero_w = panel_w * 2 + col_gap
    fig_w = margin_left + hero_w + cbar_pad + cbar_w + margin_right
    fig_h = margin_bottom + panel_h + row_gap + hero_h + margin_top
    fig = plt.figure(figsize=(fig_w, fig_h))
    x_left = margin_left
    x_right = margin_left + panel_w + col_gap
    y_bottom = margin_bottom
    y_hero = margin_bottom + panel_h + row_gap

    def _axes_at(x, y, width, height):
        return fig.add_axes([
            x / fig_w, y / fig_h, width / fig_w, height / fig_h
        ])

    # Panel B: Feature classification
    ax_a = _axes_at(x_left, y_bottom, panel_w, panel_h)

    if td is not None:
        cat_order = ["transdiagnostic", "mixed", "specific", "non_significant"]
        cat_c = {
            "transdiagnostic": DIRECTION_COLORS["transdiagnostic"],
            "mixed": DIRECTION_COLORS["mixed"],
            "specific": DIRECTION_COLORS["specific"],
            "non_significant": DIRECTION_COLORS["non_sig"],
        }
        cat_l = {
            "transdiagnostic": "Transdiag.\n(≥5 dis.)",
            "mixed": "Mixed\n(3–4)",
            "specific": "Specific\n(1–2)",
            "non_significant": "Non-sig."
        }
        counts = td["category"].value_counts()
        bh = [counts.get(c, 0) for c in cat_order]
        bars = ax_a.bar(
            range(4), bh,
            color=[cat_c[c] for c in cat_order],
            edgecolor='black', lw=0.3, width=0.55, zorder=3
        )
        ax_a.set_xticks(range(4))
        ax_a.set_xticklabels([cat_l[c] for c in cat_order], fontsize=FS_TICK, multialignment='center', linespacing=1.4)
        for b, h in zip(bars, bh):
            ax_a.text(b.get_x() + b.get_width() / 2, h + 0.5, str(int(h)), ha='center', fontsize=FS_AXLABEL, fontweight='bold')
        ax_a.set_ylim(0, max(bh) * 1.28)
        ax_a.grid(axis='y', color='#EEEEEE', lw=0.5, zorder=0)

    ax_a.set_ylabel("Number of Features", fontsize=FS_AXLABEL)
    ax_a.set_title("B. Feature Classification", fontsize=FS_TITLE, fontweight='bold', loc='left', pad=8)
    _despine(ax_a)

    # Panel C: DiffSpec vs Magnitude Specificity scatter
    ax_b = _axes_at(x_right, y_bottom, panel_w, panel_h)

    if ds is not None and cu is not None:
        mg = pd.DataFrame({'diffspec': ds.squeeze(), 'cu': cu.squeeze()})
        mg['magspec'] = sp.squeeze() if sp is not None else np.nan
        v = mg.dropna()

        if len(v) > 0:
            sc = ax_b.scatter(
                v['magspec'], v['diffspec'],
                c=v['cu'], cmap='YlOrRd', s=32, edgecolors='black', lw=0.2,
                vmin=0, vmax=0.8, zorder=5, alpha=0.85
            )
            cax = _axes_at(
                x_right + panel_w + cbar_pad,
                y_bottom + panel_h * 0.14,
                cbar_w,
                panel_h * 0.72,
            )
            cb = fig.colorbar(sc, cax=cax)
            cb.set_label("Clinical Utility", fontsize=FS_ANNOT)
            cb.ax.tick_params(labelsize=FS_TICK)

        ax_b.axhline(y=0.2, color='#AAAAAA', ls=':', lw=0.7, zorder=1)
        ax_b.text(
            0.03, 0.97,
            "All DiffSpec < 0.2\n→ Quantitative (severity)\n   not qualitative (direction)",
            transform=ax_b.transAxes, fontsize=FS_ANNOT, va='top', ha='left', style='italic', color='#B71C1C',
            bbox=dict(boxstyle='round,pad=0.35', fc='#FFF9C4', ec='#F57F17', alpha=0.93), zorder=15
        )

    ax_b.set_xlabel("Magnitude Specificity", fontsize=FS_AXLABEL)
    ax_b.set_ylabel("Differential Specificity (noise-gated)", fontsize=FS_AXLABEL)
    ax_b.set_title("C. The Specificity Paradox", fontsize=FS_TITLE, fontweight='bold', loc='left', pad=8)
    _despine(ax_b)

    # Panel A: Disorder profiles (full-width hero)
    ax_c = _axes_at(x_left, y_hero, hero_w, hero_h)

    if profiles:
        all_dims = set()
        for d, p in profiles.items():
            all_dims.update(p.get("dimension_summary", {}).keys())
        dims = [d for d in dim_order if d in all_dims]
        if not dims:
            dims = sorted(all_dims)
        n_dims = len(dims)

        _groups = globals().get('DISORDER_GROUPS', list(profiles.keys()))
        
        for disorder in _groups:
            if disorder not in profiles:
                continue
            ds_ = profiles[disorder].get("dimension_summary", {})
            vals = [ds_.get(d, {}).get("mean_abs_g", 0) if d in ds_ else 0 for d in dims]
            
            c  = get_disorder_color(disorder)
            dn = get_disorder_display(disorder)
            
            ax_c.plot(
                range(n_dims), vals, color=c, lw=1.2, alpha=0.80,
                marker='o', ms=3.5, markeredgewidth=0, zorder=4,
                label=dn
            )

        _short_map = globals().get('DIM_SHORT', {})
        _color_map = globals().get('DIM_COLORS', {})
        dl = [_short_map.get(d, str(d)[:8]) for d in dims]
        ax_c.set_xticks(range(n_dims))
        ax_c.set_xticklabels(dl, rotation=30, ha='right', rotation_mode='anchor', fontsize=FS_TICK)
        
        for i, d in enumerate(dims):
            lbl = ax_c.get_xticklabels()[i]
            lbl.set_color(_color_map.get(d, '#333'))
            ax_c.axvspan(i - 0.5, i + 0.5, alpha=0.04, color=_color_map.get(d, '#999'), zorder=0)

        for thresh, label, col in [(0.2, 'Small\n(0.2)', '#BBBBBB'), (0.5, 'Medium\n(0.5)', '#999999')]:
            ax_c.axhline(y=thresh, color=col, ls=':', lw=0.8, zorder=1)
            ax_c.text(
                1.01, thresh, label,
                transform=ax_c.get_yaxis_transform(),
                fontsize=FS_ANNOT, color=col, va='center', ha='left', clip_on=False
            )

        if n_dims > 0:
            ax_c.set_xlim(-0.5, n_dims - 0.5)
        ax_c.set_ylim(bottom=-0.02)

        ax_c.legend(fontsize=FS_LEGEND, loc='upper center', bbox_to_anchor=(0.5, -0.18), 
                    ncol=5, frameon=False, handletextpad=0.4, columnspacing=1.5)

    ax_c.set_ylabel("Mean |Hedges' g|", fontsize=FS_AXLABEL)
    ax_c.set_title("A. Disorder Profiles by Dimension", fontsize=FS_TITLE, fontweight='bold', loc='left', pad=8)
    _despine(ax_c)

    fig.suptitle(
        "Figure 4 | The Specificity Paradox: High Clinical Utility, Low Differential Specificity",
        fontsize=FS_SUPTITLE, fontweight='bold', y=0.975
    )

    _save_main_fixed(fig, "fig4_specificity")

# =====================================================================
# ED FIGURES (Complete)
# =====================================================================

def plot_ed_fig1_factor():
    """ED Fig 1: Factor analysis details."""
    logger.info("ED Fig 1: Factor Details...")
    sc = _csv(os.path.join(STATS_DIR, "factor_scores_disorders.csv"),
              index_col=0)
    ld = _csv(os.path.join(STATS_DIR, "factor_loadings_features.csv"),
              index_col=0)
    fi = _json(os.path.join(STATS_DIR, "factor_selection.json"))
    if sc is None or fi is None:
        logger.error("  Skip ED Fig 1"); return

    nf = fi.get("n_factors", 2)
    ve = fi.get("variance_explained", [])
    f2d, _ = _load_dim_map()
    sc.index = [get_disorder_display(d) for d in sc.index]
    fn = {0: "p-factor", 1: "Int. vs Neurodev.", 2: "Thought Dis."}
    for i in range(nf):
        fn.setdefault(i, f"F{i+1}")

    eig = fi.get("selection_criteria", {}).get("eigenvalues", [])
    r95 = fi.get("selection_criteria", {}).get(
        "random_95th_eigenvalues", [])
    if not eig:
        eig, r95 = _compute_eigenvalues()

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.95))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                           height_ratios=[1, 0.9],
                           left=0.08, right=0.97, top=0.93, bottom=0.10)

    # Panel A: Loadings heatmap
    ax_a = fig.add_subplot(gs[0, 0])
    if ld is not None and nf >= 2:
        ml = ld.abs().max(axis=1)
        top_feats = ml.nlargest(min(15, len(ld))).index.tolist()
        ls = ld.loc[top_feats]
        nm = TwoSlopeNorm(vmin=-8, vcenter=0, vmax=8)
        im = ax_a.imshow(ls.values, cmap=DIVERGING_CMAP, norm=nm,
                         aspect='auto')
        for i in range(ls.shape[0]):
            for j in range(ls.shape[1]):
                v = ls.iloc[i, j]
                tc = 'white' if abs(v) > 3.6 else 'black'
                ax_a.text(j, i, f'{v:.2f}', ha='center', va='center',
                          fontsize=FS_ANNOT, color=tc)
        ax_a.set_yticks(range(len(top_feats)))
        ax_a.set_yticklabels(
            [_clean_name(f)[:22] for f in top_feats], fontsize=FS_TICK, fontweight='bold')
        for i, f in enumerate(top_feats):
            ax_a.get_yticklabels()[i].set_color(
                DIM_COLORS.get(f2d.get(f, 'other'), '#333'))
        xl = [f"{fn.get(i, 'F'+str(i+1))}\n({ve[i]*100:.1f}%)"
              if i < len(ve) else f"F{i+1}" for i in range(nf)]
        ax_a.set_xticks(range(nf))
        ax_a.set_xticklabels(xl, fontsize=FS_TICK)
        cb = fig.colorbar(im, ax=ax_a, shrink=0.8, aspect=20, pad=0.02)
        cb.set_label("Loading", fontsize=FS_AXLABEL)
        cb.ax.tick_params(labelsize=FS_TICK)
    ax_a.set_title("A. Top Feature Loadings", fontsize=FS_TITLE,
                   fontweight='bold', loc='center', pad=20)
    for spine in ax_a.spines.values():
        spine.set_visible(True)
        spine.set_color('#222222')
        spine.set_linewidth(1.0)

    # Panel B: Biplot
    ax_b = fig.add_subplot(gs[0, 1])
    if nf >= 2:
        x_vals, y_vals = [], []
        
        # 1. 🎯 核心高精度微调：为挤在一起的疾病定制散开方向 (offset_x, offset_y, ha, va)
        label_adjustments = {
            "Eating Disorders":   (50, 20, 'right', 'bottom'),   # 往左上推
            "Anxiety Disorders":  (18, 10, 'right', 'bottom'),  # 往左侧推
            "Depression":         (-4, 6, 'right', 'center'),    # 往右侧推
            "Bipolar Disorder":   (2, -24, 'left', 'top'),     # 往左下推
            "ADHD":               (8, -6, 'left', 'top'),       # 往右下推
            "Schizophrenia":      (8, 0, 'left', 'center'),     # 往右侧推
            "OCD":                (5, -6, 'right', 'top'),     # 往左下躲开 Borderline
            "Borderline PD":      (4, 8, 'left', 'bottom'),     # 往右上推
            "PTSD":               (10, -4, 'left', 'center'),   # 往右侧推
            "Autism Spectrum":    (8, 6, 'left', 'bottom')      # 往右上推
        }

        for dd in sc.index:
            x, y = sc.loc[dd, sc.columns[0]], sc.loc[dd, sc.columns[1]]
            x_vals.append(x)
            y_vals.append(y)
            
            rn = [k for k, v in DISORDER_DISPLAY_NAMES.items() if v == dd]
            c = get_disorder_color(rn[0]) if rn else "#808080"
            
            # 【美化点 1】：散点增大，改用纯白描边，提升立体感与干净度
            ax_b.scatter(x, y, c=c, s=100, edgecolors='#FFFFFF', lw=1.5, zorder=10)
            
            # 获取定制的排版参数，如果没有在字典里，则使用默认向右上偏移
            off_x, off_y, halign, valign = label_adjustments.get(dd, (6, 6, 'left', 'bottom'))
            
            # 【美化点 2】：引入动态对齐与白底描边防遮挡
            txt = ax_b.annotate(
                dd, (x, y), xytext=(off_x, off_y),
                textcoords='offset points', ha=halign, va=valign,
                fontsize=FS_TICK, color=c, fontweight='bold', zorder=15,
                arrowprops=dict(arrowstyle='-', color=c, lw=0.6, alpha=0.7, shrinkA=0, shrinkB=4)
            )
            txt.set_path_effects([withStroke(linewidth=2.5, foreground='white')])

        for sp, ds_ in HITOP_SPECTRA.items():
            mem = [get_disorder_display(d) for d in ds_ if get_disorder_display(d) in sc.index]
            if len(mem) >= 3:
                try:
                    co = sc.loc[mem, [sc.columns[0], sc.columns[1]]].values
                    h = ConvexHull(co)
                    hp = np.vstack([co[h.vertices], co[h.vertices[0]]])
                    
                    # 【美化点 3】：增加淡色背景填充，强化群组归属感
                    ax_b.fill(hp[:, 0], hp[:, 1], color=HITOP_COLORS[sp], alpha=0.06, zorder=1)
                    # 虚线调得更细腻柔和
                    ax_b.plot(hp[:, 0], hp[:, 1], color=HITOP_COLORS[sp], lw=1.2, ls='--', alpha=0.6, zorder=2)
                except Exception:
                    pass
        
        # 【美化点 4】：中心十字线改为更浅的虚线，不抢数据风头
        ax_b.axhline(0, color='#E5E5E5', lw=0.8, ls='--', zorder=0)
        ax_b.axvline(0, color='#E5E5E5', lw=0.8, ls='--', zorder=0)
        
        x_margin = (max(x_vals) - min(x_vals)) * 0.35 or 0.2
        y_margin = (max(y_vals) - min(y_vals)) * 0.35 or 0.2
        ax_b.set_xlim(min(x_vals) - x_margin, max(x_vals) + x_margin)
        ax_b.set_ylim(min(y_vals) - y_margin, max(y_vals) + y_margin)
        
        v1 = ve[0]*100 if ve is not None else 0
        v2 = ve[1]*100 if ve is not None and len(ve) > 1 else 0
        
        ax_b.set_xlabel(f"{fn.get(0,'F1')} ({v1:.1f}%)", fontsize=FS_AXLABEL)
        ax_b.set_ylabel(f"{fn.get(1,'F2')} ({v2:.1f}%)", fontsize=FS_AXLABEL)
        
    ax_b.set_title("B. Disorder Factor Scores", fontsize=FS_TITLE, fontweight='bold', loc='center', pad=20)
    ax_b.tick_params(labelsize=FS_TICK)
    _despine(ax_b)

    # Panel C: Scree
    ax_c = fig.add_subplot(gs[1, 0])
    if eig:
        ns = min(len(eig), 10)
        x = range(1, ns + 1)
        bars = ax_c.bar(x, eig[:ns], color='#4393C3', alpha=0.85,
                        edgecolor='white', lw=0.3, width=0.6)
        for i in range(min(nf, ns)):
            bars[i].set_color('#D6604D'); bars[i].set_alpha(0.9)
        if r95:
            ax_c.plot(x, r95[:ns], 'D--', color='#F4A736', ms=4,
                      label='95th %ile (random)', lw=1.0)
        ax_c.axhline(y=1.0, color='#999', ls=':', lw=0.8,
                     label='Kaiser criterion')
        ax_c.set_xticks(list(x))
        if ve:
            for i, v in enumerate(eig[:ns]):
                if i < len(ve):
                    ax_c.text(i + 1, v + 0.05, f'{ve[i]*100:.1f}%',
                              ha='center', fontsize=FS_ANNOT)
    ax_c.set_xlabel("Factor Number", fontsize=FS_AXLABEL)
    ax_c.set_ylabel("Eigenvalue", fontsize=FS_AXLABEL)
    ax_c.set_title("C. Scree Plot", fontsize=FS_TITLE,
                   fontweight='bold', loc='center', pad=20)
    ax_c.legend(fontsize=FS_LEGEND, frameon=False)
    ax_c.tick_params(labelsize=FS_TICK)
    _despine(ax_c)

    # Panel D: Variance explained
    ax_d = fig.add_subplot(gs[1, 1])
    if ve:
        cum = np.cumsum(ve) * 100
        ind = range(1, len(ve) + 1)
        ax_d.bar(ind, [v*100 for v in ve], color='#4393C3', alpha=0.85,
                 edgecolor='white', lw=0.3, label='Individual')
        ax_d.plot(ind, cum, 'o-', color='#D6604D', ms=5, lw=1.2,
                  label='Cumulative')
        for i, v in enumerate(ve):
            ax_d.text(i+1, v*100+1, f'{v*100:.1f}%',
                      ha='center', fontsize=FS_ANNOT)
        ax_d.axhline(y=80, color='#999', ls=':', lw=0.8)
    ax_d.set_xlabel("Factor Number", fontsize=FS_AXLABEL)
    ax_d.set_ylabel("Variance Explained (%)", fontsize=FS_AXLABEL)
    ax_d.set_title("D. Variance Explained", fontsize=FS_TITLE,
                   fontweight='bold', loc='center', pad=20)
    ax_d.legend(fontsize=FS_LEGEND, frameon=False)
    ax_d.tick_params(labelsize=FS_TICK)
    _despine(ax_d)

    fig.suptitle("ED Figure 1 | Factor Analysis Details",
                 fontsize=FS_SUPTITLE, fontweight='bold', y=1.02)
    _save(fig, "ed_fig1_factor", "ed")


def plot_ed_fig2_clustering():
    """ED Fig 2: Dendrogram + MDS.
    
    优化版更新：
      1. [加宽画布]: 横向宽度增加 35%，为标签留出充足伸展空间。
      2. [图A纤细化]: 强制使用极细线条绘制树状图，杜绝粗糙感。
      3. [图B顶刊美学]: 引入专属偏移字典、连线防穿刺、气泡白边叠印以及动态放宽的坐标轴。
    """
    logger.info("ED Fig 2: Clustering (Optimized Layout & Connections)...")
    lp = os.path.join(STATS_DIR, "linkage_matrix.npy")
    mp = os.path.join(STATS_DIR, "disorder_mds_embedding.csv")
    if not os.path.exists(lp) or not os.path.exists(mp):
        logger.error("  Skip ED Fig 2"); return

    lm = np.load(lp)
    md = _csv(mp, index_col=0)
    ci = _json(os.path.join(STATS_DIR, "clustering_summary.json")) or {}

    # 【优化点 1】：将宽度系数从原本的 1.0 拉宽至 1.35，高度保持适中，实现宽屏舒展布局
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL * 1.35, DOUBLE_COL * 0.6))
    fig.subplots_adjust(left=0.05, right=0.96, top=0.88, bottom=0.15, wspace=0.50)

    # ==========================================
    # Panel A: Dendrogram (树状图)
    # ==========================================
    ax = axes[0]
    dl = [get_disorder_display(d) for d in md.index]
    
    # 【优化点 2】：使用 rc_context 强制将树状图的线条宽度调细至 1.0 (默认是 1.5)，使其更加高级优雅
    with plt.rc_context({'lines.linewidth': 1.0}):
        dendrogram(lm, labels=dl, orientation='left', ax=ax,
                   leaf_font_size=FS_TICK, color_threshold=0,
                   above_threshold_color='#555555')
        
    for lb in ax.get_ymajorticklabels():
        rn = [k for k, v in DISORDER_DISPLAY_NAMES.items()
              if v == lb.get_text()]
        if rn:
            lb.set_color(get_disorder_color(rn[0]))
            lb.set_fontweight('bold')
            
    ax.set_title("A. Hierarchical Clustering", fontsize=FS_TITLE,
                 fontweight='bold', loc='center', pad=20)
    ax.set_xlabel("Distance", fontsize=FS_AXLABEL)
    ax.tick_params(labelsize=FS_TICK)
    _despine(ax)

    # ==========================================
    # Panel B: MDS Embedding
    # ==========================================
    ax2 = axes[1]
    x_vals, y_vals = [], []
    
    # 【优化点 3.1】：高精度定制坐标偏移，强行炸开拥挤在中间的病症簇
    label_adjustments = {
        "Autism Spectrum":   (-8, 16, 'center', 'bottom'),    # 正上方
        "ADHD":              (15, 0, 'left', 'center'),      # 往右推
        "Schizophrenia":     (-10, 0, 'right', 'center'),    # 往左推
        "Depression":        (-12, 14, 'right', 'bottom'),   # 往左上拉开
        "Bipolar Disorder":  (10, -22, 'right', 'top'),     # 往左下避让
        "Borderline PD":     (-18, -2, 'right', 'center'),   # 往极左拉开
        "PTSD":              (6, 16, 'left', 'bottom'),      # 往右上拉开
        "OCD":               (16, 6, 'left', 'bottom'),      # 往右上偏拉开
        "Anxiety Disorders": (16, -6, 'left', 'top'),        # 往右下拉开
        "Eating Disorders":  (6, -18, 'left', 'top')         # 往右下拉开
    }

    for d in md.index:
        x, y = md.loc[d, "MDS_1"], md.loc[d, "MDS_2"]
        x_vals.append(x)
        y_vals.append(y)
        
        c = get_disorder_color(d)
        dn = get_disorder_display(d)
        
        # 散点改为白边气泡质感
        ax2.scatter(x, y, c=c, s=100, edgecolors='#FFFFFF', lw=1.5, zorder=10)
        
        # 获取专属偏移
        off_x, off_y, halign, valign = label_adjustments.get(dn, (15, 15, 'left', 'bottom'))
        
        # 【优化点 3.2】：带 arrowprops 的牵引线标注，并施加白色发光叠印防遮挡
        txt = ax2.annotate(
            dn, (x, y), xytext=(off_x, off_y),
            textcoords='offset points', ha=halign, va=valign,
            fontsize=FS_TICK, color=c, fontweight='bold', zorder=15,
            arrowprops=dict(arrowstyle='-', color=c, lw=1.2, alpha=0.7, shrinkA=0, shrinkB=4)
        )
        txt.set_path_effects([withStroke(linewidth=2.5, foreground='white')])

    # 绘制 HiTOP 光谱背景区
    for sp, ds_ in HITOP_SPECTRA.items():
        mem = [d for d in ds_ if d in md.index]
        if len(mem) >= 3:
            try:
                co = md.loc[mem, ["MDS_1", "MDS_2"]].values
                h = ConvexHull(co)
                hp = np.vstack([co[h.vertices], co[h.vertices[0]]])
                # 淡色虚线与内部半透明填充
                ax2.plot(hp[:, 0], hp[:, 1], color=HITOP_COLORS[sp], lw=1.2, ls='--', alpha=0.6, zorder=2)
                ax2.fill(hp[:, 0], hp[:, 1], color=HITOP_COLORS[sp], alpha=0.08, zorder=1)
            except Exception:
                pass

    # 【优化点 3.3】：坐标轴安全距离加倍 (从 0.35 提升到 0.45)，防止文字和引线冲出画框
    x_margin = (max(x_vals) - min(x_vals)) * 0.45 or 0.2
    y_margin = (max(y_vals) - min(y_vals)) * 0.45 or 0.2
    ax2.set_xlim(min(x_vals) - x_margin, max(x_vals) + x_margin)
    ax2.set_ylim(min(y_vals) - y_margin, max(y_vals) + y_margin)

    ax2.set_xlabel("MDS 1", fontsize=FS_AXLABEL)
    ax2.set_ylabel("MDS 2", fontsize=FS_AXLABEL)
    ax2.set_title("B. MDS Embedding", fontsize=FS_TITLE,
                  fontweight='bold', loc='center', pad=20)
    ax2.tick_params(labelsize=FS_TICK)
    
    # 增加浅色十字基准线
    ax2.axhline(0, color='#EEEEEE', lw=0.8, ls='--', zorder=0)
    ax2.axvline(0, color='#EEEEEE', lw=0.8, ls='--', zorder=0)
    
    _despine(ax2)

    # ARI & NMI 指标悬浮框
    ari = ci.get("ari_vs_hitop", np.nan)
    nmi = ci.get("nmi_vs_hitop", np.nan)
    if not (isinstance(ari, float) and np.isnan(ari)):
        ax2.text(0.02, 0.98, f"ARI = {ari:.3f}, NMI = {nmi:.3f}",
                 transform=ax2.transAxes, fontsize=FS_ANNOT, va='top', zorder=20,
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#DDDDDD', alpha=0.9))

    # 图例配置
    sh = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                 ms=6, label=s.replace('_', ' ').title(),
                 markeredgecolor='none')
          for s, c in HITOP_COLORS.items()]
          
    # 🌟 核心美化：开启边框并配置优雅的透明度与留白
    leg = ax2.legend(handles=sh, fontsize=FS_LEGEND, loc='upper right',
                     title="HiTOP", title_fontsize=FS_LEGEND, 
                     frameon=True,          # 1. 开启边框
                     framealpha=0.90,       # 2. 90%不透明度：能透出一点点底部的虚线，有毛玻璃悬浮感
                     edgecolor='#DDDDDD',   # 3. 极浅的银灰边框：避免黑色边框带来的笨重感
                     facecolor='#FFFFFF',   # 4. 纯白背景
                     borderpad=0.8,         # 5. 边框内边距：让文字和框之间有充足的呼吸感
                     labelspacing=0.6)      # 6. 图例行间距：稍微拉开一点，更显整洁

    # 🌟 进阶边框质感微调 (通过操控底层的 frame 对象)
    frame = leg.get_frame()
    frame.set_linewidth(0.8)  # 边框线宽调细
    # 设置微圆角框线 (pad控制内边距补充，rounding_size控制圆角曲率)
    frame.set_boxstyle('round,pad=0.4,rounding_size=0.3') 
    
    # 将图例的标题加粗，使其与里面的细体图例项产生层级对比
    leg.get_title().set_fontweight('bold')

    fig.suptitle("ED Figure 2 | Hierarchical Clustering & MDS Embedding",
                 fontsize=FS_SUPTITLE, fontweight='bold', y=1.02)
    _save(fig, "ed_fig2_clustering", "ed")


def plot_ed_fig3_tsne():
    """ED Fig 3: t-SNE small multiples."""
    logger.info("ED Fig 3: t-SNE...")
    td = _csv(os.path.join(STATS_DIR, "tsne_embedding.csv"))
    if td is None:
        logger.error("  Skip ED Fig 3"); return
    mc = _json(os.path.join(STATS_DIR, "multiclass_results.json"))
    pca_auc = mc.get("per_class_auroc_mean", {}) if mc else {}

    xm = [td["tSNE_1"].min(), td["tSNE_1"].max()]
    ym = [td["tSNE_2"].min(), td["tSNE_2"].max()]
    mg = [(xm[1]-xm[0])*0.08, (ym[1]-ym[0])*0.08]

    CONTROL_COLOR = '#EEEEEE'
    PATIENT_LEGEND_COLOR = '#D6604D'

    fig, axes = plt.subplots(2, 5, figsize=(DOUBLE_COL, DOUBLE_COL * 0.55))
    fig.subplots_adjust(left=0.03, right=0.99, top=0.90, bottom=0.08,
                        hspace=0.45, wspace=0.25)

    for idx, d in enumerate(DISORDER_GROUPS):
        ax = axes[idx // 5, idx % 5]
        cm = td["group"] == "control"
        ax.scatter(td.loc[cm, "tSNE_1"], td.loc[cm, "tSNE_2"],
                   c=CONTROL_COLOR, s=1, alpha=0.12, rasterized=True)
        m = (td["group"] == "patient") & (td["disorder_group"] == d)
        if m.sum() > 0:
            ax.scatter(td.loc[m, "tSNE_1"], td.loc[m, "tSNE_2"],
                       c=get_disorder_color(d), s=3, alpha=0.5,
                       rasterized=True)
        dn = get_disorder_display(d)
        av = pca_auc.get(d, np.nan)
        aucs = f"\nAUC={av:.2f}" if not pd.isna(av) else ""
        ax.set_title(f"{dn}{aucs}", fontsize=FS_TICK, fontweight='bold',
                     color=get_disorder_color(d), loc='center', pad=4)
        ax.set_xlim(xm[0] - mg[0], xm[1] + mg[0])
        ax.set_ylim(ym[0] - mg[1], ym[1] + mg[1])
        ax.set_xticks([]); ax.set_yticks([])
        _despine(ax, top=True, right=True, left=True, bottom=True)

    h_ctrl = Line2D([0], [0], marker='o', color='w',
                    markerfacecolor=CONTROL_COLOR, ms=5,
                    label='Control', markeredgecolor='#CCC', mew=0.3)
    h_pat = Line2D([0], [0], marker='o', color='w',
                    markerfacecolor=PATIENT_LEGEND_COLOR, ms=5,
                    label='Patient', markeredgecolor='#666', mew=0.3)
    fig.legend(handles=[h_ctrl, h_pat], fontsize=FS_LEGEND,
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.01),
               frameon=False)
    fig.suptitle("ED Figure 3 | t-SNE Embeddings",
                 fontsize=FS_SUPTITLE, fontweight='bold', y=1.02)
    _save(fig, "ed_fig3_tsne", "ed")


def plot_ed_fig4_confusion():
    """ED Fig 4: Confusion matrix."""
    logger.info("ED Fig 4: Confusion Matrix...")
    from matplotlib.patches import Rectangle # 引入矩形绘制模块
    
    cm = _csv(os.path.join(STATS_DIR,
                           "confusion_matrix_normalized.csv"), index_col=0)
    mc = _json(os.path.join(STATS_DIR, "multiclass_results.json"))
    if cm is None:
        logger.error("  Skip ED Fig 4"); return

    cm.index = [get_disorder_display(d) if d != "control" else "Control"
                for d in cm.index]
    cm.columns = [get_disorder_display(d) if d != "control" else "Control"
                  for d in cm.columns]

    data = cm.values.astype(float)
    finite = data[np.isfinite(data)]
    plot_data, im_kw, text_thr, cmap_log = _prepare_heatmap_layers(
        data, finite, fixed_sequential_vmax=HM_REF_VMAX)
    logger.info(f"  Colormap: {cmap_log}")

    fig, ax = _ed_heatmap_figure()
    im = _ed_heatmap_imshow(ax, plot_data, im_kw)
    _ed_heatmap_annotate(ax, data, text_thr, empty_at=0.0)

    ax.set_xticks(range(len(cm.columns)))
    ax.set_xticklabels(cm.columns, rotation=45, ha='right',
                       fontsize=FS_TICK, fontweight='bold')
    ax.set_yticks(range(len(cm.index)))
    ax.set_yticklabels(cm.index, fontsize=FS_TICK, fontweight='bold')
    ax.set_xlabel("Predicted", fontsize=FS_AXLABEL)
    ax.set_ylabel("True", fontsize=FS_AXLABEL)
    ax.tick_params(labelsize=FS_TICK)

    for lb in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        rn = [k for k, v in DISORDER_DISPLAY_NAMES.items()
              if v == lb.get_text()]
        if rn:
            lb.set_color(get_disorder_color(rn[0]))

    # ---------------------------------------------------------
    # 【修改点 2：绘制红色高亮虚线框】
    # ---------------------------------------------------------
    labels = list(cm.index)
    try:
        col_control = labels.index("Control")
        row_adhd = labels.index("ADHD")
        row_autism = labels.index("Autism Spectrum")
        col_ed = labels.index("Eating Disorders")
        row_bpd = labels.index("Borderline PD")
        
        # ax.imshow 的单元格原点在正中心，所以左下角锚点需要 -0.5
        # 高亮 ADHD -> Control
        ax.add_patch(Rectangle((col_control - 0.5, row_adhd - 0.5), 1, 1, fill=False, edgecolor='#D55E00', lw=1, ls='--', zorder=10))
        # 高亮 Autism Spectrum -> Control
        ax.add_patch(Rectangle((col_control - 0.5, row_autism - 0.5), 1, 1, fill=False, edgecolor='#D55E00', lw=1, ls='--', zorder=10))
        # 高亮 Borderline PD -> Eating Disorders
        ax.add_patch(Rectangle((col_ed - 0.5, row_bpd - 0.5), 1, 1, fill=False, edgecolor='#D55E00', lw=1, ls='--', zorder=10))
    except ValueError:
        # 防呆设计：如果某些名称没匹配上，安全跳过，防止代码报错崩溃
        pass

    # ---------------------------------------------------------
    # 【修改点 3：移动性能指标框到右下角防遮挡】
    # ---------------------------------------------------------
    if mc:
        ma = mc.get("macro_auroc_mean", np.nan)
        ba = mc.get("balanced_accuracy_mean", np.nan)
        # Keep the summary inside the fixed canvas (below the colorbar).
        fig.text(
            0.80, 0.055,
            f"Macro AUROC = {ma:.3f}\nBal. Accuracy = {ba:.3f}",
            fontsize=FS_ANNOT, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFFFFF",
                      ec="#AAAAAA", alpha=0.95),
            zorder=20,
        )

    _ed_heatmap_style_frame(ax)
    _ed_heatmap_colorbar(fig, ax, im, "Proportion")
    _ed_heatmap_title(ax, "ED Fig 4 | Confusion Matrix")
    _save_ed_fixed(fig, "ed_fig4_confusion")


def plot_ed_fig5_ablation():
    """ED Fig 5: Dimension ablation heatmap (layout aligned with ED Fig 4)."""
    logger.info("ED Fig 5: Ablation...")
    ab = _csv(os.path.join(STATS_DIR, "dimension_ablation.csv"), index_col=0)
    if ab is None:
        logger.error("  Skip ED Fig 5"); return

    dc = [c for c in ab.columns if c.startswith("delta_")]
    if not dc:
        logger.error("  Skip ED Fig 5: no delta_* columns (expect leave-one-out from s04)"); return

    dd = ab[dc].copy()
    dd.columns = [c.replace("delta_", "") for c in dc]
    dd.index = [get_disorder_display(d) for d in dd.index]
    dd = dd[dd.mean(axis=0).sort_values(ascending=False).index]

    data = dd.values.astype(float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        logger.error("  Skip ED Fig 5: no finite values"); return

    # Data sanity (delta = full AUROC − without-dimension AUROC from s04)
    n_neg = int(np.sum(finite < -1e-6))
    logger.info(
        f"  Ablation ΔAUROC: n={finite.size}, min={finite.min():+.4f}, "
        f"max={finite.max():+.4f}, n_negative={n_neg}"
    )

    plot_data, im_kw, text_thr, cmap_log = _prepare_heatmap_layers(data, finite)
    logger.info(f"  Colormap: {cmap_log}")

    fig, ax = _ed_heatmap_figure()
    im = _ed_heatmap_imshow(ax, plot_data, im_kw)
    _ed_heatmap_annotate(ax, data, text_thr, empty_at=None)

    ax.set_xticks(range(len(dd.columns)))
    dim_labels = [DIM_SHORT.get(c, c[:8]) for c in dd.columns]
    ax.set_xticklabels(dim_labels, rotation=45, ha="right",
                       rotation_mode="anchor", fontsize=FS_TICK, fontweight="bold")
    for i, c in enumerate(dd.columns):
        ax.get_xticklabels()[i].set_color(DIM_COLORS.get(c, "#333333"))

    ax.set_yticks(range(len(dd.index)))
    ax.set_yticklabels(dd.index, fontsize=FS_TICK, fontweight="bold")
    for lb in ax.get_yticklabels():
        rn = [k for k, v in DISORDER_DISPLAY_NAMES.items() if v == lb.get_text()]
        if rn:
            lb.set_color(get_disorder_color(rn[0]))

    _ed_heatmap_style_frame(ax)
    _ed_heatmap_colorbar(fig, ax, im, "\u0394AUROC (Full \u2212 Without Dim)")
    _ed_heatmap_title(ax, "ED Fig 5 | Dimension Contribution (\u0394AUROC When Removed)")
    _save_ed_fixed(fig, "ed_fig5_ablation")


def plot_ed_fig6_sensitivity():
    """ED Fig 6: Sensitivity summary (4 panels)."""
    logger.info("ED Fig 6: Sensitivity (Optimized Aesthetics)...")
    orig = _csv(os.path.join(STATS_DIR, "effect_sizes_full.csv"))

    # 加宽画布比例，让并排的4个子图拥有更宽敞的呼吸空间
    fig = plt.figure(figsize=(DOUBLE_COL * 1.35, DOUBLE_COL * 0.45))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.45,
                           width_ratios=[1, 1, 0.8, 1],
                           left=0.05, right=0.98, top=0.85, bottom=0.20)

    # Panel A: Confound adjustment scatter
    ax_a = fig.add_subplot(gs[0])
    conf_linear = _csv(os.path.join(STATS_DIR, "sensitivity_confound_linear.csv"))
    if conf_linear is not None and orig is not None:
        orig_clean = orig[~orig["feature"].isin(SELECTION_ARTIFACT_FEATURES)]
        merged = conf_linear.merge(
            orig_clean[["disorder", "feature", "hedges_g"]],
            on=["disorder", "feature"], how="inner")
        if len(merged) > 0:
            valid = merged["hedges_g_adjusted"].notna() & merged["hedges_g"].notna()
            if valid.sum() > 10:
                from scipy import stats as sp_stats
                r, p = sp_stats.pearsonr(
                    merged.loc[valid, "hedges_g_adjusted"],
                    merged.loc[valid, "hedges_g"])
                mad = np.mean(np.abs(
                    merged.loc[valid, "hedges_g_adjusted"] -
                    merged.loc[valid, "hedges_g"]))
                
                # 美化散点：适当增大并调高通透度
                ax_a.scatter(merged.loc[valid, "hedges_g"],
                             merged.loc[valid, "hedges_g_adjusted"],
                             s=8, alpha=0.30, c='#1E88E5', edgecolors='none', rasterized=True)
                lims = [merged.loc[valid, "hedges_g"].min(), merged.loc[valid, "hedges_g"].max()]
                ax_a.plot(lims, lims, color='#333333', ls='--', lw=0.8)
                
                # 悬浮指标框美化
                ax_a.text(0.05, 0.95, f"r = {r:.3f}\nMAD = {mad:.3f}",
                          transform=ax_a.transAxes, fontsize=FS_ANNOT, va='top', fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.3', fc='#FFFFFF', ec='#DDDDDD', alpha=0.9))
    ax_a.set_xlabel("Original g", fontsize=FS_AXLABEL)
    ax_a.set_ylabel("Adjusted g", fontsize=FS_AXLABEL)
    ax_a.set_title("A. Confound Adj.", fontsize=FS_TITLE, fontweight='bold', loc='left', pad=12)
    ax_a.tick_params(labelsize=FS_TICK)
    _despine(ax_a)

    # Panel B: Sample size stability (computed; reads standalone export OR the
    # sample_size_stability section of sensitivity_report.json).
    ax_b = fig.add_subplot(gs[1])
    stab = None
    for fname in ["sensitivity_sample_stability.json", "sensitivity_sample_size.json"]:
        sp = os.path.join(STATS_DIR, fname)
        if os.path.exists(sp):
            stab = _json(sp)
            if stab:
                break
    if not stab:
        rep_ss = _json(os.path.join(STATS_DIR, "sensitivity_report.json"))
        stab = rep_ss.get("sample_size_stability") if isinstance(rep_ss, dict) else None
    if not stab:
        raise FileNotFoundError(
            "ED Fig 6 panel B requires sample-size-stability results "
            "(sensitivity_sample_stability.json or "
            "sensitivity_report.json['sample_size_stability'])."
        )
    for disorder, fracs in stab.items():
        if not isinstance(fracs, dict):
            continue
        xs = sorted(float(k) for k in fracs.keys())
        ys = []
        for x in xs:
            v = fracs.get(str(x), fracs.get(x))
            if isinstance(v, dict):
                v = v.get("mean_auroc")
            ys.append(np.nan if v is None else float(v))
        ax_b.plot(xs, ys, 'o-', color=get_disorder_color(disorder), ms=4, lw=1.2,
                  label=get_disorder_display(disorder), alpha=0.85)
    leg = ax_b.legend(fontsize=FS_ANNOT, loc='lower right', ncol=2, frameon=True, 
                      framealpha=0.9, edgecolor='#DDDDDD', facecolor='#FFFFFF')
    leg.get_frame().set_boxstyle('round,pad=0.3,rounding_size=0.3')
    ax_b.set_xlabel("Sample Fraction", fontsize=FS_AXLABEL)
    ax_b.set_ylabel("AUROC", fontsize=FS_AXLABEL)
    ax_b.set_title("B. Sample Stability", fontsize=FS_TITLE, fontweight='bold', loc='left', pad=12)
    ax_b.tick_params(labelsize=FS_TICK)
    _despine(ax_b)

    # Panel C: Multiple comparison
    ax_c = fig.add_subplot(gs[2])
    if orig is not None:
        orig_clean = orig[~orig["feature"].isin(SELECTION_ARTIFACT_FEATURES)]
        pvals = orig_clean["p_value"].dropna().values
        from statsmodels.stats.multitest import multipletests
        mc_data = {}
        for name, method in [("None", None), ("Bonf.", "bonferroni"), ("FDR", "fdr_bh")]:
            if method is None:
                n_sig = int(np.sum(pvals < 0.05))
            else:
                reject, _, _, _ = multipletests(pvals, alpha=0.05, method=method)
                n_sig = int(np.sum(reject))
            mc_data[name] = n_sig / max(1, len(pvals))
        methods = list(mc_data.keys())
        props = list(mc_data.values())
        colors = ['#BDBDBD', '#D6604D', '#4393C3']
        bars = ax_c.barh(range(len(methods)), props, color=colors, edgecolor='#222222', lw=0.6, height=0.6)
        for i, p in enumerate(props):
            ax_c.text(p + 0.02, i, f'{p:.0%}', va='center', fontsize=FS_ANNOT, fontweight='bold')
        ax_c.set_yticks(range(len(methods)))
        ax_c.set_yticklabels(methods, fontsize=FS_TICK)
        ax_c.set_xlim(0, 1.05)
    ax_c.set_xlabel("Prop. Significant", fontsize=FS_AXLABEL)
    ax_c.set_title("C. Multi. Comp.", fontsize=FS_TITLE, fontweight='bold', loc='left', pad=12)
    ax_c.tick_params(labelsize=FS_TICK)
    _despine(ax_c)

    # Panel D: Outlier strategies
    ax_d = fig.add_subplot(gs[3])
    out_loaded = False
    for fname in ["sensitivity_outlier.json", "sensitivity_outlier_strategies.json"]:
        sp = os.path.join(STATS_DIR, fname)
        if os.path.exists(sp):
            out = _json(sp)
            if out and isinstance(out, dict):
                strategies = list(out.keys())
                aurocs = [out[s].get("mean_auroc", 0) for s in strategies if isinstance(out[s], dict)]
                if aurocs:
                    short_names = [s.replace('_', ' ')[:12] for s in strategies[:len(aurocs)]]
                    ax_d.barh(range(len(aurocs)), aurocs, color='#78909C', edgecolor='#222222', lw=0.6, height=0.6, alpha=0.85)
                    for i, v in enumerate(aurocs):
                        ax_d.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=FS_ANNOT, fontweight='bold')
                    ax_d.set_yticks(range(len(short_names)))
                    ax_d.set_yticklabels(short_names, fontsize=FS_TICK)
                    out_loaded = True
                    break
    if not out_loaded:
        raise FileNotFoundError(
            "ED Fig 6 panel D requires outlier-strategy output "
            "(sensitivity_outlier.json or sensitivity_outlier_strategies.json) in "
            f"{STATS_DIR}; refusing to plot fabricated values."
        )
    ax_d.set_xlabel("Mean AUROC", fontsize=FS_AXLABEL)
    ax_d.set_title("D. Outlier Strategy", fontsize=FS_TITLE, fontweight='bold', loc='left', pad=12)
    ax_d.set_xlim(0, max(aurocs)*1.25 if aurocs else 1)
    ax_d.tick_params(labelsize=FS_TICK)
    _despine(ax_d)

    fig.suptitle("ED Fig 6 | Sensitivity Analyses", fontsize=FS_SUPTITLE, fontweight='bold', y=1.02)
    _save(fig, "ed_fig6_sensitivity", "ed")


def plot_ed_fig7_strict_inconsistency():
    """ED Fig 7: Strict direction inconsistency detail."""
    logger.info("ED Fig 7: Strict Direction Inconsistency (Optimized Aesthetics)...")
    # Written by s03_effect_sizes.detect_direction_inconsistency (not direction_inconsistency.csv)
    di = _csv(os.path.join(STATS_DIR, "feature_direction_analysis.csv"))
    em = _csv(os.path.join(STATS_DIR, "effect_size_matrix_clean.csv"), index_col=0)
    if di is None or em is None:
        logger.error("  Skip ED Fig 7"); return

    def _col_bool(series):
        if series.dtype == bool:
            return series
        return series.astype(str).str.lower().eq("true")

    strict_col = "has_strict_direction_inconsistency"
    expl_col = "has_direction_inconsistency"
    if strict_col not in di.columns:
        logger.error(f"  Skip ED Fig 7: missing column {strict_col}"); return

    strict = di[_col_bool(di[strict_col])].copy()
    if "is_selection_artifact" in strict.columns:
        strict = strict[~_col_bool(strict["is_selection_artifact"])]
    if len(strict) == 0 and expl_col in di.columns:
        strict = di[_col_bool(di[expl_col])].copy()
        if "is_selection_artifact" in strict.columns:
            strict = strict[~_col_bool(strict["is_selection_artifact"])]
    if len(strict) == 0:
        logger.warning("  No direction-inconsistent features"); return

    strict = strict.sort_values("sig_g_range", ascending=True)
    n = min(len(strict), 20)
    feats = strict["feature"].values[:n]

    data = np.full((n, len(DISORDER_GROUPS)), np.nan)
    for i, f in enumerate(feats):
        if f not in em.index:
            continue
        for j, d in enumerate(DISORDER_GROUPS):
            if d in em.columns:
                g = em.loc[f, d]
                if pd.notna(g):
                    data[i, j] = float(g)

    finite = data[np.isfinite(data)]
    if finite.size == 0:
        logger.error("  Skip ED Fig 7: empty effect matrix"); return

    logger.info(
        f"  Strict-inconsistent heatmap: features={n}, "
        f"g range [{finite.min():+.3f}, {finite.max():+.3f}]"
    )

    plot_data, im_kw, text_thr, cmap_log = _prepare_heatmap_layers(data, finite)
    logger.info(f"  Colormap: {cmap_log}")

    fig, ax = _ed_heatmap_figure()
    im = _ed_heatmap_imshow(ax, plot_data, im_kw)
    _ed_heatmap_annotate(ax, data, text_thr)

    f2d, _ = _load_dim_map() if hasattr(sys.modules[__name__], '_load_dim_map') else ({}, [])
    ax.set_yticks(range(n))
    ax.set_yticklabels([_clean_name(f)[:22] for f in feats],
                       fontsize=FS_TICK, fontweight="bold")
    for i, f in enumerate(feats):
        ax.get_yticklabels()[i].set_color(DIM_COLORS.get(f2d.get(f, "other"), "#333333"))

    ax.set_xticks(range(len(DISORDER_GROUPS)))
    ax.set_xticklabels(
        [get_disorder_display(d) for d in DISORDER_GROUPS],
        rotation=45, ha="right", rotation_mode="anchor",
        fontsize=FS_TICK, fontweight="bold",
    )
    for lb in ax.get_xticklabels():
        rn = [k for k, v in DISORDER_DISPLAY_NAMES.items() if v == lb.get_text()]
        if rn:
            lb.set_color(get_disorder_color(rn[0]))

    _ed_heatmap_style_frame(ax)
    _ed_heatmap_colorbar(fig, ax, im, "Hedges' g")
    _ed_heatmap_title(
        ax,
        "ED Fig 7 | Strict Direction-Inconsistent Features\n"
        "(Blue = Control > Patient; Red = Patient > Control)",
    )
    _save_ed_fixed(fig, "ed_fig7_strict_inconsistency")


def plot_ed_fig8_effect_categories():
    """ED Fig 8: Effect size category by disorder."""
    logger.info("ED Fig 8: Effect Size Categories (Optimized Aesthetics)...")
    ef = _csv(os.path.join(STATS_DIR, "effect_sizes_full.csv"))
    if ef is None:
        logger.error("  Skip ED Fig 8"); return

    ec = ef[~ef["feature"].isin(SELECTION_ARTIFACT_FEATURES)].copy()
    cat_order = ['large', 'medium', 'small', 'minimal', 'negligible']
    cat_colors = {'large': '#D64545', 'medium': '#E8874A',
                  'small': '#F4C04A', 'minimal': '#F5E1C0',
                  'negligible': '#F7F3ED'}

    disorder_cats = {}
    for d in DISORDER_GROUPS:
        mask = ec["disorder"] == d
        sub = ec[mask]
        if len(sub) == 0: continue
        cats = sub["effect_category"].value_counts()
        disorder_cats[d] = {c: cats.get(c, 0) for c in cat_order}

    if not disorder_cats:
        logger.error("  No category data"); return

    fig, ax = _ed_bar_figure(bottom=BAR_MARGIN_BOTTOM_ROTATED)
    
    disorders = list(disorder_cats.keys())
    disorders.sort(key=lambda d: sum(disorder_cats[d].get(c, 0) for c in ['large','medium','small']), reverse=True)

    x = np.arange(len(disorders))
    bottoms = np.zeros(len(disorders))
    totals = np.array([
        sum(disorder_cats[d].get(cat, 0) for cat in cat_order)
        for d in disorders
    ])
    for cat in cat_order:
        vals = [disorder_cats[d].get(cat, 0) for d in disorders]
        ax.bar(x, vals, bottom=bottoms, color=cat_colors[cat],
               edgecolor="white", linewidth=0.5, width=BAR_STACK_W,
               label=cat.title())
        for i, v in enumerate(vals):
            # Label only segments large enough to remain legible.
            if v >= max(3, totals[i] * 0.06):
                ax.text(i, bottoms[i] + v/2, str(v), ha='center', va='center', 
                        fontsize=FS_ANNOT,
                        color='white' if cat in ('large','medium') else '#333333')
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(
        [get_disorder_display(d) for d in disorders],
        rotation=45, ha='right', rotation_mode='anchor', fontsize=FS_TICK, fontweight='bold')
        
    for lb in ax.get_xticklabels():
        rn = [k for k, v in DISORDER_DISPLAY_NAMES.items() if v == lb.get_text()]
        if rn:
            lb.set_color(get_disorder_color(rn[0]))
            
    ax.set_ylim(0, max(bottoms) * BAR_Y_PAD)
    ax.legend(fontsize=FS_LEGEND, loc="upper center",
              bbox_to_anchor=(0.5, -0.28), ncol=5, frameon=False,
              handlelength=1.2, handleheight=0.5, handletextpad=0.4,
              columnspacing=0.8)
    _ed_bar_style(
        ax,
        ylabel="Number of Features",
        title="ED Fig 8 | Effect Size Category by Disorder",
    )
    _save_ed_fixed(fig, "ed_fig8_effect_categories")


# =====================================================================
# MASTER RUNNER
# =====================================================================

def run_all_visualizations():
    logger.info("=" * 70)
    logger.info("PUBLICATION-QUALITY VISUALIZATION PIPELINE (v5.0)")
    logger.info("=" * 70)

    logger.info("\n--- Main Figures ---")
    plot_fig1_atlas()
    plot_fig2_pfactor()
    plot_fig3_detectability()
    plot_fig4_specificity()

    logger.info("\n--- Extended Data Figures ---")
    plot_ed_fig1_factor()
    plot_ed_fig2_clustering()
    plot_ed_fig3_tsne()
    plot_ed_fig4_confusion()
    plot_ed_fig5_ablation()
    plot_ed_fig6_sensitivity()
    plot_ed_fig7_strict_inconsistency()
    plot_ed_fig8_effect_categories()

    logger.info("\n" + "=" * 70)
    logger.info("ALL VISUALIZATIONS COMPLETE")
    logger.info(f"  Main:    {MAIN_DIR}")
    logger.info(f"  Extended: {ED_DIR}")
    logger.info("=" * 70)


# =====================================================================
# MASTER RUNNER & COMMAND-LINE INTERFACE (CLI)
# =====================================================================

# 建立标准化的图表注册表，将指令字符串映射到具体的绘图函数
PLOT_REGISTRY = {
    "fig1": plot_fig1_atlas,
    "fig2": plot_fig2_pfactor,
    "fig3": plot_fig3_detectability,
    "fig4": plot_fig4_specificity,
    "edfig1": plot_ed_fig1_factor,
    "edfig2": plot_ed_fig2_clustering,
    "edfig3": plot_ed_fig3_tsne,
    "edfig4": plot_ed_fig4_confusion,
    "edfig5": plot_ed_fig5_ablation,
    "edfig6": plot_ed_fig6_sensitivity,
    "edfig7": plot_ed_fig7_strict_inconsistency,
    "edfig8": plot_ed_fig8_effect_categories,
}

def run_visualizations(selected_plots):
    """
    基于用户指令模块化运行选定的绘图函数。
    """
    logger.info("=" * 70)
    logger.info("PUBLICATION-QUALITY VISUALIZATION PIPELINE (v5.1 - Modular CLI)")
    logger.info("=" * 70)

    # 如果指令包含 'all'，则加载全部模块
    if "all" in selected_plots:
        targets = list(PLOT_REGISTRY.keys())
    else:
        # 过滤并验证输入的命令
        targets = [p for p in selected_plots if p in PLOT_REGISTRY]
        invalid = [p for p in selected_plots if p not in PLOT_REGISTRY and p != "all"]
        if invalid:
            logger.warning(f"Unrecognized plot commands: {invalid}")
            logger.info(f"Available commands: {list(PLOT_REGISTRY.keys())}")

    if not targets:
        logger.error("No valid plots selected. Exiting.")
        sys.exit(1)

    logger.info(f"Targeted plots for execution: {targets}\n")

    # 按需独立执行
    for target in targets:
        func = PLOT_REGISTRY[target]
        try:
            func()
        except Exception as e:
            logger.error(f"Error executing {target}: {str(e)}")

    logger.info("\n" + "=" * 70)
    logger.info("TARGETED VISUALIZATIONS COMPLETE")
    logger.info(f"  Main:     {MAIN_DIR}")
    logger.info(f"  Extended: {ED_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    # 配置标准化的命令行参数解析器
    parser = argparse.ArgumentParser(
        description="Modular CLI for Digital Phenotyping Visualizations.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "-p", "--plots",
        nargs="+",
        default=["all"],
        help=(
            "Specify which plots to generate. You can pass multiple arguments.\n"
            "Example: python script.py -p fig1 edfig3\n"
            "Options: 'all', 'fig1'-'fig4', 'edfig1'-'edfig8'"
        )
    )

    args = parser.parse_args()
    
    # 提取参数并传入执行引擎
    run_visualizations(args.plots)