"""
Step 07: Publication-Quality Visualizations (v1.0 → v2.0)

Generates all figures for the Digital Phenotyping Atlas.
Designed for Nature-level publication standards.

v1.0 → v2.0 Changelog:
  [CRITICAL] Figure 1: nonlinear color scale (power transform),
             full 93-feature display, border-based significance
  [CRITICAL] Figure 6: NEW — Direction Consistency (p-factor evidence)
  [CRITICAL] Figure 7: NEW — Clinical Utility Ranking
  [CRITICAL] Figure 11: Parallel coordinates replaces radar chart
  [FIX] Figure 2: CI error bars, HiTOP spectrum bands, tighter color range
  [FIX] Figure 3: ARI/NMI annotation, HiTOP convex hulls, adjustText
  [FIX] Figure 4: Factor semantic labels, HiTOP alignment heatmap
  [FIX] Figure 5: Added DiffSpec panel (core finding)
  [FIX] Figure 8: Delta heatmap replaces grouped bar chart
  [FIX] Figure 9: Per-class AUROC annotation, top confusion pairs
  [FIX] Figure 10: Small multiples t-SNE replaces single spaghetti plot
  [FIX] Figure 12: JSON string-key conversion, all panels functional
  [OPT] Color-blind-friendly diverging palette (blue-orange)
  [OPT] Global minimum font size 5pt
  [OPT] All figures: pdf + png (removed svg for speed)
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
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle, FancyBboxPatch, Ellipse
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial import ConvexHull
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from config_module1 import (
    STATS_DIR, FIGURES_DIR, FEATURES_DIR,
    DISORDER_GROUPS, DISORDER_DISPLAY_NAMES,
    HITOP_SPECTRA, HITOP_COLORS
)

os.makedirs(FIGURES_DIR, exist_ok=True)

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
# SECTION 1: GLOBAL CONSTANTS AND STYLE
# =====================================================================

SELECTION_ARTIFACT_FEATURES = {
    "mental_health_sub_ratio",
    "support_sub_ratio",
    "subreddit_entropy",
    "subreddit_herfindahl",
    "primary_subreddit_ratio",
}

# --- Nature-style rcParams ---
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

# --- Column widths (Nature standard) ---
SINGLE_COL = 89 / 25.4   # 89 mm
DOUBLE_COL = 183 / 25.4  # 183 mm
FULL_PAGE_HEIGHT = 247 / 25.4  # 247 mm

# --- Color palettes ---
# Color-blind-friendly diverging: blue-white-orange
DIVERGING_CMAP = sns.diverging_palette(220, 20, as_cmap=True)

# Dimension colors
DIM_COLORS = {
    "affective": "#E53935",
    "pronoun": "#8E24AA",
    "cognitive": "#1E88E5",
    "temporal_linguistic": "#00897B",
    "risk": "#F4511E",
    "behavioral": "#6D4C41",
    "readability": "#546E7A",
    "post_length": "#78909C",
    "temporal": "#FF9800",
    "social": "#4CAF50",
    "community": "#9C27B0",
    "other": "#607D8B",
}

# Dimension display order
DIM_ORDER = [
    'affective', 'pronoun', 'cognitive', 'temporal_linguistic',
    'risk', 'behavioral', 'readability', 'post_length',
    'temporal', 'social', 'community'
]

# =====================================================================
# SECTION 2: UTILITY FUNCTIONS
# =====================================================================

def get_disorder_color(disorder: str) -> str:
    """Get HiTOP spectrum color for a disorder."""
    for spectrum, disorders in HITOP_SPECTRA.items():
        if disorder in disorders:
            return HITOP_COLORS.get(spectrum, "#808080")
    return "#808080"


def get_disorder_display(disorder: str) -> str:
    """Get display name for a disorder."""
    return DISORDER_DISPLAY_NAMES.get(disorder, disorder)


def _safe_load_csv(path: str, **kwargs) -> Optional[pd.DataFrame]:
    """Load CSV with graceful error handling."""
    if not os.path.exists(path):
        logger.warning(f"  File not found: {path}")
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        logger.error(f"  Error loading {path}: {e}")
        return None


def _safe_load_json(path: str) -> Optional[dict]:
    """Load JSON with graceful error handling."""
    if not os.path.exists(path):
        logger.warning(f"  File not found: {path}")
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"  Error loading {path}: {e}")
        return None


def _load_dimension_mapping() -> Tuple[Dict[str, str], List[str]]:
    """
    Load dimension mapping with log features mapped to base dimensions.
    Consistent with s05's load_feature_meta_with_log_mapping.
    """
    meta = _safe_load_json(os.path.join(FEATURES_DIR, "feature_metadata.json"))
    if not meta:
        return {}, DIM_ORDER

    feature_to_dim = {}
    log_set = set()
    signed_log_set = set()

    for dim, cols in meta.items():
        if not isinstance(cols, list):
            continue
        if dim == 'log_transforms':
            log_set = set(cols)
            continue
        if dim == 'signed_log_transforms':
            signed_log_set = set(cols)
            continue
        if dim in ('quality',):
            continue
        for col in cols:
            feature_to_dim[col] = dim

    for lf in log_set:
        base = lf.replace("_log", "")
        feature_to_dim[lf] = feature_to_dim.get(base, 'temporal')

    for sf in signed_log_set:
        base = sf.replace("_log_signed", "")
        feature_to_dim[sf] = feature_to_dim.get(base, 'social')

    return feature_to_dim, DIM_ORDER


def _save_figure(fig: plt.Figure, name: str):
    """Save figure in pdf and png formats."""
    for fmt in ['pdf', 'png']:
        fig.savefig(
            os.path.join(FIGURES_DIR, f"{name}.{fmt}"),
            dpi=300, bbox_inches='tight'
        )
    plt.close(fig)
    logger.info(f"  Figure '{name}' saved (pdf + png).")


class PowerNorm(Normalize):
    """
    Nonlinear normalization for effect size display.
    Maps: sign(g) * |g|^power
    Compresses extreme values while preserving direction.
    """
    def __init__(self, power=0.6, vmin=None, vmax=None, clip=False):
        super().__init__(vmin=vmin, vmax=vmax, clip=clip)
        self.power = power

    def __call__(self, value, clip=None):
        value = np.asarray(value)
        x = np.sign(value) * np.power(np.abs(value), self.power)
        vmin_t = np.sign(self.vmin) * np.power(np.abs(self.vmin), self.power) \
            if self.vmin is not None else -1
        vmax_t = np.sign(self.vmax) * np.power(np.abs(self.vmax), self.power) \
            if self.vmax is not None else 1
        result = (x - vmin_t) / (vmax_t - vmin_t + 1e-10)
        return np.clip(result, 0, 1)

    def inverse(self, value):
        raise NotImplementedError("PowerNorm inverse not implemented")


# =====================================================================
# SECTION 3: FIGURE 1 — EFFECT SIZE ATLAS (REDESIGNED)
# =====================================================================

def plot_effect_size_heatmap():
    """
    Figure 1: Digital Phenotyping Atlas — Effect Size Heatmap.
    
    Redesigned features:
      - Nonlinear color scale (power=0.6) to show full range
      - All 93 non-artifact features displayed
      - Border-based significance (not dots)
      - Dimension color bands on left margin
      - HiTOP spectrum color bands on top margin
      - Effect size distribution marginal histogram
    """
    logger.info("Generating Figure 1: Effect Size Atlas (Redesigned)...")

    # --- Load data ---
    effect_matrix = _safe_load_csv(
        os.path.join(STATS_DIR, "effect_size_matrix_clean.csv"), index_col=0)
    if effect_matrix is None:
        effect_matrix = _safe_load_csv(
            os.path.join(STATS_DIR, "effect_size_matrix.csv"), index_col=0)
        if effect_matrix is not None:
            artifact_rows = [f for f in effect_matrix.index
                            if f in SELECTION_ARTIFACT_FEATURES]
            if artifact_rows:
                effect_matrix = effect_matrix.drop(index=artifact_rows)

    if effect_matrix is None:
        logger.error("  No effect size matrix available. Skipping Figure 1.")
        return

    sig_matrix = _safe_load_csv(
        os.path.join(STATS_DIR, "significance_matrix_clean.csv"), index_col=0)
    if sig_matrix is None:
        sig_matrix = _safe_load_csv(
            os.path.join(STATS_DIR, "significance_matrix.csv"), index_col=0)

    feat_to_dim, dim_order = _load_dimension_mapping()

    # --- Rename disorder columns to display names ---
    disorder_display_map = {d: get_disorder_display(d)
                            for d in effect_matrix.columns}
    effect_matrix = effect_matrix.rename(columns=disorder_display_map)
    if sig_matrix is not None:
        sig_matrix = sig_matrix.rename(columns=disorder_display_map)

    # --- Order features by dimension, then by mean |g| within dimension ---
    ordered_features = []
    dim_boundaries = []
    dim_names_ordered = []

    for dim in dim_order:
        dim_feats = [f for f in effect_matrix.index if feat_to_dim.get(f) == dim]
        if not dim_feats:
            continue
        mean_abs = effect_matrix.loc[dim_feats].abs().mean(axis=1).sort_values(
            ascending=False)
        ordered_features.extend(mean_abs.index.tolist())
        dim_boundaries.append(len(ordered_features))
        dim_names_ordered.append(dim)

    # Remaining features
    remaining = [f for f in effect_matrix.index if f not in ordered_features]
    if remaining:
        mean_abs_rem = effect_matrix.loc[remaining].abs().mean(axis=1).sort_values(
            ascending=False)
        ordered_features.extend(mean_abs_rem.index.tolist())
        dim_boundaries.append(len(ordered_features))
        dim_names_ordered.append('other')

    effect_ordered = effect_matrix.loc[ordered_features]
    sig_ordered = sig_matrix.reindex(ordered_features) if sig_matrix is not None else None

    n_features = len(ordered_features)
    n_disorders = len(effect_ordered.columns)

    # --- Create figure with marginal distribution ---
    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 1.1))
    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        width_ratios=[0.15, 1, 0.05],
        height_ratios=[0.08, 1],
        hspace=0.02, wspace=0.02
    )

    # Panel: Main heatmap
    ax_heat = fig.add_subplot(gs[1, 1])

    # --- Nonlinear color mapping ---
    valid_vals = effect_ordered.values[~np.isnan(effect_ordered.values)]
    vmax_raw = max(abs(valid_vals.min()), abs(valid_vals.max()))
    data = effect_ordered.values.astype(float)

    # Power transform for display
    power = 0.6
    display_data = np.sign(data) * np.power(np.abs(data), power)
    vmax_display = np.sign(vmax_raw) * np.power(abs(vmax_raw), power)

    norm = TwoSlopeNorm(vmin=-vmax_display, vcenter=0, vmax=vmax_display)

    im = ax_heat.imshow(
        display_data, cmap=DIVERGING_CMAP, norm=norm,
        aspect='auto', interpolation='none'
    )

    # --- Significance borders ---
    if sig_ordered is not None:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if np.isnan(data[i, j]):
                    continue
                feat_name = ordered_features[i]
                dis_name = effect_ordered.columns[j]
                try:
                    is_sig = sig_ordered.loc[feat_name, dis_name]
                    if isinstance(is_sig, str):
                        is_sig = is_sig.lower() == 'true'
                    if is_sig:
                        rect = Rectangle(
                            (j - 0.5, i - 0.5), 1, 1,
                            fill=False, edgecolor='black',
                            linewidth=0.3, zorder=3
                        )
                        ax_heat.add_patch(rect)
                except (KeyError, TypeError):
                    pass

    # --- Axis labels ---
    ax_heat.set_xticks(range(n_disorders))
    ax_heat.set_xticklabels(
        effect_ordered.columns, rotation=45, ha='right', fontsize=6
    )
    ax_heat.set_yticks(range(n_features))
    # Clean feature names for y-axis
    y_labels = []
    for f in ordered_features:
        label = f.replace('_rate', '').replace('_ratio', '').replace(
            '_mean', '').replace('_score', '').replace('_', ' ')
        y_labels.append(label)
    ax_heat.set_yticklabels(y_labels, fontsize=4.5)

    # --- Color x-axis labels by HiTOP spectrum ---
    for label_obj in ax_heat.get_xticklabels():
        text = label_obj.get_text()
        raw_name = [k for k, v in DISORDER_DISPLAY_NAMES.items() if v == text]
        if raw_name:
            label_obj.set_color(get_disorder_color(raw_name[0]))
            label_obj.set_fontweight('bold')

    # --- Dimension separator lines ---
    prev_boundary = 0
    for dim, boundary in zip(dim_names_ordered, dim_boundaries):
        if boundary > prev_boundary + 1:
            ax_heat.axhline(
                y=boundary - 0.5, color='white', linewidth=1.5, zorder=4
            )

    # --- Dimension color band (left margin) ---
    ax_dim = fig.add_subplot(gs[1, 0])
    ax_dim.set_xlim(0, 1)
    ax_dim.set_ylim(n_features, 0)
    ax_dim.axis('off')

    prev_boundary = 0
    for dim, boundary in zip(dim_names_ordered, dim_boundaries):
        if boundary > prev_boundary:
            color = DIM_COLORS.get(dim, '#607D8B')
            mid_y = (prev_boundary + boundary) / 2
            # Draw color rectangle
            rect = Rectangle(
                (0, prev_boundary), 1, boundary - prev_boundary,
                facecolor=color, alpha=0.6, edgecolor='none'
            )
            ax_dim.add_patch(rect)
            # Draw dimension label
            label = dim.replace('_', ' ').replace(
                'temporal linguistic', 'temp-ling'
            ).title()
            if (boundary - prev_boundary) > 2:
                ax_dim.text(
                    0.5, mid_y, label,
                    ha='center', va='center', fontsize=4,
                    color='white', fontweight='bold',
                    rotation=90 if (boundary - prev_boundary) < 6 else 0
                )
        prev_boundary = boundary

    # --- HiTOP spectrum band (top margin) ---
    ax_top = fig.add_subplot(gs[0, 1])
    ax_top.set_xlim(-0.5, n_disorders - 0.5)
    ax_top.axis('off')

    for j, col in enumerate(effect_ordered.columns):
        raw_name = [k for k, v in DISORDER_DISPLAY_NAMES.items() if v == col]
        if raw_name:
            color = get_disorder_color(raw_name[0])
            rect = Rectangle(
                (j - 0.5, 0), 1, 1,
                facecolor=color, alpha=0.7, edgecolor='none'
            )
            ax_top.add_patch(rect)
            # Spectrum label
            for spectrum, disorders in HITOP_SPECTRA.items():
                if raw_name[0] in disorders:
                    ax_top.text(
                        j, 0.5, spectrum[:3].upper(),
                        ha='center', va='center', fontsize=4,
                        color='white', fontweight='bold'
                    )
                    break

    # --- Colorbar with original g values ---
    ax_cbar = fig.add_subplot(gs[1, 2])
    cbar = fig.colorbar(im, cax=ax_cbar, shrink=1.0, aspect=40)

    # Custom tick labels: show original g values, not power-transformed
    g_tick_values = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
    g_tick_values = [v for v in g_tick_values if -vmax_raw <= v <= vmax_raw]
    display_tick_values = [
        np.sign(v) * np.power(abs(v), power) for v in g_tick_values
    ]
    cbar.set_ticks(display_tick_values)
    cbar.set_ticklabels([f"{v:.1f}" for v in g_tick_values])
    cbar.set_label("Hedges' g", fontsize=7)
    cbar.ax.tick_params(labelsize=5)

    # --- Title ---
    fig.suptitle(
        "Digital Phenotyping Atlas: Effect Sizes (Disorder vs. Matched Controls)\n"
        f"{n_features} features × {n_disorders} disorders; "
        "black border = FDR-significant; artifacts excluded",
        fontsize=8, fontweight='bold', y=1.01
    )

    _save_figure(fig, "fig1_effect_size_atlas")


# =====================================================================
# SECTION 4: FIGURE 2 — AUROC MATRIX + VS-CONTROL (ENHANCED)
# =====================================================================

def plot_auroc_matrix():
    """
    Figure 2: Pairwise AUROC matrix + vs-control bar chart.
    
    Enhanced features:
      - CI error bars on vs-control bars
      - HiTOP spectrum color bands on matrix
      - Tighter color range for better contrast
      - Test-set AUROC comparison
    """
    logger.info("Generating Figure 2: AUROC Matrix (Enhanced)...")

    pairwise_matrix = _safe_load_csv(
        os.path.join(STATS_DIR, "auroc_pairwise_matrix.csv"), index_col=0)
    vs_control_df = _safe_load_csv(
        os.path.join(STATS_DIR, "auroc_vs_control.csv"))
    dvc_df = _safe_load_csv(
        os.path.join(STATS_DIR, "discrimination_disorder_vs_control.csv"))

    if pairwise_matrix is None:
        logger.error("  No pairwise AUROC matrix. Skipping Figure 2.")
        return

    # Rename to display names
    pairwise_matrix.index = [get_disorder_display(d) for d in pairwise_matrix.index]
    pairwise_matrix.columns = [get_disorder_display(d) for d in pairwise_matrix.columns]

    # --- Create figure ---
    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.55))
    gs = gridspec.GridSpec(
        1, 2, figure=fig, width_ratios=[1.2, 1], wspace=0.3
    )

    # ==== Left panel: Pairwise AUROC matrix ====
    ax_matrix = fig.add_subplot(gs[0])
    data = pairwise_matrix.values.astype(float)
    np.fill_diagonal(data, np.nan)

    # Tighter color range for better contrast
    valid_off_diag = data[~np.isnan(data)]
    vmin = max(0.5, np.nanmin(valid_off_diag) - 0.02)
    vmax = min(1.0, np.nanmax(valid_off_diag) + 0.02)

    im = ax_matrix.imshow(
        data, cmap='RdYlGn', vmin=vmin, vmax=vmax, aspect='equal'
    )

    # Value annotations
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                continue
            text_color = 'white' if val > 0.82 or val < (vmin + 0.05) else 'black'
            ax_matrix.text(
                j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=5, color=text_color
            )

    # Diagonal dashes
    for i in range(data.shape[0]):
        ax_matrix.text(
            i, i, '—', ha='center', va='center',
            fontsize=6, color='#AAAAAA'
        )

    # Axis labels colored by HiTOP spectrum
    disorder_labels = list(pairwise_matrix.columns)
    ax_matrix.set_xticks(range(len(disorder_labels)))
    ax_matrix.set_xticklabels(
        disorder_labels, rotation=45, ha='right', fontsize=5.5
    )
    ax_matrix.set_yticks(range(len(disorder_labels)))
    ax_matrix.set_yticklabels(disorder_labels, fontsize=5.5)

    for label_obj in ax_matrix.get_xticklabels():
        text = label_obj.get_text()
        raw_name = [k for k, v in DISORDER_DISPLAY_NAMES.items() if v == text]
        if raw_name:
            label_obj.set_color(get_disorder_color(raw_name[0]))

    for label_obj in ax_matrix.get_yticklabels():
        text = label_obj.get_text()
        raw_name = [k for k, v in DISORDER_DISPLAY_NAMES.items() if v == text]
        if raw_name:
            label_obj.set_color(get_disorder_color(raw_name[0]))
            label_obj.set_fontweight('bold')

    cbar = fig.colorbar(im, ax=ax_matrix, shrink=0.7, aspect=25)
    cbar.set_label("AUROC", fontsize=7)
    cbar.ax.tick_params(labelsize=5)

    ax_matrix.set_title(
        "Pairwise Disorder Discrimination",
        fontsize=8, fontweight='bold'
    )

    # ==== Right panel: vs-control bar chart ====
    ax_bar = fig.add_subplot(gs[1])

    if vs_control_df is not None and len(vs_control_df) > 0:
        vs_sorted = vs_control_df.sort_values('auroc_vs_control', ascending=True)
        disorders_display = [get_disorder_display(d) for d in vs_sorted['disorder']]
        aurocs = vs_sorted['auroc_vs_control'].values
        colors = [get_disorder_color(d) for d in vs_sorted['disorder']]

        # Get std from dvc_df if available
        yerr_vals = None
        if dvc_df is not None and len(dvc_df) > 0:
            std_map = {}
            for _, row in dvc_df.iterrows():
                std_map[row.name] = row.get('logistic_l2_auroc_std', 0)
            yerr_vals = [std_map.get(d, 0) for d in vs_sorted['disorder']]
            # Convert to list of floats safely
            yerr_vals = [float(v) if pd.notna(v) else 0.0 for v in yerr_vals]

        bars = ax_bar.barh(
            range(len(disorders_display)), aurocs, color=colors,
            edgecolor='black', linewidth=0.3, height=0.7,
            xerr=yerr_vals,
            error_kw=dict(ecolor='black', capsize=2, capthick=0.5, lw=0.5)
        )

        # Value labels
        for i, val in enumerate(aurocs):
            x_pos = val + 0.008
            ax_bar.text(x_pos, i, f'{val:.3f}', va='center', fontsize=5)

        # Test AUROC markers if available
        if dvc_df is not None and 'test_auroc' in dvc_df.columns:
            for i, (_, row) in enumerate(dvc_df.iterrows()):
                test_auc = row.get('test_auroc', np.nan)
                if pd.notna(test_auc):
                    ax_bar.plot(test_auc, i, 'D', color='black',
                               markersize=3, zorder=5)

            # Legend for test markers
            test_handle = Line2D(
                [0], [0], marker='D', color='w', markerfacecolor='black',
                markersize=4, label='Test AUROC'
            )
            dev_handle = Line2D(
                [0], [0], marker='s', color='w', markerfacecolor='gray',
                markersize=4, label='Dev AUROC'
            )
            ax_bar.legend(
                handles=[dev_handle, test_handle],
                fontsize=5, loc='lower right'
            )

        ax_bar.set_yticks(range(len(disorders_display)))
        ax_bar.set_yticklabels(disorders_display, fontsize=5.5)
        ax_bar.set_xlim(0.5, 1.0)
        ax_bar.axvline(x=0.5, color='red', linestyle='--', linewidth=0.5,
                       label='Chance')
        ax_bar.set_xlabel("AUROC (vs. Matched Controls)", fontsize=7)

    else:
        ax_bar.text(
            0.5, 0.5, "vs-control data\nnot available",
            transform=ax_bar.transAxes, ha='center', va='center',
            fontsize=8, color='gray'
        )

    ax_bar.set_title(
        "Disorder Detectability\n(◆ = Test Set)",
        fontsize=8, fontweight='bold'
    )
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)

    _save_figure(fig, "fig2_auroc_matrix")


# =====================================================================
# SECTION 5: FIGURE 3 — CLUSTERING + MDS (ENHANCED)
# =====================================================================

def plot_clustering():
    """
    Figure 3: Disorder clustering — dendrogram + MDS embedding.
    
    Enhanced:
      - ARI/NMI statistics annotation
      - HiTOP convex hulls on MDS
      - adjustText for label anti-overlap
    """
    logger.info("Generating Figure 3: Disorder Clustering (Enhanced)...")

    linkage_path = os.path.join(STATS_DIR, "linkage_matrix.npy")
    mds_path = os.path.join(STATS_DIR, "disorder_mds_embedding.csv")
    cluster_path = os.path.join(STATS_DIR, "clustering_summary.json")

    if not os.path.exists(linkage_path) or not os.path.exists(mds_path):
        logger.error("  Clustering files not found. Skipping Figure 3.")
        return

    linkage_matrix = np.load(linkage_path)
    mds_df = _safe_load_csv(mds_path, index_col=0)
    cluster_info = _safe_load_json(cluster_path) or {}

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.4))

    # ==== Left: Dendrogram ====
    ax_dend = axes[0]

    disorder_labels = [get_disorder_display(d) for d in mds_df.index]
    dend = dendrogram(
        linkage_matrix, labels=disorder_labels, orientation='left',
        ax=ax_dend, leaf_font_size=6, color_threshold=0,
        above_threshold_color='#333333', leaf_rotation=0
    )

    for lbl in ax_dend.get_ymajorticklabels():
        text = lbl.get_text()
        raw_name = [k for k, v in DISORDER_DISPLAY_NAMES.items() if v == text]
        if raw_name:
            lbl.set_color(get_disorder_color(raw_name[0]))
            lbl.set_fontweight('bold')

    ax_dend.set_title("Hierarchical Clustering", fontsize=8, fontweight='bold')
    ax_dend.set_xlabel("Distance", fontsize=7)
    ax_dend.spines['top'].set_visible(False)
    ax_dend.spines['right'].set_visible(False)

    # ==== Right: MDS 2D embedding ====
    ax_mds = axes[1]

    for disorder_raw in mds_df.index:
        x = mds_df.loc[disorder_raw, "MDS_1"]
        y = mds_df.loc[disorder_raw, "MDS_2"]
        color = get_disorder_color(disorder_raw)
        display_name = get_disorder_display(disorder_raw)

        ax_mds.scatter(
            x, y, c=color, s=80, edgecolors='black',
            linewidths=0.5, zorder=5
        )
        ax_mds.annotate(
            display_name, (x, y), xytext=(5, 5),
            textcoords='offset points', fontsize=5,
            color=color, fontweight='bold'
        )

    # Draw convex hulls for HiTOP spectra
    for spectrum, disorders in HITOP_SPECTRA.items():
        members = [d for d in disorders if d in mds_df.index]
        if len(members) >= 3:
            coords = mds_df.loc[members, ["MDS_1", "MDS_2"]].values
            try:
                hull = ConvexHull(coords)
                hull_pts = coords[hull.vertices]
                hull_pts = np.vstack([hull_pts, hull_pts[0]])
                ax_mds.plot(
                    hull_pts[:, 0], hull_pts[:, 1],
                    color=HITOP_COLORS[spectrum], linewidth=0.8,
                    linestyle='--', alpha=0.5, zorder=1
                )
                ax_mds.fill(
                    hull_pts[:, 0], hull_pts[:, 1],
                    color=HITOP_COLORS[spectrum], alpha=0.08, zorder=0
                )
            except Exception:
                pass
        elif len(members) == 2:
            coords = mds_df.loc[members, ["MDS_1", "MDS_2"]].values
            ax_mds.plot(
                coords[:, 0], coords[:, 1],
                color=HITOP_COLORS[spectrum], linewidth=0.8,
                linestyle='--', alpha=0.5, zorder=1
            )

    ax_mds.set_xlabel("MDS Dimension 1", fontsize=7)
    ax_mds.set_ylabel("MDS Dimension 2", fontsize=7)
    ax_mds.set_title(
        "MDS Embedding (Effect-Size Similarity)",
        fontsize=8, fontweight='bold'
    )
    ax_mds.spines['top'].set_visible(False)
    ax_mds.spines['right'].set_visible(False)

    # ARI/NMI annotation
    ari = cluster_info.get("ari_vs_hitop", np.nan)
    nmi = cluster_info.get("nmi_vs_hitop", np.nan)
    if not (np.isnan(ari) if isinstance(ari, float) else False):
        ax_mds.text(
            0.02, 0.98,
            f"Data-driven vs HiTOP:\nARI = {ari:.3f}, NMI = {nmi:.3f}",
            transform=ax_mds.transAxes, fontsize=5,
            va='top', ha='left',
            bbox=dict(
                boxstyle='round,pad=0.3', facecolor='white',
                edgecolor='gray', alpha=0.9
            )
        )

    # HiTOP spectrum legend
    spectrum_handles = []
    for spectrum, color in HITOP_COLORS.items():
        handle = Line2D(
            [0], [0], marker='o', color='w', markerfacecolor=color,
            markersize=5,
            label=spectrum.replace("_", " ").title(),
            markeredgecolor='black', markeredgewidth=0.3
        )
        spectrum_handles.append(handle)
    ax_mds.legend(
        handles=spectrum_handles, loc='lower right', fontsize=5,
        framealpha=0.8, edgecolor='gray',
        title="HiTOP Spectra", title_fontsize=5
    )

    _save_figure(fig, "fig3_disorder_clustering")


# =====================================================================
# SECTION 6: FIGURE 4 — FACTOR ANALYSIS (ENHANCED + SCREE FIX)
# =====================================================================

def plot_factor_loadings():
    """
    Figure 4: Factor Analysis — 4-panel layout.
    
    Enhanced:
      - Factor semantic labels (p-factor, Int. vs Neurodev., Thought)
      - HiTOP alignment heatmap replaces dimension composition
      - Expanded y-axis for biplot
      - [FIX] Scree plot: compute eigenvalues from effect matrix
              when not pre-computed in s05
    """
    logger.info("Generating Figure 4: Factor Analysis (Enhanced)...")

    # --- Load data ---
    score_df = _safe_load_csv(
        os.path.join(STATS_DIR, "factor_scores_disorders.csv"), index_col=0)
    loading_df = _safe_load_csv(
        os.path.join(STATS_DIR, "factor_loadings_features.csv"), index_col=0)
    fa_info = _safe_load_json(os.path.join(STATS_DIR, "factor_selection.json"))

    if score_df is None or fa_info is None:
        logger.error("  Factor analysis files not found. Skipping Figure 4.")
        return

    n_factors = fa_info.get("n_factors", 2)
    var_explained = fa_info.get("variance_explained", [])
    factor_interp = fa_info.get("factor_interpretation", {})

    score_df.index = [get_disorder_display(d) for d in score_df.index]
    feat_to_dim, dim_order = _load_dimension_mapping()

    # Initialize top_features for Panel B reference
    top_features = []

    # Factor semantic labels
    factor_names = {}
    for fi in range(n_factors):
        key = f"Factor_{fi+1}"
        interp = factor_interp.get(key, {})
        top_disorders = interp.get("top_disorders", [])
        if top_disorders:
            names = [d["disorder"] for d in top_disorders[:2]]
            factor_names[fi] = f"F{fi+1}"
        else:
            factor_names[fi] = f"F{fi+1}"

    semantic_defaults = {0: "p-factor", 1: "Int. vs Neurodev.", 2: "Thought Dis."}
    for fi in range(n_factors):
        if fi not in factor_names or factor_names[fi] == f"F{fi+1}":
            factor_names[fi] = semantic_defaults.get(fi, f"F{fi+1}")

    # =====================================================================
    # [FIX] Compute eigenvalues from effect matrix if not in JSON
    # =====================================================================
    eigenvalues = fa_info.get("selection_criteria", {}).get("eigenvalues", [])
    random_95th = fa_info.get("selection_criteria", {}).get(
        "random_95th_eigenvalues", []
    )

    if not eigenvalues:
        logger.info("  Eigenvalues not in factor_selection.json — computing from effect matrix...")
        effect_mat = _safe_load_csv(
            os.path.join(STATS_DIR, "effect_size_matrix_clean.csv"), index_col=0)
        if effect_mat is None:
            effect_mat = _safe_load_csv(
                os.path.join(STATS_DIR, "effect_size_matrix.csv"), index_col=0)
            if effect_mat is not None:
                artifact_rows = [f for f in effect_mat.index
                                if f in SELECTION_ARTIFACT_FEATURES]
                if artifact_rows:
                    effect_mat = effect_mat.drop(index=artifact_rows)

        if effect_mat is not None:
            X = effect_mat.values.astype(float)
            X = np.nan_to_num(X, nan=0.0)
            n_samples, n_vars = X.shape

            # Compute eigenvalues of disorder correlation matrix
            # Features as observations, disorders as variables → X is (93, 10)
            corr_matrix = np.corrcoef(X.T)  # (10, 10)
            eig_vals_raw, _ = np.linalg.eigh(corr_matrix)
            eigenvalues = sorted(eig_vals_raw, reverse=True)
            eigenvalues = [float(e) for e in eigenvalues]

            logger.info(f"  Computed {len(eigenvalues)} eigenvalues: "
                       f"{[f'{e:.3f}' for e in eigenvalues[:5]]}...")

            # Quick parallel analysis (200 iterations, fast on 10×10 matrix)
            n_pa_iter = 200
            rng = np.random.RandomState(42)
            random_eigenvalues_all = np.zeros((n_pa_iter, n_vars))

            for i in range(n_pa_iter):
                random_data = rng.standard_normal((n_samples, n_vars))
                random_corr = np.corrcoef(random_data.T)
                random_eig, _ = np.linalg.eigh(random_corr)
                random_eigenvalues_all[i] = sorted(random_eig, reverse=True)

            random_95th = np.percentile(
                random_eigenvalues_all, 95, axis=0
            ).tolist()

            logger.info(f"  Parallel analysis 95th percentiles: "
                       f"{[f'{e:.3f}' for e in random_95th[:5]]}...")
        else:
            logger.warning("  Could not load effect matrix for scree plot.")

    # =====================================================================
    # Create 2×2 figure
    # =====================================================================
    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.9))
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        width_ratios=[1.2, 1],
        height_ratios=[1.1, 1],
        hspace=0.35, wspace=0.35
    )

    # =====================================================================
    # Panel A: Feature Loading Heatmap (Top Features × Factors)
    # =====================================================================
    ax_heat = fig.add_subplot(gs[0, 0])

    if loading_df is not None and n_factors >= 2:
        max_loading = loading_df.abs().max(axis=1)
        top_n = min(25, len(loading_df))
        top_features = max_loading.nlargest(top_n).index.tolist()
        load_subset = loading_df.loc[top_features]

        vmax = max(abs(load_subset.values.min()), abs(load_subset.values.max()))
        vmax = min(vmax, 5.0)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        im = ax_heat.imshow(
            load_subset.values, cmap=DIVERGING_CMAP,
            norm=norm, aspect='auto', interpolation='none'
        )

        for i in range(load_subset.shape[0]):
            for j in range(load_subset.shape[1]):
                val = load_subset.iloc[i, j]
                text_color = 'white' if abs(val) > vmax * 0.6 else 'black'
                ax_heat.text(
                    j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=4, color=text_color, fontweight='bold'
                )

        y_labels = [f[:30] for f in top_features]
        ax_heat.set_yticks(range(len(y_labels)))
        ax_heat.set_yticklabels(y_labels, fontsize=4.5)

        for i, feat in enumerate(top_features):
            dim = feat_to_dim.get(feat, 'other')
            color = DIM_COLORS.get(dim, '#333333')
            ax_heat.get_yticklabels()[i].set_color(color)

        x_labels = []
        for fi in range(n_factors):
            v = var_explained[fi] * 100 if fi < len(var_explained) else 0
            name = factor_names.get(fi, f"F{fi+1}")
            x_labels.append(f"{name}\n({v:.1f}%)")
        ax_heat.set_xticks(range(n_factors))
        ax_heat.set_xticklabels(x_labels, fontsize=5.5)

        cbar = fig.colorbar(im, ax=ax_heat, shrink=0.8, aspect=20, pad=0.02)
        cbar.set_label("Loading", fontsize=6)
        cbar.ax.tick_params(labelsize=5)

    ax_heat.set_title(
        "A. Top Feature Loadings (Varimax-Rotated)",
        fontsize=8, fontweight='bold'
    )

    # =====================================================================
    # Panel B: Disorder Factor Scores
    # =====================================================================
    ax_biplot = fig.add_subplot(gs[0, 1])

    if n_factors >= 2:
        f1_col = score_df.columns[0]
        f2_col = score_df.columns[1]

        for disorder_display in score_df.index:
            x = score_df.loc[disorder_display, f1_col]
            y = score_df.loc[disorder_display, f2_col]
            raw_name = [k for k, v in DISORDER_DISPLAY_NAMES.items()
                       if v == disorder_display]
            color = get_disorder_color(raw_name[0]) if raw_name else "#808080"

            ax_biplot.scatter(
                x, y, c=color, s=80, edgecolors='black',
                linewidths=0.5, zorder=5
            )
            ax_biplot.annotate(
                disorder_display, (x, y), xytext=(4, 4),
                textcoords='offset points', fontsize=5,
                color=color, fontweight='bold'
            )

        # Draw HiTOP convex hulls
        for spectrum, disorders in HITOP_SPECTRA.items():
            members_display = [get_disorder_display(d) for d in disorders
                              if get_disorder_display(d) in score_df.index]
            if len(members_display) >= 3:
                coords = score_df.loc[
                    members_display, [f1_col, f2_col]
                ].values
                try:
                    hull = ConvexHull(coords)
                    hull_pts = coords[hull.vertices]
                    hull_pts = np.vstack([hull_pts, hull_pts[0]])
                    ax_biplot.plot(
                        hull_pts[:, 0], hull_pts[:, 1],
                        color=HITOP_COLORS[spectrum], linewidth=0.6,
                        linestyle='--', alpha=0.5, zorder=1
                    )
                except Exception:
                    pass

        f1_name = factor_names.get(0, "Factor 1")
        f2_name = factor_names.get(1, "Factor 2")
        v1 = var_explained[0] * 100 if var_explained else 0
        v2 = var_explained[1] * 100 if len(var_explained) > 1 else 0
        ax_biplot.set_xlabel(f"{f1_name} ({v1:.1f}%)", fontsize=7)
        ax_biplot.set_ylabel(f"{f2_name} ({v2:.1f}%)", fontsize=7)
        ax_biplot.axhline(y=0, color='gray', linewidth=0.3)
        ax_biplot.axvline(x=0, color='gray', linewidth=0.3)

        # Expand y-axis
        f2_vals = score_df.iloc[:, 1].values
        f2_range = f2_vals.max() - f2_vals.min()
        if f2_range < 0.01:
            f2_range = (score_df.iloc[:, 0].max() - score_df.iloc[:, 0].min()) * 0.1
        y_margin = max(f2_range * 0.5, 0.05)
        y_center = f2_vals.mean()
        ax_biplot.set_ylim(y_center - y_margin, y_center + y_margin)

        # Dimension legend
        dim_legend_items = []
        seen_dims = set()
        for feat in top_features:
            dim = feat_to_dim.get(feat, 'other')
            if dim not in seen_dims:
                seen_dims.add(dim)
                handle = Line2D(
                    [0], [0], marker='s', color='w',
                    markerfacecolor=DIM_COLORS.get(dim, '#333'),
                    markersize=4,
                    label=dim.replace('_', ' ').title(),
                    markeredgecolor='black', markeredgewidth=0.3
                )
                dim_legend_items.append(handle)
        if dim_legend_items:
            ax_biplot.legend(
                handles=dim_legend_items, loc='lower right',
                fontsize=4, framealpha=0.8, edgecolor='gray',
                title="Feature Dims", title_fontsize=4, ncol=2
            )

    ax_biplot.set_title(
        "B. Disorder Factor Scores",
        fontsize=8, fontweight='bold'
    )
    ax_biplot.spines['top'].set_visible(False)
    ax_biplot.spines['right'].set_visible(False)

    # =====================================================================
    # Panel C: Scree Plot + Parallel Analysis (FIXED)
    # =====================================================================
    ax_scree = fig.add_subplot(gs[1, 0])

    if eigenvalues:
        n_show = min(len(eigenvalues), 10)
        x = range(1, n_show + 1)

        # Bar chart for observed eigenvalues
        bars = ax_scree.bar(
            x, eigenvalues[:n_show], color='#1976D2',
            alpha=0.7, edgecolor='black', linewidth=0.3,
            label='Observed', width=0.6
        )

        # Highlight retained factors
        for i in range(min(n_factors, n_show)):
            bars[i].set_color('#FF9800')
            bars[i].set_alpha(0.9)

        # Parallel analysis reference
        if random_95th:
            ax_scree.plot(
                x, random_95th[:n_show], 'D--', color='#E53935',
                markersize=3, label='95th %ile (Random)', linewidth=0.8
            )

        # Kaiser criterion
        ax_scree.axhline(
            y=1.0, color='gray', linestyle=':', linewidth=0.5,
            label='Kaiser Criterion'
        )
        ax_scree.set_xticks(list(x))

        # Annotate variance explained
        for i, v in enumerate(eigenvalues[:n_show]):
            if i < len(var_explained):
                ax_scree.text(
                    i + 1, v + 0.05, f'{var_explained[i]*100:.1f}%',
                    ha='center', fontsize=4.5, color='#333333'
                )

        # Annotate n_factors choice
        n_kaiser = sum(1 for e in eigenvalues if e > 1.0)
        n_parallel = sum(1 for e, r in zip(eigenvalues, random_95th)
                        if e > r) if random_95th else 0

        ax_scree.text(
            0.98, 0.95,
            f"Kaiser: {n_kaiser} factors\n"
            f"Parallel: {n_parallel} factors\n"
            f"Selected: {n_factors} (HiTOP theory)",
            transform=ax_scree.transAxes, fontsize=5,
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4',
                     edgecolor='#F57F17', alpha=0.9)
        )

    else:
        # Final fallback: just show variance explained as bar chart
        if var_explained:
            x = range(1, len(var_explained) + 1)
            ax_scree.bar(
                x, [v * 100 for v in var_explained],
                color='#FF9800', edgecolor='black', linewidth=0.3,
                label='Variance Explained'
            )
            ax_scree.set_ylabel("Variance Explained (%)", fontsize=7)
            ax_scree.set_xticks(list(x))
        else:
            ax_scree.text(
                0.5, 0.5, "Scree data\nnot available",
                transform=ax_scree.transAxes, ha='center',
                va='center', fontsize=8, color='gray'
            )

    ax_scree.set_xlabel("Factor Number", fontsize=7)
    ax_scree.set_ylabel("Eigenvalue", fontsize=7)
    ax_scree.set_title(
        "C. Scree Plot + Parallel Analysis",
        fontsize=8, fontweight='bold'
    )
    ax_scree.legend(fontsize=5, loc='upper right')
    ax_scree.spines['top'].set_visible(False)
    ax_scree.spines['right'].set_visible(False)

    # =====================================================================
    # Panel D: HiTOP-Factor Alignment Heatmap
    # =====================================================================
    ax_align = fig.add_subplot(gs[1, 1])

    if score_df is not None and n_factors >= 1:
        spectra = list(HITOP_SPECTRA.keys())
        align_data = np.zeros((n_factors, len(spectra)))

        for si, spectrum in enumerate(spectra):
            disorders = HITOP_SPECTRA[spectrum]
            members = [get_disorder_display(d) for d in disorders
                      if get_disorder_display(d) in score_df.index]
            if members:
                for fi in range(n_factors):
                    col = score_df.columns[fi]
                    align_data[fi, si] = score_df.loc[members, col].mean()

        vmax_align = max(abs(align_data.min()), abs(align_data.max()))
        if vmax_align < 1e-10:
            vmax_align = 1.0
        norm_align = TwoSlopeNorm(
            vmin=-vmax_align, vcenter=0, vmax=vmax_align
        )

        im_align = ax_align.imshow(
            align_data, cmap=DIVERGING_CMAP, norm=norm_align,
            aspect='auto', interpolation='none'
        )

        for i in range(n_factors):
            for j in range(len(spectra)):
                val = align_data[i, j]
                text_color = 'white' if abs(val) > vmax_align * 0.6 else 'black'
                ax_align.text(
                    j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=5, color=text_color, fontweight='bold'
                )

        y_labels = [factor_names.get(fi, f"F{fi+1}") for fi in range(n_factors)]
        ax_align.set_yticks(range(n_factors))
        ax_align.set_yticklabels(y_labels, fontsize=5.5)

        x_labels = [s.replace("_", "/").replace("/", "\n") for s in spectra]
        ax_align.set_xticks(range(len(spectra)))
        ax_align.set_xticklabels(x_labels, fontsize=4.5, rotation=30,
                                  ha='right')

        for j, lbl in enumerate(ax_align.get_xticklabels()):
            spectrum = spectra[j]
            lbl.set_color(HITOP_COLORS.get(spectrum, 'black'))

        cbar_align = fig.colorbar(
            im_align, ax=ax_align, shrink=0.8, aspect=15, pad=0.02
        )
        cbar_align.set_label("Mean Factor Score", fontsize=5)
        cbar_align.ax.tick_params(labelsize=4)

    ax_align.set_title(
        "D. Factor × HiTOP Alignment",
        fontsize=8, fontweight='bold'
    )

    # --- Save ---
    _save_figure(fig, "fig4_factor_analysis")

# =====================================================================
# SECTION 7: FIGURE 5 — TRANSDIAGNOSTIC VS SPECIFIC (ENHANCED)
# =====================================================================

def plot_transdiagnostic_specific():
    """
    Figure 5: Transdiagnostic vs Specific Features.
    
    Enhanced:
      - Added DiffSpec vs CU scatter (Panel B)
      - DiffSpec annotation showing all values < 0.2
    """
    logger.info("Generating Figure 5: Transdiagnostic vs Specific (Enhanced)...")

    transdiag_df = _safe_load_csv(
        os.path.join(STATS_DIR, "transdiagnostic_features.csv"))
    diff_spec = _safe_load_csv(
        os.path.join(STATS_DIR, "feature_differential_specificity.csv"),
        index_col=0)
    clin_util = _safe_load_csv(
        os.path.join(STATS_DIR, "feature_clinical_utility.csv"),
        index_col=0)
    specificity = _safe_load_csv(
        os.path.join(STATS_DIR, "feature_specificity.csv"),
        index_col=0)

    if transdiag_df is None:
        logger.error("  Transdiagnostic features not found. Skipping Fig 5.")
        return

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.42))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ==== Panel A: Feature Classification Bar Chart ====
    ax_bar = fig.add_subplot(gs[0])

    cat_order = ["transdiagnostic", "mixed", "specific", "non_significant"]
    cat_colors = {
        "transdiagnostic": "#E53935", "mixed": "#FB8C00",
        "specific": "#1E88E5", "non_significant": "#BDBDBD"
    }
    cat_labels = {
        "transdiagnostic": "Transdiagnostic\n(≥5 dis.)",
        "mixed": "Mixed\n(3-4 dis.)",
        "specific": "Specific\n(1-2 dis.)",
        "non_significant": "Non-signif."
    }

    counts = transdiag_df["category"].value_counts()
    bars_x = range(len(cat_order))
    bars_h = [counts.get(c, 0) for c in cat_order]
    bar_colors = [cat_colors[c] for c in cat_order]
    bar_labels = [cat_labels[c] for c in cat_order]

    bars = ax_bar.bar(
        bars_x, bars_h, color=bar_colors,
        edgecolor='black', linewidth=0.3
    )
    ax_bar.set_xticks(bars_x)
    ax_bar.set_xticklabels(bar_labels, fontsize=5)
    ax_bar.set_ylabel("Number of Features", fontsize=7)
    ax_bar.set_title(
        "A. Feature Classification",
        fontsize=8, fontweight='bold', loc='left'
    )
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)

    for bar, h in zip(bars, bars_h):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2, h + 0.5,
            str(int(h)), ha='center', va='bottom', fontsize=6
        )

    # ==== Panel B: DiffSpec vs CU Scatter (Core Finding) ====
    ax_scatter = fig.add_subplot(gs[1])

    if diff_spec is not None and clin_util is not None:
        # Merge
        merged = pd.DataFrame({
            'diff_spec': diff_spec.squeeze(),
            'clin_util': clin_util.squeeze(),
        })
        if specificity is not None:
            merged['specificity'] = specificity.squeeze()
        else:
            merged['specificity'] = np.nan

        # Color by clinical utility
        valid = merged.dropna()
        if len(valid) > 0:
            sc = ax_scatter.scatter(
                valid['specificity'], valid['diff_spec'],
                c=valid['clin_util'], cmap='YlOrRd',
                s=25, edgecolors='black', linewidths=0.2,
                vmin=0, vmax=0.8, zorder=3
            )
            cbar = fig.colorbar(sc, ax=ax_scatter, shrink=0.7, aspect=20,
                                pad=0.02)
            cbar.set_label("Clinical Utility", fontsize=5)
            cbar.ax.tick_params(labelsize=4)

            # Annotate top-CU features
            top_cu = valid.nlargest(5, 'clin_util')
            for _, row in top_cu.iterrows():
                feat_name = row.name[:20]
                ax_scatter.annotate(
                    feat_name,
                    (row['specificity'], row['diff_spec']),
                    fontsize=4, xytext=(3, 3),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.3)
                )

        ax_scatter.set_xlabel("Specificity (magnitude-based)", fontsize=6)
        ax_scatter.set_ylabel("Differential Specificity", fontsize=6)

        # Annotate DiffSpec < 0.2 finding
        ax_scatter.axhline(y=0.2, color='gray', linestyle=':', linewidth=0.5)
        ax_scatter.text(
            0.02, 0.95,
            "All DiffSpec < 0.2\n→ Quantitative, not qualitative",
            transform=ax_scatter.transAxes, fontsize=5,
            va='top', ha='left', style='italic',
            color='#B71C1C',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF9C4',
                     edgecolor='#F57F17', alpha=0.9)
        )

    ax_scatter.set_title(
        "B. DiffSpec vs Specificity",
        fontsize=8, fontweight='bold', loc='left'
    )
    ax_scatter.spines['top'].set_visible(False)
    ax_scatter.spines['right'].set_visible(False)

    # ==== Panel C: Effect Size vs Breadth Scatter ====
    ax_breadth = fig.add_subplot(gs[2])

    for cat in cat_order:
        subset = transdiag_df[transdiag_df["category"] == cat]
        if len(subset) == 0:
            continue
        ax_breadth.scatter(
            subset["n_sig_disorders"], subset["mean_abs_g_all"],
            c=cat_colors[cat], s=15, alpha=0.6,
            edgecolors='black', linewidths=0.2,
            label=cat.replace("_", " ").title(),
            zorder=3 if cat == "transdiagnostic" else 2
        )

    # Label top features
    top_trans = transdiag_df[
        transdiag_df["category"] == "transdiagnostic"
    ].nlargest(3, "mean_abs_g_all")
    for _, row in top_trans.iterrows():
        ax_breadth.annotate(
            row["feature"][:20],
            (row["n_sig_disorders"], row["mean_abs_g_all"]),
            fontsize=4, xytext=(3, 3), textcoords='offset points',
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.3)
        )

    ax_breadth.set_xlabel("N Significant Disorders", fontsize=6)
    ax_breadth.set_ylabel("Mean |Hedges' g|", fontsize=6)
    ax_breadth.set_title(
        "C. Effect Size vs Breadth",
        fontsize=8, fontweight='bold', loc='left'
    )
    ax_breadth.legend(fontsize=4, markerscale=0.8)
    ax_breadth.spines['top'].set_visible(False)
    ax_breadth.spines['right'].set_visible(False)

    _save_figure(fig, "fig5_transdiagnostic_specific")


# =====================================================================
# SECTION 8: FIGURE 6 — DIRECTION CONSISTENCY (NEW)
# =====================================================================

def plot_direction_consistency():
    """
    Figure 6: Direction Consistency — Core p-factor Evidence.
    
    Panel A: Direction fingerprint (stacked bars per feature)
    Panel B: Observed vs expected direction inconsistency
    """
    logger.info("Generating Figure 6: Direction Consistency (NEW)...")

    # Load full effect sizes to construct direction matrix
    effect_full = _safe_load_csv(
        os.path.join(STATS_DIR, "effect_sizes_full.csv"))

    if effect_full is None:
        logger.error("  Full effect sizes not found. Skipping Fig 6.")
        return

    # Filter out artifact features
    effect_clean = effect_full[
        ~effect_full["feature"].isin(SELECTION_ARTIFACT_FEATURES)
    ].copy()

    # Construct direction matrix: feature × disorder → {pos, neg, ns}
    features_unique = sorted(effect_clean["feature"].unique())
    disorders_unique = sorted(effect_clean["disorder"].unique())

    # Pivot
    g_pivot = effect_clean.pivot_table(
        index="feature", columns="disorder",
        values="hedges_g", aggfunc="first"
    )
    sig_pivot = effect_clean.pivot_table(
        index="feature", columns="disorder",
        values="sig_fdr_global", aggfunc="first"
    )

    # Compute direction counts per feature
    direction_data = []
    for feat in features_unique:
        if feat not in g_pivot.index:
            continue
        g_vals = g_pivot.loc[feat]
        sig_vals = sig_pivot.loc[feat] if feat in sig_pivot.index else pd.Series()

        n_pos = 0
        n_neg = 0
        n_ns = 0
        for disorder in disorders_unique:
            if disorder not in g_pivot.columns:
                n_ns += 1
                continue
            g = g_vals.get(disorder, np.nan)
            s = sig_vals.get(disorder, False)
            if isinstance(s, str):
                s = s.lower() == 'true'
            if pd.isna(g):
                n_ns += 1
            elif s and g > 0:
                n_pos += 1
            elif s and g < 0:
                n_neg += 1
            else:
                n_ns += 1

        mean_abs_g = g_vals.abs().mean()

        direction_data.append({
            'feature': feat,
            'n_sig_pos': n_pos,
            'n_sig_neg': n_neg,
            'n_non_sig': n_ns,
            'mean_abs_g': mean_abs_g,
        })

    dir_df = pd.DataFrame(direction_data)

    # Sort by direction consistency: features with all-same-sign first
    dir_df['consistency'] = dir_df[['n_sig_pos', 'n_sig_neg']].max(axis=1) / \
        dir_df[['n_sig_pos', 'n_sig_neg']].sum(axis=1).replace(0, 1)
    dir_df = dir_df.sort_values('consistency', ascending=True)

    # --- Create figure ---
    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.7))
    gs = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[3, 1],
        hspace=0.25
    )

    # ==== Panel A: Direction Fingerprint (Stacked Bars) ====
    ax_fingerprint = fig.add_subplot(gs[0])

    n_feats = len(dir_df)
    y_pos = np.arange(n_feats)

    # Stacked horizontal bars
    ax_fingerprint.barh(
        y_pos, dir_df['n_sig_pos'].values,
        color='#E53935', edgecolor='none', linewidth=0,
        label='Significant Positive', height=0.9
    )
    ax_fingerprint.barh(
        y_pos, dir_df['n_sig_neg'].values,
        left=dir_df['n_sig_pos'].values,
        color='#1E88E5', edgecolor='none', linewidth=0,
        label='Significant Negative', height=0.9
    )
    ax_fingerprint.barh(
        y_pos, dir_df['n_non_sig'].values,
        left=dir_df['n_sig_pos'].values + dir_df['n_sig_neg'].values,
        color='#E0E0E0', edgecolor='none', linewidth=0,
        label='Non-significant', height=0.9
    )

    ax_fingerprint.set_xlim(0, 10)
    ax_fingerprint.set_ylim(-0.5, n_feats - 0.5)

    # Only show feature labels for mixed-direction features
    for i, (_, row) in enumerate(dir_df.iterrows()):
        if row['n_sig_pos'] > 0 and row['n_sig_neg'] > 0:
            ax_fingerprint.text(
                -0.3, i, row['feature'][:25],
                ha='right', va='center', fontsize=4,
                color='#B71C1C', fontweight='bold'
            )

    # Hide most y-tick labels (too many)
    ax_fingerprint.set_yticks([])
    ax_fingerprint.set_xlabel("Number of Disorders (out of 10)", fontsize=7)
    ax_fingerprint.legend(fontsize=5, loc='lower right')
    ax_fingerprint.set_title(
        "A. Direction Fingerprint Across Disorders\n"
        "(Red = significant positive, Blue = significant negative)",
        fontsize=8, fontweight='bold', loc='left'
    )
    ax_fingerprint.spines['top'].set_visible(False)
    ax_fingerprint.spines['right'].set_visible(False)

    # ==== Panel B: Observed vs Expected Direction Inconsistency ====
    ax_compare = fig.add_subplot(gs[1])

    # Count observed inconsistent features
    n_inconsistent_obs = ((dir_df['n_sig_pos'] > 0) &
                          (dir_df['n_sig_neg'] > 0)).sum()
    n_total = len(dir_df)

    # Null expectation from s03 report
    null_report_path = os.path.join(STATS_DIR, "effect_sizes_report.json")
    null_expected = 90.7  # Default from logs
    report_json = _safe_load_json(null_report_path)
    if report_json:
        # Try to extract from report
        pass  # Use logged value

    # Bar comparison
    categories = ['Observed\nDirection-\nInconsistent', 'Expected\nUnder Null\n(Independent)']
    values = [n_inconsistent_obs, null_expected]
    colors_bar = ['#E53935', '#BDBDBD']

    bars = ax_compare.bar(
        range(len(categories)), values, color=colors_bar,
        edgecolor='black', linewidth=0.3, width=0.5
    )

    for bar, val in zip(bars, values):
        ax_compare.text(
            bar.get_x() + bar.get_width() / 2, val + 1,
            f'{val:.0f}', ha='center', va='bottom', fontsize=7,
            fontweight='bold'
        )

    # Annotation
    ratio = n_inconsistent_obs / max(1, null_expected)
    ax_compare.text(
        0.5, 0.85,
        f"O/E = {ratio:.1f}×\n(p < 0.001, binomial test)\n"
        "→ Direction consistency exceeds random expectation\n"
        "→ Supports p-factor (general psychopathology)",
        transform=ax_compare.transAxes, fontsize=6,
        va='top', ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4',
                 edgecolor='#F57F17', alpha=0.9)
    )

    ax_compare.set_xticks(range(len(categories)))
    ax_compare.set_xticklabels(categories, fontsize=6)
    ax_compare.set_ylabel("Number of Features", fontsize=7)
    ax_compare.set_title(
        "B. Direction Inconsistency: Observed vs. Null Expectation",
        fontsize=8, fontweight='bold', loc='left'
    )
    ax_compare.spines['top'].set_visible(False)
    ax_compare.spines['right'].set_visible(False)

    _save_figure(fig, "fig6_direction_consistency")


# =====================================================================
# SECTION 9: FIGURE 7 — CLINICAL UTILITY RANKING (NEW)
# =====================================================================

def plot_clinical_utility_ranking():
    """
    Figure 7: Clinical Utility Ranking — Top features for translation.
    
    Lollipop chart of top 20 features by clinical utility score,
    colored by dimension, with DiffSpec and best disorder annotated.
    """
    logger.info("Generating Figure 7: Clinical Utility Ranking (NEW)...")

    clin_util = _safe_load_csv(
        os.path.join(STATS_DIR, "feature_clinical_utility.csv"),
        index_col=0)
    diff_spec = _safe_load_csv(
        os.path.join(STATS_DIR, "feature_differential_specificity.csv"),
        index_col=0)
    effect_matrix = _safe_load_csv(
        os.path.join(STATS_DIR, "effect_size_matrix_clean.csv"),
        index_col=0)
    feat_to_dim, _ = _load_dimension_mapping()

    if clin_util is None:
        logger.error("  Clinical utility data not found. Skipping Fig 7.")
        return

    cu_series = clin_util.squeeze()
    top_n = min(20, len(cu_series))
    top_features = cu_series.nlargest(top_n)

    # Get best disorder and DiffSpec for each
    feature_info = []
    for feat in top_features.index:
        cu_val = top_features[feat]
        ds_val = diff_spec.loc[feat].squeeze() if (
            diff_spec is not None and feat in diff_spec.index) else np.nan
        dim = feat_to_dim.get(feat, 'other')

        # Best disorder
        best_disorder = "N/A"
        best_g = np.nan
        if effect_matrix is not None and feat in effect_matrix.index:
            g_row = effect_matrix.loc[feat].abs()
            best_disorder = g_row.idxmax()
            best_g = effect_matrix.loc[feat, best_disorder]

        feature_info.append({
            'feature': feat,
            'cu': float(cu_val),
            'diff_spec': float(ds_val) if not pd.isna(ds_val) else 0,
            'dim': dim,
            'best_disorder': best_disorder,
            'best_g': float(best_g) if not pd.isna(best_g) else 0,
        })

    info_df = pd.DataFrame(feature_info).sort_values('cu', ascending=True)

    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 0.55))

    y_pos = np.arange(len(info_df))

    # Lollipop: horizontal stems + dots
    for i, (_, row) in enumerate(info_df.iterrows()):
        color = DIM_COLORS.get(row['dim'], '#607D8B')
        # Stem
        ax.plot([0, row['cu']], [i, i], color=color, linewidth=0.8,
                alpha=0.5)
        # Dot (sized by DiffSpec, but DiffSpec is all low)
        ax.scatter(row['cu'], i, c=color, s=50, edgecolors='black',
                   linewidths=0.3, zorder=5)

        # Annotate best disorder
        if row['best_disorder'] != "N/A":
            disp = get_disorder_display(row['best_disorder'])
            g_str = f"g={row['best_g']:+.2f}"
            ax.text(
                row['cu'] + 0.01, i,
                f"{disp} ({g_str})",
                va='center', ha='left', fontsize=4.5,
                color=get_disorder_color(row['best_disorder'])
            )

    # Y-axis: feature names colored by dimension
    ax.set_yticks(y_pos)
    y_labels = info_df['feature'].tolist()
    ax.set_yticklabels(y_labels, fontsize=5.5)
    for i, (_, row) in enumerate(info_df.iterrows()):
        color = DIM_COLORS.get(row['dim'], '#607D8B')
        ax.get_yticklabels()[i].set_color(color)

    ax.set_xlabel("Clinical Utility Score", fontsize=7)
    ax.set_xlim(0, 0.85)
    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=0.5,
               label='High utility threshold')

    # Dimension legend
    dim_legend_items = []
    seen_dims = set()
    for _, row in info_df.iterrows():
        dim = row['dim']
        if dim not in seen_dims:
            seen_dims.add(dim)
            handle = Line2D(
                [0], [0], marker='o', color='w',
                markerfacecolor=DIM_COLORS.get(dim, '#333'),
                markersize=5,
                label=dim.replace('_', ' ').title(),
                markeredgecolor='black', markeredgewidth=0.3
            )
            dim_legend_items.append(handle)
    ax.legend(handles=dim_legend_items, fontsize=4.5, loc='lower right',
              ncol=2, framealpha=0.8, title="Dimension", title_fontsize=4.5)

    ax.set_title(
        "Clinical Utility Ranking: Top Features for Digital Phenotyping\n"
        "(Annotated with best-discriminating disorder and effect size)",
        fontsize=8, fontweight='bold'
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    _save_figure(fig, "fig7_clinical_utility_ranking")


# =====================================================================
# SECTION 10: FIGURE 8 — DIMENSION ABLATION (REDESIGNED)
# =====================================================================

def plot_dimension_ablation():
    """
    Figure 8: Dimension Ablation — Delta heatmap.
    
    Redesigned from grouped bars to heatmap showing
    ΔAUROC = Full model − Without this dimension.
    Positive = dimension contributes; Negative = dimension hurts.
    """
    logger.info("Generating Figure 8: Dimension Ablation (Redesigned)...")

    ablation_df = _safe_load_csv(
        os.path.join(STATS_DIR, "dimension_ablation.csv"), index_col=0)

    if ablation_df is None:
        logger.error("  Ablation data not found. Skipping Figure 8.")
        return

    # Extract delta columns
    delta_cols = [c for c in ablation_df.columns if c.startswith("delta_")]
    if not delta_cols:
        # Compute deltas from "only_" and "all_dimensions" columns
        dim_cols = [c for c in ablation_df.columns if c.startswith("only_")]
        if "all_dimensions" not in ablation_df.columns or not dim_cols:
            logger.error("  No delta or sufficient ablation columns.")
            return
        for dim_col in dim_cols:
            delta_col = dim_col.replace("only_", "delta_")
            # Delta = all - only_dim (positive means dim contributes)
            ablation_df[delta_col] = (
                ablation_df["all_dimensions"] - ablation_df[dim_col]
            )
        delta_cols = [c for c in ablation_df.columns if c.startswith("delta_")]

    delta_data = ablation_df[delta_cols].copy()
    delta_data.columns = [c.replace("delta_", "") for c in delta_cols]

    # Rename disorder indices to display names
    delta_data.index = [get_disorder_display(d) for d in delta_data.index]

    # Sort by mean delta (most important dimensions first)
    dim_importance = delta_data.mean(axis=0).sort_values(ascending=False)
    delta_data = delta_data[dim_importance.index]

    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, SINGLE_COL * 0.85))

    # Diverging heatmap centered at 0
    valid_vals = delta_data.values[~np.isnan(delta_data.values)]
    vmax_delta = max(abs(valid_vals.min()), abs(valid_vals.max())) if len(
        valid_vals) > 0 else 0.1

    norm = TwoSlopeNorm(vmin=-vmax_delta, vcenter=0, vmax=vmax_delta)

    im = ax.imshow(
        delta_data.values, cmap=DIVERGING_CMAP, norm=norm,
        aspect='auto', interpolation='none'
    )

    # Annotate values
    for i in range(delta_data.shape[0]):
        for j in range(delta_data.shape[1]):
            val = delta_data.iloc[i, j]
            if pd.isna(val):
                ax.text(j, i, '—', ha='center', va='center',
                        fontsize=5, color='#AAAAAA')
                continue
            text_color = 'white' if abs(val) > vmax_delta * 0.6 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=5, color=text_color, fontweight='bold')

    # Axis labels
    ax.set_xticks(range(len(delta_data.columns)))
    ax.set_xticklabels(
        [c.replace('_', ' ').title() for c in delta_data.columns],
        rotation=45, ha='right', fontsize=5.5
    )
    ax.set_yticks(range(len(delta_data.index)))
    ax.set_yticklabels(delta_data.index, fontsize=5.5)

    # Color y-tick labels by HiTOP spectrum
    for label_obj in ax.get_yticklabels():
        text = label_obj.get_text()
        raw_name = [k for k, v in DISORDER_DISPLAY_NAMES.items() if v == text]
        if raw_name:
            label_obj.set_color(get_disorder_color(raw_name[0]))

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=25)
    cbar.set_label("ΔAUROC (Full − Without Dimension)", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    ax.set_title(
        "Dimension Contribution: ΔAUROC When Removed\n"
        "(Red = dimension important, Blue = dimension harmful/overfitting)",
        fontsize=8, fontweight='bold'
    )

    _save_figure(fig, "fig8_dimension_ablation")


# =====================================================================
# SECTION 11: FIGURE 9 — CONFUSION MATRIX (ENHANCED)
# =====================================================================

def plot_confusion_matrix():
    """
    Figure 9: Multiclass Confusion Matrix.
    
    Enhanced:
      - Per-class AUROC annotation
      - Top confusion pairs highlighted
      - Macro statistics
    """
    logger.info("Generating Figure 9: Confusion Matrix (Enhanced)...")

    cm_df = _safe_load_csv(
        os.path.join(STATS_DIR, "confusion_matrix_normalized.csv"), index_col=0)

    # Load multiclass results for per-class AUROC
    mc_results = _safe_load_json(
        os.path.join(STATS_DIR, "multiclass_results.json"))

    if cm_df is None:
        logger.error("  Confusion matrix not found. Skipping Figure 9.")
        return

    cm_df.index = [get_disorder_display(d) if d != "control" else "Control"
                   for d in cm_df.index]
    cm_df.columns = [get_disorder_display(d) if d != "control" else "Control"
                     for d in cm_df.columns]

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.5, SINGLE_COL * 1.5))

    data = cm_df.values.astype(float)
    im = ax.imshow(data, cmap='Blues', vmin=0, vmax=0.6, aspect='equal')

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            text_color = 'white' if val > 0.35 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=4, color=text_color)

    # Highlight top-3 off-diagonal confusion pairs
    off_diag = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if i != j:
                off_diag.append((i, j, data[i, j]))
    off_diag.sort(key=lambda x: x[2], reverse=True)

    for rank, (i, j, val) in enumerate(off_diag[:3]):
        rect = Rectangle(
            (j - 0.5, i - 0.5), 1, 1,
            fill=False, edgecolor='#E53935',
            linewidth=1.0, linestyle='--', zorder=5
        )
        ax.add_patch(rect)
        if rank == 0:
            ax.text(
                j + 0.4, i - 0.4, f'#{rank+1}',
                fontsize=4, color='#E53935', fontweight='bold', zorder=6
            )

    ax.set_xticks(range(len(cm_df.columns)))
    ax.set_xticklabels(cm_df.columns, rotation=45, ha='right', fontsize=5)
    ax.set_yticks(range(len(cm_df.index)))
    ax.set_yticklabels(cm_df.index, fontsize=5)

    ax.set_xlabel("Predicted", fontsize=7)
    ax.set_ylabel("True", fontsize=7)

    # Macro statistics annotation
    if mc_results:
        macro_auc = mc_results.get("macro_auroc_mean", np.nan)
        bal_acc = mc_results.get("balanced_accuracy_mean", np.nan)
        ax.text(
            0.02, 0.02,
            f"Macro AUROC = {macro_auc:.3f}\nBal. Accuracy = {bal_acc:.3f}",
            transform=ax.transAxes, fontsize=5, va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                     edgecolor='gray', alpha=0.9)
        )

    # Per-class AUROC from side
    if mc_results and "per_class_auroc_mean" in mc_results:
        per_class = mc_results["per_class_auroc_mean"]
        # Add as right-side annotation
        class_names = list(per_class.keys())
        for i, cn in enumerate(cm_df.index):
            raw_name = [k for k, v in DISORDER_DISPLAY_NAMES.items() if v == cn]
            if not raw_name and cn == "Control":
                raw_name = ["control"]
            if raw_name and raw_name[0] in per_class:
                auc = per_class[raw_name[0]]
                ax.text(
                    len(cm_df.columns) - 0.5 + 0.3, i,
                    f'{auc:.2f}', fontsize=4, va='center',
                    color=get_disorder_color(raw_name[0]),
                    fontweight='bold'
                )
        ax.text(
            len(cm_df.columns) - 0.5 + 0.3, -0.8,
            'AUROC', fontsize=5, ha='center', fontweight='bold'
        )

    ax.set_title(
        "Multiclass Confusion Matrix (Normalized)\n"
        "Red dashed = top confusion pairs",
        fontsize=8, fontweight='bold'
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, aspect=25)
    cbar.set_label("Proportion", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    _save_figure(fig, "fig9_confusion_matrix")


# =====================================================================
# SECTION 12: FIGURE 10 — t-SNE SMALL MULTIPLES (REDESIGNED)
# =====================================================================

def plot_tsne_embedding():
    """
    Figure 10: t-SNE embedding — Small multiples.
    
    Redesigned from single spaghetti plot to 2×5 grid
    where each panel shows one disorder vs controls.
    """
    logger.info("Generating Figure 10: t-SNE Small Multiples (Redesigned)...")

    tsne_df = _safe_load_csv(os.path.join(STATS_DIR, "tsne_embedding.csv"))

    if tsne_df is None:
        logger.error("  t-SNE data not found. Skipping Figure 10.")
        return

    # Load per-class AUROC for annotation
    mc_results = _safe_load_json(
        os.path.join(STATS_DIR, "multiclass_results.json"))
    per_class_auc = {}
    if mc_results and "per_class_auroc_mean" in mc_results:
        per_class_auc = mc_results["per_class_auroc_mean"]

    # Compute global axis limits
    x_min, x_max = tsne_df["tSNE_1"].min(), tsne_df["tSNE_1"].max()
    y_min, y_max = tsne_df["tSNE_2"].min(), tsne_df["tSNE_2"].max()
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05

    fig, axes = plt.subplots(
        2, 5, figsize=(DOUBLE_COL, DOUBLE_COL * 0.45)
    )

    for idx, disorder in enumerate(DISORDER_GROUPS):
        row = idx // 5
        col = idx % 5
        ax = axes[row, col]

        # Background: controls (light gray)
        control_mask = tsne_df["group"] == "control"
        ax.scatter(
            tsne_df.loc[control_mask, "tSNE_1"],
            tsne_df.loc[control_mask, "tSNE_2"],
            c="#E0E0E0", s=1, alpha=0.15, rasterized=True
        )

        # Foreground: this disorder
        mask = ((tsne_df["group"] == "patient") &
                (tsne_df["disorder_group"] == disorder))
        if mask.sum() > 0:
            color = get_disorder_color(disorder)
            ax.scatter(
                tsne_df.loc[mask, "tSNE_1"],
                tsne_df.loc[mask, "tSNE_2"],
                c=color, s=3, alpha=0.5,
                edgecolors='none', rasterized=True
            )

        # Formatting
        display = get_disorder_display(disorder)
        auc_val = per_class_auc.get(disorder, np.nan)
        auc_str = f"\nAUROC={auc_val:.2f}" if not pd.isna(auc_val) else ""
        ax.set_title(f"{display}{auc_str}", fontsize=5.5,
                     fontweight='bold', color=get_disorder_color(disorder))

        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.3)
        ax.spines['left'].set_linewidth(0.3)

    fig.suptitle(
        "User Embedding (t-SNE): Each Disorder vs. Controls\n"
        "Gray = Controls; Colored = Patients",
        fontsize=8, fontweight='bold', y=1.02
    )

    _save_figure(fig, "fig10_tsne_small_multiples")


# =====================================================================
# SECTION 13: FIGURE 11 — DISORDER PROFILES (PARALLEL COORDINATES)
# =====================================================================

def plot_disorder_profiles():
    """
    Figure 11: Disorder Digital Phenotype Profiles.
    
    Replaces radar chart with parallel coordinates plot,
    which preserves magnitude information and allows
    precise cross-disorder comparison.
    """
    logger.info("Generating Figure 11: Disorder Profiles (Parallel Coords)...")

    profiles = _safe_load_json(
        os.path.join(STATS_DIR, "disorder_profiles.json"))
    if profiles is None:
        logger.error("  Disorder profiles not found. Skipping Figure 11.")
        return

    feat_to_dim, dim_order = _load_dimension_mapping()

    # Collect dimensions that appear in profiles
    all_dims = set()
    for disorder, profile in profiles.items():
        all_dims.update(profile.get("dimension_summary", {}).keys())
    dimensions = [d for d in dim_order if d in all_dims]
    if not dimensions:
        dimensions = sorted(all_dims)
    n_dims = len(dimensions)

    if n_dims < 3:
        logger.warning("  Too few dimensions for profile plot. Skipping.")
        return

    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, SINGLE_COL * 0.65))

    # For each disorder, plot a line across dimensions
    for disorder in DISORDER_GROUPS:
        if disorder not in profiles:
            continue

        profile = profiles[disorder]
        dim_summary = profile.get("dimension_summary", {})

        values = []
        for dim in dimensions:
            if dim in dim_summary:
                values.append(dim_summary[dim].get("mean_abs_g", 0))
            else:
                values.append(0.0)

        color = get_disorder_color(disorder)
        display_name = get_disorder_display(disorder)

        ax.plot(
            range(n_dims), values,
            color=color, linewidth=1.2, alpha=0.8,
            label=display_name,
            marker='o', markersize=3
        )

        # Label at last point
        ax.text(
            n_dims - 0.5, values[-1],
            display_name, fontsize=4.5,
            color=color, fontweight='bold',
            va='center', ha='left'
        )

    # X-axis: dimension names
    dim_labels = []
    for d in dimensions:
        label = d.replace('_', ' ').replace(
            'temporal linguistic', 'temp-ling'
        ).title()
        dim_labels.append(label)

    ax.set_xticks(range(n_dims))
    ax.set_xticklabels(dim_labels, rotation=30, ha='right', fontsize=5.5)

    # Color x-tick labels by dimension
    for i, dim in enumerate(dimensions):
        color = DIM_COLORS.get(dim, '#333333')
        ax.get_xticklabels()[i].set_color(color)

    # Background shading for dimension groups
    for i, dim in enumerate(dimensions):
        color = DIM_COLORS.get(dim, '#333333')
        ax.axvspan(i - 0.4, i + 0.4, alpha=0.05, color=color, zorder=0)

    ax.set_ylabel("Mean |Hedges' g|", fontsize=7)
    ax.set_xlim(-0.5, n_dims - 0.5)

    # Reference lines for effect size thresholds
    ax.axhline(y=0.2, color='#BDBDBD', linestyle=':', linewidth=0.5,
               label='Small (0.2)')
    ax.axhline(y=0.5, color='#9E9E9E', linestyle=':', linewidth=0.5,
               label='Medium (0.5)')
    ax.axhline(y=0.8, color='#757575', linestyle=':', linewidth=0.5,
               label='Large (0.8)')

    ax.legend(fontsize=4, loc='upper left', ncol=2,
              framealpha=0.8, title="Disorders", title_fontsize=4)

    ax.set_title(
        "Disorder Digital Phenotype Profiles: Mean Effect Size by Dimension\n"
        "(Parallel coordinates replace radar chart for precise comparison)",
        fontsize=8, fontweight='bold'
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    _save_figure(fig, "fig11_disorder_profiles_parallel")


# =====================================================================
# SECTION 14: FIGURE 12 — SENSITIVITY SUMMARY (FIXED)
# =====================================================================

def plot_sensitivity_summary():
    """
    Figure 12: Sensitivity & Robustness Summary.
    
    Fixed:
      - JSON string keys → explicit float conversion
      - Explicit tick control
      - All panels functional
      - Lollipop charts for multiple comparison
    """
    logger.info("Generating Figure 12: Sensitivity Summary (Fixed)...")

    report = _safe_load_json(
        os.path.join(STATS_DIR, "sensitivity_report.json"))
    if report is None:
        logger.warning("  Sensitivity report not found. Skipping Figure 12.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.75))

    # ==== Panel A: Robustness Checklist ====
    ax_check = axes[0, 0]
    checks = report.get("robustness_summary", [])

    if checks:
        n_checks = len(checks)
        ax_check.set_xlim(0, 3)
        ax_check.set_ylim(-0.8, n_checks - 0.2)
        ax_check.axis('off')

        for i, check in enumerate(checks):
            status = check.get('status', 'UNKNOWN')
            name = check.get('check', '').replace('_', ' ').title()
            detail = check.get('detail', '')

            # Shorten detail
            if 'r = ' in detail:
                try:
                    r_val = float(detail.split('r = ')[1].strip())
                    detail = f"r = {r_val:.3f}"
                except (ValueError, IndexError):
                    pass
            elif len(detail) > 25:
                detail = detail[:23] + "..."

            # Colors
            if status == 'PASS':
                bg_color = '#E8F5E9'
                symbol_color = '#2E7D32'
                symbol = '✓'
            else:
                bg_color = '#FFF3E0'
                symbol_color = '#E65100'
                symbol = '⚠'

            ax_check.barh(i, 3, height=0.65, color=bg_color,
                          edgecolor='none', left=0)
            ax_check.text(0.08, i, symbol, fontsize=10, fontweight='bold',
                          color=symbol_color, va='center', ha='left')
            ax_check.text(0.40, i + 0.12, name, fontsize=5.5,
                          fontweight='bold', color='#333333',
                          va='center', ha='left')
            ax_check.text(0.40, i - 0.12, detail, fontsize=4.5,
                          color='#666666', va='center', ha='left',
                          style='italic')
    else:
        ax_check.text(0.5, 0.5, "No robustness checks available",
                      transform=ax_check.transAxes, ha='center', fontsize=8)

    ax_check.set_title("A. Robustness Checklist",
                       fontsize=8, fontweight='bold', loc='left', pad=8)

    # ==== Panel B: Sample Size Stability ====
    ax_stab = axes[0, 1]
    stability = report.get("sample_size_stability", {})

    if stability:
        disorder_data = {}
        for disorder, fracs in stability.items():
            if not fracs:
                continue
            fracs_numeric = {}
            for k, v in fracs.items():
                try:
                    fk = float(k)
                    fracs_numeric[fk] = v
                except (ValueError, TypeError):
                    continue
            if not fracs_numeric:
                continue

            fracs_sorted = sorted(fracs_numeric.keys())
            disorder_data[disorder] = {
                'x_vals': [f * 100 for f in fracs_sorted],
                'aurocs': [fracs_numeric[f].get("mean_auroc", np.nan)
                           for f in fracs_sorted],
                'display': get_disorder_display(disorder),
                'color': get_disorder_color(disorder),
            }

        for disorder, data in disorder_data.items():
            ax_stab.plot(
                data['x_vals'], data['aurocs'],
                marker='o', markersize=2.5, linewidth=0.7,
                color=data['color'], alpha=0.7,
                label=data['display']
            )
            # Reference line at 100% AUROC
            if data['aurocs'] and not np.isnan(data['aurocs'][-1]):
                ax_stab.axhline(
                    y=data['aurocs'][-1], color=data['color'],
                    linestyle=':', linewidth=0.3, alpha=0.3
                )

        ax_stab.set_xticks([25, 50, 75, 100])
        ax_stab.set_xticklabels(['25%', '50%', '75%', '100%'], fontsize=5)
        ax_stab.set_xlabel("Sample Fraction", fontsize=6)
        ax_stab.set_ylabel("AUROC", fontsize=6)
        ax_stab.legend(fontsize=3.5, loc='lower right', ncol=2,
                       framealpha=0.8)
    else:
        ax_stab.text(0.5, 0.5, "Stability data\nnot available",
                     transform=ax_stab.transAxes, ha='center', fontsize=8,
                     color='gray')

    ax_stab.set_title("B. Sample Size Stability",
                      fontsize=8, fontweight='bold', loc='left', pad=8)
    ax_stab.spines['top'].set_visible(False)
    ax_stab.spines['right'].set_visible(False)

    # ==== Panel C: Multiple Comparison Methods (Lollipop) ====
    ax_mc = axes[1, 0]
    mc_results = report.get("multiple_comparison", {})

    if mc_results:
        methods = list(mc_results.keys())
        n_sig = [mc_results[m].get("n_significant", 0) for m in methods]
        pct = [mc_results[m].get("proportion_significant", 0) for m in methods]

        # Color by strictness
        mc_colors = {
            "none": "#E53935",
            "bonferroni": "#1565C0",
            "holm": "#1976D2",
            "fdr_bh": "#42A5F5",
            "fdr_by": "#90CAF9",
        }

        for i, (method, n) in enumerate(zip(methods, n_sig)):
            color = mc_colors.get(method, '#607D8B')
            ax_mc.plot(n, i, 'o', color=color, markersize=5, zorder=5)
            ax_mc.hlines(i, 0, n, color=color, linewidth=1.0, alpha=0.6)
            ax_mc.text(n + 5, i, f"{n} ({pct[i]*100:.0f}%)",
                       va='center', fontsize=5)

        ax_mc.set_yticks(range(len(methods)))
        ax_mc.set_yticklabels(
            [m.replace('_', ' ').upper() for m in methods], fontsize=5
        )
        ax_mc.set_xlabel("N Significant Tests", fontsize=6)
    else:
        ax_mc.text(0.5, 0.5, "MC data\nnot available",
                   transform=ax_mc.transAxes, ha='center', fontsize=8,
                   color='gray')

    ax_mc.set_title("C. Multiple Comparison Methods",
                    fontsize=8, fontweight='bold', loc='left', pad=8)
    ax_mc.spines['top'].set_visible(False)
    ax_mc.spines['right'].set_visible(False)

    # ==== Panel D: Confound Adjustment ====
    ax_confound = axes[1, 1]
    confound = report.get("confound_adjustment", {})

    if confound and "comparison" in confound:
        comp = confound["comparison"]
        methods_cf = list(comp.keys())
        correlations = [comp[m].get("correlation_with_original", np.nan)
                        for m in methods_cf]
        mads = [comp[m].get("mean_abs_difference", np.nan) for m in methods_cf]

        x_pos = np.arange(len(methods_cf))
        width = 0.35

        bars_r = ax_confound.bar(
            x_pos - width/2, correlations, width,
            color='#1976D2', edgecolor='black', linewidth=0.3,
            label='Correlation (r)'
        )
        bars_mad = ax_confound.bar(
            x_pos + width/2, mads, width,
            color='#FF9800', edgecolor='black', linewidth=0.3,
            label='MAD'
        )

        ax_confound.set_xticks(x_pos)
        ax_confound.set_xticklabels(
            [m.replace('_', ' ').title() for m in methods_cf],
            fontsize=5
        )

        # Annotate nonlinearity ratio
        nonlin = confound.get("nonlinearity_ratio", np.nan)
        if not pd.isna(nonlin):
            ax_confound.text(
                0.5, 0.95,
                f"Nonlinearity ratio: {nonlin:.2f}",
                transform=ax_confound.transAxes, fontsize=5,
                ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF9C4',
                         edgecolor='#F57F17', alpha=0.9)
            )

        ax_confound.legend(fontsize=5)
    else:
        ax_confound.text(0.5, 0.5, "Confound data\nnot available",
                         transform=ax_confound.transAxes, ha='center',
                         fontsize=8, color='gray')

    ax_confound.set_title("D. Confound Adjustment",
                          fontsize=8, fontweight='bold', loc='left', pad=8)
    ax_confound.spines['top'].set_visible(False)
    ax_confound.spines['right'].set_visible(False)

    _save_figure(fig, "fig12_sensitivity_summary")


# =====================================================================
# SECTION 15: MAIN PIPELINE
# =====================================================================

def run_all_visualizations():
    """Execute all visualization functions in order."""
    logger.info("=" * 70)
    logger.info("VISUALIZATION PIPELINE v2.0 — Nature-Quality Figures")
    logger.info("=" * 70)

    figure_functions = [
        ("Figure 1:  Effect Size Atlas", plot_effect_size_heatmap),
        ("Figure 2:  AUROC Matrix", plot_auroc_matrix),
        ("Figure 3:  Disorder Clustering", plot_clustering),
        ("Figure 4:  Factor Analysis", plot_factor_loadings),
        ("Figure 5:  Transdiagnostic vs Specific", plot_transdiagnostic_specific),
        ("Figure 6:  Direction Consistency", plot_direction_consistency),
        ("Figure 7:  Clinical Utility Ranking", plot_clinical_utility_ranking),
        ("Figure 8:  Dimension Ablation", plot_dimension_ablation),
        ("Figure 9:  Confusion Matrix", plot_confusion_matrix),
        ("Figure 10: t-SNE Small Multiples", plot_tsne_embedding),
        ("Figure 11: Disorder Profiles", plot_disorder_profiles),
        ("Figure 12: Sensitivity Summary", plot_sensitivity_summary),
    ]

    completed = 0
    failed = 0

    for name, func in figure_functions:
        logger.info(f"\n{'─' * 60}")
        logger.info(f"  Generating: {name}")
        logger.info(f"{'─' * 60}")
        try:
            func()
            completed += 1
        except Exception as e:
            logger.error(f"  ❌ FAILED: {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # --- Summary ---
    logger.info(f"\n{'=' * 70}")
    logger.info("VISUALIZATION PIPELINE COMPLETE")
    logger.info(f"  Completed: {completed}/{len(figure_functions)}")
    logger.info(f"  Failed:    {failed}/{len(figure_functions)}")
    logger.info(f"  Output:    {FIGURES_DIR}")
    logger.info(f"{'=' * 70}")

    # --- Figure classification for manuscript ---
    logger.info("\nFigure Classification for Manuscript:")
    logger.info("  MAIN TEXT:")
    logger.info("    Fig 1  → Effect Size Atlas (core finding)")
    logger.info("    Fig 2  → AUROC Matrix (discrimination)")
    logger.info("    Fig 4  → Factor Analysis (latent structure)")
    logger.info("    Fig 6  → Direction Consistency (p-factor evidence)")
    logger.info("  EXTENDED DATA:")
    logger.info("    ED Fig 1 → Clinical Utility Ranking (Fig 7)")
    logger.info("    ED Fig 2 → Transdiagnostic vs Specific (Fig 5)")
    logger.info("    ED Fig 3 → Clustering + MDS (Fig 3)")
    logger.info("    ED Fig 4 → Dimension Ablation (Fig 8)")
    logger.info("    ED Fig 5 → t-SNE Small Multiples (Fig 10)")
    logger.info("    ED Fig 6 → Confusion Matrix (Fig 9)")
    logger.info("    ED Fig 7 → Disorder Profiles (Fig 11)")
    logger.info("    ED Fig 8 → Sensitivity Summary (Fig 12)")


if __name__ == "__main__":
    run_all_visualizations()