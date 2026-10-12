"""
Step 07: Publication-Quality Visualizations (v1.0 → v1.2)

Generates all figures for the digital phenotyping atlas.
Designed for Nature-level publication standards.

v1.0 → v1.2 Changelog:
  [CRITICAL] Figure 1: use effect_size_matrix_clean.csv (exclude artifacts)
  [CRITICAL] Figure 2: load vs-control separately, fix NaN diagonal
  [CRITICAL] Figure 4: read correct s05 output filenames
  [FIX] Dimension grouping: use s05's log-mapped metadata
  [FIX] AUROC matrix: separate color scales for diagonal vs off-diagonal
  [FIX] Feature loading figure: show feature loadings (not disorder scores)
  [NEW] Figure 2b: vs-control bar chart as complement to pairwise matrix
  [NEW] Robustness summary figure from s06 sensitivity report
  [OPT] Graceful degradation when files missing
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import squareform
from collections import defaultdict
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

# Selection artifact features (consistent with s03-s06)
SELECTION_ARTIFACT_FEATURES = {
    "mental_health_sub_ratio",
    "support_sub_ratio",
    "subreddit_entropy",
    "subreddit_herfindahl",
    "primary_subreddit_ratio",
}

# ======================== STYLE CONFIGURATION ========================

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

DIVERGING_CMAP = sns.diverging_palette(240, 10, as_cmap=True)
SEQUENTIAL_CMAP = sns.color_palette("YlOrRd", as_cmap=True)

SINGLE_COL = 89 / 25.4
DOUBLE_COL = 183 / 25.4
FULL_PAGE_HEIGHT = 247 / 25.4

dim_colors_map = {
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
def get_disorder_color(disorder):
    for spectrum, disorders in HITOP_SPECTRA.items():
        if disorder in disorders:
            return HITOP_COLORS.get(spectrum, "#808080")
    return "#808080"


def get_disorder_display(disorder):
    return DISORDER_DISPLAY_NAMES.get(disorder, disorder)


def _safe_load_csv(path, **kwargs):
    """Load CSV with graceful error handling."""
    if not os.path.exists(path):
        logger.warning(f"  File not found: {path}")
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        logger.error(f"  Error loading {path}: {e}")
        return None


def _safe_load_json(path):
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


def _load_dimension_mapping():
    """
    [FIX v1.2] Load dimension mapping with log features mapped to base dims.
    Consistent with s05's load_feature_meta_with_log_mapping.
    """
    meta = _safe_load_json(os.path.join(FEATURES_DIR, "feature_metadata.json"))
    if not meta:
        return {}, []

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

    dim_order = ['affective', 'pronoun', 'cognitive', 'temporal_linguistic',
                 'risk', 'behavioral', 'readability', 'post_length',
                 'temporal', 'social', 'community']

    return feature_to_dim, dim_order


# ======================== FIGURE 1: EFFECT SIZE HEATMAP ========================

def plot_effect_size_heatmap():
    """
    [FIX v1.2] Main atlas figure using CLEAN effect size matrix.
    Artifacts excluded to avoid circular-reasoning dominance.
    """
    logger.info("Generating Figure 1: Effect Size Heatmap...")

    # [CRITICAL v1.2] Use clean matrix
    effect_matrix = _safe_load_csv(
        os.path.join(STATS_DIR, "effect_size_matrix_clean.csv"), index_col=0)
    if effect_matrix is None:
        # Fallback: load full and filter manually
        effect_matrix = _safe_load_csv(
            os.path.join(STATS_DIR, "effect_size_matrix.csv"), index_col=0)
        if effect_matrix is not None:
            artifact_rows = [f for f in effect_matrix.index
                            if f in SELECTION_ARTIFACT_FEATURES]
            if artifact_rows:
                effect_matrix = effect_matrix.drop(index=artifact_rows)
                logger.info(f"  Manually removed {len(artifact_rows)} artifact features")

    if effect_matrix is None:
        logger.error("  No effect size matrix available. Skipping Figure 1.")
        return

    sig_matrix = _safe_load_csv(
        os.path.join(STATS_DIR, "significance_matrix_clean.csv"), index_col=0)
    if sig_matrix is None:
        sig_matrix = _safe_load_csv(
            os.path.join(STATS_DIR, "significance_matrix.csv"), index_col=0)

    # Load dimension mapping
    feat_to_dim, dim_order = _load_dimension_mapping()

    # Rename disorders
    effect_matrix.columns = [get_disorder_display(d) for d in effect_matrix.columns]
    if sig_matrix is not None:
        sig_matrix.columns = [get_disorder_display(d) for d in sig_matrix.columns]

    # Order features by dimension, then by mean |g| within dimension
    ordered_features = []
    dim_boundaries = []
    dim_names_ordered = []

    for dim in dim_order:
        dim_feats = [f for f in effect_matrix.index if feat_to_dim.get(f) == dim]
        if not dim_feats:
            continue
        mean_abs = effect_matrix.loc[dim_feats].abs().mean(axis=1).sort_values(ascending=False)
        ordered_features.extend(mean_abs.index.tolist())
        dim_boundaries.append(len(ordered_features))
        dim_names_ordered.append(dim)

    # Add remaining features
    remaining = [f for f in effect_matrix.index if f not in ordered_features]
    if remaining:
        mean_abs_rem = effect_matrix.loc[remaining].abs().mean(axis=1).sort_values(ascending=False)
        ordered_features.extend(mean_abs_rem.index.tolist())
        dim_boundaries.append(len(ordered_features))
        dim_names_ordered.append('other')

    # Select top N features for readability
    top_n = 50
    mean_abs_all = effect_matrix.loc[ordered_features].abs().mean(axis=1).sort_values(ascending=False)
    top_features = mean_abs_all.head(top_n).index.tolist()

    # Re-order top features by dimension
    top_ordered = []
    top_dim_boundaries = []
    top_dim_names = []

    for dim in dim_names_ordered:
        dim_feats = [f for f in top_features if feat_to_dim.get(f) == dim]
        if not dim_feats:
            continue
        mean_abs_dim = effect_matrix.loc[dim_feats].abs().mean(axis=1).sort_values(ascending=False)
        top_ordered.extend(mean_abs_dim.index.tolist())
        top_dim_boundaries.append(len(top_ordered))
        top_dim_names.append(dim)

    remaining_top = [f for f in top_features if f not in top_ordered]
    if remaining_top:
        top_ordered.extend(remaining_top)
        top_dim_boundaries.append(len(top_ordered))
        top_dim_names.append('other')

    effect_top = effect_matrix.loc[top_ordered]
    sig_top = sig_matrix.reindex(top_ordered) if sig_matrix is not None else None

    # Create figure
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 0.85))

    # Symmetric color normalization
    valid_vals = effect_top.values[~np.isnan(effect_top.values)]
    vmax = min(max(abs(valid_vals.min()), abs(valid_vals.max())), 1.5)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    data = effect_top.values.astype(float)
    im = ax.imshow(data, cmap=DIVERGING_CMAP, norm=norm, aspect='auto', interpolation='none')

    # Significance markers
    if sig_top is not None:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if np.isnan(data[i, j]):
                    continue
                feat_name = top_ordered[i]
                dis_name = effect_top.columns[j]
                try:
                    is_sig = sig_top.loc[feat_name, dis_name]
                    if isinstance(is_sig, str):
                        is_sig = is_sig.lower() == 'true'
                    if is_sig:
                        ax.text(j, i, '•', ha='center', va='center',
                                fontsize=4, color='black', fontweight='bold')
                except (KeyError, TypeError):
                    pass

    # Axis labels
    ax.set_xticks(range(len(effect_top.columns)))
    ax.set_xticklabels(effect_top.columns, rotation=45, ha='right', fontsize=6)
    ax.set_yticks(range(len(top_ordered)))
    ax.set_yticklabels(top_ordered, fontsize=5)

    # Dimension separators
    dim_colors = dim_colors_map

    prev_boundary = 0
    for dim, boundary in zip(top_dim_names, top_dim_boundaries):
        if boundary > prev_boundary:
            ax.axhline(y=boundary - 0.5, color='white', linewidth=1.5)
            mid_y = (prev_boundary + boundary) / 2 - 0.5
            label = dim.replace('_', ' ').replace('temporal linguistic', 'temporal-ling').title()
            ax.text(-1.5, mid_y, label,
                    ha='right', va='center', fontsize=5,
                    color=dim_colors.get(dim, 'black'),
                    fontweight='bold', rotation=0)
        prev_boundary = boundary

    # Color bar
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, aspect=30, pad=0.02)
    cbar.set_label("Hedges' g", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax.set_title("Digital Phenotyping Atlas: Effect Sizes (Disorder vs. Matched Controls)\n"
                 "Artifact features excluded; • = FDR-significant",
                 fontsize=8, fontweight='bold', pad=10)

    # Color x-axis by HiTOP spectrum
    for label in ax.get_xticklabels():
        text = label.get_text()
        raw_name = [k for k, v in DISORDER_DISPLAY_NAMES.items() if v == text]
        if raw_name:
            label.set_color(get_disorder_color(raw_name[0]))

    plt.tight_layout()

    for fmt in ['pdf', 'png', 'svg']:
        fig.savefig(os.path.join(FIGURES_DIR, f"fig1_effect_size_heatmap.{fmt}"),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Figure 1 saved.")


# ======================== FIGURE 2: AUROC MATRIX ========================

def plot_auroc_matrix():
    """
    [FIX v1.2] Pairwise AUROC matrix + vs-control bar chart.
    
    Separates the two semantically different metrics:
      - Off-diagonal: disorder vs disorder (discriminability)
      - Bar chart: disorder vs control (detectability)
    """
    logger.info("Generating Figure 2: AUROC Matrix...")

    pairwise_matrix = _safe_load_csv(
        os.path.join(STATS_DIR, "auroc_pairwise_matrix.csv"), index_col=0)

    # [FIX v1.2] Load vs-control separately
    vs_control_df = _safe_load_csv(
        os.path.join(STATS_DIR, "auroc_vs_control.csv"))

    if pairwise_matrix is None:
        logger.error("  No pairwise AUROC matrix. Skipping Figure 2.")
        return

    # Rename
    pairwise_matrix.index = [get_disorder_display(d) for d in pairwise_matrix.index]
    pairwise_matrix.columns = [get_disorder_display(d) for d in pairwise_matrix.columns]

    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.55),
                             width_ratios=[1.2, 1])

    # ---- Left: Pairwise matrix (disorder vs disorder) ----
    ax_matrix = axes[0]
    data = pairwise_matrix.values.astype(float)

    # Fill diagonal with NaN for display (it's meaningless for pairwise)
    np.fill_diagonal(data, np.nan)

    im = ax_matrix.imshow(data, cmap='RdYlGn', vmin=0.45, vmax=0.95, aspect='equal')

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                continue
            text_color = 'white' if val > 0.8 or val < 0.55 else 'black'
            ax_matrix.text(j, i, f'{val:.2f}', ha='center', va='center',
                           fontsize=5, color=text_color)

    # Mark diagonal as "--"
    for i in range(data.shape[0]):
        ax_matrix.text(i, i, '—', ha='center', va='center',
                       fontsize=6, color='#AAAAAA')

    ax_matrix.set_xticks(range(len(pairwise_matrix.columns)))
    ax_matrix.set_xticklabels(pairwise_matrix.columns, rotation=45, ha='right', fontsize=5)
    ax_matrix.set_yticks(range(len(pairwise_matrix.index)))
    ax_matrix.set_yticklabels(pairwise_matrix.index, fontsize=5)

    cbar = fig.colorbar(im, ax=ax_matrix, shrink=0.7, aspect=25)
    cbar.set_label("AUROC", fontsize=7)
    cbar.ax.tick_params(labelsize=5)

    ax_matrix.set_title("Pairwise Disorder Discrimination",
                        fontsize=8, fontweight='bold')

    # ---- Right: vs-control bar chart ----
    ax_bar = axes[1]

    if vs_control_df is not None and len(vs_control_df) > 0:
        # Sort by AUROC
        vs_control_sorted = vs_control_df.sort_values('auroc_vs_control', ascending=True)

        disorders_display = [get_disorder_display(d) for d in vs_control_sorted['disorder']]
        aurocs = vs_control_sorted['auroc_vs_control'].values
        colors = [get_disorder_color(d) for d in vs_control_sorted['disorder']]

        bars = ax_bar.barh(range(len(disorders_display)), aurocs, color=colors,
                           edgecolor='black', linewidth=0.3, height=0.7)

        # Add value labels
        for i, (val, bar) in enumerate(zip(aurocs, bars)):
            ax_bar.text(val + 0.005, i, f'{val:.3f}', va='center', fontsize=5)

        ax_bar.set_yticks(range(len(disorders_display)))
        ax_bar.set_yticklabels(disorders_display, fontsize=5)
        ax_bar.set_xlim(0.5, 1.0)
        ax_bar.axvline(x=0.5, color='red', linestyle='--', linewidth=0.5, label='Chance')
        ax_bar.set_xlabel("AUROC (vs. Controls)", fontsize=7)
        ax_bar.set_title("Disorder Detectability\n(vs. Matched Controls)",
                         fontsize=8, fontweight='bold')
    else:
        # Fallback: use diagonal from pairwise matrix if vs-control not available
        ax_bar.text(0.5, 0.5, "vs-control data\nnot available",
                    transform=ax_bar.transAxes, ha='center', va='center',
                    fontsize=8, color='gray')

    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)

    plt.tight_layout()
    for fmt in ['pdf', 'png', 'svg']:
        fig.savefig(os.path.join(FIGURES_DIR, f"fig2_auroc_matrix.{fmt}"),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Figure 2 saved.")


# ======================== FIGURE 3: DENDROGRAM + MDS ========================

def plot_clustering():
    """Disorder clustering: dendrogram + MDS 2D embedding."""
    logger.info("Generating Figure 3: Disorder Clustering...")

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

    # Left: Dendrogram
    ax_dend = axes[0]
    disorder_labels = [get_disorder_display(d) for d in mds_df.index]

    dend = dendrogram(
        linkage_matrix, labels=disorder_labels, orientation='left',
        ax=ax_dend, leaf_font_size=6, color_threshold=0,
        above_threshold_color='#333333', leaf_rotation=0,
    )

    for lbl in ax_dend.get_ymajorticklabels():
        text = lbl.get_text()
        raw_name = [k for k, v in DISORDER_DISPLAY_NAMES.items() if v == text]
        if raw_name:
            lbl.set_color(get_disorder_color(raw_name[0]))
            lbl.set_fontweight('bold')

    ax_dend.set_title("Hierarchical Clustering", fontsize=8, fontweight='bold')
    ax_dend.set_xlabel("Ward Distance", fontsize=7)
    ax_dend.spines['top'].set_visible(False)
    ax_dend.spines['right'].set_visible(False)

    # Right: MDS 2D embedding
    ax_mds = axes[1]

    for disorder_raw in mds_df.index:
        x = mds_df.loc[disorder_raw, "MDS_1"]
        y = mds_df.loc[disorder_raw, "MDS_2"]
        color = get_disorder_color(disorder_raw)
        display_name = get_disorder_display(disorder_raw)

        ax_mds.scatter(x, y, c=color, s=80, edgecolors='black',
                       linewidths=0.5, zorder=5)
        ax_mds.annotate(display_name, (x, y), xytext=(5, 5),
                        textcoords='offset points', fontsize=5,
                        color=color, fontweight='bold')

    ax_mds.set_xlabel("MDS Dimension 1", fontsize=7)
    ax_mds.set_ylabel("MDS Dimension 2", fontsize=7)
    ax_mds.set_title("MDS Embedding (Effect-Size Similarity)",
                     fontsize=8, fontweight='bold')
    ax_mds.spines['top'].set_visible(False)
    ax_mds.spines['right'].set_visible(False)

    # HiTOP legend
    spectrum_handles = []
    for spectrum, color in HITOP_COLORS.items():
        handle = Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                        markersize=5, label=spectrum.replace("_", " ").title(),
                        markeredgecolor='black', markeredgewidth=0.3)
        spectrum_handles.append(handle)
    ax_mds.legend(handles=spectrum_handles, loc='lower right', fontsize=5,
                  framealpha=0.8, edgecolor='gray', title="HiTOP Spectra",
                  title_fontsize=5)

    plt.tight_layout()
    for fmt in ['pdf', 'png', 'svg']:
        fig.savefig(os.path.join(FIGURES_DIR, f"fig3_disorder_clustering.{fmt}"),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Figure 3 saved.")


# ======================== FIGURE 4: FACTOR ANALYSIS ========================

def plot_factor_loadings():
    """
    [FIX v1.2→v1.3] Complete Factor Analysis Figure (4 panels).
    
    Panel A: Feature loading heatmap (features × factors)
    Panel B: Disorder scores in factor space (with aspect ratio fix)
    Panel C: Scree plot + parallel analysis
    Panel D: Factor interpretation summary (dimension composition)
    """
    logger.info("Generating Figure 4: Factor Analysis (4-panel)...")

    # ---- Load data ----
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

    # Rename disorders
    score_df.index = [get_disorder_display(d) for d in score_df.index]

    # Load dimension mapping
    feat_to_dim, dim_order = _load_dimension_mapping()

    # ============================================================
    # Create 2×2 figure
    # ============================================================
    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.9))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           width_ratios=[1.2, 1],
                           height_ratios=[1.1, 1],
                           hspace=0.35, wspace=0.35)

    # ============================================================
    # Panel A: Feature Loading Heatmap (Top Features × Factors)
    # ============================================================
    ax_heat = fig.add_subplot(gs[0, 0])

    if loading_df is not None and n_factors >= 2:
        # Select top 25 features by max absolute loading across factors
        max_loading = loading_df.abs().max(axis=1)
        top_n = min(25, len(loading_df))
        top_features = max_loading.nlargest(top_n).index.tolist()
        load_subset = loading_df.loc[top_features]

        # Color: diverging, centered at 0
        vmax = max(abs(load_subset.values.min()), abs(load_subset.values.max()))
        vmax = min(vmax, 5.0)  # Cap for readability

        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        im = ax_heat.imshow(load_subset.values, cmap=DIVERGING_CMAP,
                            norm=norm, aspect='auto', interpolation='none')

        # Annotate values
        for i in range(load_subset.shape[0]):
            for j in range(load_subset.shape[1]):
                val = load_subset.iloc[i, j]
                text_color = 'white' if abs(val) > vmax * 0.6 else 'black'
                ax_heat.text(j, i, f'{val:.2f}', ha='center', va='center',
                           fontsize=4, color=text_color, fontweight='bold')

        # Y-axis: feature names (truncated) colored by dimension
        y_labels = [f[:30] for f in top_features]
        ax_heat.set_yticks(range(len(y_labels)))
        ax_heat.set_yticklabels(y_labels, fontsize=4.5)

        for i, feat in enumerate(top_features):
            dim = feat_to_dim.get(feat, 'other')
            # 使用模块级 dim_colors_map
            color = dim_colors_map.get(dim, '#333333')
            ax_heat.get_yticklabels()[i].set_color(color)

        # X-axis: factor names with variance
        x_labels = []
        for i in range(n_factors):
            v = var_explained[i] * 100 if i < len(var_explained) else 0
            x_labels.append(f"F{i+1}\n({v:.1f}%)")
        ax_heat.set_xticks(range(n_factors))
        ax_heat.set_xticklabels(x_labels, fontsize=6)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax_heat, shrink=0.8, aspect=20, pad=0.02)
        cbar.set_label("Loading", fontsize=6)
        cbar.ax.tick_params(labelsize=5)

        ax_heat.set_title("A. Top Feature Loadings", fontsize=8, fontweight='bold')
    else:
        ax_heat.text(0.5, 0.5, "Loading data unavailable",
                    transform=ax_heat.transAxes, ha='center', fontsize=8)

    # ============================================================
    # Panel B: Disorder Factor Scores (with ASPECT RATIO FIX)
    # ============================================================
    ax_biplot = fig.add_subplot(gs[0, 1])

    if n_factors >= 2:
        f1_col = score_df.columns[0]
        f2_col = score_df.columns[1]

        for disorder_display in score_df.index:
            x = score_df.loc[disorder_display, f1_col]
            y = score_df.loc[disorder_display, f2_col]
            raw_name = [k for k, v in DISORDER_DISPLAY_NAMES.items() if v == disorder_display]
            color = get_disorder_color(raw_name[0]) if raw_name else "#808080"

            ax_biplot.scatter(x, y, c=color, s=80, edgecolors='black',
                             linewidths=0.5, zorder=5)
            ax_biplot.annotate(disorder_display, (x, y), xytext=(4, 4),
                              textcoords='offset points', fontsize=5,
                              color=color, fontweight='bold')

        f1_label = f"Factor 1 ({var_explained[0]*100:.1f}%)" if var_explained else "Factor 1"
        f2_label = f"Factor 2 ({var_explained[1]*100:.1f}%)" if len(var_explained) > 1 else "Factor 2"
        ax_biplot.set_xlabel(f1_label, fontsize=7)
        ax_biplot.set_ylabel(f2_label, fontsize=7)
        ax_biplot.axhline(y=0, color='gray', linewidth=0.3)
        ax_biplot.axvline(x=0, color='gray', linewidth=0.3)

        # [FIX] Expand y-axis to make Factor 2 variation visible
        f2_vals_all = score_df.iloc[:, 1].values
        f2_range = f2_vals_all.max() - f2_vals_all.min()
        if f2_range < 0.01:
            f2_range = (score_df.iloc[:, 0].max() - score_df.iloc[:, 0].min()) * 0.1
        y_margin = max(f2_range * 0.5, 0.05)
        y_center = f2_vals_all.mean()
        ax_biplot.set_ylim(y_center - y_margin, y_center + y_margin)

        # Dimension legend for Panel A (placed in Panel B's space)
        dim_legend_items = []
        seen_dims = set()
        for feat in (top_features if loading_df is not None else []):
            dim = feat_to_dim.get(feat, 'other')
            if dim not in seen_dims:
                seen_dims.add(dim)
                handle = Line2D([0], [0], marker='s', color='w',
                              markerfacecolor=dim_colors_map.get(dim, '#333'),
                              markersize=4, label=dim.replace('_', ' ').title(),
                              markeredgecolor='black', markeredgewidth=0.3)
                dim_legend_items.append(handle)
        if dim_legend_items:
            ax_biplot.legend(handles=dim_legend_items, loc='lower right',
                           fontsize=4, framealpha=0.8, edgecolor='gray',
                           title="Feature Dims", title_fontsize=4, ncol=2)

    ax_biplot.set_title("B. Disorder Factor Scores",
                        fontsize=8, fontweight='bold')
    ax_biplot.spines['top'].set_visible(False)
    ax_biplot.spines['right'].set_visible(False)

    # ============================================================
    # Panel C: Scree Plot + Parallel Analysis
    # ============================================================
    ax_scree = fig.add_subplot(gs[1, 0])

    eigenvalues = fa_info.get("selection_criteria", {}).get("eigenvalues", [])
    random_95th = fa_info.get("selection_criteria", {}).get("random_95th_eigenvalues", [])

    if eigenvalues:
        n_show = min(len(eigenvalues), 10)
        x = range(1, n_show + 1)

        # Bar chart for observed eigenvalues
        bars = ax_scree.bar(x, eigenvalues[:n_show], color='#1976D2',
                           alpha=0.7, edgecolor='black', linewidth=0.3,
                           label='Observed', width=0.6)

        # Highlight retained factors
        for i in range(min(n_factors, n_show)):
            bars[i].set_color('#FF9800')
            bars[i].set_alpha(0.9)

        if random_95th:
            ax_scree.plot(x, random_95th[:n_show], 'D--', color='#E53935',
                         markersize=3, label='95th %ile (Random)', linewidth=0.8)

        ax_scree.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.5,
                         label='Kaiser Criterion')
        ax_scree.set_xticks(list(x))

        # Annotate variance explained
        for i, v in enumerate(eigenvalues[:n_show]):
            if i < len(var_explained):
                ax_scree.text(i + 1, v + 0.05,
                             f'{var_explained[i]*100:.1f}%',
                             ha='center', fontsize=4.5, color='#333333')

    ax_scree.set_xlabel("Factor Number", fontsize=7)
    ax_scree.set_ylabel("Eigenvalue", fontsize=7)
    ax_scree.set_title("C. Scree Plot + Parallel Analysis",
                       fontsize=8, fontweight='bold')
    ax_scree.legend(fontsize=5, loc='upper right')
    ax_scree.spines['top'].set_visible(False)
    ax_scree.spines['right'].set_visible(False)

    # ============================================================
    # Panel D: Factor Interpretation — Dimension Composition
    # ============================================================
    ax_interp = fig.add_subplot(gs[1, 1])

    if loading_df is not None and n_factors >= 1:
        # For each factor, compute average absolute loading per dimension
        dim_factor_loadings = defaultdict(lambda: defaultdict(list))

        for feat in loading_df.index:
            dim = feat_to_dim.get(feat, 'other')
            for fi in range(n_factors):
                dim_factor_loadings[dim][fi].append(abs(loading_df.iloc[
                    loading_df.index.get_loc(feat), fi]))

        # Compute mean |loading| per dimension per factor
        dim_names_ordered = []
        dim_factor_means = {}

        for dim in dim_order:
            if dim in dim_factor_loadings:
                dim_names_ordered.append(dim)
                dim_factor_means[dim] = [
                    np.mean(dim_factor_loadings[dim][fi])
                    for fi in range(n_factors)
                ]

        if dim_names_ordered:
            n_dims = len(dim_names_ordered)
            y_pos = np.arange(n_dims)
            bar_height = 0.35

            # Factor 1 bars
            f1_vals = [dim_factor_means[d][0] for d in dim_names_ordered]
            ax_interp.barh(y_pos + bar_height/2, f1_vals, bar_height,
                          color='#FF9800', edgecolor='black', linewidth=0.3,
                          label=f'Factor 1 ({var_explained[0]*100:.1f}%)'
                          if var_explained else 'Factor 1')

            # Factor 2 bars
            if n_factors >= 2:
                f2_vals = [dim_factor_means[d][1] for d in dim_names_ordered]
                ax_interp.barh(y_pos - bar_height/2, f2_vals, bar_height,
                              color='#1976D2', edgecolor='black', linewidth=0.3,
                              label=f'Factor 2 ({var_explained[1]*100:.1f}%)'
                              if len(var_explained) > 1 else 'Factor 2')

            ax_interp.set_yticks(y_pos)
            dim_display = [d.replace('_', ' ').replace('temporal linguistic',
                            'temp-ling').title() for d in dim_names_ordered]
            ax_interp.set_yticklabels(dim_display, fontsize=5)
            ax_interp.set_xlabel("Mean |Loading|", fontsize=6)
            ax_interp.legend(fontsize=5, loc='lower right')

    ax_interp.set_title("D. Factor Composition by Dimension",
                        fontsize=8, fontweight='bold')
    ax_interp.spines['top'].set_visible(False)
    ax_interp.spines['right'].set_visible(False)

    # ============================================================
    # Save
    # ============================================================
    for fmt in ['pdf', 'png', 'svg']:
        fig.savefig(os.path.join(FIGURES_DIR, f"fig4_factor_analysis.{fmt}"),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Figure 4 saved (4-panel layout).")

# ======================== FIGURE 5: TRANSDIAGNOSTIC VS SPECIFIC ========================

def plot_transdiagnostic_specific():
    """Visualization of transdiagnostic vs disorder-specific features."""
    logger.info("Generating Figure 5: Transdiagnostic vs Specific Features...")

    transdiag_df = _safe_load_csv(
        os.path.join(STATS_DIR, "transdiagnostic_features.csv"))

    if transdiag_df is None:
        logger.error("  Transdiagnostic features not found. Skipping Figure 5.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.4))

    # Left: Bar chart
    ax_bar = axes[0]
    cat_order = ["transdiagnostic", "mixed", "specific", "non_significant"]
    cat_colors = {"transdiagnostic": "#E53935", "mixed": "#FB8C00",
                  "specific": "#1E88E5", "non_significant": "#BDBDBD"}
    cat_labels = {"transdiagnostic": "Transdiagnostic\n(≥5 disorders)",
                  "mixed": "Mixed\n(3-4 disorders)",
                  "specific": "Specific\n(1-2 disorders)",
                  "non_significant": "Non-significant"}

    counts = transdiag_df["category"].value_counts()
    bars_x = range(len(cat_order))
    bars_h = [counts.get(c, 0) for c in cat_order]
    bar_colors = [cat_colors[c] for c in cat_order]
    bar_labels = [cat_labels[c] for c in cat_order]

    bars = ax_bar.bar(bars_x, bars_h, color=bar_colors,
                      edgecolor='black', linewidth=0.3)
    ax_bar.set_xticks(bars_x)
    ax_bar.set_xticklabels(bar_labels, fontsize=5)
    ax_bar.set_ylabel("Number of Features", fontsize=7)
    ax_bar.set_title("Feature Classification", fontsize=8, fontweight='bold')
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)

    for bar, h in zip(bars, bars_h):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    str(int(h)), ha='center', va='bottom', fontsize=6)

    # Right: Scatter plot
    ax_scatter = axes[1]
    for cat in cat_order:
        subset = transdiag_df[transdiag_df["category"] == cat]
        if len(subset) == 0:
            continue
        ax_scatter.scatter(
            subset["n_sig_disorders"], subset["mean_abs_g_all"],
            c=cat_colors[cat], s=15, alpha=0.6,
            edgecolors='black', linewidths=0.2,
            label=cat.replace("_", " ").title(),
            zorder=3 if cat == "transdiagnostic" else 2
        )

    # Label top features
    top_trans = transdiag_df[transdiag_df["category"] == "transdiagnostic"].nlargest(
        3, "mean_abs_g_all")
    if "specific" in transdiag_df["category"].values:
        top_spec = transdiag_df[transdiag_df["category"] == "specific"].nlargest(
            3, "max_abs_g")
        for _, row in pd.concat([top_trans, top_spec]).iterrows():
            ax_scatter.annotate(
                row["feature"][:25],
                (row["n_sig_disorders"], row["mean_abs_g_all"]),
                fontsize=4, xytext=(3, 3), textcoords='offset points',
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.3)
            )
    else:
        for _, row in top_trans.iterrows():
            ax_scatter.annotate(
                row["feature"][:25],
                (row["n_sig_disorders"], row["mean_abs_g_all"]),
                fontsize=4, xytext=(3, 3), textcoords='offset points'
            )

    ax_scatter.set_xlabel("Number of Significant Disorders", fontsize=7)
    ax_scatter.set_ylabel("Mean |Hedges' g|", fontsize=7)
    ax_scatter.set_title("Feature Specificity vs. Effect Size",
                         fontsize=8, fontweight='bold')
    ax_scatter.legend(fontsize=5, markerscale=0.8)
    ax_scatter.spines['top'].set_visible(False)
    ax_scatter.spines['right'].set_visible(False)

    plt.tight_layout()
    for fmt in ['pdf', 'png', 'svg']:
        fig.savefig(os.path.join(FIGURES_DIR, f"fig5_transdiagnostic_specific.{fmt}"),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Figure 5 saved.")


# ======================== FIGURE 6: DIMENSION ABLATION ========================

def plot_dimension_ablation():
    """Ablation analysis: contribution of each feature dimension."""
    logger.info("Generating Figure 6: Dimension Ablation...")

    ablation_df = _safe_load_csv(
        os.path.join(STATS_DIR, "dimension_ablation.csv"), index_col=0)

    if ablation_df is None:
        logger.error("  Ablation data not found. Skipping Figure 6.")
        return

    dim_cols_only = [c for c in ablation_df.columns if c.startswith("only_")]
    dim_names = [c.replace("only_", "") for c in dim_cols_only]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 0.35))

    disorders_display = [get_disorder_display(d) for d in ablation_df.index]
    n_disorders = len(disorders_display)
    n_dims = len(dim_names) + 1

    bar_width = 0.8 / n_dims
    x = np.arange(n_disorders)

    dim_colors_map = {
        "affective": "#E53935", "pronoun": "#8E24AA",
        "cognitive": "#1E88E5", "temporal_linguistic": "#00897B",
        "risk": "#F4511E", "behavioral": "#6D4C41",
        "readability": "#546E7A", "post_length": "#78909C",
        "temporal": "#FF9800", "social": "#4CAF50",
        "community": "#9C27B0",
    }

    # All dimensions
    all_dim_vals = ablation_df["all_dimensions"].values
    ax.bar(x - 0.4 + bar_width * 0, all_dim_vals, bar_width,
           label="All", color="#333333", edgecolor='black', linewidth=0.3)

    for i, (dim_col, dim_name) in enumerate(zip(dim_cols_only, dim_names)):
        vals = ablation_df[dim_col].values
        color = dim_colors_map.get(dim_name, "#808080")
        ax.bar(x - 0.4 + bar_width * (i + 1), vals, bar_width,
               label=dim_name.replace('_', ' ').title(),
               color=color, edgecolor='black', linewidth=0.3)

    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=0.5, label='Chance')

    ax.set_xticks(x)
    ax.set_xticklabels(disorders_display, rotation=45, ha='right', fontsize=5)
    ax.set_ylabel("AUROC", fontsize=7)
    ax.set_title("Dimension Contribution to Disorder Discrimination",
                 fontsize=8, fontweight='bold')
    ax.legend(fontsize=4, loc='upper right', ncol=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0.4, 1.0)

    plt.tight_layout()
    for fmt in ['pdf', 'png', 'svg']:
        fig.savefig(os.path.join(FIGURES_DIR, f"fig6_dimension_ablation.{fmt}"),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Figure 6 saved.")


# ======================== FIGURE 7: CONFUSION MATRIX ========================

def plot_confusion_matrix():
    """Multiclass confusion matrix."""
    logger.info("Generating Figure 7: Confusion Matrix...")

    cm_df = _safe_load_csv(
        os.path.join(STATS_DIR, "confusion_matrix_normalized.csv"), index_col=0)

    if cm_df is None:
        logger.error("  Confusion matrix not found. Skipping Figure 7.")
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

    ax.set_xticks(range(len(cm_df.columns)))
    ax.set_xticklabels(cm_df.columns, rotation=45, ha='right', fontsize=5)
    ax.set_yticks(range(len(cm_df.index)))
    ax.set_yticklabels(cm_df.index, fontsize=5)

    ax.set_xlabel("Predicted", fontsize=7)
    ax.set_ylabel("True", fontsize=7)
    ax.set_title("Multiclass Confusion Matrix (Normalized)",
                 fontsize=8, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, aspect=25)
    cbar.set_label("Proportion", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    plt.tight_layout()
    for fmt in ['pdf', 'png', 'svg']:
        fig.savefig(os.path.join(FIGURES_DIR, f"fig7_confusion_matrix.{fmt}"),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Figure 7 saved.")


# ======================== FIGURE 8: t-SNE EMBEDDING ========================

def plot_tsne_embedding():
    """t-SNE embedding of users colored by disorder group."""
    logger.info("Generating Figure 8: t-SNE Embedding...")

    tsne_df = _safe_load_csv(os.path.join(STATS_DIR, "tsne_embedding.csv"))

    if tsne_df is None:
        logger.error("  t-SNE data not found. Skipping Figure 8.")
        return

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.3, SINGLE_COL * 1.3))

    # Controls (background)
    control_mask = tsne_df["group"] == "control"
    ax.scatter(
        tsne_df.loc[control_mask, "tSNE_1"],
        tsne_df.loc[control_mask, "tSNE_2"],
        c="#E0E0E0", s=3, alpha=0.3, label="Control", zorder=1, rasterized=True
    )

    # Disorders
    for disorder in DISORDER_GROUPS:
        mask = (tsne_df["group"] == "patient") & (tsne_df["disorder_group"] == disorder)
        if mask.sum() == 0:
            continue
        color = get_disorder_color(disorder)
        display = get_disorder_display(disorder)
        ax.scatter(
            tsne_df.loc[mask, "tSNE_1"], tsne_df.loc[mask, "tSNE_2"],
            c=color, s=8, alpha=0.5, label=display, zorder=2,
            edgecolors='black', linewidths=0.1, rasterized=True
        )

    ax.set_xlabel("t-SNE 1", fontsize=7)
    ax.set_ylabel("t-SNE 2", fontsize=7)
    ax.set_title("User Embedding (t-SNE)", fontsize=8, fontweight='bold')
    ax.legend(fontsize=4, markerscale=1.5, loc='upper right', ncol=2,
              framealpha=0.8, edgecolor='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    for fmt in ['pdf', 'png', 'svg']:
        fig.savefig(os.path.join(FIGURES_DIR, f"fig8_tsne_embedding.{fmt}"),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Figure 8 saved.")


# ======================== FIGURE 9: DISORDER PROFILES ========================

def plot_disorder_profiles():
    """Radar charts showing digital phenotype profiles."""
    logger.info("Generating Figure 9: Disorder Profiles...")

    profiles = _safe_load_json(os.path.join(STATS_DIR, "disorder_profiles.json"))
    if profiles is None:
        logger.error("  Disorder profiles not found. Skipping Figure 9.")
        return

    feat_to_dim, dim_order = _load_dimension_mapping()

    # Collect unique dimensions that actually appear in profiles
    all_dims = set()
    for disorder, profile in profiles.items():
        all_dims.update(profile.get("dimension_summary", {}).keys())
    dimensions = [d for d in dim_order if d in all_dims]
    if not dimensions:
        dimensions = sorted(all_dims)
    n_dims = len(dimensions)

    if n_dims < 3:
        logger.warning("  Too few dimensions for radar chart. Skipping.")
        return

    n_cols = 5
    n_rows = 2
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(DOUBLE_COL, DOUBLE_COL * 0.5),
        subplot_kw=dict(projection='polar')
    )

    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]

    disorder_list = [d for d in DISORDER_GROUPS if d in profiles]

    for idx, disorder in enumerate(disorder_list):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        profile = profiles[disorder]
        dim_summary = profile.get("dimension_summary", {})

        values = []
        for dim in dimensions:
            if dim in dim_summary:
                values.append(dim_summary[dim].get("mean_abs_g", 0))
            else:
                values.append(0.0)
        values += values[:1]

        color = get_disorder_color(disorder)
        display_name = get_disorder_display(disorder)

        ax.fill(angles, values, alpha=0.25, color=color)
        ax.plot(angles, values, color=color, linewidth=1.0)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([d[:6] for d in dimensions], fontsize=3)
        ax.set_ylim(0, max(max(values) * 1.2, 0.5))
        ax.set_title(display_name, fontsize=6, fontweight='bold',
                     pad=10, color=color)
        ax.tick_params(axis='y', labelsize=3)

    for idx in range(len(disorder_list), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    plt.suptitle("Disorder Digital Phenotype Profiles (Mean |Hedges' g| by Dimension)",
                 fontsize=9, fontweight='bold', y=1.02)
    plt.tight_layout()
    for fmt in ['pdf', 'png', 'svg']:
        fig.savefig(os.path.join(FIGURES_DIR, f"fig9_disorder_profiles.{fmt}"),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Figure 9 saved.")


# ======================== FIGURE 10: SENSITIVITY SUMMARY ========================

def plot_sensitivity_summary():
    """
    [FIX v1.3] Redesigned Sensitivity Summary Figure.
    
    Key fixes:
      - JSON string keys → explicit float conversion before arithmetic
      - Explicit tick control on every axis
      - Horizontal bars for Panel C
      - Shortened detail text in Panel A
      - Reduced legend in Panel B
    """
    logger.info("Generating Figure 10: Sensitivity Summary...")

    report = _safe_load_json(os.path.join(STATS_DIR, "sensitivity_report.json"))
    if report is None:
        logger.warning("  Sensitivity report not found. Skipping Figure 10.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.75))

    # ============================================================
    # Panel A: Robustness Checklist
    # ============================================================
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

            # --- Shorten detail text ---
            if 'r = ' in detail:
                try:
                    r_val = float(detail.split('r = ')[1].strip())
                    detail = f"r = {r_val:.3f}"
                except (ValueError, IndexError):
                    pass
            elif 'high I' in detail or 'I²' in detail:
                pass  # keep as-is
            elif len(detail) > 25:
                detail = detail[:23] + "..."

            # --- Colors ---
            if status == 'PASS':
                bg_color = '#E8F5E9'
                symbol_color = '#2E7D32'
                symbol = '✓'
            else:
                bg_color = '#FFF3E0'
                symbol_color = '#E65100'
                symbol = '⚠'

            # --- Draw row ---
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

    ax_check.set_title("A. Robustness Checklist",
                       fontsize=8, fontweight='bold', loc='left', pad=8)

    # ============================================================
    # Panel B: Sample Size Stability
    # [CRITICAL FIX] JSON keys are strings → must convert to float
    # ============================================================
    ax_stab = axes[0, 1]
    stability = report.get("sample_size_stability", {})

    if stability:
        disorder_data = {}
        for disorder, fracs in stability.items():
            if not fracs:
                continue

            # [CRITICAL FIX] Convert JSON string keys to float
            # JSON serializes {0.25: ...} as {"0.25": ...}
            # Without this, "0.25" * 100 = "0.250.25..." (string repeat!)
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

            # [CRITICAL FIX] Now f is float, f*100 = 25.0 (not string repeat)
            disorder_data[disorder] = {
                'x_vals': [f * 100 for f in fracs_sorted],  # [25, 50, 75, 100]
                'aurocs': [fracs_numeric[f]["mean_auroc"] for f in fracs_sorted],
                'display': get_disorder_display(disorder),
                'color': get_disorder_color(disorder),
            }

        # Plot
        for disorder, data in disorder_data.items():
            ax_stab.plot(data['x_vals'], data['aurocs'],
                         marker='o', markersize=2.5, linewidth=0.7,
                         color=data['color'], alpha=0.7,
                         label=data['display'])

        # [FIX] Explicit x-ticks
        ax_stab.set_xticks([25, 50, 75, 100])
        ax_stab.set_xticklabels(['25%', '50%', '75%', '100%'], fontsize=5)

        # [FIX] Explicit y-ticks: compute tight range, 5 ticks max
        all_aurocs = [a for d in disorder_data.values() for a in d['aurocs']]
        if all_aurocs:
            y_lo = np.floor(min(all_aurocs) * 50) / 50  # round down to 0.02
            y_hi = np.ceil(max(all_aurocs) * 50) / 50    # round up to 0.02
            if y_hi - y_lo < 0.04:
                y_lo -= 0.02
                y_hi += 0.02
            ax_stab.set_ylim(y_lo, y_hi)
            n_y_ticks = 5
            y_step = (y_hi - y_lo) / (n_y_ticks - 1)
            y_ticks = np.arange(y_lo, y_hi + y_step * 0.1, y_step)
            ax_stab.set_yticks(y_ticks)
            ax_stab.set_yticklabels([f'{v:.2f}' for v in y_ticks], fontsize=5)

        ax_stab.set_xlabel("Sample Fraction", fontsize=6)
        ax_stab.set_ylabel("AUROC", fontsize=6)

        # [FIX] Legend: top 5 by AUROC spread only
        if disorder_data:
            sorted_dis = sorted(disorder_data.items(),
                                key=lambda x: max(x[1]['aurocs']) -
                                              min(x[1]['aurocs']),
                                reverse=True)
            top5 = sorted_dis[:5]
            handles = [plt.Line2D([0], [0], color=d['color'],
                                  linewidth=0.8, marker='o', markersize=3,
                                  label=d['display'])
                       for _, d in top5]
            ax_stab.legend(handles=handles, fontsize=3.5,
                           loc='lower right', framealpha=0.85,
                           ncol=2, columnspacing=0.5, handletextpad=0.3)

    ax_stab.set_title("B. Sample Size Stability",
                      fontsize=8, fontweight='bold', loc='left', pad=8)
    ax_stab.spines['top'].set_visible(False)
    ax_stab.spines['right'].set_visible(False)
    ax_stab.tick_params(axis='both', labelsize=5)

    # ============================================================
    # Panel C: Multiple Comparison (HORIZONTAL bars)
    # ============================================================
    ax_mc = axes[1, 0]
    mc = report.get("multiple_comparison", {})

    if mc:
        methods = list(mc.keys())
        n_sig = [mc[m]["n_significant"] for m in methods]
        total = mc[methods[0]]["n_total"] if methods else 1

        display_map = {
            'none': 'None',
            'bonferroni': 'Bonferroni',
            'holm': 'Holm',
            'fdr_bh': 'FDR-BH',
            'fdr_by': 'FDR-BY',
        }
        short_names = [display_map.get(m, m) for m in methods]

        color_map = {
            'none': '#9E9E9E',
            'bonferroni': '#E53935',
            'holm': '#FF7043',
            'fdr_bh': '#1E88E5',
            'fdr_by': '#42A5F5',
        }
        colors_mc = [color_map.get(m, '#757575') for m in methods]

        y_pos = np.arange(len(methods))
        bars = ax_mc.barh(y_pos, n_sig, color=colors_mc,
                          edgecolor='black', linewidth=0.3, height=0.55)

        for i, (val, name) in enumerate(zip(n_sig, short_names)):
            pct = val / total * 100
            ax_mc.text(val + total * 0.01, i,
                       f'{val} ({pct:.0f}%)',
                       va='center', fontsize=4.5, color='#333333')

        ax_mc.set_yticks(y_pos)
        ax_mc.set_yticklabels(short_names, fontsize=5.5)

        # [FIX] X-axis: 3-4 clean integer ticks
        max_sig = max(n_sig) if n_sig else total
        x_step = max(1, int(np.ceil(max_sig / 4 / 50) * 50))
        x_ticks = list(range(0, max_sig + x_step, x_step))
        ax_mc.set_xticks(x_ticks)
        ax_mc.set_xticklabels([str(v) for v in x_ticks], fontsize=5)
        ax_mc.set_xlabel("N Significant Tests", fontsize=6)

        ax_mc.axvline(x=total, color='#BDBDBD', linestyle='--', linewidth=0.5)
        ax_mc.text(total, -0.6, f'N={total}', fontsize=4,
                   color='#9E9E9E', ha='center')

    ax_mc.set_title("C. Multiple Comparison Sensitivity",
                    fontsize=8, fontweight='bold', loc='left', pad=8)
    ax_mc.spines['top'].set_visible(False)
    ax_mc.spines['right'].set_visible(False)
    ax_mc.tick_params(axis='both', labelsize=5)

    # ============================================================
    # Panel D: Heterogeneity Distribution
    # ============================================================
    ax_het = axes[1, 1]
    het = report.get("heterogeneity", {})

    if het:
        mean_i2 = het.get('mean_I_squared', 0)

        categories = ['Low\n(I²<25%)', 'Moderate\n(25-75%)', 'High\n(I²>75%)']
        values = [het.get('pct_low', 0),
                  het.get('pct_moderate', 0),
                  het.get('pct_high', 0)]
        colors_het = ['#4CAF50', '#FF9800', '#E53935']

        bars_h = ax_het.bar(range(3), values, color=colors_het,
                            edgecolor='black', linewidth=0.3, width=0.55)

        for bar_h, val in zip(bars_h, values):
            ax_het.text(bar_h.get_x() + bar_h.get_width() / 2,
                       val + 1.5,
                       f'{val:.0f}%', ha='center', fontsize=5.5,
                       fontweight='bold')

        ax_het.set_xticks(range(3))
        ax_het.set_xticklabels(categories, fontsize=5)

        # [FIX] Y-axis: clean integer ticks
        y_max = max(values) * 1.15 if max(values) > 0 else 100
        ax_het.set_ylim(0, y_max)
        y_step = max(5, int(np.ceil(y_max / 4 / 5) * 5))
        y_ticks = list(range(0, int(y_max) + y_step, y_step))
        ax_het.set_yticks(y_ticks)
        ax_het.set_yticklabels([str(v) for v in y_ticks], fontsize=5)
        ax_het.set_ylabel("% of Features", fontsize=6)

        ax_het.text(0.97, 0.92,
                    f'Mean I² = {mean_i2:.0f}%',
                    transform=ax_het.transAxes, ha='right', va='top',
                    fontsize=5.5, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3',
                             facecolor='#FFF3E0',
                             edgecolor='#FF9800', alpha=0.9))

        ax_het.text(0.97, 0.72,
                    'High I² reflects genuine\ndisorder heterogeneity,\nnot method failure',
                    transform=ax_het.transAxes, ha='right', va='top',
                    fontsize=3.5, color='#757575', style='italic')

    ax_het.set_title("D. Effect Size Heterogeneity",
                     fontsize=8, fontweight='bold', loc='left', pad=8)
    ax_het.spines['top'].set_visible(False)
    ax_het.spines['right'].set_visible(False)
    ax_het.tick_params(axis='both', labelsize=5)

    # ============================================================
    # Overall assessment banner
    # ============================================================
    overall = report.get("overall_robustness", "")
    is_pass = '2/3' in overall or '3/3' in overall
    fig.text(0.5, 0.005,
             f"Overall Robustness: {overall}",
             ha='center', fontsize=7, fontweight='bold',
             color='#2E7D32' if is_pass else '#E65100')

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    for fmt in ['pdf', 'png', 'svg']:
        fig.savefig(os.path.join(FIGURES_DIR,
                                f"fig10_sensitivity_summary.{fmt}"),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Figure 10 saved.")


# ======================== MAIN ========================

def run_visualization():
    """Generate all figures for Module 1."""
    logger.info("=" * 80)
    logger.info("MODULE 1 - Step 07: Visualization (v1.2)")
    logger.info("=" * 80)

    figures = [
        ("Figure 1: Effect Size Heatmap", plot_effect_size_heatmap),
        ("Figure 2: AUROC Matrix", plot_auroc_matrix),
        ("Figure 3: Disorder Clustering", plot_clustering),
        ("Figure 4: Factor Analysis", plot_factor_loadings),
        ("Figure 5: Transdiagnostic vs Specific", plot_transdiagnostic_specific),
        ("Figure 6: Dimension Ablation", plot_dimension_ablation),
        ("Figure 7: Confusion Matrix", plot_confusion_matrix),
        ("Figure 8: t-SNE Embedding", plot_tsne_embedding),
        ("Figure 9: Disorder Profiles", plot_disorder_profiles),
        ("Figure 10: Sensitivity Summary", plot_sensitivity_summary),
    ]

    n_success = 0
    n_fail = 0

    for name, func in figures:
        try:
            func()
            n_success += 1
        except Exception as e:
            logger.error(f"{name} failed: {e}")
            n_fail += 1

    logger.info(f"\nVisualization complete: {n_success} succeeded, {n_fail} failed")
    logger.info(f"Output directory: {FIGURES_DIR}")


if __name__ == "__main__":
    run_visualization()