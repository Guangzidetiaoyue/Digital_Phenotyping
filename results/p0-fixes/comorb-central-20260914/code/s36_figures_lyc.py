"""
Module 3 Publication Figures (lyc) — Nature/Science-grade
=========================================================
Main -> module3_results/figures_lyc/main/
SI   -> module3_results/figures_lyc/ed/

CLI: python s36_figures_lyc.py -p fig1 edfig3
      python s36_figures_lyc.py -p all
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
from matplotlib.patheffects import withStroke
from matplotlib.lines import Line2D
from scipy import stats

from config_module3 import (
    STATS_DIR as M3_STATS,
    FIGURES_DIR,
    DISORDER_DISPLAY_NAMES,
    HITOP_SPECTRA,
    HITOP_COLORS,
    DSM5_CHAPTERS,
    ComorbidityMatrix,
    RANDOM_SEED,
)

# External, per-pair CITED comorbidity dict (single source; no uncited fallback).
CLINICAL_COMORBIDITY = ComorbidityMatrix().pairs

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
logging.getLogger("fontTools.subset").setLevel(logging.WARNING)

# ---- global style (设计参考.md / Module1 lyc) ----
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
    "savefig.bbox": None,
    "savefig.pad_inches": 0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

DISPLAY = dict(DISORDER_DISPLAY_NAMES)


def get_disorder_color(d):
    for s, ds in HITOP_SPECTRA.items():
        if d in ds:
            return HITOP_COLORS.get(s, "#808080")
    return "#808080"


def _save(fig, name, subdir="main"):
    """Save the designed canvas unchanged; never crop fixed margins."""
    d = MAIN_DIR if subdir == "main" else ED_DIR
    for fmt in ("pdf", "png"):
        fig.savefig(
            os.path.join(d, f"{name}.{fmt}"),
            dpi=300,
            bbox_inches=None,
            pad_inches=0,
        )
    plt.close(fig)
    logger.info(f"  Saved: {subdir}/{name}")


def _despine(ax, top=True, right=True, left=False, bottom=False):
    if top: ax.spines["top"].set_visible(False)
    if right: ax.spines["right"].set_visible(False)
    if left: ax.spines["left"].set_visible(False)
    if bottom: ax.spines["bottom"].set_visible(False)
    ax.tick_params(direction="out")


def _require_columns(df, required, source):
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")


def _csv(name, required=(), **kw):
    p = os.path.join(M3_STATS, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required figure input not found: {p}")
    df = pd.read_csv(p, **kw)
    _require_columns(df, required, name)
    return df


def _json(name, required=()):
    p = os.path.join(M3_STATS, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required figure input not found: {p}")
    with open(p) as f:
        data = json.load(f)
    missing = sorted(set(required) - set(data))
    if missing:
        raise ValueError(f"{name} missing required keys: {missing}")
    return data


def _add_axes_inches(fig, fig_w, fig_h, left, bottom, width, height):
    """Add an axes using physical-inch geometry."""
    return fig.add_axes([
        left / fig_w, bottom / fig_h, width / fig_w, height / fig_h
    ])


def _three_panel_figure(left_text=0.55, right_text=0.25,
                        bottom_text=0.50, top_text=0.40):
    """Fixed top-wide + two-bottom-panel layout from lyc specification."""
    blank = 0.35
    left = blank + left_text
    bottom = blank + bottom_text
    plot_w, plot_h = 2.35, 2.05
    top_w, top_h = 5.45, 2.35
    col_gap, row_gap = 0.75, 1.10
    fig_w = left + top_w + right_text + blank
    fig_h = bottom + plot_h + row_gap + top_h + top_text + blank
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax_a = _add_axes_inches(
        fig, fig_w, fig_h, left, bottom + plot_h + row_gap, top_w, top_h
    )
    ax_b = _add_axes_inches(fig, fig_w, fig_h, left, bottom, plot_w, plot_h)
    ax_c = _add_axes_inches(
        fig, fig_w, fig_h, left + plot_w + col_gap, bottom, plot_w, plot_h
    )
    return fig, (ax_a, ax_b, ax_c)


def _single_square_figure(left=0.90, right=0.45, bottom=0.75, top=0.65,
                          plot_size=3.60):
    fig_w = left + plot_size + right
    fig_h = bottom + plot_size + top
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = _add_axes_inches(fig, fig_w, fig_h, left, bottom, plot_size, plot_size)
    return fig, ax


def _unordered_lookup(mapping, a, b, default=np.nan):
    """Look up pair-valued data without assuming tuple orientation."""
    return mapping.get((a, b), mapping.get((b, a), default))


def _geometry_summary_from_embedding(emb, n_perm=10000):
    """Recompute displayed geometry metrics from the plotted embedding.

    This avoids silently plotting stale reports produced by older versions of
    s32 that had an ineffective permutation and orientation-sensitive pair
    lookup.
    """
    pat = emb[emb["group"] == "patient"].dropna(
        subset=["disorder_group", "umap_x", "umap_y"]
    )
    disorders = sorted(set(pat["disorder_group"]) & set(DISPLAY))
    if len(disorders) < 3:
        raise ValueError("At least three patient disorders are required")
    centroids = {
        d: pat.loc[
            pat["disorder_group"] == d, ["umap_x", "umap_y"]
        ].to_numpy().mean(axis=0)
        for d in disorders
    }
    spectrum_of = {
        d: spectrum for spectrum, members in HITOP_SPECTRA.items()
        for d in members
    }
    dsm_of = {
        d: chapter for chapter, members in DSM5_CHAPTERS.items()
        for d in members
    }
    distances, within, between = [], [], []
    hitop_same, dsm_same, comorbidity = [], [], []
    for i, a in enumerate(disorders):
        for b in disorders[i + 1:]:
            distance = float(np.linalg.norm(centroids[a] - centroids[b]))
            distances.append(distance)
            same_spectrum = float(spectrum_of[a] == spectrum_of[b])
            hitop_same.append(same_spectrum)
            dsm_same.append(float(dsm_of[a] == dsm_of[b]))
            comorbidity.append(_unordered_lookup(CLINICAL_COMORBIDITY, a, b))
            (within if same_spectrum else between).append(distance)

    within = np.asarray(within)
    between = np.asarray(between)
    observed = abs(within.mean() - between.mean())
    pooled = np.concatenate([within, between])
    rng = np.random.RandomState(RANDOM_SEED)
    count = 0
    for _ in range(n_perm):
        permuted = rng.permutation(pooled)
        delta = abs(
            permuted[:len(within)].mean() - permuted[len(within):].mean()
        )
        count += delta >= observed
    perm_p = (count + 1) / (n_perm + 1)

    distances = np.asarray(distances)
    comorbidity = np.asarray(comorbidity, dtype=float)
    valid_comorb = np.isfinite(comorbidity)
    structural = {
        "hitop_r": float(np.corrcoef(distances, hitop_same)[0, 1]),
        "dsm5_r": float(np.corrcoef(distances, dsm_same)[0, 1]),
        "comorbidity_r": float(
            np.corrcoef(distances[valid_comorb], comorbidity[valid_comorb])[0, 1]
        ),
    }
    return {
        "spectrum_separation": {
            "within_mean": float(within.mean()),
            "between_mean": float(between.mean()),
            "perm_p": float(perm_p),
        },
        "structural_fit": structural,
    }


def _prepare_bridge_data(bridge):
    """Normalize legacy bridge tables to a common all-disorder denominator."""
    first_col = bridge.columns[0]
    if first_col in ("mean_bridge", "n_disorders_bridge"):
        raise ValueError("bridge_centrality_by_disorder.csv has no feature column")
    bridge = bridge.rename(columns={first_col: "feature"}).copy()
    disorder_cols = [d for d in DISPLAY if d != "control"]
    for disorder in disorder_cols:
        if disorder not in bridge:
            logger.warning(
                "Treating absent bridge column '%s' as zero (legacy output)",
                disorder,
            )
            bridge[disorder] = 0.0
    bridge[disorder_cols] = bridge[disorder_cols].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0.0)
    bridge["mean_bridge"] = bridge[disorder_cols].mean(axis=1)
    bridge["n_disorders_bridge"] = (bridge[disorder_cols] > 0).sum(axis=1)
    return bridge, disorder_cols


def _mantel_from_pair_table(pw, n_perm=10000):
    """Compute a proper matrix-label Mantel test from pairwise AUROCs."""
    disorders = sorted(set(pw["disorder_a"]) | set(pw["disorder_b"]))
    index = {d: i for i, d in enumerate(disorders)}
    auroc = np.full((len(disorders), len(disorders)), np.nan)
    comorbidity = np.full_like(auroc, np.nan)
    for _, row in pw.iterrows():
        i, j = index[row["disorder_a"]], index[row["disorder_b"]]
        auroc[i, j] = auroc[j, i] = row["auroc"]
    for i, a in enumerate(disorders):
        for j, b in enumerate(disorders):
            if i != j:
                comorbidity[i, j] = _unordered_lookup(
                    CLINICAL_COMORBIDITY, a, b
                )
    tri = np.triu_indices_from(auroc, k=1)
    valid = np.isfinite(auroc[tri]) & np.isfinite(comorbidity[tri])
    observed = float(np.corrcoef(auroc[tri][valid], comorbidity[tri][valid])[0, 1])
    rng = np.random.RandomState(RANDOM_SEED)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(disorders))
        perm_comorbidity = comorbidity[np.ix_(perm, perm)]
        perm_valid = np.isfinite(auroc[tri]) & np.isfinite(perm_comorbidity[tri])
        perm_r = np.corrcoef(
            auroc[tri][perm_valid], perm_comorbidity[tri][perm_valid]
        )[0, 1]
        count += abs(perm_r) >= abs(observed)
    return observed, (count + 1) / (n_perm + 1)


# =====================================================================
#  FIGURE M3-1: Pairwise distinguishability (M1 fig3-C style heatmap)
# =====================================================================
def fig1_pairwise():
    """(a) 45-pair AUROC heatmap in M1 fig3-C style (in-cell values,
    RdBu_r diverging, HiTOP-colored labels); (b) per-disorder mean ± SD."""
    pw = _csv(
        "pairwise_distinguishability.csv",
        required=("disorder_a", "disorder_b", "auroc"),
    )
    mantel_r, mantel_p = _mantel_from_pair_table(pw)

    # Fixed physical geometry: square heatmap + tall forest panel.
    fig_w, fig_h = 7.80, 4.75
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = _add_axes_inches(fig, fig_w, fig_h, 1.00, 1.05, 2.80, 2.80)

    # (a) AUROC heatmap with in-cell values
    disorders = sorted(set(pw["disorder_a"]) | set(pw["disorder_b"]))
    mat = np.full((len(disorders), len(disorders)), np.nan)
    for _, r in pw.iterrows():
        i, j = disorders.index(r["disorder_a"]), disorders.index(r["disorder_b"])
        mat[i, j] = mat[j, i] = r["auroc"]
    np.fill_diagonal(mat, np.nan)

    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        raise ValueError("pairwise_distinguishability.csv has no finite AUROC values")
    vmin = min(0.5, float(finite.min()))
    vmax = max(0.85, float(finite.max()))
    vcenter = min(max(0.7, vmin + 1e-6), vmax - 1e-6)
    im = ax.imshow(
        mat, cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax),
        aspect="auto",
    )
    ax.set_box_aspect(1)
    ax.set_xlim(-0.5, len(disorders) - 0.5)
    ax.set_ylim(len(disorders) - 0.5, -0.5)

    threshold = max(abs(vmin - vcenter), abs(vmax - vcenter)) * 0.58
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "\u2014", ha="center", va="center",
                        fontsize=FS_ANNOT, color="#AAAAAA")
            else:
                color = "white" if abs(v - vcenter) >= threshold else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=FS_ANNOT, color=color)

    ax.set_xticks(range(len(disorders)))
    ax.set_xticklabels(
        [DISPLAY.get(d, d) for d in disorders],
        rotation=45, ha="right", rotation_mode="anchor",
        fontsize=FS_TICK, fontweight="bold",
    )
    ax.set_yticks(range(len(disorders)))
    ax.set_yticklabels(
        [DISPLAY.get(d, d) for d in disorders],
        fontsize=FS_TICK, fontweight="bold",
    )
    for labels, keys in (
        (ax.get_xticklabels(), disorders),
        (ax.get_yticklabels(), disorders),
    ):
        for label, disorder in zip(labels, keys):
            label.set_color(get_disorder_color(disorder))
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#222222")
        spine.set_linewidth(1.0)
    cax = _add_axes_inches(fig, fig_w, fig_h, 3.88, 1.44, 0.10, 2.02)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("AUROC", fontsize=FS_AXLABEL)
    cb.ax.tick_params(labelsize=FS_TICK)
    cb.outline.set_linewidth(1.0)
    cb.outline.set_edgecolor("#222222")
    ax.set_title("A. Pairwise distinguishability", loc="left", pad=20)

    # (b) Per-disorder mean AUROC + Mantel
    ax = _add_axes_inches(fig, fig_w, fig_h, 5.30, 1.05, 2.10, 2.80)
    summary = []
    for d in disorders:
        vals = pw.loc[
            (pw["disorder_a"] == d) | (pw["disorder_b"] == d), "auroc"
        ].dropna()
        if not vals.empty:
            summary.append((d, vals.mean(), vals.std(ddof=1)))
    summary.sort(key=lambda row: row[1])
    ys = np.arange(len(summary))
    for yi, (d, mean_auc, sd_auc) in enumerate(summary):
        ax.errorbar(
            mean_auc, yi, xerr=sd_auc, fmt="o",
            color=get_disorder_color(d), ms=5, capsize=2,
            elinewidth=0.8, markeredgecolor="white", markeredgewidth=0.5,
        )
    grand_mean = float(np.mean([row[1] for row in summary]))
    ax.axvline(grand_mean, color="#999999", lw=0.6, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels(
        [DISPLAY.get(row[0], row[0]) for row in summary],
        fontsize=FS_TICK, fontweight="bold",
    )
    for label, (d, _, _) in zip(ax.get_yticklabels(), summary):
        label.set_color(get_disorder_color(d))
    limits = [
        value
        for _, mean_auc, sd_auc in summary
        for value in (mean_auc - sd_auc, mean_auc + sd_auc)
        if np.isfinite(value)
    ]
    span = max(limits) - min(limits)
    pad = max(0.02, span * 0.12)
    ax.set_xlim(max(0.5, min(limits) - pad), min(1.0, max(limits) + pad))
    ax.set_xlabel("Mean pairwise AUROC (±SD)")
    ax.text(
        0.98, 0.04,
        f"Mantel r={mantel_r:.2f}, p={mantel_p:.3f}",
        transform=ax.transAxes, fontsize=FS_ANNOT, ha="right",
        va="bottom", style="italic",
    )
    ax.set_title("B. Mean distinguishability", loc="left", pad=20)
    _despine(ax)

    fig.text(
        0.5, 0.955,
        "Cross-disorder distinguishability of digital phenotypes",
        ha="center", va="top", fontsize=FS_SUPTITLE, fontweight="bold",
    )
    _save(fig, "fig1_pairwise_distinguishability")


# =====================================================================
#  FIGURE M3-2: Phenotype geometry (M1 MDS-style + t-SNE small multiples)
# =====================================================================
def fig2_geometry():
    """(a) UMAP universe in M1 MDS style (bubbles + leader lines + HiTOP
    hulls); (b) spectrum separation; (c) structural fit."""
    emb = _csv(
        "phenotype_universe_embedding.csv",
        required=("group", "disorder_group", "umap_x", "umap_y"),
    )
    geo = _geometry_summary_from_embedding(emb)

    fig, (ax, ax_b, ax_c) = _three_panel_figure(
        left_text=0.55, right_text=0.30, bottom_text=0.48, top_text=0.52
    )

    # (a) UMAP universe (M1 MDS aesthetic)
    ctrl = emb[emb["group"] == "control"]
    pat = emb[emb["group"] == "patient"]
    if pat.empty:
        raise ValueError("phenotype_universe_embedding.csv has no patient rows")
    ax.scatter(ctrl["umap_x"], ctrl["umap_y"], c="#EEEEEE", s=2, alpha=0.3,
               rasterized=True)
    # HiTOP hulls (patients only)
    for sp, ds in HITOP_SPECTRA.items():
        mem = [d for d in ds if d in set(pat["disorder_group"].dropna())]
        if not mem:
            continue
        co = pat[pat["disorder_group"].isin(mem)][["umap_x", "umap_y"]].values
        if len(co) >= 3:
            try:
                from scipy.spatial import ConvexHull
                h = ConvexHull(co)
                hp = np.vstack([co[h.vertices], co[h.vertices[0]]])
                ax.plot(hp[:, 0], hp[:, 1], color=HITOP_COLORS[sp], lw=1.0,
                        ls="--", alpha=0.5, zorder=2)
                ax.fill(hp[:, 0], hp[:, 1], color=HITOP_COLORS[sp], alpha=0.05,
                        zorder=1)
            except Exception:
                pass
    # disorder centroids with leader lines
    label_adjust = {
        "Depression": (-50, 38, "right", "bottom"),
        "Bipolar": (-18, -55, "right", "top"),
        "Borderline PD": (-62, -25, "right", "top"),
        "PTSD": (15, 52, "left", "bottom"),
        "OCD": (58, 34, "left", "bottom"),
        "Anxiety": (68, -15, "left", "top"),
        "Eating Disorders": (38, -52, "left", "top"),
        "ADHD": (72, 8, "left", "center"),
        "Schizophrenia": (-68, 8, "right", "center"),
        "Autism": (-18, 62, "right", "bottom"),
    }
    for d in DISPLAY:
        if d == "control":
            continue
        sub = pat[pat["disorder_group"] == d]
        if sub.empty:
            continue
        cx, cy = sub["umap_x"].mean(), sub["umap_y"].mean()
        c = get_disorder_color(d)
        ax.scatter(cx, cy, c=c, s=60, edgecolors="#FFFFFF", lw=1.0, zorder=10)
        dn = DISPLAY.get(d, d)
        off = label_adjust.get(dn, (15, 15, "left", "bottom"))
        txt = ax.annotate(
            dn, (cx, cy), xytext=(off[0], off[1]),
            textcoords="offset points", ha=off[2], va=off[3],
            fontsize=FS_TICK, color=c, fontweight="bold", zorder=15,
            annotation_clip=False,
            arrowprops=dict(
                arrowstyle="-", color=c, lw=1.0, alpha=0.6,
                shrinkA=0, shrinkB=4,
            ),
        )
        txt.set_path_effects([withStroke(linewidth=2.5, foreground="white")])
    ax.margins(x=0.10, y=0.13)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("A. Digital phenotype universe", loc="left", pad=20)
    _despine(ax, left=True, bottom=True)

    # (b) Spectrum separation
    ax = ax_b
    sep = geo.get("spectrum_separation", {})
    w, b = sep.get("within_mean", np.nan), sep.get("between_mean", np.nan)
    if not np.isfinite([w, b]).all():
        raise ValueError("phenotype_geometry_report.json has invalid separation means")
    bars = ax.bar(
        ["Within", "Between"], [w, b], width=0.55,
        color=["#4DBBD5", "#E64B35"], edgecolor="#222222", linewidth=0.6,
    )
    ymax = max(w, b) * 1.32
    for bar, value in zip(bars, (w, b)):
        ax.text(bar.get_x() + bar.get_width() / 2, value + ymax * 0.025,
                f"{value:.2f}", ha="center", fontsize=FS_ANNOT)
    ax.set_ylabel("Centroid distance")
    ax.set_ylim(0, ymax)
    ax.text(
        0.98, 0.96, f"Permutation p={sep.get('perm_p', np.nan):.3f}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=FS_ANNOT, style="italic",
    )
    ax.tick_params(axis="x", labelsize=FS_TICK)
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    ax.set_title("B. Spectrum separation", loc="left", pad=20)
    _despine(ax)

    # (c) Structural fit
    ax = ax_c
    sf = geo.get("structural_fit", {})
    labels = ["HiTOP", "DSM-5", "Comorb."]
    rs = np.array([
        sf.get("hitop_r", np.nan),
        sf.get("dsm5_r", np.nan),
        sf.get("comorbidity_r", np.nan),
    ], dtype=float)
    if not np.isfinite(rs).all():
        raise ValueError("Embedding-derived structural-fit correlations are invalid")
    x = np.arange(3)
    bars = ax.bar(
        x, rs, width=0.55,
        color=["#9B59B6", "#3498DB", "#E74C3C"],
        edgecolor="#222222", linewidth=0.6,
    )
    ax.axhline(0, color="#999999", lw=0.6)
    yrange = max(0.6, float(np.max(np.abs(rs))) * 1.35)
    ax.set_ylim(-yrange, yrange)
    for bar, value in zip(bars, rs):
        offset = 0.035 * yrange
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (offset if value >= 0 else -offset),
            f"{value:.2f}", ha="center",
            va="bottom" if value >= 0 else "top", fontsize=FS_ANNOT,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FS_TICK, fontweight="bold")
    ax.set_ylabel("Mantel correlation (r)")
    ax.set_title("C. Structural fit", loc="left", pad=20)
    _despine(ax)

    fig.suptitle(
        "Geometry of the digital phenotype space",
        fontsize=FS_SUPTITLE, fontweight="bold", y=0.975,
    )
    _save(fig, "fig2_phenotype_geometry")


# =====================================================================
#  FIGURE M3-3: Disorder networks
# =====================================================================
def fig3_networks():
    """(a) network density vs n_users (sample-size effect), (b) bridge
    features, (c) edge consistency."""
    topo = _csv(
        "disorder_network_topology.csv",
        required=("disorder", "n_users", "density"),
    )
    bridge = _csv(
        "bridge_centrality_by_disorder.csv",
        required=("mean_bridge", "n_disorders_bridge"),
    )
    ec = _csv("network_edge_consistency.csv", required=("n_disorders",))

    fig, (ax, ax_b, ax_c) = _three_panel_figure(
        left_text=1.35, right_text=0.30, bottom_text=0.50, top_text=0.50
    )

    # (a) density vs n_users
    xpad = max(float(topo["n_users"].max()) * 0.015, 1)
    for _, row in topo.iterrows():
        c = get_disorder_color(row["disorder"])
        ax.scatter(
            row["n_users"], row["density"], s=45, c=c,
            edgecolors="white", lw=0.5, zorder=3,
        )
        ax.annotate(
            DISPLAY.get(row["disorder"], row["disorder"]),
            (row["n_users"], row["density"]),
            xytext=(5, 0), textcoords="offset points",
            fontsize=FS_ANNOT, va="center", color=c, fontweight="bold",
            annotation_clip=False,
        )
    ax.set_xlabel("Users per disorder")
    ax.set_ylabel("Network density")
    ax.set_xlim(
        max(0, float(topo["n_users"].min()) - xpad * 3),
        float(topo["n_users"].max()) + xpad * 18,
    )
    ax.margins(y=0.15)
    ax.set_title("A. Density versus sample size", loc="left", pad=20)
    _despine(ax)

    # (b) bridge features
    ax = ax_b
    bridge, _ = _prepare_bridge_data(bridge)
    top = bridge.sort_values("mean_bridge", ascending=False).head(8)
    ax.barh(
        [str(c)[:28] for c in top["feature"]], top["mean_bridge"],
        color="#E64B35", height=0.70, edgecolor="#222222", linewidth=0.6,
    )
    ax.set_xlabel("Mean bridge centrality")
    ax.set_xlim(left=0, right=max(top["mean_bridge"].max() * 1.12, 1e-6))
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=FS_TICK)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.set_title("B. Bridge features", loc="left", pad=20)
    _despine(ax)

    # (c) edge consistency
    ax = ax_c
    counts = ec["n_disorders"].value_counts().sort_index()
    xvals = counts.index.astype(int).to_numpy()
    yvals = counts.to_numpy()
    ax.plot(xvals, yvals, "o-", color="#3C5488", ms=4, lw=1.5)
    ax.fill_between(xvals, 0, yvals, alpha=0.15, color="#3C5488")
    ax.set_xlabel("Disorders sharing edge")
    ax.set_ylabel("Number of edges")
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(bottom=0)
    ax.set_title("C. Edge consistency", loc="left", pad=20)
    _despine(ax)

    fig.suptitle(
        "Cross-disorder digital phenotype networks",
        fontsize=FS_SUPTITLE, fontweight="bold", y=0.975,
    )
    _save(fig, "fig3_disorder_networks")


# =====================================================================
#  FIGURE M3-4: Comorbidity
# =====================================================================
def fig4_comorbidity():
    """(a) n-disorders histogram, (b) top comorbidity pairs (small multiples
    style bar), (c) intermediate fraction per pair."""
    stats_ = _json(
        "multi_diagnosis_stats.json",
        required=("n_disorders_dist", "top_comorbidity_pairs", "pct_users_multi"),
    )
    inter = _csv(
        "comorbidity_intermediate_results.csv",
        required=("disorder_a", "disorder_b", "frac_intermediate"),
    )
    inter = inter.dropna(subset=["frac_intermediate"]).sort_values(
        "frac_intermediate"
    )
    if inter.empty:
        raise ValueError("comorbidity_intermediate_results.csv has no finite fractions")

    # A/B are stacked on the left; the many-row pair panel gets the full height.
    fig_w, fig_h = 7.80, 6.70
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = _add_axes_inches(fig, fig_w, fig_h, 1.55, 3.75, 2.35, 2.05)
    ax_b = _add_axes_inches(fig, fig_w, fig_h, 1.55, 0.75, 2.35, 2.05)
    ax_c = _add_axes_inches(fig, fig_w, fig_h, 4.85, 0.75, 2.55, 5.05)

    # (a) n_disorders histogram
    dist = stats_.get("n_disorders_dist", {})
    if not dist:
        raise ValueError("multi_diagnosis_stats.json n_disorders_dist is empty")
    zero_count = int(dist.get("0", 0))
    if zero_count:
        logger.warning(
            "Ignoring %d legacy rows with zero validated disorders; "
            "rerun s34_multi_diagnosis_parser.py to regenerate corrected stats",
            zero_count,
        )
    ks = sorted(int(k) for k in dist if int(k) >= 1)
    vs = [int(dist[str(k)]) for k in ks]
    n_valid = sum(vs)
    n_multi = sum(v for k, v in zip(ks, vs) if k > 1)
    pct_multi = n_multi / n_valid if n_valid else np.nan
    bars = ax.bar(
        ks, vs, color="#4DBBD5", width=0.55,
        edgecolor="#222222", linewidth=0.6,
    )
    bars[0].set_color("#AAAAAA")
    ymax = max(vs) * 1.20
    ax.set_xlabel("Disorders reported")
    ax.set_ylabel("Users")
    ax.set_xticks(ks)
    ax.set_ylim(0, ymax)
    ax.ticklabel_format(axis="y", style="plain")
    for xi, value in zip(ks, vs):
        ax.text(
            xi, value + ymax * 0.025, f"{value:,}",
            ha="center", fontsize=FS_ANNOT,
        )
    ax.text(
        0.98, 0.96,
        f"{pct_multi * 100:.1f}% multi-diagnosis",
        transform=ax.transAxes, ha="right", va="top", fontsize=FS_ANNOT,
    )
    ax.set_title(
        "A. Self-report diagnosis distribution", loc="left", pad=20
    )
    _despine(ax)

    # (b) top comorbidity pairs
    ax = ax_b
    pairs = stats_.get("top_comorbidity_pairs", [])[:10]
    if not pairs:
        raise ValueError("multi_diagnosis_stats.json top_comorbidity_pairs is empty")
    labels = [
        "–".join(DISPLAY.get(d, d).split()[0] for d in pair["pair"])
        for pair in pairs
    ][::-1]
    vals_k = np.array([pair["n"] for pair in pairs][::-1], dtype=float) / 1000
    ax.barh(
        labels, vals_k, color="#E64B35", height=0.70,
        edgecolor="#222222", linewidth=0.6,
    )
    x_max = max(vals_k) * 1.18
    for yi, value in enumerate(vals_k):
        ax.text(value + x_max * 0.018, yi, f"{value:.1f}",
                va="center", fontsize=FS_ANNOT)
    ax.set_xlabel("Users (thousands)")
    ax.set_xlim(0, x_max)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.set_title(
        "B. Self-report comorbidity pairs", loc="left", pad=20
    )
    _despine(ax)

    # (c) intermediate fraction
    ax = ax_c
    ys = np.arange(len(inter))
    colors = [get_disorder_color(a) for a in inter["disorder_a"]]
    ax.scatter(
        inter["frac_intermediate"], ys, s=25, c=colors,
        edgecolors="white", lw=0.4, zorder=3,
    )
    ax.axvline(0.5, color="#E64B35", lw=0.8, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels(
        [
            f"{DISPLAY.get(a, a)[:3]}–{DISPLAY.get(b, b)[:3]}"
            for a, b in zip(inter["disorder_a"], inter["disorder_b"])
        ],
        fontsize=FS_TICK, fontweight="bold",
    )
    ax.set_xlabel("Fraction classified as intermediate")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.8, len(inter) - 0.2)
    ax.text(
        0.51, 0.99, "Null = 0.50",
        transform=ax.get_xaxis_transform(), fontsize=FS_ANNOT,
        color="#E64B35", va="top",
    )
    ax.set_title(
        "C. Feature-linked intermediate states", loc="left", pad=20
    )
    _despine(ax)

    fig.suptitle(
        "Comorbidity in the digital phenotype space",
        fontsize=FS_SUPTITLE, fontweight="bold", y=0.975,
    )
    _save(fig, "fig4_comorbidity")


# =====================================================================
#  ED FIGURES
# =====================================================================
def ed_fig1_mantel_scatter():
    """ED: AUROC vs comorbidity per pair."""
    pw = _csv(
        "pairwise_distinguishability.csv",
        required=("disorder_a", "disorder_b", "auroc"),
    )
    fig, ax = _single_square_figure()
    xs, ys = [], []
    for _, r in pw.iterrows():
        rate = _unordered_lookup(
            CLINICAL_COMORBIDITY, r["disorder_a"], r["disorder_b"]
        )
        if np.isfinite(rate) and np.isfinite(r["auroc"]):
            xs.append(rate)
            ys.append(r["auroc"])
    if len(xs) != len(pw):
        logger.warning(
            "ED Fig 1 matched %d/%d AUROC pairs to comorbidity rates",
            len(xs), len(pw),
        )
    if len(xs) < 3:
        raise ValueError("Fewer than three matched AUROC/comorbidity pairs")
    ax.scatter(xs, ys, s=40, color="#3C5488", alpha=0.75, edgecolors="white",
               lw=0.5)
    slope, intercept, rval, pval, _ = stats.linregress(xs, ys)
    xx = np.linspace(min(xs), max(xs), 100)
    ax.plot(xx, slope * xx + intercept, color="#E64B35", lw=1.5)
    ax.text(
        0.04, 0.04, f"Pearson r={rval:.2f}, p={pval:.3f}; n={len(xs)} pairs",
        transform=ax.transAxes, fontsize=FS_ANNOT,
    )
    ax.set_xlabel("Clinical comorbidity rate (%)")
    ax.set_ylabel("Pairwise AUROC")
    ax.margins(x=0.08, y=0.10)
    ax.set_title(
        "ED Fig. 1 | Digital distinguishability versus clinical comorbidity",
        pad=20,
    )
    _despine(ax)
    _save(fig, "ed_fig1_mantel", "ed")


def ed_fig2_bridge_matrix():
    """ED: bridge centrality heatmap (features x disorders) in M1 style."""
    bridge = _csv(
        "bridge_centrality_by_disorder.csv",
        required=("mean_bridge", "n_disorders_bridge"),
    )
    bridge, cols = _prepare_bridge_data(bridge)
    top = bridge.sort_values("mean_bridge", ascending=False).head(20)
    mat = top[cols].to_numpy(dtype=float)
    feats = top["feature"].astype(str).tolist()

    fig_w, fig_h = 6.40, 5.76
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = _add_axes_inches(fig, fig_w, fig_h, 1.45, 1.00, 3.70, 3.70)
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    ax.set_box_aspect(1)
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(len(feats) - 0.5, -0.5)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(
        [DISPLAY.get(c, c) for c in cols],
        rotation=45, ha="right", rotation_mode="anchor",
        fontsize=FS_TICK, fontweight="bold",
    )
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(
        [f[:28] for f in feats], fontsize=FS_TICK, fontweight="bold"
    )
    for label, disorder in zip(ax.get_xticklabels(), cols):
        label.set_color(get_disorder_color(disorder))
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#222222")
        spine.set_linewidth(1.0)
    cax = _add_axes_inches(fig, fig_w, fig_h, 5.30, 1.52, 0.10, 2.66)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Bridge centrality", fontsize=FS_AXLABEL)
    cb.ax.tick_params(labelsize=FS_TICK)
    cb.outline.set_linewidth(1.0)
    cb.outline.set_edgecolor("#222222")
    ax.set_title("ED Fig. 2 | Bridge centrality across disorders", pad=20)
    _save(fig, "ed_fig2_bridge_matrix", "ed")


def ed_fig3_intermediate_detail():
    """ED: intermediate-state detail - residual ratio vs fraction."""
    inter = _csv(
        "comorbidity_intermediate_results.csv",
        required=("frac_intermediate", "res_ratio"),
    ).dropna(subset=["frac_intermediate", "res_ratio"])
    if inter.empty:
        raise ValueError("No finite intermediate-state detail rows")
    fig, ax = _single_square_figure()
    ax.scatter(
        inter["frac_intermediate"], inter["res_ratio"], s=40,
        color="#4DBBD5", alpha=0.75, edgecolors="white", lw=0.5,
    )
    ax.axhline(1, color="#888888", lw=0.6, ls="--")
    ax.axvline(0.5, color="#888888", lw=0.6, ls="--")
    ax.set_xlabel("Fraction classified as intermediate")
    ax.set_ylabel("Residual ratio (dual/single)")
    ax.set_xlim(0, 1)
    ymin, ymax = inter["res_ratio"].min(), inter["res_ratio"].max()
    span = max(ymax - ymin, 0.1)
    ax.set_ylim(max(0, ymin - span * 0.12), ymax + span * 0.12)
    ax.set_title("ED Fig. 3 | Intermediate-state geometry", pad=20)
    _despine(ax)
    _save(fig, "ed_fig3_intermediate_detail", "ed")


def ed_fig4_umap_small_multiples():
    """ED: UMAP small multiples (M1 t-SNE style) - one panel per disorder,
    control as grey background, patients colored, title with mean AUROC."""
    emb = _csv(
        "phenotype_universe_embedding.csv",
        required=("group", "disorder_group", "umap_x", "umap_y"),
    )
    pw = _csv(
        "pairwise_distinguishability.csv",
        required=("disorder_a", "disorder_b", "auroc"),
    )
    fig_w, fig_h = 7.20, 4.30
    fig = plt.figure(figsize=(fig_w, fig_h))
    panel_w = panel_h = 1.15
    left, bottom = 0.35, 0.65
    col_gap, row_gap = 0.18, 0.48
    axes = np.empty((2, 5), dtype=object)
    for row in range(2):
        for col in range(5):
            axes[row, col] = _add_axes_inches(
                fig, fig_w, fig_h,
                left + col * (panel_w + col_gap),
                bottom + (1 - row) * (panel_h + row_gap),
                panel_w, panel_h,
            )
    # mean distinguishability per disorder
    mean_auc = {}
    if not pw.empty:
        for d in set(pw["disorder_a"]) | set(pw["disorder_b"]):
            vals = pw.loc[(pw["disorder_a"] == d) | (pw["disorder_b"] == d), "auroc"]
            mean_auc[d] = vals.mean()
    ctrl = emb[emb["group"] == "control"]
    pat = emb[emb["group"] == "patient"]
    disorder_list = [d for d in DISPLAY if d != "control"]
    if len(disorder_list) != 10:
        raise ValueError(f"Expected 10 disorders, found {len(disorder_list)}")
    xmin, xmax = emb["umap_x"].min(), emb["umap_x"].max()
    ymin, ymax = emb["umap_y"].min(), emb["umap_y"].max()
    xpad = max((xmax - xmin) * 0.03, 1e-6)
    ypad = max((ymax - ymin) * 0.03, 1e-6)
    for idx, d in enumerate(disorder_list):
        ax = axes[idx // 5, idx % 5]
        ax.scatter(ctrl["umap_x"], ctrl["umap_y"], c="#EEEEEE", s=1,
                   alpha=0.12, rasterized=True)
        m = pat["disorder_group"] == d
        if m.sum() > 0:
            ax.scatter(pat.loc[m, "umap_x"], pat.loc[m, "umap_y"],
                       c=get_disorder_color(d), s=3, alpha=0.5, rasterized=True)
        dn = DISPLAY.get(d, d)
        aucs = (
            f"\nMean pairwise\nAUROC={mean_auc[d]:.2f}"
            if d in mean_auc else ""
        )
        ax.set_title(f"{dn}{aucs}", fontsize=FS_TICK, fontweight="bold",
                     color=get_disorder_color(d), loc="center", pad=4)
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([])
        _despine(ax, top=True, right=True, left=True, bottom=True)
    h_ctrl = Line2D([0], [0], marker="o", color="w", markerfacecolor="#EEEEEE",
                    ms=5, label="Control", markeredgecolor="#CCC", mew=0.3)
    h_pat = Line2D([0], [0], marker="o", color="w", markerfacecolor="#D6604D",
                   ms=5, label="Patient", markeredgecolor="#666", mew=0.3)
    fig.legend(
        handles=[h_ctrl, h_pat], fontsize=FS_LEGEND,
        loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.03),
        frameon=False,
    )
    fig.suptitle(
        "ED Fig. 4 | UMAP embeddings by disorder",
        fontsize=FS_SUPTITLE, fontweight="bold", y=0.965,
    )
    _save(fig, "ed_fig4_umap_small_multiples", "ed")


PLOT_REGISTRY = {
    "fig1": fig1_pairwise,
    "fig2": fig2_geometry,
    "fig3": fig3_networks,
    "fig4": fig4_comorbidity,
    "edfig1": ed_fig1_mantel_scatter,
    "edfig2": ed_fig2_bridge_matrix,
    "edfig3": ed_fig3_intermediate_detail,
    "edfig4": ed_fig4_umap_small_multiples,
}


def run_visualizations(selected_plots):
    logger.info("=" * 70)
    logger.info("Module 3 Publication Visualizations (lyc — Modular CLI)")
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
    failures = []
    for target in targets:
        func = PLOT_REGISTRY[target]
        try:
            func()
        except Exception as e:
            logger.error(f"Error executing {target}: {e}")
            failures.append((target, e))

    logger.info("\n" + "=" * 70)
    logger.info("TARGETED VISUALIZATIONS COMPLETE")
    logger.info(f"  Main: {MAIN_DIR}")
    logger.info(f"  ED:   {ED_DIR}")
    logger.info("=" * 70)
    if failures:
        failed_names = ", ".join(name for name, _ in failures)
        raise RuntimeError(f"Failed plots: {failed_names}")


def run_all():
    run_visualizations(["all"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Modular CLI for Module 3 visualizations.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-p", "--plots",
        nargs="+",
        default=["all"],
        help=(
            "Plots to generate.\n"
            "Example: python s36_figures_lyc.py -p fig1 fig3 edfig2\n"
            "Options: 'all', 'fig1'-'fig4', 'edfig1'-'edfig4'"
        ),
    )
    args = parser.parse_args()
    run_visualizations(args.plots)
