"""
Module 3 Publication Figures (v2) - M1-compatible Nature/Science style
=======================================================================
Redrawn to match M1's aesthetic (s07_visualization_new_v3_lyc.py):
  - AUROC matrix: RdBu_r + TwoSlopeNorm, in-cell value annotations with
    auto-contrast color, disorder labels colored by HiTOP spectrum, full border
  - Embedding: white-edged bubble scatter + leader-line annotations with white
    halo, HiTOP spectrum convex-hull background
  - t-SNE-style small multiples: one panel per disorder, control as grey
    background, disorder-colored patients, per-panel AUC in title
  - Typography: titles 8pt, axis 7pt, ticks 6pt, annotations 6pt

Main -> module3_results/figures/main/
SI   -> module3_results/figures/ed/
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
from matplotlib.lines import Line2D
from scipy import stats

PROJECT_ROOT = "/path/to/project"
M3_STATS = os.path.join(PROJECT_ROOT, "module3_results", "statistics")
M3_FIGS = os.path.join(PROJECT_ROOT, "module3_results", "figures")
MAIN_DIR = os.path.join(M3_FIGS, "main")
ED_DIR = os.path.join(M3_FIGS, "ed")
os.makedirs(MAIN_DIR, exist_ok=True)
os.makedirs(ED_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---- typography (M1 standard) ----
FS_TITLE, FS_AXLABEL, FS_TICK, FS_ANNOT, FS_SUPTITLE, FS_LEGEND = 8, 7, 6, 6, 9, 6.5
DOUBLE_COL = 7.2
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": FS_TITLE,
    "axes.labelsize": FS_AXLABEL,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#333333",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.2,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

DISPLAY = {
    "depression": "Depression", "bipolar": "Bipolar", "anxiety": "Anxiety",
    "ocd": "OCD", "ptsd": "PTSD", "bpd": "Borderline PD",
    "eating_disorders": "Eating Disorders", "schizophrenia": "Schizophrenia",
    "adhd": "ADHD", "autism": "Autism", "control": "Control",
}
HITOP_SPECTRA = {
    "Internalizing": ["depression", "anxiety", "eating_disorders"],
    "Thought Disorder": ["schizophrenia", "bipolar"],
    "Externalizing": ["adhd"],
    "Detachment/Personality": ["bpd", "ptsd", "ocd"],
    "Neurodevelopmental": ["autism"],
}
HITOP_COLORS = {
    "Internalizing": "#E74C3C", "Thought Disorder": "#9B59B6",
    "Externalizing": "#F39C12", "Detachment/Personality": "#3498DB",
    "Neurodevelopmental": "#2ECC71",
}
def get_disorder_color(d):
    for s, ds in HITOP_SPECTRA.items():
        if d in ds:
            return HITOP_COLORS[s]
    return "#808080"

def _save(fig, name, subdir="main"):
    d = MAIN_DIR if subdir == "main" else ED_DIR
    for fmt in ("pdf", "png"):
        fig.savefig(os.path.join(d, f"{name}.{fmt}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {subdir}/{name}")

def _despine(ax, top=True, right=True, left=False, bottom=False):
    if top: ax.spines["top"].set_visible(False)
    if right: ax.spines["right"].set_visible(False)
    if left: ax.spines["left"].set_visible(False)
    if bottom: ax.spines["bottom"].set_visible(False)

def _csv(name, **kw):
    p = os.path.join(M3_STATS, name)
    if not os.path.exists(p):
        logger.warning(f"  Missing: {name}")
        return pd.DataFrame()
    return pd.read_csv(p, **kw)

def _json(name):
    p = os.path.join(M3_STATS, name)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


# =====================================================================
#  FIGURE M3-1: Pairwise distinguishability (M1 fig3-C style heatmap)
# =====================================================================
def fig1_pairwise():
    """(a) 45-pair AUROC heatmap in M1 fig3-C style (in-cell values,
    RdBu_r diverging, HiTOP-colored labels); (b) per-disorder mean AUROC
    errorbar dot plot with Mantel result."""
    pw = _csv("pairwise_distinguishability.csv")
    rep = _json("pairwise_distinguishability_report.json")

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.62))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.25)

    # (a) AUROC heatmap with in-cell values
    ax = fig.add_subplot(gs[0])
    if not pw.empty:
        disorders = sorted(set(pw["disorder_a"]) | set(pw["disorder_b"]))
        mat = np.full((len(disorders), len(disorders)), np.nan)
        for _, r in pw.iterrows():
            i, j = disorders.index(r["disorder_a"]), disorders.index(r["disorder_b"])
            mat[i, j] = mat[j, i] = r["auroc"]
        np.fill_diagonal(mat, np.nan)

        cmap = plt.get_cmap("RdBu_r")
        norm = TwoSlopeNorm(vmin=0.5, vcenter=0.7, vmax=0.85)
        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    ax.text(j, i, "\u2014", ha="center", va="center",
                            fontsize=FS_TICK, color="#AAA")
                    continue
                tc = "white" if (v > 0.80 or v < 0.60) else "black"
                v_str = f"{v:.2f}".lstrip("0") if v < 1.0 else "1.0"
                ax.text(j, i, v_str, ha="center", va="center",
                        fontsize=FS_ANNOT, color=tc)

        ax.set_xticks(range(len(disorders)))
        ax.set_xticklabels([DISPLAY.get(d, d) for d in disorders],
                           rotation=45, ha="right", rotation_mode="anchor",
                           fontsize=FS_TICK, fontweight="bold")
        ax.set_yticks(range(len(disorders)))
        ax.set_yticklabels([DISPLAY.get(d, d) for d in disorders],
                           fontsize=FS_TICK, fontweight="bold")
        # color labels by HiTOP spectrum
        for lb in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            rn = [k for k, v in DISPLAY.items() if v == lb.get_text()]
            if rn:
                lb.set_color(get_disorder_color(rn[0]))

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#222222")
            spine.set_linewidth(1.0)

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.15)
        cb = fig.colorbar(im, cax=cax)
        cb.set_ticks([0.5, 0.6, 0.7, 0.8])
        cb.set_label("AUROC", fontsize=FS_AXLABEL)
        cb.ax.tick_params(labelsize=FS_TICK)
    ax.set_title("A. Pairwise Distinguishability", fontsize=FS_TITLE,
                 fontweight="bold", loc="center", pad=18)

    # (b) Per-disorder mean AUROC + Mantel
    ax = fig.add_subplot(gs[1])
    if not pw.empty:
        di = {}
        for d in set(pw["disorder_a"]) | set(pw["disorder_b"]):
            vals = pw.loc[(pw["disorder_a"] == d) | (pw["disorder_b"] == d), "auroc"]
            di[d] = (vals.mean(), vals.std())
        s = pd.Series({k: v[0] for k, v in di.items()}).sort_values()
        ys = np.arange(len(s))
        colors = [get_disorder_color(d) for d in s.index]
        for yi, (d, (m, sd)) in zip(ys, di.items()):
            ax.errorbar(m, s.index.tolist().index(d), xerr=1.96 * sd, fmt="o",
                        color=get_disorder_color(d), ms=6, capsize=2,
                        elinewidth=0.8, markeredgecolor="white", lw=0.5)
        ax.axvline(s.mean(), color="#888", lw=1, ls="--")
        ax.text(s.mean() + 0.002, len(s) - 0.5, f"mean={s.mean():.3f}",
                color="#555", fontsize=6)
        ax.set_yticks(ys)
        ax.set_yticklabels([DISPLAY.get(d, d) for d in s.index],
                           fontsize=FS_TICK, fontweight="bold")
        for lb, d in zip(ax.get_yticklabels(), s.index):
            lb.set_color(get_disorder_color(d))
        ax.set_xlabel("Mean pairwise AUROC (±95% CI)")
        ax.set_xlim(0.55, 0.85)
        mr = rep.get("mantel_r", np.nan)
        mp = rep.get("mantel_p", np.nan)
        ax.text(0.05, 0.05,
                f"Mantel r={mr:.2f} (p={mp:.3f})",
                transform=ax.transAxes, fontsize=6, va="bottom", style="italic")
    ax.set_title("B. Mean Distinguishability", fontsize=FS_TITLE,
                 fontweight="bold", loc="center", pad=18)
    _despine(ax)

    fig.suptitle("Cross-Disorder Distinguishability of Digital Phenotypes",
                 fontsize=FS_SUPTITLE, fontweight="bold", y=0.98)
    _save(fig, "fig1_pairwise_distinguishability")


# =====================================================================
#  FIGURE M3-2: Phenotype geometry (M1 MDS-style + t-SNE small multiples)
# =====================================================================
def fig2_geometry():
    """(a) UMAP universe in M1 MDS style (bubbles + leader lines + HiTOP
    hulls); (b) spectrum separation; (c) structural fit."""
    emb = _csv("phenotype_universe_embedding.csv")
    geo = _json("phenotype_geometry_report.json")

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1, 1], wspace=0.3)

    # (a) UMAP universe (M1 MDS aesthetic)
    ax = fig.add_subplot(gs[0])
    if not emb.empty:
        ctrl = emb[emb["group"] == "control"]
        ax.scatter(ctrl["umap_x"], ctrl["umap_y"], c="#EEEEEE", s=2, alpha=0.3,
                   rasterized=True)
        # HiTOP hulls (patients only)
        pat = emb[emb["group"] == "patient"]
        for sp, ds in HITOP_SPECTRA.items():
            mem = [d for d in ds if d in pat["disorder_group"].values]
            if len(mem) < 3:
                continue
            co = pat[pat["disorder_group"].isin(mem)][["umap_x", "umap_y"]].values
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
            "Depression": (-12, 14, "right", "bottom"),
            "Bipolar": (10, -22, "right", "top"),
            "Borderline PD": (-18, -2, "right", "center"),
            "PTSD": (6, 16, "left", "bottom"),
            "OCD": (16, 6, "left", "bottom"),
            "Anxiety": (16, -6, "left", "top"),
            "Eating Disorders": (6, -18, "left", "top"),
            "ADHD": (15, 0, "left", "center"),
            "Schizophrenia": (-10, 0, "right", "center"),
            "Autism": (-8, 16, "center", "bottom"),
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
            txt = ax.annotate(dn, (cx, cy), xytext=(off[0], off[1]),
                              textcoords="offset points", ha=off[2], va=off[3],
                              fontsize=FS_TICK, color=c, fontweight="bold",
                              zorder=15,
                              arrowprops=dict(arrowstyle="-", color=c, lw=1.0,
                                              alpha=0.6, shrinkA=0, shrinkB=4))
            txt.set_path_effects([withStroke(linewidth=2.5, foreground="white")])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
    ax.set_title("A. Digital Phenotype Universe", fontsize=FS_TITLE,
                 fontweight="bold", loc="center", pad=18)
    _despine(ax, left=True, bottom=True)

    # (b) Spectrum separation
    ax = fig.add_subplot(gs[1])
    sep = geo.get("spectrum_separation", {})
    if sep:
        w, b = sep.get("within_mean", 0), sep.get("between_mean", 0)
        ax.bar(["Within", "Between"], [w, b], width=0.55,
               color=["#4DBBD5", "#E64B35"], edgecolor="white", lw=0.5)
        ax.text(0, w + 0.006, f"{w:.2f}", ha="center", fontsize=6)
        ax.text(1, b + 0.006, f"{b:.2f}", ha="center", fontsize=6)
        ax.set_ylabel("Centroid distance")
        ax.set_ylim(0, max(w, b) * 1.25)
        ax.text(0.5, max(w, b) * 1.12, f"perm p={sep.get('perm_p', 1):.2f}",
                ha="center", fontsize=6, style="italic")
    ax.set_title("B. Spectrum Separation", fontsize=FS_TITLE,
                 fontweight="bold", loc="center", pad=18)
    _despine(ax)

    # (c) Structural fit
    ax = fig.add_subplot(gs[2])
    sf = geo.get("structural_fit", {})
    if sf:
        labels = ["HiTOP", "DSM-5", "Comorb."]
        rs = [sf.get("hitop_r", np.nan), sf.get("dsm5_r", np.nan),
              sf.get("comorbidity_r", np.nan)]
        x = np.arange(3)
        ax.bar(x, rs, width=0.55,
               color=["#9B59B6", "#3498DB", "#E74C3C"], edgecolor="white", lw=0.5)
        for xi, r in zip(x, rs):
            ax.text(xi, r + 0.01, f"{r:.2f}", ha="center", fontsize=6)
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=FS_TICK)
        ax.set_ylabel("Correlation")
        ax.set_ylim(-0.5, 0.6)
    ax.set_title("C. Structural Fit", fontsize=FS_TITLE,
                 fontweight="bold", loc="center", pad=18)
    _despine(ax)

    fig.suptitle("Geometry of the Digital Phenotype Space",
                 fontsize=FS_SUPTITLE, fontweight="bold", y=0.98)
    _save(fig, "fig2_phenotype_geometry")


# =====================================================================
#  FIGURE M3-3: Disorder networks
# =====================================================================
def fig3_networks():
    """(a) network density vs n_users (sample-size effect), (b) bridge
    features, (c) edge consistency."""
    topo = _csv("disorder_network_topology.csv")
    bridge = _csv("bridge_centrality_by_disorder.csv")

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.3)

    # (a) density vs n_users
    ax = fig.add_subplot(gs[0])
    if not topo.empty:
        for _, r in topo.iterrows():
            c = get_disorder_color(r["disorder"])
            ax.scatter(r["n_users"], r["density"], s=45, c=c,
                       edgecolors="white", lw=0.5, zorder=3)
            ax.text(r["n_users"] + 800, r["density"],
                    DISPLAY.get(r["disorder"], r["disorder"])[:3],
                    fontsize=4.5, va="center", color=c, fontweight="bold")
        ax.set_xlabel("N users (disorder)")
        ax.set_ylabel("Network density")
        ax.set_xlim(0, topo["n_users"].max() * 1.3)
    ax.set_title("A. Density vs Sample Size", fontsize=FS_TITLE,
                 fontweight="bold", loc="center", pad=18)
    _despine(ax)

    # (b) bridge features
    ax = fig.add_subplot(gs[1])
    if not bridge.empty and "mean_bridge" in bridge.columns:
        bridge = bridge.rename(columns={bridge.columns[0]: "feature"})
        top = bridge.sort_values("mean_bridge", ascending=False).head(8)
        ax.barh([c[:22] for c in top["feature"]], top["mean_bridge"],
                color="#E64B35", height=0.6, edgecolor="white", lw=0.5)
        ax.set_xlabel("Mean bridge centrality")
        ax.invert_yaxis()
    ax.set_title("B. Bridge Features", fontsize=FS_TITLE,
                 fontweight="bold", loc="center", pad=18)
    _despine(ax)

    # (c) edge consistency
    ax = fig.add_subplot(gs[2])
    ec = _csv("network_edge_consistency.csv")
    if not ec.empty:
        counts = ec["n_disorders"].value_counts().sort_index()
        ax.plot(counts.index.astype(int), counts.values, "o-",
                color="#3C5488", ms=4, lw=1.2)
        ax.fill_between(counts.index.astype(int), 0, counts.values,
                        alpha=0.15, color="#3C5488")
        ax.set_xlabel("Disorders sharing edge")
        ax.set_ylabel("Edges")
        ax.set_xticks(range(1, 11))
        ax.set_xlim(0.5, 10.5)
    ax.set_title("C. Edge Consistency", fontsize=FS_TITLE,
                 fontweight="bold", loc="center", pad=18)
    _despine(ax)

    fig.suptitle("Cross-Disorder Digital Phenotype Networks",
                 fontsize=FS_SUPTITLE, fontweight="bold", y=0.98)
    _save(fig, "fig3_disorder_networks")


# =====================================================================
#  FIGURE M3-4: Comorbidity
# =====================================================================
def fig4_comorbidity():
    """(a) n-disorders histogram, (b) top comorbidity pairs (small multiples
    style bar), (c) intermediate fraction per pair."""
    stats_ = _json("multi_diagnosis_stats.json")
    inter = _csv("comorbidity_intermediate_results.csv")

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.5))
    gs = fig.add_gridspec(1, 3, wspace=0.3)

    # (a) n_disorders histogram
    ax = fig.add_subplot(gs[0])
    dist = stats_.get("n_disorders_dist", {})
    if dist:
        ks = sorted(int(k) for k in dist.keys())
        vs = [dist[str(k)] for k in ks]
        bars = ax.bar(ks, vs, color="#4DBBD5", edgecolor="white", lw=0.5)
        bars[0].set_color("#AAAAAA")
        ax.set_xlabel("N disorders reported")
        ax.set_ylabel("Users")
        ax.set_xticks(ks)
        for xi, v in zip(ks, vs):
            ax.text(xi, v + 1000, f"{v//1000}k", ha="center", fontsize=5)
        ax.text(0.95, 0.95,
                f"{stats_.get('pct_users_multi', 0)*100:.1f}% multi-dx",
                transform=ax.transAxes, ha="right", va="top", fontsize=6)
    ax.set_title("A. Multi-Diagnosis Distribution", fontsize=FS_TITLE,
                 fontweight="bold", loc="center", pad=18)
    _despine(ax)

    # (b) top comorbidity pairs
    ax = fig.add_subplot(gs[1])
    pairs = stats_.get("top_comorbidity_pairs", [])[:10]
    if pairs:
        labels = ["-".join(DISPLAY.get(d, d).split()[0] for d in p["pair"])
                  for p in pairs][::-1]
        vals = [p["n"] for p in pairs][::-1]
        ax.barh(labels, vals, color="#E64B35", height=0.6,
                edgecolor="white", lw=0.5)
        for yi, v in enumerate(vals):
            ax.text(v + 300, yi, f"{v//1000}k", va="center", fontsize=5)
        ax.set_xlabel("Users (×1000)")
        ax.set_xlim(0, max(vals) * 1.15)
    ax.set_title("B. Top Comorbidity Pairs", fontsize=FS_TITLE,
                 fontweight="bold", loc="center", pad=18)
    _despine(ax)

    # (c) intermediate fraction
    ax = fig.add_subplot(gs[2])
    if not inter.empty:
        inter = inter.sort_values("frac_intermediate")
        ys = np.arange(len(inter))
        colors = [get_disorder_color(a) for a in inter["disorder_a"]]
        ax.scatter(inter["frac_intermediate"], ys, s=25, c=colors,
                   edgecolors="white", lw=0.4, zorder=3)
        ax.axvline(0.5, color="#E64B35", lw=1, ls="--")
        ax.axvline(0.25, color="#888", lw=0.8, ls=":")
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{DISPLAY.get(a,'')[:3]}-{DISPLAY.get(b,'')[:3]}"
                            for a, b in zip(inter["disorder_a"], inter["disorder_b"])],
                           fontsize=4.5)
        ax.set_xlabel("Fraction intermediate")
        ax.set_xlim(0, 1)
        ax.text(0.52, len(inter) - 0.5, "0.5 (midpoint)", fontsize=5,
                color="#E64B35")
    ax.set_title("C. Intermediate State", fontsize=FS_TITLE,
                 fontweight="bold", loc="center", pad=18)
    _despine(ax)

    fig.suptitle("Comorbidity in the Digital Phenotype Space",
                 fontsize=FS_SUPTITLE, fontweight="bold", y=0.98)
    _save(fig, "fig4_comorbidity")


# =====================================================================
#  ED FIGURES
# =====================================================================
def ed_fig1_mantel_scatter():
    """ED: AUROC vs comorbidity per pair."""
    pw = _csv("pairwise_distinguishability.csv")
    ext_path = os.path.join(M3_STATS, "external_comorbidity.csv")
    if not os.path.exists(ext_path):
        raise FileNotFoundError(
            f"ED Fig 1 requires the externally sourced matrix {ext_path} "
            "(columns disorder_a,disorder_b,comorbidity_pct); refusing to use a "
            "hard-coded subset."
        )
    ext = pd.read_csv(ext_path)
    need = {"disorder_a", "disorder_b", "comorbidity_pct"}
    if not need.issubset(ext.columns):
        raise ValueError(f"external_comorbidity.csv missing columns {need - set(ext.columns)}")
    # orientation-agnostic lookup (sort BOTH the dict keys and the query keys)
    comorb = {tuple(sorted([str(r["disorder_a"]), str(r["disorder_b"])])):
              float(r["comorbidity_pct"]) for _, r in ext.iterrows()}
    measure_label = "Clinical comorbidity"
    if "measure" in ext.columns and ext["measure"].notna().any():
        measure_label = f"Clinical comorbidity ({str(ext['measure'].dropna().iloc[0])})"
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    xs, ys = [], []
    for _, r in pw.iterrows():
        k = tuple(sorted([r["disorder_a"], r["disorder_b"]]))
        if k in comorb:
            xs.append(comorb[k]); ys.append(r["auroc"])
    if len(xs) != len(comorb):
        raise RuntimeError(
            f"ED Fig 1: matched {len(xs)} of {len(comorb)} sourced pairs; check "
            "disorder-name alignment between the pairwise table and the external matrix."
        )
    ax.scatter(xs, ys, s=40, color="#3C5488", alpha=0.75, edgecolors="white",
               lw=0.5)
    if len(xs) > 2:
        slope, intercept, rval, pval, se = stats.linregress(xs, ys)
        xx = np.linspace(min(xs), max(xs), 50)
        ax.plot(xx, slope * xx + intercept, color="#E64B35", lw=1.5)
        ax.text(0.05, 0.05, f"r={rval:.2f}, p={pval:.3f}",
                transform=ax.transAxes, fontsize=6)
    ax.set_xlabel(measure_label)
    ax.set_ylabel("Pairwise AUROC")
    _despine(ax)
    _save(fig, "ed_fig1_mantel", "ed")


def ed_fig2_bridge_matrix():
    """ED: bridge centrality heatmap (features x disorders) in M1 style."""
    bridge = _csv("bridge_centrality_by_disorder.csv")
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 0.55))
    if not bridge.empty:
        bridge = bridge.rename(columns={bridge.columns[0]: "feature"})
        top = bridge.sort_values("mean_bridge", ascending=False).head(20)
        mat = top.drop(columns=["feature", "mean_bridge", "n_disorders_bridge"]).values.astype(float)
        feats = top["feature"].tolist()
        cols = [c for c in top.columns if c not in ("feature", "mean_bridge", "n_disorders_bridge")]
        cmap = plt.get_cmap("YlOrRd")
        im = ax.imshow(mat, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([DISPLAY.get(c, c) for c in cols],
                           rotation=45, ha="right", rotation_mode="anchor",
                           fontsize=FS_TICK)
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels([f[:20] for f in feats], fontsize=FS_TICK)
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.15)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Bridge centrality", fontsize=FS_AXLABEL)
        cb.ax.tick_params(labelsize=FS_TICK)
    _save(fig, "ed_fig2_bridge_matrix", "ed")


def ed_fig3_intermediate_detail():
    """ED: intermediate-state detail - residual ratio vs fraction."""
    inter = _csv("comorbidity_intermediate_results.csv")
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    if not inter.empty:
        ax.scatter(inter["frac_intermediate"], inter["res_ratio"], s=40,
                   color="#4DBBD5", alpha=0.75, edgecolors="white", lw=0.5)
        ax.axhline(1, color="#888", lw=0.8, ls="--")
        ax.axvline(0.5, color="#888", lw=0.8, ls="--")
        ax.set_xlabel("Fraction intermediate")
        ax.set_ylabel("Residual ratio (dual/single)")
        ax.set_xlim(0, 0.6)
    _despine(ax)
    _save(fig, "ed_fig3_intermediate_detail", "ed")


def ed_fig4_umap_small_multiples():
    """ED: UMAP small multiples (M1 t-SNE style) - one panel per disorder,
    control as grey background, patients colored, title with mean AUROC."""
    emb = _csv("phenotype_universe_embedding.csv")
    pw = _csv("pairwise_distinguishability.csv")
    fig, axes = plt.subplots(2, 5, figsize=(DOUBLE_COL, DOUBLE_COL * 0.55))
    fig.subplots_adjust(left=0.03, right=0.99, top=0.90, bottom=0.08,
                        hspace=0.45, wspace=0.25)
    if emb.empty:
        _save(fig, "ed_fig4_umap_small_multiples", "ed")
        return
    # mean distinguishability per disorder
    mean_auc = {}
    if not pw.empty:
        for d in set(pw["disorder_a"]) | set(pw["disorder_b"]):
            vals = pw.loc[(pw["disorder_a"] == d) | (pw["disorder_b"] == d), "auroc"]
            mean_auc[d] = vals.mean()
    ctrl = emb[emb["group"] == "control"]
    pat = emb[emb["group"] == "patient"]
    disorder_list = [d for d in DISPLAY if d != "control"]
    for idx, d in enumerate(disorder_list):
        ax = axes[idx // 5, idx % 5]
        ax.scatter(ctrl["umap_x"], ctrl["umap_y"], c="#EEEEEE", s=1,
                   alpha=0.12, rasterized=True)
        m = pat["disorder_group"] == d
        if m.sum() > 0:
            ax.scatter(pat.loc[m, "umap_x"], pat.loc[m, "umap_y"],
                       c=get_disorder_color(d), s=3, alpha=0.5, rasterized=True)
        dn = DISPLAY.get(d, d)
        aucs = f"\nAUC={mean_auc.get(d, np.nan):.2f}" if d in mean_auc else ""
        ax.set_title(f"{dn}{aucs}", fontsize=FS_TICK, fontweight="bold",
                     color=get_disorder_color(d), loc="center", pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        _despine(ax, top=True, right=True, left=True, bottom=True)
    h_ctrl = Line2D([0], [0], marker="o", color="w", markerfacecolor="#EEEEEE",
                    ms=5, label="Control", markeredgecolor="#CCC", mew=0.3)
    h_pat = Line2D([0], [0], marker="o", color="w", markerfacecolor="#D6604D",
                   ms=5, label="Patient", markeredgecolor="#666", mew=0.3)
    fig.legend(handles=[h_ctrl, h_pat], fontsize=FS_LEGEND,
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01),
               frameon=False)
    fig.suptitle("UMAP Embeddings by Disorder (with mean pairwise AUROC)",
                 fontsize=FS_SUPTITLE, fontweight="bold", y=1.02)
    _save(fig, "ed_fig4_umap_small_multiples", "ed")


def run_all():
    logger.info("=== M3 figures v2 (M1-compatible) ===")
    fig1_pairwise()
    fig2_geometry()
    fig3_networks()
    fig4_comorbidity()
    ed_fig1_mantel_scatter()
    ed_fig2_bridge_matrix()
    ed_fig3_intermediate_detail()
    ed_fig4_umap_small_multiples()
    logger.info("DONE")


if __name__ == "__main__":
    run_all()
