"""NMH Figure 4 - Predominantly shared, dimensional cross-diagnostic structure.

Each panel uses the chart type that best fits its data and message:
  a  Heatmap   : full pairwise case-vs-case AUROC matrix (45 pairs)
  b  Scatter   : 18k-user UMAP; centroids labelled, 95% ellipses shown as a zoom inset
  c  Box plot  : distribution of feature loadings on each factor
  d  Scatter   : external comorbidity vs phenotypic dissimilarity (Mantel-style)
  e  Heatmap   : bridge centrality (features x disorders)
  f  Dendrogram: clustering of disorders vs HiTOP
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, ConnectionPatch
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import squareform
from scipy.stats import chi2

from common_nmh import (apply_style, mm2inch, despine, panel_label, MAIN_DIR,
                        DISORDERS, abbr, disorder_color, spectrum_of, SPECTRUM_COLORS,
                        SEQ_CMAP, ACCENT_BLUE, ACCENT_RED)

B = "/path/to/project"
M3 = os.path.join(B, "module3_results", "statistics")
M1 = os.path.join(B, "module1_results", "statistics")
AUC_NPY = os.path.join(M3, "auroc_matrix.npy")
REPORT = os.path.join(M3, "pairwise_distinguishability_report.json")
GEOM = os.path.join(M3, "phenotype_geometry_report.json")
EMB = os.path.join(M3, "phenotype_universe_embedding.csv")
FLOAD = os.path.join(M1, "factor_loadings_features.csv")
FSEL = os.path.join(M1, "factor_selection.json")
EXT = os.path.join(M3, "external_comorbidity.csv")
PW = os.path.join(M3, "pairwise_distinguishability.csv")
BRIDGE = os.path.join(M3, "bridge_centrality_by_disorder.csv")
CLUST = os.path.join(M1, "clustering_summary.json")


def _auroc_matrix():
    A = np.load(AUC_NPY).astype(float)
    order = json.load(open(REPORT))["disorders"]
    return A, order


def _spectrum_strip(ax, labels, size=0.9, offset=0.5):
    for i, d in enumerate(labels):
        c = SPECTRUM_COLORS.get(spectrum_of(d), "#CCCCCC")
        ax.add_patch(Rectangle((-offset - size, i - 0.5), size, 1, facecolor=c,
                               edgecolor="none", clip_on=False))


def panel_a(ax):
    A, order = _auroc_matrix()
    idx = [DISORDERS.index(d) for d in order]
    A = A[np.ix_(idx, idx)]
    D = 1.0 - A
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    o = leaves_list(Z)
    labels = [order[i] for i in o]
    M = A[np.ix_(o, o)].copy()
    np.fill_diagonal(M, np.nan)
    ax.imshow(np.ma.masked_invalid(M), cmap=SEQ_CMAP, vmin=0.60, vmax=0.85, aspect="auto")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i < j and np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=5.0,
                        color="white" if M[i, j] > 0.79 else "#333333")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([abbr(d) for d in labels], rotation=90, fontsize=5.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([abbr(d) for d in labels], fontsize=5.8)
    ax.tick_params(axis="y", pad=12)
    _spectrum_strip(ax, labels)
    ax.set_title("Pairwise separability (AUROC)", loc="left", fontsize=7.8)
    panel_label(ax, "a")


def panel_b(ax):
    e = pd.read_csv(EMB, usecols=["group", "disorder_group", "umap_x", "umap_y"])
    cen = {}
    for d in DISORDERS:
        sub = e[e["disorder_group"] == d]
        if len(sub) < 5:
            continue
        pts = sub[["umap_x", "umap_y"]].values
        mu = pts.mean(0)
        cov = np.cov(pts.T)
        w, v = np.linalg.eigh(cov)
        angd = np.degrees(np.arctan2(v[1, -1], v[0, -1]))
        sfac = np.sqrt(chi2.ppf(0.95, 2))
        cen[d] = (mu, angd, 2 * sfac * np.sqrt(w[-1]), 2 * sfac * np.sqrt(w[0]))
    ds = list(cen.keys())
    cxs = np.array([cen[d][0][0] for d in ds])
    cys = np.array([cen[d][0][1] for d in ds])
    xa, ya = cxs.mean(), cys.mean()
    span = max(cxs.max() - cxs.min(), cys.max() - cys.min()) or 1.0
    for d in ds:
        mu = cen[d][0]
        ax.scatter([mu[0]], [mu[1]], s=30, color=disorder_color(d), edgecolors="white",
                   linewidths=0.6, zorder=5)
    R = 1.35 * span
    n = len(ds)
    inset_pos = [0.55, 0.48, 0.44, 0.44]
    Rx, Ry = 1.35 * span, 1.85 * span
    angs = np.deg2rad(90) + np.arange(n) * (np.deg2rad(270) / n)
    for d, th in zip(ds, angs):
        mx, my = cen[d][0]
        ixr, iyr = xa + 0.62 * Rx * np.cos(th), ya + 0.62 * Ry * np.sin(th)
        ax.plot([mx, ixr], [my, iyr], color="#BBBBBB", lw=0.4, zorder=3)
        ax.plot([ixr, xa + 0.84 * Rx * np.cos(th)], [iyr, ya + 0.84 * Ry * np.sin(th)],
                color="#BBBBBB", lw=0.4, zorder=3)
        ax.annotate(abbr(d), (xa + Rx * np.cos(th), ya + Ry * np.sin(th)), fontsize=5.2,
                    ha="center", va="center", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.85))
    x0_ = (xa + Rx * np.cos(angs)).min() - 0.22 * Rx
    x1_ = (xa + Rx * np.cos(angs)).max() + 0.62 * Rx
    y0_ = (ya + Ry * np.sin(angs)).min() - 0.22 * Ry
    y1_ = (ya + Ry * np.sin(angs)).max() + 0.22 * Ry
    ax.set_xlim(x0_, x1_)
    ax.set_ylim(y0_, y1_)
    axin = ax.inset_axes(inset_pos)
    ctrl = e[e["group"] == "control"]
    axin.scatter(ctrl["umap_x"], ctrl["umap_y"], s=0.5, color="#CFCFCF", alpha=0.06,
                 lw=0, zorder=1)
    specs = [s for s in SPECTRUM_COLORS if any(spectrum_of(d) == s for d in DISORDERS)]
    for sp in specs:
        sub = e[e["disorder_group"].isin([d for d in DISORDERS if spectrum_of(d) == sp])]
        axin.scatter(sub["umap_x"], sub["umap_y"], s=0.5, color=SPECTRUM_COLORS[sp],
                     alpha=0.06, lw=0, zorder=2)
    ex0, ex1, ey0, ey1 = np.inf, -np.inf, np.inf, -np.inf
    for d, (mu, angd, wd, ht) in cen.items():
        axin.add_patch(Ellipse(mu, wd, ht, angle=angd, facecolor="none",
                               edgecolor=disorder_color(d), lw=0.7, alpha=0.9))
        axin.scatter([mu[0]], [mu[1]], s=7, color=disorder_color(d), edgecolors="none",
                     zorder=5)
        thr = np.deg2rad(angd)
        dx = 0.5 * np.hypot(wd * np.cos(thr), ht * np.sin(thr))
        dy = 0.5 * np.hypot(wd * np.sin(thr), ht * np.cos(thr))
        ex0 = min(ex0, mu[0] - dx); ex1 = max(ex1, mu[0] + dx)
        ey0 = min(ey0, mu[1] - dy); ey1 = max(ey1, mu[1] + dy)
    mx_ = 0.06 * max(ex1 - ex0, ey1 - ey0)
    axin.set_xlim(ex0 - mx_, ex1 + mx_)
    axin.set_ylim(ey0 - mx_, ey1 + mx_)
    axin.set_aspect("equal", adjustable="box")
    axin.set_anchor("N")
    axin.set_xticks([]); axin.set_yticks([])
    for s in axin.spines.values():
        s.set_visible(True)
        s.set_color("#AAAAAA")
        s.set_linewidth(0.5)
    ax.text(inset_pos[0], inset_pos[1] + inset_pos[3] + 0.004,
            "18,000 users\n95% covariance ellipses", transform=ax.transAxes,
            fontsize=5.2, ha="left", va="bottom", color="#333333")
    con = ConnectionPatch(xyA=(0.0, 0.0), coordsA=axin.transAxes, xyB=(xa, ya),
                          coordsB=ax.transData, color="#AAAAAA", lw=0.5, ls=(0, (3, 2)))
    ax.add_artist(con)
    ax.set_xlabel("Phenotype space (UMAP coordinates)")
    ax.set_ylabel("UMAP-2")
    despine(ax)
    ax.set_title("Disorder centroids", loc="left", fontsize=7.8)
    panel_label(ax, "b")


def panel_c(ax):
    fl = pd.read_csv(FLOAD, index_col=0)[["Factor_1", "Factor_2", "Factor_3"]]
    data = [np.abs(fl[c].values) for c in fl.columns]
    cap = float(np.percentile(np.concatenate(data), 97.5))
    bp = ax.boxplot(data, positions=[0, 1, 2], widths=0.55, patch_artist=True,
                    showfliers=False, medianprops=dict(color="#333333", lw=1.0),
                    whiskerprops=dict(color="#666666", lw=0.8),
                    capprops=dict(color="#666666", lw=0.8))
    for patch, c in zip(bp["boxes"], [ACCENT_BLUE, "#9BB8D3", "#C7D6E5"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.45)
        patch.set_edgecolor("#666666")
        patch.set_linewidth(0.8)
    rng = np.random.default_rng(42)
    for gi, g in enumerate(data):
        lo = g[g <= cap]
        hi = g[g > cap]
        ax.scatter(gi + rng.uniform(-0.18, 0.18, len(lo)), lo, s=5, color="#444444",
                   alpha=0.35, lw=0, zorder=4)
        if len(hi):
            ax.scatter([gi] * len(hi), [cap * 1.02] * len(hi), marker="^", s=10,
                       color="#444444", lw=0, zorder=5)
    ax.set_ylim(0, cap * 1.28)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["F1", "F2", "F3"], fontsize=6.2)
    ax.set_ylabel("|feature loading|")
    despine(ax)
    ttl = "Loadings on one factor"
    try:
        fs = json.load(open(FSEL))
        ttl = f"Loadings on one factor (F1 {fs['variance_explained'][0]*100:.0f}% var)"
    except Exception:
        pass
    ax.set_title(ttl, loc="left", fontsize=7.8, pad=5)
    panel_label(ax, "c")


def panel_d(ax):
    ext = pd.read_csv(EXT)
    pw = pd.read_csv(PW)
    pw["key"] = [tuple(sorted([a, b])) for a, b in zip(pw["disorder_a"], pw["disorder_b"])]
    ext["key"] = [tuple(sorted([a, b])) for a, b in zip(ext["disorder_a"], ext["disorder_b"])]
    j = pw.merge(ext[["key", "value", "source"]], on="key", how="inner")
    x = j["value"].astype(float).values
    y = 1.0 - j["auroc"].astype(float).values
    same = np.array([spectrum_of(a) == spectrum_of(b)
                     for a, b in zip(j["disorder_a"], j["disorder_b"])])
    ax.scatter(x[same], y[same], s=18, color=ACCENT_BLUE, edgecolors="white", lw=0.5,
               zorder=4, label="within spectrum")
    ax.scatter(x[~same], y[~same], s=18, color=ACCENT_RED, edgecolors="white", lw=0.5,
               zorder=4, label="across spectra")
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 60)
    rng = np.random.default_rng(42)
    boot = np.array([np.polyfit(x[i], y[i], 1) for i in
                     (rng.integers(0, len(x), len(x)) for _ in range(500))])
    lo = np.percentile(boot[:, 0], 2.5) * xs + np.percentile(boot[:, 1], 2.5)
    hi = np.percentile(boot[:, 0], 97.5) * xs + np.percentile(boot[:, 1], 97.5)
    ax.fill_between(xs, lo, hi, color=ACCENT_RED, alpha=0.12, lw=0)
    ax.plot(xs, a + b * xs, color=ACCENT_RED, lw=1.1)
    r = np.corrcoef(x, y)[0, 1]
    ax.margins(0.14)
    ax.legend(fontsize=5.6, loc="lower right", frameon=True, framealpha=0.85,
              edgecolor="#CCCCCC", handletextpad=0.3, borderpad=0.3)
    ax.set_xlabel("External comorbidity (adjusted OR)")
    ax.set_ylabel("Phenotypic dissimilarity (1 - AUROC)")
    ax.text(0.03, 0.97, f"r={r:+.2f}, n={len(x)}, n.s.", transform=ax.transAxes,
            fontsize=6.5, va="top", ha="left", color="#333333")
    despine(ax)
    ax.set_title("Comorbidity vs separability", loc="left", fontsize=7.8)
    panel_label(ax, "d")


def panel_e(ax):
    br = pd.read_csv(BRIDGE, index_col=0)
    cols = [c for c in br.columns if c in DISORDERS]
    top = br.sort_values("mean_bridge", ascending=False).head(15)
    M = top[cols].values.astype(float)
    im = ax.imshow(M, cmap=SEQ_CMAP, aspect="auto", vmin=0, vmax=np.nanmax(M))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([abbr(c) for c in cols], rotation=90, fontsize=5.6)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([str(f)[:18] for f in top.index], fontsize=5.0)
    for i, d in enumerate(cols):
        ax.add_patch(Rectangle((i - 0.5, -0.7), 1, 0.7,
                               facecolor=SPECTRUM_COLORS.get(spectrum_of(d), "#CCCCCC"),
                               edgecolor="none", clip_on=False))
    cax = ax.inset_axes([1.03, 0.0, 0.045, 0.55])
    cb = ax.figure.colorbar(im, cax=cax)
    cb.set_label("bridge centrality", fontsize=5.2)
    cb.ax.tick_params(labelsize=5.0)
    ax.set_title("Cross-disorder bridge features", loc="left", fontsize=7.8)
    panel_label(ax, "e")


def panel_f(ax):
    A, order = _auroc_matrix()
    idx = [DISORDERS.index(d) for d in order]
    A = A[np.ix_(idx, idx)]
    D = 1.0 - A
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    dendrogram(Z, labels=[abbr(d) for d in order], ax=ax, orientation="left",
               color_threshold=0, above_threshold_color="#888888")
    for t in ax.get_yticklabels():
        lab = t.get_text()
        d = next((x for x in order if abbr(x) == lab), None)
        if d:
            t.set_color(SPECTRUM_COLORS.get(spectrum_of(d), "#333333"))
    ax.tick_params(axis="y", labelsize=5.6)
    ttl = "Clustering vs HiTOP"
    try:
        ari = json.load(open(CLUST))["ari_vs_hitop"]
        ttl = f"Clustering vs HiTOP (ARI = {ari:.2f})"
    except Exception:
        pass
    ax.set_ylabel("Disorder (leaf order)", fontsize=6.0)
    ax.set_xlabel("Digital distance (1 - AUROC)")
    despine(ax)
    ax.set_title(ttl, loc="left", fontsize=7.8)
    panel_label(ax, "f")


def main():
    apply_style()
    fig = plt.figure(figsize=(mm2inch(180), mm2inch(170)))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], wspace=0.62, hspace=0.42,
                          left=0.13, right=0.96, top=0.95, bottom=0.10)
    panel_a(fig.add_subplot(gs[0, 0]))
    panel_b(fig.add_subplot(gs[0, 1]))
    panel_c(fig.add_subplot(gs[0, 2]))
    panel_d(fig.add_subplot(gs[1, 0]))
    panel_e(fig.add_subplot(gs[1, 1]))
    panel_f(fig.add_subplot(gs[1, 2]))

    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Helvetica Neue", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    from audit_panel_alignment import require_matplotlib_panel_alignment
    out = os.path.join(MAIN_DIR, "fig4_crossdiagnostic_structure")
    require_matplotlib_panel_alignment(
        fig, json_out=out + ".alignment.json", overlay_svg=out + ".alignment.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(out + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(out + ".pdf", dpi=600, bbox_inches="tight")
    fig.savefig(out + ".svg", dpi=600, bbox_inches="tight")
    fig.savefig(out + ".tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Exported:", out + ".{png,pdf,svg,tiff}")


if __name__ == "__main__":
    main()
