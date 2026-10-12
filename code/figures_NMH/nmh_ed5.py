"""NMH Extended Data Figure ED5 - Factor and component robustness.
(a) PCA scree; (b) top feature loadings; (c) disorder factor scores; (d) disorder PCA scores.
Complements Fig. 4 (dendrogram, MDS, loadings box) with different views. Source data only.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common_nmh import (apply_style, mm2inch, despine, panel_label, ED_DIR, DISORDERS,
                        abbr, spectrum_of, SPECTRUM_COLORS, DIV_CMAP, ACCENT_BLUE,
                        ring_labels)

M1 = "/path/to/project/module1_results/statistics"


def panel_a(ax):
    j = json.load(open(os.path.join(M1, "pca_summary.json")))
    v = np.array(j["variance_explained"], float) * 100.0
    x = np.arange(1, len(v) + 1)
    ax.plot(x, v, "-o", color=ACCENT_BLUE, ms=3.0, lw=1.1)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_xticks(x)
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("PCA scree", loc="left", fontsize=7.8)
    panel_label(ax, "a")


def panel_b(ax):
    d = pd.read_csv(os.path.join(M1, "factor_loadings_features.csv"), index_col=0)
    cols = ["Factor_1", "Factor_2", "Factor_3"]
    top = d.assign(m=d[cols].abs().max(axis=1)).sort_values("m", ascending=False).head(12)
    M = top[cols].values.astype(float)
    vmax = np.nanmax(np.abs(M))
    ax.imshow(M, cmap=DIV_CMAP, aspect="auto", vmin=-vmax, vmax=vmax)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=5.0,
                    color="white" if abs(M[i, j]) > 0.6 * vmax else "#333333")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["F1", "F2", "F3"], fontsize=6.0)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([str(f)[:18] for f in top.index], fontsize=5.0)
    ax.tick_params(axis="both", pad=4)
    despine(ax)
    ax.set_title("Top feature loadings", loc="left", fontsize=7.8)
    panel_label(ax, "b")


def panel_c(ax):
    d = pd.read_csv(os.path.join(M1, "factor_scores_disorders.csv"), index_col=0)
    ds = [x for x in DISORDERS if x in d.index]
    xs = [d.loc[x, "Factor_1"] for x in ds]
    ys = [d.loc[x, "Factor_2"] for x in ds]
    for x, y, dd in zip(xs, ys, ds):
        ax.scatter([x], [y], s=22, color=SPECTRUM_COLORS[spectrum_of(dd)],
                   edgecolors="white", lw=0.5, zorder=4)
    ring_labels(ax, xs, ys, [abbr(dd) for dd in ds], pad=0.75, min_gap_deg=34)
    ax.set_xlabel("Factor 1 score")
    ax.set_ylabel("Factor 2 score")
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Disorder factor scores", loc="left", fontsize=7.8)
    panel_label(ax, "c")


def panel_d(ax):
    d = pd.read_csv(os.path.join(M1, "disorder_pca_scores.csv"), index_col=0)
    ds = [x for x in DISORDERS if x in d.index]
    xs = [d.loc[x, "DPC1"] for x in ds]
    ys = [d.loc[x, "DPC2"] for x in ds]
    for x, y, dd in zip(xs, ys, ds):
        ax.scatter([x], [y], s=22, color=SPECTRUM_COLORS[spectrum_of(dd)],
                   edgecolors="white", lw=0.5, zorder=4)
    ring_labels(ax, xs, ys, [abbr(dd) for dd in ds], pad=0.75, min_gap_deg=34)
    ax.set_xlabel("Disorder PC 1")
    ax.set_ylabel("Disorder PC 2")
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Disorder PCA scores", loc="left", fontsize=7.8)
    panel_label(ax, "d")


def main():
    apply_style()
    fig = plt.figure(figsize=(mm2inch(180), mm2inch(150)))
    gs = fig.add_gridspec(2, 2, wspace=0.55, hspace=0.45, left=0.11, right=0.965,
                          top=0.94, bottom=0.09)
    panel_a(fig.add_subplot(gs[0, 0]))
    panel_b(fig.add_subplot(gs[0, 1]))
    panel_c(fig.add_subplot(gs[1, 0]))
    panel_d(fig.add_subplot(gs[1, 1]))

    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Helvetica Neue", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    from audit_panel_alignment import require_matplotlib_panel_alignment
    out = os.path.join(ED_DIR, "ed5_factor_robustness")
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
