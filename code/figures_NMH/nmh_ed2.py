"""NMH Extended Data Figure ED2 - Feature-level robustness.

Complements Fig. 2 (atlas, volcano, dimension contribution) with:
  a  effect-size category counts
  b  per-feature significance consistency (number of disorders)
  c  significance matrix (feature x disorder, ordered by consistency)
  d  top features by stored diagnostic value
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from common_nmh import (apply_style, mm2inch, despine, panel_label, ED_DIR, DISORDERS,
                        abbr, ACCENT_BLUE, NEUTRAL)

B = "/path/to/project"
M1 = os.path.join(B, "module1_results", "statistics")
EFF = os.path.join(M1, "effect_sizes_full.csv")
SIG = os.path.join(M1, "significance_matrix_clean.csv")
DV = os.path.join(M1, "feature_diagnostic_value.csv")

CAT_ORDER = ["negligible", "minimal", "small", "medium", "large"]
CAT_COL = {"negligible": "#E2E2E2", "minimal": "#BFCFE2", "small": "#8AA6C4",
           "medium": "#5F84AD", "large": "#355C8A"}


def panel_a(ax):
    d = pd.read_csv(EFF, usecols=["effect_category"])
    cats = [c for c in CAT_ORDER if c in set(d["effect_category"].dropna())]
    cats += [c for c in sorted(set(d["effect_category"].dropna())) if c not in cats]
    counts = [int((d["effect_category"] == c).sum()) for c in cats]
    ys = np.arange(len(cats))[::-1]
    for y, c, n in zip(ys, cats, counts):
        ax.barh(y, n, height=0.6, color=CAT_COL.get(c, "#BBBBBB"), edgecolor="none")
        ax.text(n + max(counts) * 0.015, y, str(n), va="center", ha="left", fontsize=5.0,
                color="#333333")
    ax.set_yticks(ys)
    ax.set_yticklabels(cats, fontsize=5.6)
    ax.set_xlabel("Feature x disorder effects")
    ax.set_xlim(0, max(counts) * 1.18)
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Effect-size categories", loc="left", fontsize=7.8)
    panel_label(ax, "a")


def panel_b(ax):
    d = pd.read_csv(EFF, usecols=["feature", "sig_fdr_global"])
    vals = d.groupby("feature")["sig_fdr_global"].sum().values.astype(int)
    ax.hist(vals, bins=np.arange(0, 12) - 0.5, color=ACCENT_BLUE, edgecolor="white",
            lw=0.4)
    ax.set_xlabel("Significant disorders (of 10)")
    ax.set_ylabel("Features")
    ax.set_xticks(range(0, 11))
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Per-feature consistency", loc="left", fontsize=7.8)
    panel_label(ax, "b")


def panel_c(ax):
    d = pd.read_csv(SIG)
    cols = [c for c in DISORDERS if c in d.columns]
    M = d[cols].astype(float).values
    order = np.argsort(-M.sum(axis=1))
    M = M[order]
    ax.imshow(M, cmap=ListedColormap(["#F2F2F2", "#355C8A"]), aspect="auto",
              vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([abbr(c) for c in cols], rotation=90, fontsize=5.4)
    ax.set_yticks([])
    ax.set_ylabel("Features (ordered)")
    ax.tick_params(axis="both", pad=4)
    despine(ax)
    ax.set_title("Significance matrix", loc="left", fontsize=7.8)
    panel_label(ax, "c")


def panel_d(ax):
    d = pd.read_csv(DV, index_col=0)
    v = pd.to_numeric(d["diagnostic_value"], errors="coerce").dropna()
    top = v.sort_values(ascending=False).head(15).iloc[::-1]
    ys = np.arange(len(top))
    ax.barh(ys, top.values, height=0.6, color=ACCENT_BLUE, edgecolor="none")
    ax.set_yticks(ys)
    ax.set_yticklabels([str(f)[:18] for f in top.index], fontsize=5.0)
    ax.set_xlabel("Stored diagnostic value")
    ax.set_xlim(0, top.values.max() * 1.15)
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Top features by diagnostic value", loc="left", fontsize=7.8)
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
    out = os.path.join(ED_DIR, "ed2_feature_robustness")
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
