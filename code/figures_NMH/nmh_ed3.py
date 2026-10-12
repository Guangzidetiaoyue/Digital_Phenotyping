"""NMH Extended Data Figure ED3 - Discrimination detail.

Complements Fig. 2 (aggregate volcano / AUROC forest) with:
  a  normalised multiclass confusion matrix
  b  per-class AUROC (mean, s.d.)
  c  multiclass coefficients (top features x class)
  d  t-SNE embedding by group (controls grey)
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common_nmh import (apply_style, mm2inch, despine, panel_label, ED_DIR, abbr,
                        spectrum_of, SPECTRUM_COLORS, SEQ_CMAP, DIV_CMAP, ACCENT_BLUE,
                        NEUTRAL)

B = "/path/to/project"
M1 = os.path.join(B, "module1_results", "statistics")
CONF = os.path.join(M1, "confusion_matrix_normalized.csv")
MC = os.path.join(M1, "multiclass_results.json")
COEF = os.path.join(M1, "multiclass_coefficients.csv")
TSNE = os.path.join(M1, "tsne_embedding.csv")


def panel_a(ax):
    d = pd.read_csv(CONF, index_col=0)
    labels = [abbr(c) for c in d.index]
    M = d.values.astype(float)
    im = ax.imshow(M, cmap=SEQ_CMAP, aspect="auto", vmin=0, vmax=np.nanmax(M))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5.4)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=5.4)
    cax = ax.inset_axes([1.03, 0.0, 0.045, 0.5])
    cb = ax.figure.colorbar(im, cax=cax)
    cb.set_label("fraction", fontsize=5.0)
    cb.ax.tick_params(labelsize=5.0)
    ax.set_title("Multiclass confusion", loc="left", fontsize=7.8)
    panel_label(ax, "a")


def panel_b(ax):
    j = json.load(open(MC))
    names = j["class_names"]
    per = j["per_class_auroc_mean"]
    std = j["per_class_auroc_std"]
    rows = []
    for k, nm in enumerate(names):
        if isinstance(per, dict):
            rows.append((abbr(nm), float(per[nm]), float(std[nm])))
        else:
            rows.append((abbr(nm), float(per[k]), float(std[k])))
    rows.sort(key=lambda t: t[1])
    ys = np.arange(len(rows))
    for y, (lab, m, s) in zip(ys, rows):
        ax.plot([m - s, m + s], [y, y], color=NEUTRAL, lw=0.9, zorder=2)
        ax.scatter([m], [y], s=18, color=ACCENT_BLUE, zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=5.6)
    ax.set_xlabel("Per-class AUROC")
    mv = [r[1] for r in rows]
    ax.set_xlim(min(mv) - 0.06, max(mv) + 0.06)
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Per-class discrimination", loc="left", fontsize=7.8)
    panel_label(ax, "b")


def panel_c(ax):
    d = pd.read_csv(COEF, index_col=0)
    classes = list(d.index)
    feats = d.abs().max(axis=0).sort_values(ascending=False).head(8).index
    M = d[feats].T.values.astype(float)
    vmax = np.nanmax(np.abs(M))
    ax.imshow(M, cmap=DIV_CMAP, aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([abbr(c) for c in classes], rotation=90, fontsize=5.0)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([str(f)[:16] for f in feats], fontsize=5.0)
    cax = ax.inset_axes([1.03, 0.0, 0.045, 0.5])
    cb = ax.figure.colorbar(ax.images[-1], cax=cax)
    cb.set_label("coefficient", fontsize=5.0)
    cb.ax.tick_params(labelsize=5.0)
    ax.set_title("Multiclass coefficients", loc="left", fontsize=7.8)
    panel_label(ax, "c")


def panel_d(ax):
    d = pd.read_csv(TSNE)
    if len(d) > 4000:
        d = d.sample(4000, random_state=42)
    ctrl = d[d["group"] == "control"]
    ax.scatter(ctrl["tSNE_1"], ctrl["tSNE_2"], s=1.2, color="#CFCFCF", alpha=0.25, lw=0,
               zorder=1)
    for sp in sorted({spectrum_of(x) for x in d["disorder_group"].unique()
                      if x != "control"}):
        ds = [x for x in d["disorder_group"].unique()
              if x != "control" and spectrum_of(x) == sp]
        sub = d[d["disorder_group"].isin(ds)]
        ax.scatter(sub["tSNE_1"], sub["tSNE_2"], s=1.2, color=SPECTRUM_COLORS[sp],
                   alpha=0.30, lw=0, zorder=2)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_xticks([])
    ax.set_yticks([])
    despine(ax)
    ax.set_title("t-SNE embedding", loc="left", fontsize=7.8)
    panel_label(ax, "d")


def main():
    apply_style()
    fig = plt.figure(figsize=(mm2inch(180), mm2inch(150)))
    gs = fig.add_gridspec(2, 2, wspace=0.55, hspace=0.45, left=0.11, right=0.955,
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
    out = os.path.join(ED_DIR, "ed3_discrimination_detail")
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
