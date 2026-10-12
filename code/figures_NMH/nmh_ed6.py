"""NMH Extended Data Figure ED6 - GMM detail (4 panels).
(a) BIC; (b) silhouette; (c) subtype predictors; (d) robustness (seed and bootstrap ARI).
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common_nmh import (apply_style, mm2inch, despine, panel_label, ED_DIR, ACCENT_BLUE,
                        NEUTRAL)

M2 = "/path/to/project/module2_results/statistics"
R4 = "/path/to/project/module4_results/validation_r4"


def _sel():
    return pd.read_csv(os.path.join(M2, "gmm_model_selection.csv"))


def panel_a(ax):
    d = _sel().dropna(subset=["bic"])
    ax.plot(d["n_components"], d["bic"], "-o", color=ACCENT_BLUE, ms=3.0, lw=1.1)
    ax.set_xlabel("Components (k)")
    ax.set_ylabel("BIC")
    ax.set_xticks(sorted(d["n_components"].unique()))
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("BIC", loc="left", fontsize=7.8)
    panel_label(ax, "a")


def panel_b(ax):
    d = _sel().dropna(subset=["silhouette"])
    ax.plot(d["n_components"], d["silhouette"], "-o", color=ACCENT_BLUE, ms=3.0, lw=1.1)
    ax.set_xlabel("Components (k)")
    ax.set_ylabel("Silhouette")
    ax.set_xticks(sorted(d["n_components"].unique()))
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Silhouette", loc="left", fontsize=7.8)
    panel_label(ax, "b")


def panel_c(ax):
    d = pd.read_csv(os.path.join(M2, "subtype_predictors.csv"))
    d = d.sort_values("importance_mean", ascending=False).head(10).iloc[::-1]
    ys = np.arange(len(d))
    for y, (_, r) in zip(ys, d.iterrows()):
        m, s = r["importance_mean"], r["importance_std"]
        ax.plot([max(0, m - s), m + s], [y, y], color=NEUTRAL, lw=0.9, zorder=2)
        ax.scatter([m], [y], s=18, color=ACCENT_BLUE, zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([str(f)[:16] for f in d["feature"]], fontsize=5.0)
    ax.set_xlabel("Predictor importance")
    ax.set_xlim(left=0)
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Subtype predictors", loc="left", fontsize=7.8)
    panel_label(ax, "c")


def panel_d(ax):
    g = json.load(open(os.path.join(R4, "r4_gmm_robustness_results.json")))
    seeds = g["k2_init_seed_ari"]["seeds"]
    ari_seed = g["k2_init_seed_ari"]["ari_vs_2025"]
    boot = g["k2_bootstrap_ari"]
    ax.scatter(ari_seed, np.zeros(len(ari_seed)), s=24, color=ACCENT_BLUE, zorder=4,
               label="init seed")
    ax.scatter([boot["mean"]], [0.35], s=24, color=NEUTRAL, zorder=4,
               label="bootstrap mean")
    ax.plot([boot["min"], boot["mean"] + 2 * boot["std"]], [0.35, 0.35], color=NEUTRAL,
            lw=0.9, zorder=3)
    ax.set_yticks([0, 0.35])
    ax.set_yticklabels(["seed", "bootstrap"], fontsize=5.6)
    ax.set_ylim(-0.4, 0.8)
    ax.set_xlabel("Adjusted Rand index")
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Subtype robustness (ARI)", loc="left", fontsize=7.8)
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
    out = os.path.join(ED_DIR, "ed6_gmm_detail")
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
