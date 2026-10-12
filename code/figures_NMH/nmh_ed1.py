"""NMH Extended Data Figure ED1 - Validity detail (leakage, transfer, pairwise separability).

Complements Fig. 2e (leakage dumbbell), Fig. 3f (transfer heatmap) and Fig. 4a (pairwise
AUROC matrix) with finer views:
  a  leakage trial-level AUROC spread per disorder
  b  temporal-transfer delta per disorder (mean and range over era pairs)
  c  all 45 pairwise case-vs-case AUROC with 95% CI
  d  original vs no-diagnosis AUROC scatter

All values are read from the stored R4 / module3 result files.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common_nmh import (apply_style, mm2inch, despine, panel_label, ED_DIR, DISORDERS,
                        abbr, spectrum_of, SPECTRUM_COLORS, ACCENT_BLUE, ACCENT_RED,
                        NEUTRAL)

B = "/path/to/project"
M3 = os.path.join(B, "module3_results", "statistics")
R4 = os.path.join(B, "module4_results", "validation_r4")
LEAK = os.path.join(R4, "r4_leakage_a_report.json")
TRANS = os.path.join(R4, "r4_temporal_split_93train_results.csv")
PAIR = os.path.join(M3, "pairwise_distinguishability.csv")


def panel_a(ax):
    rep = json.load(open(LEAK))["results"]
    by = {r["disorder"]: r for r in rep}
    ds = [d for d in DISORDERS if d in by]
    ds.sort(key=lambda d: by[d]["auroc_nodiag"])
    ys = np.arange(len(ds))
    for y, d in zip(ys, ds):
        r = by[d]
        ax.plot([r["auroc_nodiag_min"], r["auroc_nodiag_max"]], [y, y], color=NEUTRAL,
                lw=2.6, solid_capstyle="round", zorder=2)
        ax.scatter(r["trial_aurocs"], [y] * len(r["trial_aurocs"]), s=6, color="#BBBBBB",
                   edgecolors="none", zorder=3)
        ax.scatter([r["auroc_nodiag"]], [y], s=20, color=ACCENT_BLUE, zorder=4)
        ax.scatter([r["auroc_original"]], [y], s=20, facecolors="none",
                   edgecolors=ACCENT_RED, lw=0.9, zorder=4)
        ax.text(0.964, y, f"{r['delta']:+.3f}", va="center", ha="right", fontsize=5.0,
                color="#333333")
    ax.set_yticks(ys)
    ax.set_yticklabels([abbr(d) for d in ds], fontsize=5.6)
    ax.set_xlabel("AUROC")
    ax.set_xlim(0.82, 0.965)
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Leakage: trials (dot) and original (ring)", loc="left", fontsize=7.8)
    panel_label(ax, "a")


def panel_b(ax):
    d = pd.read_csv(TRANS)
    g = d.groupby("disorder")["delta_vs_atlas"]
    ds = [x for x in DISORDERS if x in g.groups]
    rows = [(x, g.get_group(x).mean(), g.get_group(x).min(), g.get_group(x).max())
            for x in ds]
    rows.sort(key=lambda t: t[1])
    ys = np.arange(len(rows))
    for y, (x, m, lo, hi) in zip(ys, rows):
        ax.plot([lo, hi], [y, y], color=NEUTRAL, lw=0.9, zorder=2)
        c = ACCENT_BLUE if m >= 0 else ACCENT_RED
        ax.scatter([m], [y], s=20, color=c, zorder=4)
        ax.text(0.034, y, f"{m:+.3f}", va="center", ha="right", fontsize=5.0,
                color="#333333")
    ax.axvline(0, color="#999999", lw=0.7, ls=":")
    ax.set_yticks(ys)
    ax.set_yticklabels([abbr(r[0]) for r in rows], fontsize=5.6)
    ax.set_xlabel("Delta AUROC vs atlas (mean, range)")
    ax.set_xlim(-0.035, 0.035)
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Temporal transfer per disorder", loc="left", fontsize=7.8)
    panel_label(ax, "b")


def panel_c(ax):
    d = pd.read_csv(PAIR)
    d = d.sort_values("auroc").reset_index(drop=True)
    ys = np.arange(len(d))
    same = [spectrum_of(a) == spectrum_of(b)
            for a, b in zip(d["disorder_a"], d["disorder_b"])]
    for y, r, s in zip(ys, d.itertuples(), same):
        c = SPECTRUM_COLORS[spectrum_of(r.disorder_a)] if s else NEUTRAL
        ax.plot([r.auroc_ci_low, r.auroc_ci_high], [y, y], color=c, lw=0.7, alpha=0.8,
                zorder=3)
    ax.scatter(ys, d["auroc"], s=6, color=[SPECTRUM_COLORS[spectrum_of(r.disorder_a)]
               if s else NEUTRAL for r, s in zip(d.itertuples(), same)], zorder=4)
    ax.set_xlabel("Pairwise case-vs-case AUROC (95% CI)")
    ax.set_ylabel("Disorder pair (sorted)")
    ax.set_yticks([])
    ax.margins(x=0.06)
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("All 45 disorder pairs", loc="left", fontsize=7.8)
    panel_label(ax, "c")


def panel_d(ax):
    rep = json.load(open(LEAK))["results"]
    by = {r["disorder"]: r for r in rep}
    ds = [d for d in DISORDERS if d in by]
    x = [by[d]["auroc_original"] for d in ds]
    y = [by[d]["auroc_nodiag"] for d in ds]
    ax.plot([0.84, 0.94], [0.84, 0.94], color="#BBBBBB", lw=0.8, ls="--")
    for d, xx, yy in zip(ds, x, y):
        ax.scatter([xx], [yy], s=22, color=SPECTRUM_COLORS[spectrum_of(d)],
                   edgecolors="white", lw=0.5, zorder=4)
    ax.set_xlabel("Original AUROC")
    ax.set_ylabel("No-diagnosis AUROC")
    ax.set_xlim(0.84, 0.94)
    ax.set_ylim(0.84, 0.94)
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Leakage effect: original vs no-diagnosis", loc="left", fontsize=7.8)
    panel_label(ax, "d")


def main():
    apply_style()
    fig = plt.figure(figsize=(mm2inch(180), mm2inch(150)))
    gs = fig.add_gridspec(2, 2, wspace=0.45, hspace=0.42, left=0.10, right=0.965,
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
    out = os.path.join(ED_DIR, "ed1_validity_detail")
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
