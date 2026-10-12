"""NMH Extended Data Figure ED4 - Sensitivity suite.
(a) heterogeneity I^2; (b) significant effects by confound model; (c) mean AUROC by outlier
treatment; (d) subsample-size stability per disorder.
All values are read from the stored M1 result files.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common_nmh import (apply_style, mm2inch, despine, panel_label, ED_DIR, DISORDERS,
                        abbr, spectrum_of, SPECTRUM_COLORS, ACCENT_BLUE, NEUTRAL)

M1 = "/path/to/project/module1_results/statistics"


def panel_a(ax):
    d = pd.read_csv(os.path.join(M1, "sensitivity_heterogeneity.csv"))
    v = pd.to_numeric(d["I_squared"], errors="coerce").dropna().values * 100.0
    ax.hist(v, bins=20, color=ACCENT_BLUE, edgecolor="white", lw=0.4)
    ax.set_xlabel("I squared (%)")
    ax.set_ylabel("Features")
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Heterogeneity", loc="left", fontsize=7.8)
    panel_label(ax, "a")


def panel_b(ax):
    frames = []
    for fn, lab in [("sensitivity_confound_linear.csv", "linear"),
                    ("sensitivity_confound_poly2.csv", "poly2"),
                    ("sensitivity_confound_adjusted.csv", "adjusted")]:
        d = pd.read_csv(os.path.join(M1, fn))
        d["_m"] = lab
        frames.append(d)
    d = pd.concat(frames)
    g = d.groupby("_m")["significant_adjusted"].sum()
    order = ["linear", "poly2", "adjusted"]
    ys = np.arange(len(order))[::-1]
    vals = [int(g.get(o, 0)) for o in order]
    for y, o, n in zip(ys, order, vals):
        ax.barh(y, n, height=0.55, color=ACCENT_BLUE if o == "adjusted" else NEUTRAL)
        ax.text(n + max(vals) * 0.02, y, str(n), va="center", ha="left", fontsize=5.0,
                color="#333333")
    ax.set_yticks(ys)
    ax.set_yticklabels(order, fontsize=5.6)
    ax.set_xlabel("Significant effects")
    ax.set_xlim(0, max(vals) * 1.2)
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Confound models", loc="left", fontsize=7.8)
    panel_label(ax, "b")


def panel_c(ax):
    j = json.load(open(os.path.join(M1, "sensitivity_outlier_strategies.json")))
    keys = list(j.keys())
    vals = [float(j[k]["mean_auroc"]) for k in keys]
    rows = sorted(zip(keys, vals), key=lambda t: t[1])
    ys = np.arange(len(rows))
    for y, (k, v) in zip(ys, rows):
        ax.scatter([v], [y], s=20, color=ACCENT_BLUE, zorder=4)
        ax.text(v + 0.002, y, f"{v:.3f}", va="center", ha="left", fontsize=5.0,
                color="#333333")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0].replace("_", " ") for r in rows], fontsize=5.0)
    ax.set_xlabel("Mean AUROC")
    vv = [r[1] for r in rows]
    ax.set_xlim(min(vv) - 0.004, max(vv) + 0.02)
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Outlier treatment", loc="left", fontsize=7.8)
    panel_label(ax, "c")


def panel_d(ax):
    j = json.load(open(os.path.join(M1, "sensitivity_report.json")))["sample_size_stability"]
    fracs = ["0.25", "0.5", "0.75"]
    for d in DISORDERS:
        if d not in j:
            continue
        c = SPECTRUM_COLORS[spectrum_of(d)]
        ys = [j[d][f]["mean_auroc"] for f in fracs if f in j[d]]
        xs = [float(f) for f in fracs if f in j[d]]
        ax.plot(xs, ys, marker="o", ms=2.2, lw=0.9, color=c, alpha=0.9)
    ax.set_xlabel("Subsample fraction")
    ax.set_ylabel("Mean AUROC")
    ax.set_xticks([0.25, 0.5, 0.75])
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Subsample stability", loc="left", fontsize=7.8)
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
    out = os.path.join(ED_DIR, "ed4_sensitivity_suite")
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
