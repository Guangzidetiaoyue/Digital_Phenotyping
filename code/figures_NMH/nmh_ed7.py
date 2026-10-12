"""NMH Extended Data Figure ED7 - Trajectory detail (4 panels).
(a) multivariate change-point lead distribution; (b) change-point window-index;
(c) total change-points per user; (d) PELT penalty sensitivity.
Duplicates removed: t0-by-era (was duplicate of Fig. 3e) and lead-time quartiles (was
duplicate of Fig. 3c).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common_nmh import (apply_style, mm2inch, despine, panel_label, ED_DIR, ACCENT_BLUE,
                        NEUTRAL)

M2 = "/path/to/project/module2_results/statistics"
R4 = "/path/to/project/module4_results/validation_r4"


def panel_a(ax):
    d = pd.read_csv(os.path.join(M2, "multivariate_changepoints.csv"))
    v = np.abs(pd.to_numeric(d["day_difference"], errors="coerce").values)
    v = v[np.isfinite(v)]
    ax.hist(v, bins=30, color=ACCENT_BLUE, edgecolor="white", lw=0.3)
    ax.set_xlabel("Absolute change-point lead (days)")
    ax.set_ylabel("Patients")
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Change-point lead time", loc="left", fontsize=7.8)
    panel_label(ax, "a")


def panel_b(ax):
    d = pd.read_csv(os.path.join(M2, "changepoint_analysis.csv"))
    d = d[pd.to_numeric(d["is_pre_t0"], errors="coerce") == 1].dropna(
        subset=["changepoint_window_index"])
    win = pd.to_numeric(d["changepoint_window_index"], errors="coerce").astype(int)
    bins = np.arange(win.min() - 0.5, win.max() + 1.5)
    ax.hist(win, bins=bins, color=ACCENT_BLUE, edgecolor="white", lw=0.3)
    ax.set_xlabel("Window index (pre-t0)")
    ax.set_ylabel("Change-points")
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Pre-t0 window distribution", loc="left", fontsize=7.8)
    panel_label(ax, "b")


def panel_c(ax):
    d = pd.read_csv(os.path.join(M2, "changepoint_analysis.csv"))
    col = pd.to_numeric(d["n_total_changepoints"], errors="coerce")
    col = col.dropna().astype(int).values
    bins = np.arange(0, col.max() + 2) - 0.5
    ax.hist(col, bins=bins, color=ACCENT_BLUE, edgecolor="white", lw=0.3)
    ax.set_xlabel("Change-points per user")
    ax.set_ylabel("Users")
    ax.set_xticks(range(0, int(col.max()) + 1))
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Change-points per user", loc="left", fontsize=7.8)
    panel_label(ax, "c")


def panel_d(ax):
    d = pd.read_csv(os.path.join(M2, "pelt_sensitivity_analysis.csv"))
    x = pd.to_numeric(d["penalty_multiplier"], errors="coerce")
    y1 = pd.to_numeric(d["avg_cps_per_user"], errors="coerce")
    ax.plot(x, y1, "-o", color=ACCENT_BLUE, ms=2.6, lw=1.0)
    ax.set_xlabel("PELT penalty multiplier")
    ax.set_ylabel("Average change-points / user")
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("PELT penalty sensitivity", loc="left", fontsize=7.8)
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
    out = os.path.join(ED_DIR, "ed7_trajectory_detail")
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